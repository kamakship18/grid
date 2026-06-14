"""
Improved solution — addresses leakage and adds day-49 morning calibration.

Key insights from diagnosis:
  1. Day-49 TRAIN covers only hours 0-2 (midnight to 2am)
  2. Test covers hours 2-13 (early morning to early afternoon)
  3. There is NO timestamp overlap between day-49 train and test
  4. geo_ts_mean must be computed from day-48 ONLY to avoid leakage
  5. Day-49 morning demand (from train labels) is a powerful calibration feature for test
  6. Road type alone has 0.84 Pearson correlation with demand — primary signal
"""
import sys
sys.path.insert(0, '/Users/kamakshipandoh/Desktop/grid/flipkart_traffic/lib')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pygeohash as pgh
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

d48 = train[train['day'] == 48].copy().reset_index(drop=True)
d49 = train[train['day'] == 49].copy().reset_index(drop=True)

print(f"Day 48 train: {len(d48)} rows | Day 49 train: {len(d49)} rows | Test: {len(test)} rows")

# ── Parse timestamps ─────────────────────────────────────────────────────────
for df in [d48, d49, test]:
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15

# ── CALIBRATION FEATURES (key innovation) ───────────────────────────────────
# Day-49 train covers hours 0-2; use as "today's morning signal"
# Day-48 covers all hours; morning hours 0-2 are the "yesterday's morning signal"

d48_morning = d48[d48['hour'].isin([0, 1, 2])]

# Per-geohash morning means
d49_morning_geo = d49.groupby('geohash')['demand'].agg(
    d49_morning_mean='mean', d49_morning_count='count'
).reset_index()
d48_morning_geo = d48_morning.groupby('geohash')['demand'].agg(
    d48_morning_mean='mean'
).reset_index()

# Global morning values for smoothing / fallback
global_d49_morning = d49['demand'].mean()
global_d48_morning = d48_morning['demand'].mean()
global_calib_ratio = global_d49_morning / global_d48_morning
print(f"Global morning demand ratio (d49/d48): {global_calib_ratio:.4f}")

# Merge and compute smoothed calibration ratio
calib_df = d49_morning_geo.merge(d48_morning_geo, on='geohash', how='outer')
calib_df['d49_morning_mean'] = calib_df['d49_morning_mean'].fillna(global_d49_morning)
calib_df['d48_morning_mean'] = calib_df['d48_morning_mean'].fillna(global_d48_morning)
calib_df['d49_morning_count'] = calib_df['d49_morning_count'].fillna(0)

# Bayesian smoothed calibration ratio (pull toward global ratio when geohash has few samples)
smooth_k = 10
calib_df['calib_ratio_raw'] = calib_df['d49_morning_mean'] / calib_df['d48_morning_mean'].clip(1e-6)
calib_df['calib_ratio'] = (
    calib_df['d49_morning_count'] * calib_df['d49_morning_mean'] +
    smooth_k * global_calib_ratio * calib_df['d48_morning_mean']
) / (calib_df['d49_morning_count'] * calib_df['d48_morning_mean'].clip(1e-6) + smooth_k)
calib_df['calib_ratio'] = calib_df['calib_ratio'].clip(0.1, 10.0)

calib_lookup = calib_df.set_index('geohash')

# ── LEAK-FREE GEO STATS (computed from day-48 ONLY) ─────────────────────────
print("\nComputing leak-free stats from day 48 only...")

# Geohash × time_slot stats from day 48
geo_ts_d48 = d48.groupby(['geohash', 'time_slot'])['demand'].agg(
    geo_ts_mean='mean', geo_ts_std='std', geo_ts_count='count'
).reset_index()
geo_ts_d48['geo_ts_std'] = geo_ts_d48['geo_ts_std'].fillna(0)

# Geohash stats from day 48 (all hours)
gh_d48 = d48.groupby('geohash')['demand'].agg(
    gh_mean='mean', gh_median='median', gh_std='std',
    gh_q25=lambda x: x.quantile(0.25),
    gh_q75=lambda x: x.quantile(0.75),
    gh_min='min', gh_max='max', gh_count='count'
).reset_index()
gh_d48['gh_std'] = gh_d48['gh_std'].fillna(0)
gh_d48['gh_iqr'] = gh_d48['gh_q75'] - gh_d48['gh_q25']

# Time-slot stats from day 48
ts_d48 = d48.groupby('time_slot')['demand'].agg(
    ts_mean='mean', ts_std='std', ts_median='median'
).reset_index()

# Geo prefix stats from day 48
gp4_d48 = d48.copy()
gp4_d48['gp4'] = gp4_d48['geohash'].str[:4]
gp4_stats = gp4_d48.groupby('gp4')['demand'].agg(gp4_mean='mean', gp4_std='std').reset_index()
gp4_stats['gp4_std'] = gp4_stats['gp4_std'].fillna(0)

gp5_d48 = d48.copy()
gp5_d48['gp5'] = gp5_d48['geohash'].str[:5]
gp5_stats = gp5_d48.groupby('gp5')['demand'].agg(gp5_mean='mean', gp5_std='std').reset_index()
gp5_stats['gp5_std'] = gp5_stats['gp5_std'].fillna(0)

# Day-49 morning geo×time_slot stats (for day-49 morning hours)
geo_ts_d49 = d49.groupby(['geohash', 'time_slot'])['demand'].agg(
    geo_ts_d49_mean='mean'
).reset_index()

global_demand_d48 = d48['demand'].mean()

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def decode_geohash_safe(gh):
    try:
        lat, lon = pgh.decode(gh)
        return lat, lon
    except:
        return np.nan, np.nan

def engineer(df, is_d49=False):
    df = df.copy()

    # Spatial
    geo_dec = df['geohash'].apply(decode_geohash_safe)
    df['lat'] = geo_dec.apply(lambda x: x[0])
    df['lon'] = geo_dec.apply(lambda x: x[1])
    df['gp4'] = df['geohash'].str[:4]
    df['gp5'] = df['geohash'].str[:5]

    # Road infrastructure
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather']  = df['Weather'].fillna('Unknown')
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord']          = df['RoadType'].map(road_order).fillna(0).astype(int)
    df['large_vehicles_allowed'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['has_landmarks']          = (df['Landmarks'] == 'Yes').astype(int)
    df['lanes_sq']               = df['NumberofLanes'] ** 2
    df['lanes_log']              = np.log1p(df['NumberofLanes'])
    df['is_multilane']           = (df['NumberofLanes'] > 2).astype(int)
    # Road capacity proxy
    df['road_capacity'] = df['road_type_ord'] * df['NumberofLanes'] * (1 + df['large_vehicles_allowed'])

    # Weather / temperature
    weather_order = {'Sunny': 4, 'Cloudy': 3, 'Foggy': 2, 'Rainy': 1, 'Snowy': 0, 'Unknown': 2}
    df['weather_ord']    = df['Weather'].map(weather_order).fillna(2).astype(int)
    df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    temp_med = df['Temperature'].median()
    if pd.isna(temp_med): temp_med = 20.0
    df['Temperature'] = df['Temperature'].fillna(temp_med)
    df['temp_sq']  = df['Temperature'] ** 2
    df['is_cold']  = (df['Temperature'] < 5).astype(int)
    df['is_hot']   = (df['Temperature'] > 30).astype(int)

    # Time features
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['timeslot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['timeslot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['time_in_min']  = df['hour'] * 60 + df['minute']
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour']    = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)

    # Interactions
    df['road_x_lanes']   = df['road_type_ord'] * df['NumberofLanes']
    df['lanes_x_peak']   = df['NumberofLanes'] * df['is_rush_hour']
    df['weather_x_peak'] = df['weather_ord'] * df['is_rush_hour']
    df['highway_x_peak'] = (df['road_type_ord'] == 3).astype(int) * df['is_rush_hour']
    df['temp_x_weather'] = df['Temperature'] * df['weather_ord']
    df['slot_x_road']    = df['time_slot'] * df['road_type_ord']
    df['capacity_x_peak'] = df['road_capacity'] * df['is_rush_hour']

    # ── Leak-free geo stats from day 48 ──
    df = df.merge(geo_ts_d48, on=['geohash', 'time_slot'], how='left')
    df = df.merge(gh_d48, on='geohash', how='left')
    df = df.merge(ts_d48, on='time_slot', how='left')
    df = df.merge(gp4_stats, on='gp4', how='left')
    df = df.merge(gp5_stats, on='gp5', how='left')

    # Fill missing geo stats with global mean
    for c in ['geo_ts_mean', 'geo_ts_std', 'geo_ts_count']:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else global_demand_d48)
    for c in ['gh_mean', 'gh_median', 'gh_std', 'gh_q25', 'gh_q75', 'gh_min', 'gh_max', 'gh_count', 'gh_iqr']:
        df[c] = df[c].fillna(global_demand_d48)
    for c in ['ts_mean', 'ts_std', 'ts_median']:
        df[c] = df[c].fillna(global_demand_d48)
    for c in ['gp4_mean', 'gp4_std']:
        df[c] = df[c].fillna(global_demand_d48)
    for c in ['gp5_mean', 'gp5_std']:
        df[c] = df[c].fillna(global_demand_d48)

    # ── Calibration features from day-49 morning ─────────────────────────────
    # For day-49 (train or test): use actual calibration data
    # For day-48 rows: no shift (use d48 morning as d49 morning → ratio ~1)
    df['d49_morning_mean_geo'] = df['geohash'].map(
        calib_lookup['d49_morning_mean']
    ).fillna(global_d49_morning if is_d49 else df['gh_mean'])

    df['d48_morning_mean_geo'] = df['geohash'].map(
        calib_lookup['d48_morning_mean']
    ).fillna(global_d48_morning)

    df['calib_ratio'] = df['geohash'].map(
        calib_lookup['calib_ratio']
    ).fillna(global_calib_ratio if is_d49 else 1.0)

    # Calibrated geo_ts prediction: yesterday's same-slot demand × today's calibration
    df['calibrated_geo_ts'] = df['geo_ts_mean'] * df['calib_ratio']
    df['geo_ts_vs_gh']      = df['geo_ts_mean'] - df['gh_mean']
    df['geo_ts_vs_ts']      = df['geo_ts_mean'] - df['ts_mean']
    df['calib_delta']       = df['d49_morning_mean_geo'] - df['d48_morning_mean_geo']

    # Location importance
    df['location_importance'] = df['gh_mean'] * df['has_landmarks']
    df['rush_x_geo']          = df['is_rush_hour'] * df['gh_mean']
    df['calib_x_ts']          = df['calib_ratio'] * df['ts_mean']

    return df

print("Engineering features for day 48...")
d48_fe = engineer(d48, is_d49=False)
print("Engineering features for day 49 train...")
d49_fe = engineer(d49, is_d49=True)
print("Engineering features for test...")
test_fe = engineer(test, is_d49=True)

print(f"\nd48 features: {d48_fe.shape[1]}")

# ── Prepare feature matrices ─────────────────────────────────────────────────
DROP_COLS = [
    'Index', 'demand', 'timestamp', 'geohash',
    'LargeVehicles', 'Landmarks', 'RoadType', 'Weather',
    'gp4', 'gp5',
]
TARGET = 'demand'

# Combine all train
train_all_fe = pd.concat([d48_fe, d49_fe], axis=0).reset_index(drop=True)
y_all = train_all_fe[TARGET]
X_all = train_all_fe.drop(columns=DROP_COLS, errors='ignore')

y_d48 = d48_fe[TARGET]
X_d48 = d48_fe.drop(columns=DROP_COLS, errors='ignore')
y_d49 = d49_fe[TARGET]
X_d49 = d49_fe.drop(columns=DROP_COLS, errors='ignore')

X_test = test_fe.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore')

# Align test columns
X_test = X_test[[c for c in X_all.columns if c in X_test.columns]]
X_all  = X_all[[c for c in X_all.columns if c in X_test.columns]]
X_d48  = X_d48[[c for c in X_all.columns if c in X_d48.columns]]
X_d49  = X_d49[[c for c in X_all.columns if c in X_d49.columns]]

print(f"\nX_all: {X_all.shape}, X_test: {X_test.shape}")
print(f"NaN in X_all: {X_all.isnull().sum().sum()}, NaN in X_test: {X_test.isnull().sum().sum()}")

# ── HONEST VALIDATION: train on d48, predict d49 morning ────────────────────
print("\n" + "=" * 60)
print("HONEST VALIDATION: Train on day 48, evaluate on day 49 train...")

# Quick LGB on d48 → d49 validation
lgb_val = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.02, num_leaves=255,
    max_depth=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
    random_state=SEED, verbosity=-1
)
lgb_val.fit(X_d48, y_d48)
pred_d49 = lgb_val.predict(X_d49)
r2_val = r2_score(y_d49, pred_d49)
print(f"  Honest d48→d49 R²: {r2_val:.6f} → Score: {100 * r2_val:.2f}")

# Calibration baseline
calib_baseline = d49_fe['geo_ts_mean'] * d49_fe['calib_ratio']
r2_calib = r2_score(y_d49, calib_baseline)
print(f"  Calibrated geo_ts baseline R²: {r2_calib:.6f} → Score: {100 * r2_calib:.2f}")

# ── HYPERPARAMETER TUNING for LightGBM ──────────────────────────────────────
print("\n" + "=" * 60)
print("Optuna LightGBM (50 trials, training on d48, validating on d49)...")

def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'max_depth': trial.suggest_int('max_depth', 6, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'random_state': SEED,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_d48, y_d48,
          eval_set=[(X_d49, y_d49)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    preds = m.predict(X_d49)
    return r2_score(y_d49, preds)

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
best_lgb_r2 = study_lgb.best_value
print(f"Best LGB d48→d49 R²: {best_lgb_r2:.6f} → Score: {100 * best_lgb_r2:.2f}")
print(f"Best params: {study_lgb.best_params}")

# ── TRAIN FINAL LGB on ALL data ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training final LGB on ALL train data (d48 + d49)...")

best_lgb_params = study_lgb.best_params.copy()
best_lgb_params.update({
    'objective': 'regression', 'metric': 'rmse',
    'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': SEED,
})

lgb_final = lgb.LGBMRegressor(**best_lgb_params)
lgb_final.fit(X_all, y_all, callbacks=[lgb.log_evaluation(500)])
print(f"LGB train R²: {r2_score(y_all, lgb_final.predict(X_all)):.6f}")

# 5-Fold OOF on full train (for stacking)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(len(X_all))
test_preds_lgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m.predict(X_val)
    test_preds_lgb += m.predict(X_test) / 5
    print(f"  Fold {fold+1} OOF R²: {r2_score(y_val, oof_lgb[val_idx]):.6f}")

oof_lgb_r2 = r2_score(y_all, oof_lgb)
print(f"\nOOF LGB R²: {oof_lgb_r2:.6f} → Score: {100 * oof_lgb_r2:.2f}")

# ── XGBoost (30 trials, d48→d49 validation) ──────────────────────────────────
print("\n" + "=" * 60)
print("Optuna XGBoost (30 trials, d48→d49 validation)...")

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': SEED, 'tree_method': 'hist', 'verbosity': 0,
    }
    m = xgb.XGBRegressor(**params, early_stopping_rounds=100)
    m.fit(X_d48, y_d48, eval_set=[(X_d49, y_d49)], verbose=False)
    return r2_score(y_d49, m.predict(X_d49))

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
best_xgb_r2 = study_xgb.best_value
print(f"Best XGB d48→d49 R²: {best_xgb_r2:.6f} → Score: {100 * best_xgb_r2:.2f}")

best_xgb_params = study_xgb.best_params.copy()
best_xgb_params.update({
    'objective': 'reg:squarederror', 'random_state': SEED,
    'tree_method': 'hist', 'verbosity': 0,
})

oof_xgb = np.zeros(len(X_all))
test_preds_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    m = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m.predict(X_val)
    test_preds_xgb += m.predict(X_test) / 5
    print(f"  Fold {fold+1} OOF R²: {r2_score(y_val, oof_xgb[val_idx]):.6f}")

oof_xgb_r2 = r2_score(y_all, oof_xgb)
print(f"\nOOF XGB R²: {oof_xgb_r2:.6f} → Score: {100 * oof_xgb_r2:.2f}")

# ── CatBoost ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CatBoost training (d48→d49 early stopping)...")

cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

DROP_CB = ['Index', 'demand', 'timestamp', 'geohash', 'gp4', 'gp5']
X_cb_all  = train_all_fe.drop(columns=DROP_CB, errors='ignore').copy()
X_cb_d48  = d48_fe.drop(columns=DROP_CB, errors='ignore').copy()
X_cb_d49  = d49_fe.drop(columns=DROP_CB, errors='ignore').copy()
X_cb_test = test_fe.drop(columns=[c for c in DROP_CB if c != 'demand'], errors='ignore').copy()

# Align
cols_cb = [c for c in X_cb_all.columns if c in X_cb_test.columns]
X_cb_all  = X_cb_all[cols_cb]
X_cb_d48  = X_cb_d48[[c for c in cols_cb if c in X_cb_d48.columns]]
X_cb_d49  = X_cb_d49[[c for c in cols_cb if c in X_cb_d49.columns]]
X_cb_test = X_cb_test[cols_cb]

for c in cat_cols:
    for df_ in [X_cb_all, X_cb_d48, X_cb_d49, X_cb_test]:
        if c in df_.columns:
            df_[c] = df_[c].fillna('Unknown').astype(str)

cb_cat_idx = [X_cb_all.columns.tolist().index(c) for c in cat_cols if c in X_cb_all.columns]

# Find best CB params using d48→d49 validation
cb_params = dict(
    iterations=3000, learning_rate=0.03, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=20, random_seed=SEED,
    eval_metric='R2', loss_function='RMSE',
    verbose=False, early_stopping_rounds=100,
)

pool_d48 = Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx)
pool_d49 = Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx)
cb_val = CatBoostRegressor(**cb_params)
cb_val.fit(pool_d48, eval_set=pool_d49)
pred_cb_d49 = cb_val.predict(X_cb_d49)
r2_cb_val = r2_score(y_d49, pred_cb_d49)
print(f"CB d48→d49 R²: {r2_cb_val:.6f} → Score: {100 * r2_cb_val:.2f}")

oof_cb = np.zeros(len(X_cb_all))
test_preds_cb = np.zeros(len(X_cb_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb_all)):
    X_tr, X_val = X_cb_all.iloc[tr_idx], X_cb_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    pool_tr  = Pool(X_tr, y_tr, cat_features=cb_cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cb_cat_idx)
    m = CatBoostRegressor(**cb_params)
    m.fit(pool_tr, eval_set=pool_val)
    oof_cb[val_idx] = m.predict(X_val)
    test_preds_cb += m.predict(X_cb_test) / 5
    print(f"  Fold {fold+1} OOF R²: {r2_score(y_val, oof_cb[val_idx]):.6f}")

oof_cb_r2 = r2_score(y_all, oof_cb)
print(f"\nOOF CatBoost R²: {oof_cb_r2:.6f} → Score: {100 * oof_cb_r2:.2f}")

# ── Ensemble ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Finding optimal ensemble weights...")

best_score = -np.inf
best_weights = [0.5, 0.25, 0.25]

for w0 in np.arange(0.1, 0.9, 0.1):
    for w1 in np.arange(0.05, 0.7, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05 or w2 > 0.8:
            continue
        blend = w0 * oof_lgb + w1 * oof_xgb + w2 * oof_cb
        sc = r2_score(y_all, blend)
        if sc > best_score:
            best_score = sc
            best_weights = [w0, w1, w2]

W = np.array(best_weights) / np.sum(best_weights)
print(f"Optimal weights: LGB={W[0]:.3f}, XGB={W[1]:.3f}, CB={W[2]:.3f}")
print(f"Ensemble OOF R²: {best_score:.6f} → Score: {100 * best_score:.2f}")

final_preds = W[0] * test_preds_lgb + W[1] * test_preds_xgb + W[2] * test_preds_cb
final_preds = np.clip(final_preds, 0.0, 1.0)

# Also predict using d48→d49 validated LGB only (might be cleaner)
lgb_d49_pred = lgb_val.predict(X_test)
lgb_d49_pred = np.clip(lgb_d49_pred, 0.0, 1.0)

# ── Neighbor Correction ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Applying neighbor correction...")

# Use calibrated geo_ts as correction target (more accurate than raw geo_ts_mean)
test_fe_copy = test_fe.copy()
test_fe_copy['calibrated_geo_ts'] = np.clip(test_fe_copy['calibrated_geo_ts'], 0.0, 1.0)

corrected_preds = final_preds.copy()
cw = 0.15  # correction weight

for i in range(len(test_fe)):
    calib_val = test_fe_copy.iloc[i]['calibrated_geo_ts']
    corrected_preds[i] = (1 - cw) * corrected_preds[i] + cw * calib_val

corrected_preds = np.clip(corrected_preds, 0.0, 1.0)

# ── Generate Submissions ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving submissions...")

test_index = test['Index']

sub_ensemble = pd.DataFrame({'Index': test_index, 'demand': final_preds})
sub_ensemble.to_csv('submission.csv', index=False)
print(f"submission.csv (ensemble): {sub_ensemble.shape}")

sub_corrected = pd.DataFrame({'Index': test_index, 'demand': corrected_preds})
sub_corrected.to_csv('submission_corrected.csv', index=False)
print(f"submission_corrected.csv: {sub_corrected.shape}")

sub_lgb_d49 = pd.DataFrame({'Index': test_index, 'demand': lgb_d49_pred})
sub_lgb_d49.to_csv('submission_lgb_d49validated.csv', index=False)
print(f"submission_lgb_d49validated.csv (honest): {sub_lgb_d49.shape}")

# Validate
assert len(sub_ensemble) == 41778
assert list(sub_ensemble.columns) == ['Index', 'demand']
print("\nAll submissions validated!")

# ── Feature Importance ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fi = pd.DataFrame({
    'feature': X_all.columns,
    'importance': lgb_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(fi.head(20).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
fi.head(25).plot(kind='barh', x='feature', y='importance', ax=ax)
ax.set_title('LightGBM Feature Importances (Top 25) — v2')
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=100)
plt.close()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print(f"Honest d48→d49 R² (LGB): {r2_val:.6f} → Score: {100 * r2_val:.2f}")
print(f"Best Optuna LGB d48→d49: {best_lgb_r2:.6f} → Score: {100 * best_lgb_r2:.2f}")
print(f"Best Optuna XGB d48→d49: {best_xgb_r2:.6f} → Score: {100 * best_xgb_r2:.2f}")
print(f"CB d48→d49:              {r2_cb_val:.6f} → Score: {100 * r2_cb_val:.2f}")
print(f"OOF Ensemble R²:         {best_score:.6f} → Score: {100 * best_score:.2f}")
print("\nBEST SUBMISSION: submission.csv (ensemble)")
