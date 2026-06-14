"""
v5 — Clean two-stage with smarter Stage 2.

Diagnosis from v4 results:
  - Stage 1 alone: 86.05% (WEAK — calibration features contaminated Stage 1)
  - Two-stage: 90.32% (Stage 2 added +4.27%)
  - v3 best: 89.97% (no two-stage, but cleaner Stage 1)
  - Theoretical ceiling: 94.63%

Root cause: v4 Stage 1 included calibration features (calib_delta, calib_k1/5/30 etc.)
  which are computed from d49 data. The model overfits to them in d48 OOF but they
  confuse cross-day predictions. Stage 1 scored only 86.05% (vs v3's ~90%).

Fix:
  Stage 1: CLEAN model — only d48-generalizable features (road, time, weather, geo_ts_mean)
           Train on d48 ONLY (honest cross-day base)
  Stage 2: Smarter calibration correction model
           - Training data: d48 rows (target=0) + d49 rows (target=actual resid)
           - d49 upweighted 50x so morning signal dominates but daytime≈0 constrains
           - Explicit decay feature: correction decreases for later time_slots
           - This teaches Stage 2 WHEN to apply calibration (morning=large, daytime=small)

Expected: Stage1 ≈ 89-90% + Stage2 ≈ +3-4% = 92-94%
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

print("=" * 60)
print("Loading data...")
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

d48 = train[train['day'] == 48].copy().reset_index(drop=True)
d49 = train[train['day'] == 49].copy().reset_index(drop=True)

for df in [d48, d49, test]:
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['RoadType'] = df['RoadType'].fillna('Unknown')

print(f"Day 48: {len(d48)} rows | Day 49: {len(d49)} rows | Test: {len(test)} rows")

# ── CALIBRATION (for Stage 2 features) ──────────────────────────────────────
d48_morning = d48[d48['hour'].isin([0, 1, 2])].copy()

# Road-type calibration
rt_calib = {}
for rt in list(d49['RoadType'].unique()) + ['Unknown']:
    d48_rt = d48_morning[d48_morning['RoadType'] == rt]['demand'].mean()
    d49_rt = d49[d49['RoadType'] == rt]['demand'].mean()
    if pd.notna(d48_rt) and pd.notna(d49_rt) and d48_rt > 0:
        rt_calib[rt] = np.clip(d49_rt / d48_rt, 0.2, 5.0)
    else:
        rt_calib[rt] = 1.0
global_rt_calib = d49['demand'].mean() / d48_morning['demand'].mean()

# Road-type additive delta
rt_delta = {}
for rt in rt_calib:
    d48_lvl = d48_morning[d48_morning['RoadType'] == rt]['demand'].mean()
    d49_lvl = d49[d49['RoadType'] == rt]['demand'].mean()
    rt_delta[rt] = (d49_lvl - d48_lvl) if (pd.notna(d48_lvl) and pd.notna(d49_lvl)) else 0.0

print(f"RT calibration: {rt_calib}")

# Per-geohash stats
d49_geo  = d49.groupby('geohash')['demand'].agg(['mean', 'count', 'std']).rename(
    columns={'mean': 'm', 'count': 'n', 'std': 's'})
d48_geo_morn = d48_morning.groupby('geohash')['demand'].mean()
d48_geo_all  = d48.groupby('geohash')['demand'].mean()
geo_rt       = d48.groupby('geohash')['RoadType'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
global_d48   = d48['demand'].mean()
global_d48_morn = d48_morning['demand'].mean()

# Multi-resolution multiplicative calibration
def compute_calib(k):
    c = {}
    for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
        rt   = geo_rt.get(gh, 'Unknown')
        prior = rt_calib.get(rt, global_rt_calib)
        if gh not in d49_geo.index:
            c[gh] = prior; continue
        n   = d49_geo.loc[gh, 'n']
        m49 = d49_geo.loc[gh, 'm']
        m48 = d48_geo_morn.get(gh, global_d48_morn)
        if m48 < 0.005:
            c[gh] = prior; continue
        raw = np.clip(m49 / m48, 0.05, 20.0)
        c[gh] = np.clip((n * raw + k * prior) / (n + k), 0.05, 20.0)
    return c

# Additive calibration delta per geohash
def compute_delta():
    c = {}
    for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
        rt    = geo_rt.get(gh, 'Unknown')
        prior = rt_delta.get(rt, 0.0)
        if gh not in d49_geo.index:
            c[gh] = prior; continue
        n   = d49_geo.loc[gh, 'n']
        m49 = d49_geo.loc[gh, 'm']
        m48 = d48_geo_morn.get(gh, global_d48_morn)
        raw = m49 - m48
        c[gh] = (n * raw + 3 * prior) / (n + 3)
    return c

print("Computing calibration...")
calib_k1  = compute_calib(k=1)
calib_k5  = compute_calib(k=5)
calib_k30 = compute_calib(k=30)
calib_delta_geo = compute_delta()

# ── D48 LOOKUP TABLES ────────────────────────────────────────────────────────
geo_ts_d48_df = d48.groupby(['geohash', 'time_slot'])['demand'].agg(
    geo_ts_mean='mean', geo_ts_std='std').reset_index().fillna({'geo_ts_std': 0})
gh_d48_df = d48.groupby('geohash')['demand'].agg(
    gh_mean='mean', gh_std='std',
    gh_q25=lambda x: x.quantile(0.25),
    gh_q75=lambda x: x.quantile(0.75)).reset_index().fillna(0)
gh_d48_df['gh_iqr'] = gh_d48_df['gh_q75'] - gh_d48_df['gh_q25']
ts_d48_df = d48.groupby('time_slot')['demand'].agg(ts_mean='mean').reset_index()
rt_ts_d48_df = d48.groupby(['RoadType', 'time_slot'])['demand'].agg(rt_ts_mean='mean').reset_index()
d48_morning_gh_df = d48_morning.groupby('geohash')['demand'].agg(
    d48_morning_mean='mean', d48_morning_std='std').reset_index().fillna({'d48_morning_std': 0})

geo_ts_d48_dict = d48.groupby(['geohash', 'time_slot'])['demand'].mean().to_dict()
gh_d48_dict = d48.groupby('geohash')['demand'].mean().to_dict()

def decode_gh(gh):
    try:
        lat, lon = pgh.decode(gh)
        return lat, lon
    except:
        return np.nan, np.nan

# ── STAGE 1 FEATURES (pure d48 features — NO calibration) ───────────────────
def engineer_s1(df):
    df = df.copy()
    geo_dec = df['geohash'].apply(decode_gh)
    df['lat'] = geo_dec.apply(lambda x: x[0])
    df['lon'] = geo_dec.apply(lambda x: x[1])
    df['gp4'] = df['geohash'].str[:4]
    df['gp5'] = df['geohash'].str[:5]

    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather']  = df['Weather'].fillna('Unknown')
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord']  = df['RoadType'].map(road_order).fillna(0).astype(int)
    df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['has_landmarks']  = (df['Landmarks'] == 'Yes').astype(int)
    df['lanes_sq']       = df['NumberofLanes'] ** 2
    df['lanes_log']      = np.log1p(df['NumberofLanes'])
    df['road_capacity']  = df['road_type_ord'] * df['NumberofLanes'] * (1 + df['large_vehicles'])
    df['is_highway']     = (df['road_type_ord'] == 3).astype(int)

    weather_order = {'Sunny': 4, 'Cloudy': 3, 'Foggy': 2, 'Rainy': 1, 'Snowy': 0, 'Unknown': 2}
    df['weather_ord']    = df['Weather'].map(weather_order).fillna(2).astype(int)
    df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    temp_med = df['Temperature'].median()
    df['Temperature']    = df['Temperature'].fillna(temp_med if not pd.isna(temp_med) else 20.0)
    df['temp_sq']        = df['Temperature'] ** 2
    df['is_cold']        = (df['Temperature'] < 5).astype(int)
    df['is_hot']         = (df['Temperature'] > 30).astype(int)

    df['hour_sin']        = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']        = np.cos(2 * np.pi * df['hour'] / 24)
    df['timeslot_sin']    = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['timeslot_cos']    = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour']    = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)

    df['road_x_lanes']    = df['road_type_ord'] * df['NumberofLanes']
    df['lanes_x_peak']    = df['NumberofLanes'] * df['is_rush_hour']
    df['highway_x_peak']  = df['is_highway'] * df['is_rush_hour']
    df['weather_x_peak']  = df['weather_ord'] * df['is_rush_hour']
    df['capacity_x_peak'] = df['road_capacity'] * df['is_rush_hour']
    df['slot_x_road']     = df['time_slot'] * df['road_type_ord']
    df['temp_x_weather']  = df['Temperature'] * df['weather_ord']

    df = df.merge(geo_ts_d48_df, on=['geohash', 'time_slot'], how='left')
    df = df.merge(gh_d48_df, on='geohash', how='left')
    df = df.merge(ts_d48_df, on='time_slot', how='left')
    df = df.merge(rt_ts_d48_df, on=['RoadType', 'time_slot'], how='left')
    df = df.merge(d48_morning_gh_df, on='geohash', how='left')

    df['geo_ts_mean']    = df['geo_ts_mean'].fillna(global_d48)
    df['geo_ts_std']     = df['geo_ts_std'].fillna(0)
    df['gh_mean']        = df['gh_mean'].fillna(global_d48)
    df['gh_std']         = df['gh_std'].fillna(0)
    df['gh_iqr']         = df['gh_iqr'].fillna(0)
    df['ts_mean']        = df['ts_mean'].fillna(global_d48)
    df['rt_ts_mean']     = df['rt_ts_mean'].fillna(global_d48)
    df['d48_morning_mean'] = df['d48_morning_mean'].fillna(global_d48_morn)

    # Geo deviations
    df['geo_ts_vs_gh']  = df['geo_ts_mean'] - df['gh_mean']
    df['geo_ts_vs_ts']  = df['geo_ts_mean'] - df['ts_mean']
    df['rt_ts_vs_ts']   = df['rt_ts_mean']  - df['ts_mean']
    df['gh_vs_rt']      = df['gh_mean']     - df['rt_ts_mean']
    df['rush_x_geo']    = df['is_rush_hour'] * df['gh_mean']
    return df

# ── STAGE 2 FEATURES (calibration-focused) ──────────────────────────────────
def engineer_s2(df):
    df = df.copy()
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord']  = df['RoadType'].map(road_order).fillna(0).astype(int)
    df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['is_highway']     = (df['road_type_ord'] == 3).astype(int)
    df['road_capacity']  = df['road_type_ord'] * df['NumberofLanes'] * (1 + df['large_vehicles'])

    df['hour_sin']        = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']        = np.cos(2 * np.pi * df['hour'] / 24)
    df['timeslot_sin']    = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['timeslot_cos']    = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['is_rush_hour']    = (((df['hour'] >= 7) & (df['hour'] <= 10)) |
                               ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
    df['is_morning_win']  = (df['hour'] <= 2).astype(int)  # d49 morning window
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # KEY: explicit decay feature (correction decays for later time_slots)
    # Morning (slot 0-8) = 1.0, midday (slot 48) ≈ 0.5, evening (slot 72) ≈ 0.2
    df['calib_decay'] = np.exp(-df['time_slot'] * 0.02)

    # Calibration features
    df['calib_k1']       = df['geohash'].map(calib_k1).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_k5']       = df['geohash'].map(calib_k5).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_k30']      = df['geohash'].map(calib_k30).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_rt']       = df['RoadType'].map(rt_calib).fillna(global_rt_calib)
    df['calib_delta_gh'] = df['geohash'].map(calib_delta_geo).fillna(
        df['RoadType'].map(rt_delta).fillna(0.0))

    df['d49_morn_mean']  = df['geohash'].map(d49_geo['m']).fillna(0.0)
    df['d49_morn_count'] = df['geohash'].map(d49_geo['n']).fillna(0.0)
    df['d48_morn_mean']  = df['geohash'].map(d48_geo_morn).fillna(global_d48_morn)
    df['gh_mean_d48']    = df['geohash'].map(d48_geo_all).fillna(global_d48)
    df['rt_ts_mean']     = df.set_index(['RoadType', 'time_slot']).index.map(
        d48.groupby(['RoadType', 'time_slot'])['demand'].mean()).values
    df['rt_ts_mean']     = df['rt_ts_mean'].fillna(global_d48)

    # Calibration interactions
    df['delta_x_decay']  = df['calib_delta_gh'] * df['calib_decay']
    df['delta_x_road']   = df['calib_delta_gh'] * df['road_type_ord']
    df['delta_x_rush']   = df['calib_delta_gh'] * df['is_rush_hour']
    df['k5_x_decay']     = (df['calib_k5'] - 1) * df['calib_decay']
    df['k5_x_road']      = (df['calib_k5'] - 1) * df['road_type_ord']
    df['morning_ratio']  = (df['d49_morn_mean'] / df['d48_morn_mean'].clip(0.005)).clip(0.1, 10)

    # Uncertainty measure: how noisy is the per-geo calibration?
    df['calib_spread']   = df['calib_k1'] - df['calib_k30']  # high spread = noisy
    df['calib_reliability'] = df['d49_morn_count'] / (df['d49_morn_count'] + 5)  # 0-1 reliability

    return df

print("Engineering features...")
d48_s1 = engineer_s1(d48); d49_s1 = engineer_s1(d49); test_s1 = engineer_s1(test)
d48_s2 = engineer_s2(d48); d49_s2 = engineer_s2(d49); test_s2 = engineer_s2(test)
print(f"Stage 1 features: {d48_s1.shape[1]}")

DROP_COLS = ['Index', 'demand', 'timestamp', 'geohash',
             'LargeVehicles', 'Landmarks', 'RoadType', 'Weather', 'gp4', 'gp5', 'day']
TARGET = 'demand'

y_d48 = d48_s1[TARGET]; y_d49 = d49_s1[TARGET]
X_d48_s1 = d48_s1.drop(columns=DROP_COLS, errors='ignore')
X_d49_s1 = d49_s1.drop(columns=DROP_COLS, errors='ignore')
X_test_s1 = test_s1.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore')

feat_s1 = [c for c in X_d48_s1.columns if c in X_test_s1.columns]
X_d48_s1  = X_d48_s1[feat_s1]
X_d49_s1  = X_d49_s1[[c for c in feat_s1 if c in X_d49_s1.columns]]
X_test_s1 = X_test_s1[feat_s1]

print(f"X_d48_s1: {X_d48_s1.shape} | NaN: {X_d48_s1.isnull().sum().sum()}")

# ── STAGE 1: OPTUNA LGB (d48 only, cross-day validation) ────────────────────
print("\n" + "=" * 60)
print("Stage 1 LGB Optuna (100 trials, d48 train → d49 validate)...")

def objective_lgb_s1(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 8000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'max_depth': trial.suggest_int('max_depth', 6, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'random_state': SEED,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_d48_s1, y_d48, eval_set=[(X_d49_s1, y_d49)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    return r2_score(y_d49, m.predict(X_d49_s1))

study_lgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb_s1, n_trials=100, show_progress_bar=True)
best_lgb = {**study_lgb.best_params, 'objective': 'regression', 'metric': 'rmse',
            'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': SEED}
print(f"Best LGB d48→d49 R²: {study_lgb.best_value:.4f} → Score: {100*study_lgb.best_value:.2f}")

# ── STAGE 1: OPTUNA XGB ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1 XGB Optuna (50 trials)...")

def objective_xgb_s1(trial):
    params = {
        'objective': 'reg:squarederror', 'random_state': SEED,
        'tree_method': 'hist', 'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 7000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    m = xgb.XGBRegressor(**params, early_stopping_rounds=100)
    m.fit(X_d48_s1, y_d48, eval_set=[(X_d49_s1, y_d49)], verbose=False)
    return r2_score(y_d49, m.predict(X_d49_s1))

study_xgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb_s1, n_trials=50, show_progress_bar=True)
best_xgb = {**study_xgb.best_params, 'objective': 'reg:squarederror',
            'random_state': SEED, 'tree_method': 'hist', 'verbosity': 0}
print(f"Best XGB d48→d49 R²: {study_xgb.best_value:.4f} → Score: {100*study_xgb.best_value:.2f}")

# ── STAGE 1: CatBoost ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1 CatBoost...")
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
DROP_CB = ['Index', 'demand', 'timestamp', 'geohash', 'gp4', 'gp5', 'day']

def make_cb_s1(df):
    df2 = d48_s1.drop(columns=DROP_COLS, errors='ignore').copy() if df is d48 else \
          d49_s1.drop(columns=DROP_COLS, errors='ignore').copy() if df is d49 else \
          test_s1.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore').copy()
    for c in cat_cols:
        if c in df2.columns:
            df2[c] = df2[c].fillna('Unknown').astype(str)
    return df2

X_cb_d48 = make_cb_s1(d48)[[c for c in feat_s1 if c in make_cb_s1(d48).columns]]
X_cb_d49 = make_cb_s1(d49)[[c for c in feat_s1 if c in make_cb_s1(d49).columns]]
X_cb_test = make_cb_s1(test)[[c for c in feat_s1 if c in make_cb_s1(test).columns]]
cb_cat_idx = [list(X_cb_d48.columns).index(c) for c in cat_cols if c in X_cb_d48.columns]

cb_params = dict(iterations=5000, learning_rate=0.02, depth=8,
                 l2_leaf_reg=5, min_data_in_leaf=20, random_seed=SEED,
                 eval_metric='R2', loss_function='RMSE', verbose=False, early_stopping_rounds=100)

cb_probe = CatBoostRegressor(**cb_params)
cb_probe.fit(Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx),
             eval_set=Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx))
cb_r2 = r2_score(y_d49, cb_probe.predict(X_cb_d49))
print(f"CB d48→d49 R²: {cb_r2:.4f} → Score: {100*cb_r2:.2f}")

# ── STAGE 1: TRAIN FINAL MODELS (on d48 only) ────────────────────────────────
print("\n" + "=" * 60)
print("Training Stage 1 final models (d48 only)...")
lgb_s1_final = lgb.LGBMRegressor(**best_lgb)
lgb_s1_final.fit(X_d48_s1, y_d48, eval_set=[(X_d49_s1, y_d49)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

xgb_s1_final = xgb.XGBRegressor(**best_xgb, early_stopping_rounds=100)
xgb_s1_final.fit(X_d48_s1, y_d48, eval_set=[(X_d49_s1, y_d49)], verbose=False)

cb_s1_final = CatBoostRegressor(**cb_params)
cb_s1_final.fit(Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx),
                eval_set=Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx))

# Validate individually
r2_lgb_val = r2_score(y_d49, lgb_s1_final.predict(X_d49_s1))
r2_xgb_val = r2_score(y_d49, xgb_s1_final.predict(X_d49_s1))
r2_cb_val  = r2_score(y_d49, cb_s1_final.predict(X_cb_d49))
print(f"LGB val: {r2_lgb_val:.4f} | XGB val: {r2_xgb_val:.4f} | CB val: {r2_cb_val:.4f}")

# Optimal ensemble on d49 (honest cross-day)
best_score_s1, best_W_s1 = -np.inf, [0.5, 0.3, 0.2]
p_lgb = lgb_s1_final.predict(X_d49_s1)
p_xgb = xgb_s1_final.predict(X_d49_s1)
p_cb  = cb_s1_final.predict(X_cb_d49)
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.05, 0.7, 0.05):
        w2 = 1.0 - w0 - w1
        if not (0.05 <= w2 <= 0.8): continue
        sc = r2_score(y_d49, w0*p_lgb + w1*p_xgb + w2*p_cb)
        if sc > best_score_s1:
            best_score_s1, best_W_s1 = sc, [w0, w1, w2]
W_s1 = np.array(best_W_s1) / sum(best_W_s1)
print(f"S1 optimal weights: LGB={W_s1[0]:.3f}, XGB={W_s1[1]:.3f}, CB={W_s1[2]:.3f}")
print(f"S1 ensemble d48→d49 R²: {best_score_s1:.4f} → Score: {100*best_score_s1:.2f}")

# Stage 1 base predictions
d48_s1_pred = np.clip(W_s1[0]*lgb_s1_final.predict(X_d48_s1) +
                       W_s1[1]*xgb_s1_final.predict(X_d48_s1) +
                       W_s1[2]*cb_s1_final.predict(X_cb_d48), 0, 1)
d49_s1_pred = np.clip(W_s1[0]*p_lgb + W_s1[1]*p_xgb + W_s1[2]*p_cb, 0, 1)
test_s1_pred = np.clip(W_s1[0]*lgb_s1_final.predict(X_test_s1) +
                        W_s1[1]*xgb_s1_final.predict(X_test_s1) +
                        W_s1[2]*cb_s1_final.predict(X_cb_test), 0, 1)

print(f"\nStage 1 test pred: mean={test_s1_pred.mean():.4f}, std={test_s1_pred.std():.4f}")

# ── STAGE 2: CALIBRATION RESIDUAL MODEL ──────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 2: Calibration residual model...")

d48_resid = y_d48.values - d48_s1_pred  # should be ~0 (Stage 1 trained on d48)
d49_resid = y_d49.values - d49_s1_pred  # cross-day residuals
print(f"D48 residuals: mean={d48_resid.mean():.4f}, std={d48_resid.std():.4f}")
print(f"D49 residuals: mean={d49_resid.mean():.4f}, std={d49_resid.std():.4f}")
print(f"D49 Stage 1 R²: {r2_score(y_d49, d49_s1_pred):.4f}")

# Build Stage 2 features
DROP_S2 = ['Index', 'demand', 'timestamp', 'geohash', 'LargeVehicles', 'Landmarks',
           'RoadType', 'Weather', 'gp4', 'gp5', 'day']

X_d48_s2_raw = d48_s2.drop(columns=DROP_S2, errors='ignore')
X_d49_s2_raw = d49_s2.drop(columns=DROP_S2, errors='ignore')
X_test_s2_raw = test_s2.drop(columns=[c for c in DROP_S2 if c != 'demand'], errors='ignore')

feat_s2 = [c for c in X_d48_s2_raw.columns if c in X_test_s2_raw.columns]
X_d48_s2  = X_d48_s2_raw[feat_s2].fillna(0)
X_d49_s2  = X_d49_s2_raw[[c for c in feat_s2 if c in X_d49_s2_raw.columns]].fillna(0)
X_test_s2 = X_test_s2_raw[[c for c in feat_s2 if c in X_test_s2_raw.columns]].fillna(0)
feat_s2_aligned = [c for c in feat_s2 if c in X_d49_s2.columns and c in X_test_s2.columns]
X_d48_s2  = X_d48_s2[feat_s2_aligned]
X_d49_s2  = X_d49_s2[feat_s2_aligned]
X_test_s2 = X_test_s2[feat_s2_aligned]

print(f"Stage 2 features: {len(feat_s2_aligned)}")

# CRITICAL: Include BOTH d48 (target=0) and d49 (target=resid)
# d49 upweighted 50x so morning signal dominates but daytime≈0 constrains model
n_d48, n_d49 = len(d48_resid), len(d49_resid)
upweight_d49 = 50
X_s2_all = pd.concat([X_d48_s2, X_d49_s2], axis=0).reset_index(drop=True)
y_s2_all = np.concatenate([d48_resid, d49_resid])
w_s2_all = np.concatenate([np.ones(n_d48), np.full(n_d49, upweight_d49)])

print(f"Stage 2 training: {n_d48} d48 rows (target≈0) + {n_d49} d49 rows (target=resid)")
print(f"Effective weights: d48={n_d48:.0f}, d49={n_d49*upweight_d49:.0f}")

# Optuna for Stage 2 (cross-val on d49 residuals)
from sklearn.model_selection import cross_val_score

def objective_s2(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'random_state': SEED,
    }
    # Use 5-fold CV on d49 residuals only (honest Stage 2 eval)
    m = lgb.LGBMRegressor(**params)
    scores = cross_val_score(m, X_d49_s2, d49_resid, cv=5, scoring='r2')
    return scores.mean()

study_s2 = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_s2.optimize(objective_s2, n_trials=50, show_progress_bar=True)
print(f"Stage 2 best CV R² on d49 residuals: {study_s2.best_value:.4f}")

# Train Stage 2 on full d48+d49 combined
best_s2 = {**study_s2.best_params, 'objective': 'regression', 'metric': 'rmse',
           'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': SEED}
lgb_s2 = lgb.LGBMRegressor(**best_s2)
lgb_s2.fit(X_s2_all, y_s2_all, sample_weight=w_s2_all)

s2_test_pred = lgb_s2.predict(X_test_s2)
s2_d49_pred  = lgb_s2.predict(X_d49_s2)

print(f"Stage 2 test pred: mean={s2_test_pred.mean():.4f}, std={s2_test_pred.std():.4f}")
print(f"Stage 2 d49 check: {r2_score(y_d49, np.clip(d49_s1_pred + s2_d49_pred, 0, 1)):.4f}")

# ── FINAL PREDICTIONS ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Building final predictions...")

# Two-stage prediction
two_stage = np.clip(test_s1_pred + s2_test_pred, 0, 1)

# Try different Stage 2 blend strengths
for alpha in [0.5, 0.7, 1.0, 1.3, 1.5]:
    pred = np.clip(test_s1_pred + alpha * s2_test_pred, 0, 1)
    # Validate on d49
    val_pred = np.clip(d49_s1_pred + alpha * s2_d49_pred, 0, 1)
    val_r2 = r2_score(y_d49, val_pred)
    print(f"  Stage2 alpha={alpha}: d49_val_R²={val_r2:.4f}, test_mean={pred.mean():.4f}")

# Also compare with pure calibration formulas
additive_test = np.array([
    np.clip(geo_ts_d48_dict.get((row['geohash'], row['time_slot']),
            gh_d48_dict.get(row['geohash'], global_d48)) +
            calib_delta_geo.get(row['geohash'], 0.0), 0, 1)
    for _, row in test.iterrows()
])

# ── KFOLD OOF for better ensemble estimation ─────────────────────────────────
# Train on d48+d49 combined for OOF-based final ensemble
print("\nKFold OOF on d48+d49 combined (Stage 1 with d49 included)...")
X_all_s1 = pd.concat([X_d48_s1, X_d49_s1], axis=0).reset_index(drop=True)
y_all = pd.concat([y_d48, y_d49], axis=0).reset_index(drop=True)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

n_d48, n_d49 = len(d48), len(d49)
w_d49_mult = n_d48 / n_d49
sw = np.concatenate([np.ones(n_d48), np.full(n_d49, w_d49_mult)])

oof_lgb = np.zeros(len(X_all_s1))
test_lgb_kfold = np.zeros(len(X_test_s1))
for fold, (tr, va) in enumerate(kf.split(X_all_s1)):
    m = lgb.LGBMRegressor(**best_lgb)
    m.fit(X_all_s1.iloc[tr], y_all.iloc[tr], sample_weight=sw[tr],
          eval_set=[(X_all_s1.iloc[va], y_all.iloc[va])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[va] = m.predict(X_all_s1.iloc[va])
    test_lgb_kfold += m.predict(X_test_s1) / 5

oof_xgb = np.zeros(len(X_all_s1))
test_xgb_kfold = np.zeros(len(X_test_s1))
X_all_cb = pd.concat([X_cb_d48, X_cb_d49], axis=0).reset_index(drop=True)
for fold, (tr, va) in enumerate(kf.split(X_all_s1)):
    m = xgb.XGBRegressor(**best_xgb, early_stopping_rounds=100)
    m.fit(X_all_s1.iloc[tr], y_all.iloc[tr], sample_weight=sw[tr],
          eval_set=[(X_all_s1.iloc[va], y_all.iloc[va])], verbose=False)
    oof_xgb[va] = m.predict(X_all_s1.iloc[va])
    test_xgb_kfold += m.predict(X_test_s1) / 5

oof_cb = np.zeros(len(X_all_cb))
test_cb_kfold = np.zeros(len(X_cb_test))
for fold, (tr, va) in enumerate(kf.split(X_all_cb)):
    m = CatBoostRegressor(**cb_params)
    m.fit(Pool(X_all_cb.iloc[tr], y_all.iloc[tr], cat_features=cb_cat_idx, weight=sw[tr]),
          eval_set=Pool(X_all_cb.iloc[va], y_all.iloc[va], cat_features=cb_cat_idx))
    oof_cb[va] = m.predict(X_all_cb.iloc[va])
    test_cb_kfold += m.predict(X_cb_test) / 5

# Best OOF ensemble weights
best_oof, best_W = -np.inf, [0.5, 0.25, 0.25]
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.05, 0.7, 0.05):
        w2 = 1.0 - w0 - w1
        if not (0.05 <= w2 <= 0.8): continue
        sc = r2_score(y_all, w0*oof_lgb + w1*oof_xgb + w2*oof_cb)
        if sc > best_oof:
            best_oof, best_W = sc, [w0, w1, w2]
W_oof = np.array(best_W) / sum(best_W)
print(f"OOF weights: LGB={W_oof[0]:.3f}, XGB={W_oof[1]:.3f}, CB={W_oof[2]:.3f}")
print(f"OOF R²: {best_oof:.4f}")

kfold_pred = np.clip(W_oof[0]*test_lgb_kfold + W_oof[1]*test_xgb_kfold + W_oof[2]*test_cb_kfold, 0, 1)

# ── SUBMISSIONS ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
idx = test['Index']

# S1: Pure Stage 1 (d48-only base, optimally ensembled)
pd.DataFrame({'Index': idx, 'demand': test_s1_pred}).to_csv('submission_v5_s1.csv', index=False)

# S2: Two-stage (Stage1 + Stage2 calibration)
pd.DataFrame({'Index': idx, 'demand': two_stage}).to_csv('submission_v5_twostage.csv', index=False)

# S3: Two-stage with stronger Stage 2 (alpha=1.3)
pred_s2_strong = np.clip(test_s1_pred + 1.3 * s2_test_pred, 0, 1)
pd.DataFrame({'Index': idx, 'demand': pred_s2_strong}).to_csv('submission_v5_s2strong.csv', index=False)

# S4: KFold ensemble (d48+d49) + Stage 2
kfold_twostage = np.clip(kfold_pred + s2_test_pred, 0, 1)
pd.DataFrame({'Index': idx, 'demand': kfold_twostage}).to_csv('submission_v5_kfold_ts.csv', index=False)

# S5: KFold alone (without Stage 2)
pd.DataFrame({'Index': idx, 'demand': kfold_pred}).to_csv('submission_v5_kfold.csv', index=False)

assert all(pd.read_csv(f).shape[0] == 41778 for f in
           ['submission_v5_s1.csv','submission_v5_twostage.csv',
            'submission_v5_s2strong.csv','submission_v5_kfold_ts.csv','submission_v5_kfold.csv'])
print("All submissions saved!")

print("\n" + "=" * 60)
print("SUMMARY")
print(f"Stage 1 d48→d49 R²: LGB={100*study_lgb.best_value:.2f}, XGB={100*study_xgb.best_value:.2f}, CB={100*cb_r2:.2f}")
print(f"Stage 1 ensemble:    {100*best_score_s1:.2f}%")
print(f"Stage 2 CV R²:       {100*study_s2.best_value:.2f}%")
print(f"S2-corrected d49:    {100*r2_score(y_d49, np.clip(d49_s1_pred+s2_d49_pred,0,1)):.2f}%")
print(f"OOF ensemble R²:     {100*best_oof:.2f}%")
print(f"\nPrediction means:")
print(f"  s1={test_s1_pred.mean():.4f}, twostage={two_stage.mean():.4f}, kfold={kfold_pred.mean():.4f}")
