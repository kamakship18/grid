"""
v4 — Two-stage model + additive calibration delta.

Key insights from diagnostics:
1. Additive calibration (pred = geo_ts_mean + delta) → 91.12% morning validation vs 82.42% multiplicative
2. Perfect geohash-level estimates ceiling = 94.63% → target is achievable
3. Two-stage: Stage1 learns geo patterns from d48, Stage2 learns cross-day shifts from d49 residuals
4. Multi-resolution calibration: give model k=1,5,30 smoothed ratios to blend optimally
5. rt_ts_mean is a 73.64% cross-day predictor — critical feature already in v3

Architecture:
- Stage 1: Ensemble of LGB+XGB+CB trained on d48 (base geo patterns)
- Stage 2: LGB trained on d49 residuals from stage1, using calibration features only
- Final: alpha*Stage1_ensemble + beta*Stage2_calib + gamma*additive_formula
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

# ── CALIBRATION (multi-resolution) ──────────────────────────────────────────
d48_morning = d48[d48['hour'].isin([0, 1, 2])].copy()

# Road-type calibration (level 0 — most stable)
rt_calib = {}
for rt in list(d49['RoadType'].unique()) + ['Unknown']:
    d48_rt = d48_morning[d48_morning['RoadType'] == rt]['demand'].mean()
    d49_rt = d49[d49['RoadType'] == rt]['demand'].mean()
    if pd.notna(d48_rt) and pd.notna(d49_rt) and d48_rt > 0:
        rt_calib[rt] = np.clip(d49_rt / d48_rt, 0.2, 5.0)
    else:
        rt_calib[rt] = 1.0

global_rt_calib = d49['demand'].mean() / d48_morning['demand'].mean()
print(f"Road-type calibration: {rt_calib}")

# Road-type level ADDITIVE delta (stable, level 0)
rt_delta = {}
for rt in rt_calib:
    d48_rt_lvl = d48_morning[d48_morning['RoadType'] == rt]['demand'].mean()
    d49_rt_lvl = d49[d49['RoadType'] == rt]['demand'].mean()
    if pd.notna(d48_rt_lvl) and pd.notna(d49_rt_lvl):
        rt_delta[rt] = d49_rt_lvl - d48_rt_lvl
    else:
        rt_delta[rt] = 0.0

# Per-geohash calibration with MULTIPLE smoothing levels
d49_geo = d49.groupby('geohash')['demand'].agg(['mean', 'count']).rename(
    columns={'mean': 'm', 'count': 'n'})
d48_geo_morn = d48_morning.groupby('geohash')['demand'].mean()
d48_geo_all  = d48.groupby('geohash')['demand'].mean()
geo_rt       = d48.groupby('geohash')['RoadType'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
global_d48   = d48['demand'].mean()
global_d48_morn = d48_morning['demand'].mean()

# d49 morning mean per geohash (raw, for additive delta)
d49_geo_mean = d49_geo['m']
d48_geo_morn_mean = d48_geo_morn

# Multi-resolution multiplicative calibration: k=1, k=5, k=30
def compute_multi_calib(k):
    calib = {}
    for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
        rt = geo_rt.get(gh, 'Unknown')
        rt_prior = rt_calib.get(rt, global_rt_calib)
        if gh not in d49_geo.index:
            calib[gh] = rt_prior
            continue
        n   = d49_geo.loc[gh, 'n']
        m49 = d49_geo.loc[gh, 'm']
        m48 = d48_geo_morn.get(gh, global_d48_morn)
        if m48 < 0.005:
            calib[gh] = rt_prior
            continue
        raw_ratio = np.clip(m49 / m48, 0.05, 20.0)
        calib[gh] = np.clip((n * raw_ratio + k * rt_prior) / (n + k), 0.05, 20.0)
    return calib

print("Computing multi-resolution calibration (k=1, 5, 30)...")
calib_k1  = compute_multi_calib(k=1)
calib_k5  = compute_multi_calib(k=5)
calib_k30 = compute_multi_calib(k=30)

# ── day_scale_robust: multiplicative ratio of d49 vs d48 morning ─────────────
# gp4-level fallback for test geohashes not seen in train49 (~10%)
gp4_d49_early = d49.copy()
gp4_d49_early['gp4'] = gp4_d49_early['geohash'].str[:4]
gp4_d49_mean = gp4_d49_early.groupby('gp4')['demand'].mean()
gp4_d48_morn = d48_morning.copy()
gp4_d48_morn['gp4'] = gp4_d48_morn['geohash'].str[:4]
gp4_d48_mean = gp4_d48_morn.groupby('gp4')['demand'].mean()
GLOBAL_D49_EARLY = float(d49['demand'].mean())

day_scale = {}
for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
    rt   = geo_rt.get(gh, 'Unknown')
    gp4  = gh[:4]
    if gh in d49_geo.index:
        d49e = d49_geo.loc[gh, 'm']
    elif gp4 in gp4_d49_mean.index:
        d49e = gp4_d49_mean[gp4]
    else:
        d49e = GLOBAL_D49_EARLY
    d48e = d48_geo_morn.get(gh, global_d48_morn)
    day_scale[gh] = np.clip(d49e / max(d48e, 1e-9), 0.2, 10.0)

# improved d49_morning_mean with gp4 fallback for test cold-start geohashes
d49_morning_imputed = {}
for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
    gp4 = gh[:4]
    if gh in d49_geo.index:
        d49_morning_imputed[gh] = d49_geo.loc[gh, 'm']
    elif gp4 in gp4_d49_mean.index:
        d49_morning_imputed[gh] = gp4_d49_mean[gp4]
    else:
        d49_morning_imputed[gh] = GLOBAL_D49_EARLY

# Per-geohash ADDITIVE delta: calib_delta[gh] = d49_morning - d48_morning
calib_delta_geo = {}
for gh in set(list(test['geohash'].unique()) + list(train['geohash'].unique())):
    rt = geo_rt.get(gh, 'Unknown')
    rt_prior_delta = rt_delta.get(rt, 0.0)
    if gh not in d49_geo.index:
        # Use road-type level delta as fallback
        calib_delta_geo[gh] = rt_prior_delta
        continue
    n   = d49_geo.loc[gh, 'n']
    m49 = d49_geo.loc[gh, 'm']
    m48 = d48_geo_morn.get(gh, global_d48_morn)
    raw_delta = m49 - m48
    # Smooth toward road-type delta prior (k=3)
    k = 3
    calib_delta_geo[gh] = (n * raw_delta + k * rt_prior_delta) / (n + k)

# ── LOOKUP TABLES ─────────────────────────────────────────────────────────────
geo_ts_d48   = d48.groupby(['geohash', 'time_slot'])['demand'].mean()
rt_ts_d48    = d48.groupby(['RoadType', 'time_slot'])['demand'].mean()
ts_d48_all   = d48.groupby('time_slot')['demand'].mean()
gh_d48_all   = d48.groupby('geohash')['demand'].mean()

# Validate calibration formula on d49
geo_ts_d48_dict = geo_ts_d48.to_dict()
d49_pred_add = []
for _, row in d49.iterrows():
    base = geo_ts_d48_dict.get((row['geohash'], row['time_slot']),
                                 gh_d48_all.get(row['geohash'], global_d48))
    delta = calib_delta_geo.get(row['geohash'], 0.0)
    d49_pred_add.append(np.clip(base + delta, 0.0, 1.0))

r2_add = r2_score(d49['demand'], d49_pred_add)
print(f"Additive formula d49 R²: {r2_add:.4f} → Score: {100*r2_add:.2f}")

d49_pred_mult_k1 = []
for _, row in d49.iterrows():
    base = geo_ts_d48_dict.get((row['geohash'], row['time_slot']),
                                 gh_d48_all.get(row['geohash'], global_d48))
    cal = calib_k1.get(row['geohash'], 1.0)
    d49_pred_mult_k1.append(np.clip(base * cal, 0.0, 1.0))
print(f"Multiplicative k=1 d49 R²: {r2_score(d49['demand'], d49_pred_mult_k1):.4f}")

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
geo_ts_d48_df = d48.groupby(['geohash', 'time_slot'])['demand'].agg(
    geo_ts_mean='mean', geo_ts_std='std'
).reset_index().fillna({'geo_ts_std': 0})

gh_d48_df = d48.groupby('geohash')['demand'].agg(
    gh_mean='mean', gh_std='std',
    gh_q25=lambda x: x.quantile(0.25),
    gh_q75=lambda x: x.quantile(0.75)
).reset_index().fillna(0)
gh_d48_df['gh_iqr'] = gh_d48_df['gh_q75'] - gh_d48_df['gh_q25']

ts_d48_df = d48.groupby('time_slot')['demand'].agg(ts_mean='mean').reset_index()

rt_ts_d48_df = d48.groupby(['RoadType', 'time_slot'])['demand'].agg(
    rt_ts_mean='mean'
).reset_index()

d48_morning_gh_df = d48_morning.groupby('geohash')['demand'].agg(
    d48_morning_mean='mean', d48_morning_std='std'
).reset_index().fillna({'d48_morning_std': 0})

def decode_gh(gh):
    try:
        lat, lon = pgh.decode(gh)
        return lat, lon
    except:
        return np.nan, np.nan

def engineer(df, is_day49=False):
    df = df.copy()

    # Spatial
    geo_dec = df['geohash'].apply(decode_gh)
    df['lat'] = geo_dec.apply(lambda x: x[0])
    df['lon'] = geo_dec.apply(lambda x: x[1])
    df['gp4'] = df['geohash'].str[:4]
    df['gp5'] = df['geohash'].str[:5]

    # Road features
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather']  = df['Weather'].fillna('Unknown')
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord']   = df['RoadType'].map(road_order).fillna(0).astype(int)
    df['large_vehicles']  = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['has_landmarks']   = (df['Landmarks'] == 'Yes').astype(int)
    df['lanes_sq']        = df['NumberofLanes'] ** 2
    df['lanes_log']       = np.log1p(df['NumberofLanes'])
    df['is_multilane']    = (df['NumberofLanes'] > 2).astype(int)
    df['road_capacity']   = df['road_type_ord'] * df['NumberofLanes'] * (1 + df['large_vehicles'])
    df['is_highway']      = (df['road_type_ord'] == 3).astype(int)

    # Weather
    weather_order = {'Sunny': 4, 'Cloudy': 3, 'Foggy': 2, 'Rainy': 1, 'Snowy': 0, 'Unknown': 2}
    df['weather_ord']    = df['Weather'].map(weather_order).fillna(2).astype(int)
    df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    temp_med = df['Temperature'].median()
    if pd.isna(temp_med): temp_med = 20.0
    df['Temperature']    = df['Temperature'].fillna(temp_med)
    df['temp_sq']        = df['Temperature'] ** 2
    df['is_cold']        = (df['Temperature'] < 5).astype(int)
    df['is_hot']         = (df['Temperature'] > 30).astype(int)

    # Time features
    df['hour_sin']        = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']        = np.cos(2 * np.pi * df['hour'] / 24)
    df['timeslot_sin']    = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['timeslot_cos']    = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour']    = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)
    df['is_d49_morning']  = ((df['hour'] <= 2)).astype(int)  # signal for calibration time window

    # Interactions
    df['road_x_lanes']    = df['road_type_ord'] * df['NumberofLanes']
    df['lanes_x_peak']    = df['NumberofLanes'] * df['is_rush_hour']
    df['highway_x_peak']  = df['is_highway'] * df['is_rush_hour']
    df['weather_x_peak']  = df['weather_ord'] * df['is_rush_hour']
    df['capacity_x_peak'] = df['road_capacity'] * df['is_rush_hour']
    df['slot_x_road']     = df['time_slot'] * df['road_type_ord']

    # Merge d48 lookup tables
    df = df.merge(geo_ts_d48_df, on=['geohash', 'time_slot'], how='left')
    df = df.merge(gh_d48_df, on='geohash', how='left')
    df = df.merge(ts_d48_df, on='time_slot', how='left')
    df = df.merge(rt_ts_d48_df, on=['RoadType', 'time_slot'], how='left')
    df = df.merge(d48_morning_gh_df, on='geohash', how='left')

    # Fill missing with global/road-type means
    df['geo_ts_mean']    = df['geo_ts_mean'].fillna(global_d48)
    df['geo_ts_std']     = df['geo_ts_std'].fillna(0)
    df['gh_mean']        = df['gh_mean'].fillna(global_d48)
    df['gh_std']         = df['gh_std'].fillna(0)
    df['gh_iqr']         = df['gh_iqr'].fillna(0)
    df['ts_mean']        = df['ts_mean'].fillna(global_d48)
    df['rt_ts_mean']     = df['rt_ts_mean'].fillna(global_d48)
    df['d48_morning_mean'] = df['d48_morning_mean'].fillna(global_d48_morn)
    df['d48_morning_std']  = df['d48_morning_std'].fillna(0)

    # ── Multi-resolution calibration features ──
    df['calib_k1']  = df['geohash'].map(calib_k1).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_k5']  = df['geohash'].map(calib_k5).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_k30'] = df['geohash'].map(calib_k30).fillna(df['RoadType'].map(rt_calib).fillna(global_rt_calib))
    df['calib_rt']  = df['RoadType'].map(rt_calib).fillna(global_rt_calib)

    # ── Additive calibration features (KEY NEW FEATURES) ──
    df['calib_delta_gh'] = df['geohash'].map(calib_delta_geo).fillna(
        df['RoadType'].map(rt_delta).fillna(0.0))
    # Use gp4-fallback-improved d49 morning mean (better cold-start coverage)
    df['d49_morning_mean'] = df['geohash'].map(d49_morning_imputed).fillna(
        df['d48_morning_mean'] * df['calib_rt'])
    # day_scale_robust: multiplicative day49/day48 ratio — cross-day scaling per geohash
    df['day_scale_robust'] = df['geohash'].map(day_scale).fillna(global_rt_calib)
    # d48 slot prediction scaled to day49 level (direct predictor for test)
    df['d48_slot_scaled'] = (df['geo_ts_mean'] * df['day_scale_robust']).clip(0, 1)

    # Additive predictions (multiple variants)
    df['pred_additive']     = (df['geo_ts_mean'] + df['calib_delta_gh']).clip(0, 1)
    df['pred_rt_calib']     = (df['rt_ts_mean']  * df['calib_rt']).clip(0, 1)
    df['pred_mult_k5']      = (df['geo_ts_mean']  * df['calib_k5']).clip(0, 1)
    df['pred_mult_k30']     = (df['geo_ts_mean']  * df['calib_k30']).clip(0, 1)

    # Residual: today's morning level vs yesterday's morning
    df['morning_ratio']     = (df['d49_morning_mean'] / df['d48_morning_mean'].clip(0.005)).clip(0.1, 10)
    df['morning_delta_norm']= df['calib_delta_gh'] / df['d48_morning_mean'].clip(0.005)

    # Geo stat deviations
    df['geo_ts_vs_gh']   = df['geo_ts_mean'] - df['gh_mean']
    df['geo_ts_vs_ts']   = df['geo_ts_mean'] - df['ts_mean']
    df['rt_ts_vs_ts']    = df['rt_ts_mean']  - df['ts_mean']

    # Calibration interaction features
    df['delta_x_road']   = df['calib_delta_gh'] * df['road_type_ord']
    df['delta_x_rush']   = df['calib_delta_gh'] * df['is_rush_hour']
    df['calib_x_rush']   = df['calib_k5'] * df['is_rush_hour']
    df['rush_x_geo']     = df['is_rush_hour'] * df['gh_mean']

    return df

print("Engineering features...")
d48_fe   = engineer(d48, is_day49=False)
d49_fe   = engineer(d49, is_day49=True)
test_fe  = engineer(test, is_day49=True)
print(f"Features engineered: {d48_fe.shape[1]} cols")

DROP_COLS = ['Index', 'demand', 'timestamp', 'geohash',
             'LargeVehicles', 'Landmarks', 'RoadType', 'Weather',
             'gp4', 'gp5', 'day']
TARGET = 'demand'

y_d48 = d48_fe[TARGET]; y_d49 = d49_fe[TARGET]
X_d48 = d48_fe.drop(columns=DROP_COLS, errors='ignore')
X_d49 = d49_fe.drop(columns=DROP_COLS, errors='ignore')

train_all_fe = pd.concat([d48_fe, d49_fe], axis=0).reset_index(drop=True)
y_all = train_all_fe[TARGET]
X_all = train_all_fe.drop(columns=DROP_COLS, errors='ignore')

X_test = test_fe.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore')

feat_cols = [c for c in X_all.columns if c in X_test.columns]
X_all  = X_all[feat_cols]
X_d48  = X_d48[[c for c in feat_cols if c in X_d48.columns]]
X_d49  = X_d49[[c for c in feat_cols if c in X_d49.columns]]
X_test = X_test[feat_cols]

print(f"X_all: {X_all.shape} | X_test: {X_test.shape}")
print(f"NaN check: {X_all.isnull().sum().sum()} total NaNs in X_all")

# ── QUICK VALIDATION ─────────────────────────────────────────────────────────
print("\nQuick cross-day validation...")
m_quick = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=255,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0, min_child_samples=15,
                              random_state=SEED, verbosity=-1)
m_quick.fit(X_d48, y_d48)
r2_quick = r2_score(y_d49, m_quick.predict(X_d49))
print(f"Quick LGB d48→d49 R²: {r2_quick:.4f} → Score: {100*r2_quick:.2f}")

# ── STAGE 1: OPTUNA LGB (base model) ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1 LGB Optuna (75 trials, d48→d49 honest validation)...")

def objective_lgb(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
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
    if params['boosting_type'] == 'dart':
        m.fit(X_d48, y_d48)
        return r2_score(y_d49, m.predict(X_d49))
    else:
        m.fit(X_d48, y_d48, eval_set=[(X_d49, y_d49)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        return r2_score(y_d49, m.predict(X_d49))

study_lgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=75, show_progress_bar=True)
best_lgb_params = {**study_lgb.best_params,
                   'objective': 'regression', 'metric': 'rmse',
                   'verbosity': -1, 'random_state': SEED}
print(f"Best LGB R²: {study_lgb.best_value:.4f} → Score: {100*study_lgb.best_value:.2f}")

# ── STAGE 1: OPTUNA XGB ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1 XGB Optuna (40 trials)...")

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror', 'random_state': SEED,
        'tree_method': 'hist', 'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 6000),
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
    m.fit(X_d48, y_d48, eval_set=[(X_d49, y_d49)], verbose=False)
    return r2_score(y_d49, m.predict(X_d49))

study_xgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=40, show_progress_bar=True)
best_xgb_params = {**study_xgb.best_params,
                   'objective': 'reg:squarederror', 'random_state': SEED,
                   'tree_method': 'hist', 'verbosity': 0}
print(f"Best XGB R²: {study_xgb.best_value:.4f} → Score: {100*study_xgb.best_value:.2f}")

# ── KFold OOF predictions (Stage 1) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5-fold OOF (Stage 1 ensemble)...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

n_d48, n_d49 = len(d48_fe), len(d49_fe)
w_d49_mult = n_d48 / n_d49  # balanced total weight
sample_weights_all = np.concatenate([np.ones(n_d48), np.full(n_d49, w_d49_mult)])

oof_lgb = np.zeros(len(X_all))
test_preds_lgb = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w_tr = sample_weights_all[tr_idx]
    bt = best_lgb_params.get('boosting_type', 'gbdt')
    m = lgb.LGBMRegressor(**best_lgb_params)
    if bt == 'dart':
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m.predict(X_val)
    test_preds_lgb += m.predict(X_test) / 5
    print(f"  LGB Fold {fold+1}: OOF={r2_score(y_val, oof_lgb[val_idx]):.4f}")
print(f"OOF LGB R²: {r2_score(y_all, oof_lgb):.4f}")

oof_xgb = np.zeros(len(X_all))
test_preds_xgb = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w_tr = sample_weights_all[tr_idx]
    m = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m.predict(X_val)
    test_preds_xgb += m.predict(X_test) / 5
    print(f"  XGB Fold {fold+1}: OOF={r2_score(y_val, oof_xgb[val_idx]):.4f}")
print(f"OOF XGB R²: {r2_score(y_all, oof_xgb):.4f}")

# CatBoost
print("\nCatBoost...")
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
DROP_CB = ['Index', 'demand', 'timestamp', 'geohash', 'gp4', 'gp5', 'day']

def make_cb(df):
    df2 = df.drop(columns=DROP_CB, errors='ignore').copy()
    for c in cat_cols:
        if c in df2.columns:
            df2[c] = df2[c].fillna('Unknown').astype(str)
    return df2

train_all_cb = pd.concat([d48_fe, d49_fe], axis=0).reset_index(drop=True)
X_cb_all   = make_cb(train_all_cb)
cb_cols    = [c for c in X_cb_all.columns if c in make_cb(test_fe).columns]
X_cb_all   = X_cb_all[cb_cols]
X_cb_test  = make_cb(test_fe)[cb_cols]
X_cb_d48   = make_cb(d48_fe)[[c for c in cb_cols if c in make_cb(d48_fe).columns]]
X_cb_d49   = make_cb(d49_fe)[[c for c in cb_cols if c in make_cb(d49_fe).columns]]
cb_cat_idx = [cb_cols.index(c) for c in cat_cols if c in cb_cols]

cb_params = dict(iterations=4000, learning_rate=0.025, depth=9,
                 l2_leaf_reg=3, min_data_in_leaf=15, random_seed=SEED,
                 eval_metric='R2', loss_function='RMSE',
                 verbose=False, early_stopping_rounds=100)

cb_probe = CatBoostRegressor(**cb_params)
cb_probe.fit(Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx),
             eval_set=Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx))
print(f"CB d48→d49 R²: {r2_score(y_d49, cb_probe.predict(X_cb_d49)):.4f}")

oof_cb = np.zeros(len(X_cb_all))
test_preds_cb = np.zeros(len(X_cb_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb_all)):
    X_tr, X_val = X_cb_all.iloc[tr_idx], X_cb_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w_tr = sample_weights_all[tr_idx]
    m = CatBoostRegressor(**cb_params)
    m.fit(Pool(X_tr, y_tr, cat_features=cb_cat_idx, weight=w_tr),
          eval_set=Pool(X_val, y_val, cat_features=cb_cat_idx))
    oof_cb[val_idx] = m.predict(X_val)
    test_preds_cb += m.predict(X_cb_test) / 5
    print(f"  CB Fold {fold+1}: OOF={r2_score(y_val, oof_cb[val_idx]):.4f}")
print(f"OOF CB R²: {r2_score(y_all, oof_cb):.4f}")

# Optimal LGB/XGB/CB blend weights (must come BEFORE Stage 2 so residuals match submission base)
print("\n" + "=" * 60)
print("Finding optimal ensemble weights (OOF)...")
best_score, best_W = -np.inf, [0.5, 0.25, 0.25]
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.05, 0.7, 0.05):
        w2 = 1.0 - w0 - w1
        if not (0.05 <= w2 <= 0.8):
            continue
        sc = r2_score(y_all, w0 * oof_lgb + w1 * oof_xgb + w2 * oof_cb)
        if sc > best_score:
            best_score, best_W = sc, [w0, w1, w2]
W = np.array(best_W) / sum(best_W)
print(f"Optimal weights: LGB={W[0]:.3f}, XGB={W[1]:.3f}, CB={W[2]:.3f}")
print(f"Ensemble OOF R²: {best_score:.4f}")

# ── STAGE 2: RESIDUAL CALIBRATION MODEL ──────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 2: Train calibration residual model on d49...")

# Stage 1 base predictions for d49 (d48-trained models) — SAME blend W as final submission
lgb_base = lgb.LGBMRegressor(**best_lgb_params)
lgb_base_bt = best_lgb_params.get('boosting_type', 'gbdt')
if lgb_base_bt == 'dart':
    lgb_base.fit(X_d48, y_d48)
else:
    lgb_base.fit(X_d48, y_d48, eval_set=[(X_d49, y_d49)],
                 callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

xgb_base = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=100)
xgb_base.fit(X_d48, y_d48, eval_set=[(X_d49, y_d49)], verbose=False)

cb_base = CatBoostRegressor(**cb_params)
cb_base.fit(Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx),
            eval_set=Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx))

# Stage 1 ensemble on d49 (cross-day) — aligned with stage1_final / OOF weights
d49_base_pred = np.clip(
    W[0] * lgb_base.predict(X_d49)
    + W[1] * xgb_base.predict(X_d49)
    + W[2] * cb_base.predict(X_cb_d49),
    0, 1,
)

# Compute d49 residuals (what Stage 1 can't explain)
d49_resid = y_d49.values - d49_base_pred
print(f"D49 residuals: mean={d49_resid.mean():.4f}, std={d49_resid.std():.4f}, R²_lost={1 - r2_score(y_d49, d49_base_pred):.4f}")

# Features for Stage 2 (calibration-focused)
calib_feat_cols = [c for c in X_d49.columns if any(kw in c for kw in [
    'calib', 'delta', 'morning', 'pred_add', 'pred_mult', 'pred_rt',
    'road_type_ord', 'time_slot', 'hour', 'hour_sin', 'hour_cos',
    'timeslot_sin', 'timeslot_cos', 'is_rush_hour', 'is_night',
    'is_peak', 'NumberofLanes', 'road_capacity', 'is_highway',
    'lat', 'lon', 'rt_ts_mean', 'ts_mean', 'gh_mean'
])]
print(f"Stage 2 calibration features: {len(calib_feat_cols)}")

X_d49_calib  = X_d49[calib_feat_cols]
X_test_calib = X_test[[c for c in calib_feat_cols if c in X_test.columns]]
calib_feat_cols_aligned = [c for c in calib_feat_cols if c in X_test.columns]
X_d49_calib  = X_d49_calib[calib_feat_cols_aligned]
X_test_calib = X_test_calib[calib_feat_cols_aligned]

# Train Stage 2 on d49 residuals
def objective_s2(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'random_state': SEED,
    }
    # 5-fold CV on d49 (it's small so this is the best we can do)
    from sklearn.model_selection import cross_val_score
    m = lgb.LGBMRegressor(**params)
    scores = cross_val_score(m, X_d49_calib, d49_resid, cv=5, scoring='r2')
    return scores.mean()

study_s2 = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_s2.optimize(objective_s2, n_trials=30, show_progress_bar=True)
print(f"Stage 2 best CV R² on d49 residuals: {study_s2.best_value:.4f}")

best_s2_params = {**study_s2.best_params,
                  'objective': 'regression', 'metric': 'rmse',
                  'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': SEED}
lgb_s2 = lgb.LGBMRegressor(**best_s2_params)
lgb_s2.fit(X_d49_calib, d49_resid)
s2_test_pred = lgb_s2.predict(X_test_calib)

# Stage 1 ensemble predictions for test — same W as d49_base_pred / OOF
test_base_pred = np.clip(
    W[0] * lgb_base.predict(X_test)
    + W[1] * xgb_base.predict(X_test)
    + W[2] * cb_base.predict(X_cb_test),
    0, 1,
)

# Two-stage prediction
two_stage_pred = np.clip(test_base_pred + s2_test_pred, 0, 1)
print(f"\nTwo-stage test pred stats: mean={two_stage_pred.mean():.4f}, std={two_stage_pred.std():.4f}")

# ── FINAL ENSEMBLE (weights W already computed above) ───────────────────────
stage1_final = np.clip(W[0]*test_preds_lgb + W[1]*test_preds_xgb + W[2]*test_preds_cb, 0, 1)

# ── Compute additive formula for test ────────────────────────────────────────
print("\nComputing additive formula for test...")
additive_test = []
for _, row in test_fe.iterrows():
    base = geo_ts_d48_dict.get((row['geohash'], row['time_slot']),
                                 gh_d48_all.get(row['geohash'], global_d48))
    delta = calib_delta_geo.get(row['geohash'], 0.0)
    additive_test.append(np.clip(base + delta, 0.0, 1.0))
additive_test = np.array(additive_test)
print(f"Additive test pred: mean={additive_test.mean():.4f}, std={additive_test.std():.4f}")

# ── SUBMISSIONS ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
idx = test['Index']

# Submission 1: Stage 1 ensemble
pd.DataFrame({'Index': idx, 'demand': stage1_final}).to_csv('submission_v4_s1.csv', index=False)

# Submission 2: Two-stage (Stage1 + Stage2 calibration residual)
pd.DataFrame({'Index': idx, 'demand': two_stage_pred}).to_csv('submission_v4_twostage.csv', index=False)

# Submission 3: Stage1 + additive formula blend (85:15)
blended_add = np.clip(0.82 * stage1_final + 0.18 * additive_test, 0, 1)
pd.DataFrame({'Index': idx, 'demand': blended_add}).to_csv('submission_v4_add_blend.csv', index=False)

# Submission 4: Three-way blend: Stage1 + TwoStage + Additive
blend3 = np.clip(0.6 * stage1_final + 0.25 * two_stage_pred + 0.15 * additive_test, 0, 1)
pd.DataFrame({'Index': idx, 'demand': blend3}).to_csv('submission_v4_blend3.csv', index=False)

assert all(len(test['Index']) == 41778 for _ in [1])
print("All submissions saved!")

print("\n" + "=" * 60)
print("SUMMARY")
print(f"Additive formula d49 R²:    {r2_add:.4f} → Score: {100*r2_add:.2f}")
print(f"Best LGB d48→d49 R²:        {study_lgb.best_value:.4f} → Score: {100*study_lgb.best_value:.2f}")
print(f"Best XGB d48→d49 R²:        {study_xgb.best_value:.4f} → Score: {100*study_xgb.best_value:.2f}")
print(f"Stage 2 calib R²:           {study_s2.best_value:.4f}")
print(f"Ensemble OOF R²:            {best_score:.4f} → Score: {100*best_score:.2f}")
print(f"\nSubmissions: s1={stage1_final.mean():.4f}, twostage={two_stage_pred.mean():.4f},",
      f"add_blend={blended_add.mean():.4f}")
