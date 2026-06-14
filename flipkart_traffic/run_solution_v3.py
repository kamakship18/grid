"""
v3 — Road-type specific calibration priors (the key fix).

Root cause of v2 limitation:
  Global calib ratio (1.46) was inflated by road-type composition shifts.
  True road-type calibration ratios are:
    Highway:     1.055  (demand barely changed)
    Street:      0.994  (demand slightly decreased)
    Residential: 1.172  (demand increased more)
  Bayesian smoothing toward the wrong global prior (1.46) over-corrected
  highway predictions, which dominate R² due to high absolute demand.

Fix: use road-type specific priors at every smoothing level.
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

# ── CALIBRATION (hierarchical, road-type specific) ───────────────────────────
d48_morning = d48[d48['hour'].isin([0, 1, 2])].copy()
d48_all = d48.copy()

# Level 1: Road-type specific calibration ratios
rt_calib = {}
for rt in d49['RoadType'].unique():
    d48_rt_morn = d48_morning[d48_morning['RoadType'] == rt]['demand'].mean()
    d49_rt_morn = d49[d49['RoadType'] == rt]['demand'].mean()
    if d48_rt_morn > 0 and not np.isnan(d49_rt_morn):
        rt_calib[rt] = np.clip(d49_rt_morn / d48_rt_morn, 0.2, 5.0)
    else:
        rt_calib[rt] = 1.0

# Global fallback
global_rt_calib = d49['demand'].mean() / d48_morning['demand'].mean()
rt_calib['Unknown'] = rt_calib.get('Unknown', global_rt_calib)
print(f"Road-type calibration ratios: {rt_calib}")
print(f"Global calib: {global_rt_calib:.4f}")

# Level 2: Geo-prefix-4 calibration
d49['gp4'] = d49['geohash'].str[:4]
d48_morning['gp4'] = d48_morning['geohash'].str[:4]
d48_all['gp4'] = d48_all['geohash'].str[:4]

gp4_d49 = d49.groupby('gp4')['demand'].agg(['mean', 'count']).rename(columns={'mean': 'm', 'count': 'n'})
gp4_d48m = d48_morning.groupby('gp4')['demand'].mean()
gp4_rt   = d49.groupby('gp4')['RoadType'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')

def gp4_calib(gp4_val):
    if gp4_val not in gp4_d49.index:
        return 1.0
    n    = gp4_d49.loc[gp4_val, 'n']
    m49  = gp4_d49.loc[gp4_val, 'm']
    m48  = gp4_d48m.get(gp4_val, d48_morning['demand'].mean())
    rt   = gp4_rt.get(gp4_val, 'Unknown')
    prior = rt_calib.get(rt, global_rt_calib)
    smooth_k = 20
    return np.clip((n * m49 / max(m48, 1e-6) + smooth_k * prior) / (n + smooth_k), 0.1, 10.0)

# Level 3: Geo-prefix-5 calibration
d49['gp5'] = d49['geohash'].str[:5]
d48_morning['gp5'] = d48_morning['geohash'].str[:5]

gp5_d49 = d49.groupby('gp5')['demand'].agg(['mean', 'count']).rename(columns={'mean': 'm', 'count': 'n'})
gp5_d48m = d48_morning.groupby('gp5')['demand'].mean()
gp5_rt   = d49.groupby('gp5')['RoadType'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')

def gp5_calib(gp5_val, gp4_prior):
    if gp5_val not in gp5_d49.index:
        return gp4_prior
    n    = gp5_d49.loc[gp5_val, 'n']
    m49  = gp5_d49.loc[gp5_val, 'm']
    m48  = gp5_d48m.get(gp5_val, d48_morning['demand'].mean())
    smooth_k = 15
    return np.clip((n * m49 / max(m48, 1e-6) + smooth_k * gp4_prior) / (n + smooth_k), 0.1, 10.0)

# Level 4: Per-geohash calibration (smoothed with road-type + prefix priors)
d49_geo = d49.groupby('geohash')['demand'].agg(['mean', 'count', 'std']).rename(
    columns={'mean': 'm', 'count': 'n', 'std': 's'})
d48_geo_morn = d48_morning.groupby('geohash')['demand'].mean()
d48_geo_all  = d48_all.groupby('geohash')['demand'].mean()
geo_rt = d49.groupby('geohash')['RoadType'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')

def compute_geo_calib(gh, rt, gp5_val, gp4_val):
    # Compute priors at each level
    rt_prior = rt_calib.get(rt, global_rt_calib)
    gp4_prior_val = gp4_calib(gp4_val)
    gp5_prior_val = gp5_calib(gp5_val, gp4_prior_val)

    if gh not in d49_geo.index:
        return gp5_prior_val

    n   = d49_geo.loc[gh, 'n']
    m49 = d49_geo.loc[gh, 'm']
    m48 = d48_geo_morn.get(gh, d48_morning['demand'].mean())

    # Additive delta (stable for low-demand geohashes)
    delta = m49 - m48

    # Multiplicative ratio
    if m48 > 0.02:  # sufficient base demand for ratio
        raw_ratio = m49 / m48
        # Use rt-specific smooth_k: less smoothing for highway (more reliable ratio)
        rt_smooth = {'Highway': 5, 'Street': 8, 'Residential': 15, 'Unknown': 12}.get(rt, 12)
        smooth_k = rt_smooth
        geo_ratio = np.clip(
            (n * raw_ratio + smooth_k * gp5_prior_val) / (n + smooth_k),
            0.1, 10.0
        )
    else:
        # Low base demand: use additive delta anchored on gp5
        expected_delta = (gp5_prior_val - 1.0) * m48
        smooth_k = 15
        smoothed_delta = (n * delta + smooth_k * expected_delta) / (n + smooth_k)
        geo_ratio = np.clip(1.0 + smoothed_delta / max(m48, 0.01), 0.1, 10.0)

    return geo_ratio

print("\nComputing hierarchical calibration for all geohashes...")
all_ghs = set(test['geohash'].unique()) | set(train['geohash'].unique())
calib_cache = {}
for gh in all_ghs:
    rt = geo_rt.get(gh, d49[d49['geohash'] == gh]['RoadType'].mode()[0] if (d49['geohash'] == gh).any()
                    else (d48[d48['geohash'] == gh]['RoadType'].mode()[0] if (d48['geohash'] == gh).any()
                          else (test[test['geohash'] == gh]['RoadType'].mode()[0] if (test['geohash'] == gh).any() else 'Unknown')))
    gp4_val = gh[:4]
    gp5_val = gh[:5]
    calib_cache[gh] = compute_geo_calib(gh, rt, gp5_val, gp4_val)

# Validate on d49
geo_ts_d48 = d48_all.set_index(['geohash', 'time_slot'])['demand']
gh_d48_all  = d48_all.groupby('geohash')['demand'].mean()
ts_d48_all  = d48_all.groupby('time_slot')['demand'].mean()
global_d48  = d48_all['demand'].mean()

d49_preds_hier = []
for _, row in d49.iterrows():
    base = geo_ts_d48.get((row['geohash'], row['time_slot']),
           gh_d48_all.get(row['geohash'], global_d48))
    calib = calib_cache.get(row['geohash'], 1.0)
    d49_preds_hier.append(np.clip(base * calib, 0.0, 1.0))

r2_hier = r2_score(d49['demand'], d49_preds_hier)
print(f"Hierarchical calib d49 R²: {r2_hier:.6f} → Score: {100*r2_hier:.2f}")

# ── LEAK-FREE GEO STATS ──────────────────────────────────────────────────────
print("\nComputing day-48 geo stats (leak-free)...")

geo_ts_d48_df = d48_all.groupby(['geohash', 'time_slot'])['demand'].agg(
    geo_ts_mean='mean', geo_ts_std='std', geo_ts_count='count'
).reset_index()
geo_ts_d48_df['geo_ts_std'] = geo_ts_d48_df['geo_ts_std'].fillna(0)

gh_d48_df = d48_all.groupby('geohash')['demand'].agg(
    gh_mean='mean', gh_median='median', gh_std='std',
    gh_q25=lambda x: x.quantile(0.25), gh_q75=lambda x: x.quantile(0.75),
    gh_count='count'
).reset_index()
gh_d48_df['gh_std'] = gh_d48_df['gh_std'].fillna(0)
gh_d48_df['gh_iqr'] = gh_d48_df['gh_q75'] - gh_d48_df['gh_q25']

ts_d48_df = d48_all.groupby('time_slot')['demand'].agg(
    ts_mean='mean', ts_std='std', ts_median='median'
).reset_index()

# Road-type × time_slot from d48
rt_ts_d48 = d48_all.groupby(['RoadType', 'time_slot'])['demand'].agg(
    rt_ts_mean='mean', rt_ts_std='std'
).reset_index()
rt_ts_d48['rt_ts_std'] = rt_ts_d48['rt_ts_std'].fillna(0)

gp4_d48_df = d48_all.copy()
gp4_d48_df['gp4'] = gp4_d48_df['geohash'].str[:4]
gp4_stats = gp4_d48_df.groupby('gp4')['demand'].agg(gp4_mean='mean', gp4_std='std').reset_index()

gp5_d48_df = d48_all.copy()
gp5_d48_df['gp5'] = gp5_d48_df['geohash'].str[:5]
gp5_stats = gp5_d48_df.groupby('gp5')['demand'].agg(gp5_mean='mean', gp5_std='std').reset_index()

# Morning stats from d48 per geohash
d48_geo_morn_df = d48_morning.groupby('geohash')['demand'].agg(
    d48_morning_mean='mean', d48_morning_std='std'
).reset_index()
d48_geo_morn_df['d48_morning_std'] = d48_geo_morn_df['d48_morning_std'].fillna(0)

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def decode_geohash_safe(gh):
    try:
        lat, lon = pgh.decode(gh)
        return lat, lon
    except:
        return np.nan, np.nan

def engineer(df, is_day49=False):
    df = df.copy()

    # Spatial
    geo_dec = df['geohash'].apply(decode_geohash_safe)
    df['lat'] = geo_dec.apply(lambda x: x[0])
    df['lon'] = geo_dec.apply(lambda x: x[1])
    df['gp4'] = df['geohash'].str[:4]
    df['gp5'] = df['geohash'].str[:5]

    # Road
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather']  = df['Weather'].fillna('Unknown')
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord']          = df['RoadType'].map(road_order).fillna(0).astype(int)
    df['large_vehicles_allowed'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['has_landmarks']          = (df['Landmarks'] == 'Yes').astype(int)
    df['lanes_sq']               = df['NumberofLanes'] ** 2
    df['lanes_log']              = np.log1p(df['NumberofLanes'])
    df['is_multilane']           = (df['NumberofLanes'] > 2).astype(int)
    df['road_capacity'] = df['road_type_ord'] * df['NumberofLanes'] * (1 + df['large_vehicles_allowed'])

    # Weather
    weather_order = {'Sunny': 4, 'Cloudy': 3, 'Foggy': 2, 'Rainy': 1, 'Snowy': 0, 'Unknown': 2}
    df['weather_ord']    = df['Weather'].map(weather_order).fillna(2).astype(int)
    df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    temp_med = df['Temperature'].median()
    if pd.isna(temp_med): temp_med = 20.0
    df['Temperature'] = df['Temperature'].fillna(temp_med)
    df['temp_sq']  = df['Temperature'] ** 2
    df['is_cold']  = (df['Temperature'] < 5).astype(int)
    df['is_hot']   = (df['Temperature'] > 30).astype(int)

    # Time
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
    df['road_x_lanes']    = df['road_type_ord'] * df['NumberofLanes']
    df['lanes_x_peak']    = df['NumberofLanes'] * df['is_rush_hour']
    df['weather_x_peak']  = df['weather_ord'] * df['is_rush_hour']
    df['highway_x_peak']  = (df['road_type_ord'] == 3).astype(int) * df['is_rush_hour']
    df['temp_x_weather']  = df['Temperature'] * df['weather_ord']
    df['slot_x_road']     = df['time_slot'] * df['road_type_ord']
    df['capacity_x_peak'] = df['road_capacity'] * df['is_rush_hour']

    # Geo stats from d48
    df = df.merge(geo_ts_d48_df, on=['geohash', 'time_slot'], how='left')
    df = df.merge(gh_d48_df, on='geohash', how='left')
    df = df.merge(ts_d48_df, on='time_slot', how='left')
    df = df.merge(gp4_stats, on='gp4', how='left')
    df = df.merge(gp5_stats, on='gp5', how='left')
    df = df.merge(rt_ts_d48, on=['RoadType', 'time_slot'], how='left')
    df = df.merge(d48_geo_morn_df, on='geohash', how='left')

    # Fill missing
    for c in ['geo_ts_mean', 'geo_ts_std', 'geo_ts_count']:
        df[c] = df[c].fillna(global_d48)
    for c in ['gh_mean', 'gh_median', 'gh_std', 'gh_q25', 'gh_q75', 'gh_count', 'gh_iqr']:
        df[c] = df[c].fillna(global_d48)
    for c in ['ts_mean', 'ts_std', 'ts_median']:
        df[c] = df[c].fillna(global_d48)
    for c in ['gp4_mean', 'gp4_std', 'gp5_mean', 'gp5_std']:
        df[c] = df[c].fillna(global_d48)
    for c in ['rt_ts_mean', 'rt_ts_std']:
        df[c] = df[c].fillna(global_d48)
    for c in ['d48_morning_mean', 'd48_morning_std']:
        df[c] = df[c].fillna(global_d48)

    # ── Hierarchical calibration features ──
    df['calib_hier'] = df['geohash'].map(calib_cache).fillna(1.0 if not is_day49 else global_rt_calib)
    df['calib_rt']   = df['RoadType'].map(rt_calib).fillna(global_rt_calib)

    # Day-49 morning demand per geohash (actual observed today)
    df['d49_morning_mean'] = df['geohash'].map(d49_geo['m']).fillna(
        global_rt_calib * df['gh_mean'] if not is_day49 else d49['demand'].mean()
    )
    df['d49_morning_count'] = df['geohash'].map(d49_geo['n']).fillna(0)

    # Calibrated predictions
    df['calib_geo_ts']    = df['geo_ts_mean'] * df['calib_hier']
    df['calib_rt_ts']     = df['rt_ts_mean'] * df['calib_rt']
    df['calib_ts_mean']   = df['ts_mean'] * df['calib_hier']

    # Delta calibration (additive, more stable for low-demand)
    df['calib_delta']       = df['d49_morning_mean'] - df['d48_morning_mean']
    df['calib_geo_ts_add']  = (df['geo_ts_mean'] + df['calib_delta']).clip(0, 1)

    # Geo stat differences
    df['geo_ts_vs_gh']  = df['geo_ts_mean'] - df['gh_mean']
    df['geo_ts_vs_ts']  = df['geo_ts_mean'] - df['ts_mean']
    df['rt_ts_vs_ts']   = df['rt_ts_mean'] - df['ts_mean']

    # Spatial/contextual
    df['location_importance'] = df['gh_mean'] * df['has_landmarks']
    df['rush_x_geo']          = df['is_rush_hour'] * df['gh_mean']
    df['calib_x_rush']        = df['calib_hier'] * df['is_rush_hour']

    return df

print("Engineering features...")
d48_fe = engineer(d48, is_day49=False)
d49_fe = engineer(d49, is_day49=True)
test_fe = engineer(test, is_day49=True)

print(f"Features: {d48_fe.shape[1]}")

DROP_COLS = ['Index', 'demand', 'timestamp', 'geohash',
             'LargeVehicles', 'Landmarks', 'RoadType', 'Weather',
             'gp4', 'gp5']
TARGET = 'demand'

train_all_fe = pd.concat([d48_fe, d49_fe], axis=0).reset_index(drop=True)
y_all = train_all_fe[TARGET]
X_all = train_all_fe.drop(columns=DROP_COLS, errors='ignore')

y_d48 = d48_fe[TARGET]
X_d48 = d48_fe.drop(columns=DROP_COLS, errors='ignore')
y_d49 = d49_fe[TARGET]
X_d49 = d49_fe.drop(columns=DROP_COLS, errors='ignore')

X_test = test_fe.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore')

# Align columns
feat_cols = [c for c in X_all.columns if c in X_test.columns]
X_all  = X_all[feat_cols]
X_d48  = X_d48[[c for c in feat_cols if c in X_d48.columns]]
X_d49  = X_d49[[c for c in feat_cols if c in X_d49.columns]]
X_test = X_test[feat_cols]

print(f"\nX_all: {X_all.shape} | X_test: {X_test.shape}")
print(f"NaN in X_all: {X_all.isnull().sum().sum()} | NaN in X_test: {X_test.isnull().sum().sum()}")

# ── VALIDATION ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Honest d48→d49 validation...")
lgb_val = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.02, num_leaves=255,
    max_depth=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=15,
    random_state=SEED, verbosity=-1
)
lgb_val.fit(X_d48, y_d48)
r2_quick = r2_score(y_d49, lgb_val.predict(X_d49))
print(f"Quick LGB d48→d49 R²: {r2_quick:.6f} → Score: {100*r2_quick:.2f}")

# ── OPTUNA FOR LGB ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Optuna LGB (50 trials, d48→d49 honest validation)...")

# Upweight day-49 rows to make model focus on calibration
n_d48, n_d49 = len(d48_fe), len(d49_fe)
w_d49 = n_d48 / n_d49  # balance: give d49 rows same total weight as d48
sample_weights_all = np.concatenate([
    np.ones(n_d48),
    np.full(n_d49, w_d49)
])

def objective_lgb(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 6000),
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
    m.fit(X_d48, y_d48,
          eval_set=[(X_d49, y_d49)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    return r2_score(y_d49, m.predict(X_d49))

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
print(f"Best LGB d48→d49 R²: {study_lgb.best_value:.6f} → Score: {100*study_lgb.best_value:.2f}")

best_lgb_params = {**study_lgb.best_params,
                   'objective': 'regression', 'metric': 'rmse',
                   'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': SEED}

# ── OPTUNA FOR XGB ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Optuna XGB (30 trials)...")

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror', 'random_state': SEED,
        'tree_method': 'hist', 'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
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

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
print(f"Best XGB d48→d49 R²: {study_xgb.best_value:.6f} → Score: {100*study_xgb.best_value:.2f}")

best_xgb_params = {**study_xgb.best_params,
                   'objective': 'reg:squarederror', 'random_state': SEED,
                   'tree_method': 'hist', 'verbosity': 0}

# ── KFold OOF predictions ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-fold OOF on all train (with d49 upweighting)...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_all))
test_preds_lgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w_tr = sample_weights_all[tr_idx]
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m.predict(X_val)
    test_preds_lgb += m.predict(X_test) / 5
    print(f"  Fold {fold+1}: {r2_score(y_val, oof_lgb[val_idx]):.6f}")

print(f"OOF LGB R²: {r2_score(y_all, oof_lgb):.6f}")

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
    print(f"  Fold {fold+1}: {r2_score(y_val, oof_xgb[val_idx]):.6f}")

print(f"OOF XGB R²: {r2_score(y_all, oof_xgb):.6f}")

# ── CatBoost ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CatBoost...")
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
DROP_CB = ['Index', 'demand', 'timestamp', 'geohash', 'gp4', 'gp5']

def make_cb(df):
    df2 = df.drop(columns=DROP_CB, errors='ignore').copy()
    for c in cat_cols:
        if c in df2.columns:
            df2[c] = df2[c].fillna('Unknown').astype(str)
    return df2

X_cb_all  = make_cb(train_all_fe)[[c for c in make_cb(train_all_fe).columns if c in make_cb(test_fe).columns]]
X_cb_test = make_cb(test_fe)[X_cb_all.columns]
X_cb_d48  = make_cb(d48_fe)[[c for c in X_cb_all.columns if c in make_cb(d48_fe).columns]]
X_cb_d49  = make_cb(d49_fe)[[c for c in X_cb_all.columns if c in make_cb(d49_fe).columns]]

cb_cat_idx = [X_cb_all.columns.tolist().index(c) for c in cat_cols if c in X_cb_all.columns]

cb_params = dict(iterations=3000, learning_rate=0.03, depth=8,
                 l2_leaf_reg=5, min_data_in_leaf=20, random_seed=SEED,
                 eval_metric='R2', loss_function='RMSE', verbose=False, early_stopping_rounds=100)

pool_d48 = Pool(X_cb_d48, y_d48, cat_features=cb_cat_idx)
pool_d49 = Pool(X_cb_d49, y_d49, cat_features=cb_cat_idx)
cb_probe = CatBoostRegressor(**cb_params)
cb_probe.fit(pool_d48, eval_set=pool_d49)
print(f"CB d48→d49 R²: {r2_score(y_d49, cb_probe.predict(X_cb_d49)):.6f}")

oof_cb = np.zeros(len(X_cb_all))
test_preds_cb = np.zeros(len(X_cb_test))
w_all = sample_weights_all

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb_all)):
    X_tr, X_val = X_cb_all.iloc[tr_idx], X_cb_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w_tr = w_all[tr_idx]
    pool_tr  = Pool(X_tr, y_tr, cat_features=cb_cat_idx, weight=w_tr)
    pool_val = Pool(X_val, y_val, cat_features=cb_cat_idx)
    m = CatBoostRegressor(**cb_params)
    m.fit(pool_tr, eval_set=pool_val)
    oof_cb[val_idx] = m.predict(X_val)
    test_preds_cb += m.predict(X_cb_test) / 5
    print(f"  Fold {fold+1}: {r2_score(y_val, oof_cb[val_idx]):.6f}")

print(f"OOF CB R²: {r2_score(y_all, oof_cb):.6f}")

# ── Ensemble ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
best_score, best_W = -np.inf, [0.5, 0.25, 0.25]
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.05, 0.7, 0.05):
        w2 = 1.0 - w0 - w1
        if not (0.05 <= w2 <= 0.8): continue
        sc = r2_score(y_all, w0*oof_lgb + w1*oof_xgb + w2*oof_cb)
        if sc > best_score:
            best_score, best_W = sc, [w0, w1, w2]

W = np.array(best_W) / np.sum(best_W)
print(f"Optimal weights: LGB={W[0]:.3f}, XGB={W[1]:.3f}, CB={W[2]:.3f}")
print(f"Ensemble OOF R²: {best_score:.6f} → Score: {100*best_score:.2f}")

final_preds = np.clip(W[0]*test_preds_lgb + W[1]*test_preds_xgb + W[2]*test_preds_cb, 0, 1)

# Also include calibration formula as a meta-correction
test_fe_full = test_fe.copy()
test_fe_full['calib_formula'] = [
    np.clip(
        geo_ts_d48.get((row['geohash'], row['time_slot']),
                        gh_d48_all.get(row['geohash'], global_d48))
        * calib_cache.get(row['geohash'], 1.0),
        0.0, 1.0
    )
    for _, row in test_fe_full.iterrows()
]

# Blend: model ensemble + calibration formula
alpha = 0.85  # trust model more; calib formula as regularizer
final_blended = np.clip(alpha * final_preds + (1 - alpha) * test_fe_full['calib_formula'].values, 0, 1)

# ── Submissions ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
idx = test['Index']

pd.DataFrame({'Index': idx, 'demand': final_preds}).to_csv('submission.csv', index=False)
pd.DataFrame({'Index': idx, 'demand': final_blended}).to_csv('submission_v3_blended.csv', index=False)

# Pure calibration formula
calib_only = test_fe_full['calib_formula'].values
pd.DataFrame({'Index': idx, 'demand': calib_only}).to_csv('submission_calib_only.csv', index=False)

assert len(idx) == 41778
print("All submissions saved!")

print("\n" + "=" * 60)
print("SUMMARY")
print(f"Hierarchical calib formula d49 R²: {r2_hier:.6f} → Score: {100*r2_hier:.2f}")
print(f"Best LGB d48→d49:                  {study_lgb.best_value:.6f} → Score: {100*study_lgb.best_value:.2f}")
print(f"Best XGB d48→d49:                  {study_xgb.best_value:.6f} → Score: {100*study_xgb.best_value:.2f}")
print(f"Ensemble OOF R²:                   {best_score:.6f} → Score: {100*best_score:.2f}")
print(f"\nPrediction means: ensemble={final_preds.mean():.5f}, blended={final_blended.mean():.5f}")
