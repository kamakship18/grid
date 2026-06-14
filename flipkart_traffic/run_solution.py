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

# ── CELL 2: Load Data ───────────────────────────────────────────────────────
print("=" * 60)
print("CELL 2: Loading data...")
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target (demand) stats:\n{train['demand'].describe()}")

# ── CELL 3: Feature Engineering ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 3: Feature Engineering...")

def decode_geohash_safe(gh):
    try:
        lat, lon = pgh.decode(gh)
        return lat, lon
    except:
        return np.nan, np.nan

def engineer_features(df, geohash_demand_stats=None, is_train=True):
    df = df.copy()
    
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['time_in_minutes'] = df['hour'] * 60 + df['minute']
    
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin']   = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos']   = np.cos(2 * np.pi * df['minute'] / 60)
    df['timeslot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['timeslot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday']       = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_rush_hour']    = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)
    
    geo_decoded = df['geohash'].apply(decode_geohash_safe)
    df['lat'] = geo_decoded.apply(lambda x: x[0])
    df['lon'] = geo_decoded.apply(lambda x: x[1])
    
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather']  = df['Weather'].fillna('Unknown')
    
    df['large_vehicles_allowed'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['has_landmarks']          = (df['Landmarks'] == 'Yes').astype(int)
    
    road_order = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['road_type_ord'] = df['RoadType'].map(road_order).fillna(0).astype(int)
    
    df['lanes_sq']     = df['NumberofLanes'] ** 2
    df['lanes_log']    = np.log1p(df['NumberofLanes'])
    df['is_multilane'] = (df['NumberofLanes'] > 2).astype(int)
    
    weather_order = {'Sunny': 4, 'Cloudy': 3, 'Foggy': 2, 'Rainy': 1, 'Snowy': 0, 'Unknown': 2}
    df['weather_ord']    = df['Weather'].map(weather_order).fillna(2).astype(int)
    df['is_bad_weather'] = (df['Weather'].isin(['Rainy', 'Snowy', 'Foggy'])).astype(int)
    
    temp_median = df['Temperature'].median()
    df['Temperature'] = df['Temperature'].fillna(temp_median)
    df['temp_sq']    = df['Temperature'] ** 2
    df['temp_abs']   = df['Temperature'].abs()
    df['is_cold']    = (df['Temperature'] < 5).astype(int)
    df['is_hot']     = (df['Temperature'] > 30).astype(int)
    df['temp_bins']  = pd.cut(df['Temperature'], bins=10, labels=False).astype(int)
    
    df['lanes_x_peak']      = df['NumberofLanes'] * df['is_rush_hour']
    df['road_x_lanes']      = df['road_type_ord'] * df['NumberofLanes']
    df['weather_x_peak']    = df['weather_ord'] * df['is_rush_hour']
    df['lanes_x_landmarks'] = df['NumberofLanes'] * df['has_landmarks']
    df['temp_x_weather']    = df['Temperature'] * df['weather_ord']
    df['highway_x_peak']    = (df['RoadType'] == 'Highway').astype(int) * df['is_rush_hour']
    df['slot_x_road']       = df['time_slot'] * df['road_type_ord']
    
    if geohash_demand_stats is not None:
        df = df.merge(geohash_demand_stats, on='geohash', how='left')
        for col in [c for c in geohash_demand_stats.columns if c != 'geohash']:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def compute_geohash_stats(train_df):
    stats = train_df.groupby('geohash')['demand'].agg(
        gh_mean='mean', gh_median='median', gh_std='std',
        gh_min='min', gh_max='max',
        gh_q25=lambda x: x.quantile(0.25),
        gh_q75=lambda x: x.quantile(0.75),
        gh_count='count', gh_skew='skew',
    ).reset_index()
    stats['gh_std'] = stats['gh_std'].fillna(0)
    stats['gh_iqr'] = stats['gh_q75'] - stats['gh_q25']
    return stats

def compute_timeslot_stats(train_df):
    return train_df.groupby('time_slot')['demand'].agg(
        ts_mean='mean', ts_std='std', ts_median='median'
    ).reset_index()

def compute_geo_timeslot_stats(train_df):
    stats = train_df.groupby(['geohash', 'time_slot'])['demand'].agg(
        geo_ts_mean='mean', geo_ts_std='std'
    ).reset_index()
    stats['geo_ts_std'] = stats['geo_ts_std'].fillna(0)
    return stats

def compute_geo_prefix_stats(train_df, prefix_len=4):
    col = f'geo_prefix_{prefix_len}'
    train_df = train_df.copy()
    train_df[col] = train_df['geohash'].str[:prefix_len]
    stats = train_df.groupby(col)['demand'].agg(
        **{f'gp{prefix_len}_mean': 'mean', f'gp{prefix_len}_std': 'std'}
    ).reset_index()
    stats[f'gp{prefix_len}_std'] = stats[f'gp{prefix_len}_std'].fillna(0)
    return stats


_train_init = train.copy()
_train_init[['hour', 'minute']] = _train_init['timestamp'].str.split(':', expand=True).astype(int)
_train_init['time_slot'] = _train_init['hour'] * 4 + _train_init['minute'] // 15

gh_stats     = compute_geohash_stats(_train_init)
ts_stats     = compute_timeslot_stats(_train_init)
geo_ts_stats = compute_geo_timeslot_stats(_train_init)
gp4_stats    = compute_geo_prefix_stats(_train_init, 4)
gp5_stats    = compute_geo_prefix_stats(_train_init, 5)

print(f"Unique geohashes: {gh_stats.shape[0]}, geo×timeslot combos: {geo_ts_stats.shape[0]}")

# ── CELL 4: Full Feature Pipeline ───────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 4: Full feature pipeline...")

def full_feature_pipeline(df, gh_stats, ts_stats, geo_ts_stats, gp4_stats, gp5_stats):
    df = engineer_features(df, geohash_demand_stats=gh_stats)
    df = df.merge(ts_stats, on='time_slot', how='left')
    df = df.merge(geo_ts_stats, on=['geohash', 'time_slot'], how='left')
    df['geo_ts_mean'] = df['geo_ts_mean'].fillna(df['gh_mean'])
    df['geo_ts_std']  = df['geo_ts_std'].fillna(0)
    df = df.merge(gp4_stats, on='geo_prefix_4', how='left')
    df = df.merge(gp5_stats, on='geo_prefix_5', how='left')
    for c in ['gp4_mean', 'gp4_std', 'gp5_mean', 'gp5_std']:
        df[c] = df[c].fillna(df[c].median())
    df['demand_vs_geo_mean'] = df.get('gh_mean', 0)
    df['geo_ts_vs_geo_mean'] = df['geo_ts_mean'] - df['gh_mean']
    df['effective_capacity']  = df['NumberofLanes'] * df['large_vehicles_allowed'].replace(0, 0.5)
    df['location_importance'] = df['gh_mean'] * df['has_landmarks']
    df['rush_x_geo']          = df['is_rush_hour'] * df['gh_mean']
    return df

train_fe = full_feature_pipeline(train, gh_stats, ts_stats, geo_ts_stats, gp4_stats, gp5_stats)
test_fe  = full_feature_pipeline(test,  gh_stats, ts_stats, geo_ts_stats, gp4_stats, gp5_stats)
print(f"Features after engineering: {train_fe.shape[1]}")

# ── CELL 5: Prepare ML Features ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 5: Preparing ML features...")

DROP_COLS = ['Index', 'demand', 'timestamp', 'geohash',
             'LargeVehicles', 'Landmarks', 'RoadType', 'Weather',
             'geo_prefix_4', 'geo_prefix_5', 'demand_vs_geo_mean']
TARGET = 'demand'

X = train_fe.drop(columns=DROP_COLS, errors='ignore')
y = train_fe[TARGET]
X_test = test_fe.drop(columns=[c for c in DROP_COLS if c != 'demand'], errors='ignore')
X_test = X_test[X.columns]

print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
print(f"NaN in X: {X.isnull().sum().sum()}, NaN in X_test: {X_test.isnull().sum().sum()}")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── CELL 6+7: LightGBM — use best params from previous run (skip Optuna) ────
print("\n" + "=" * 60)
print("CELL 6+7: LightGBM with pre-tuned best params...")

# Best params found in previous Optuna run (50 trials, best R²=0.99588)
best_lgb_params = {
    'n_estimators': 5275,
    'learning_rate': 0.005570938642013016,
    'num_leaves': 298,
    'max_depth': 10,
    'min_child_samples': 27,
    'subsample': 0.932000768506098,
    'colsample_bytree': 0.9256409501083814,
    'reg_alpha': 0.0021973326058218265,
    'reg_lambda': 0.17418586820329307,
    'min_split_gain': 0.0003114028735063698,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': SEED,
}

lgb_model_final = lgb.LGBMRegressor(**best_lgb_params)
lgb_model_final.fit(X, y, callbacks=[lgb.log_evaluation(500)])
lgb_train_r2 = r2_score(y, lgb_model_final.predict(X))
print(f"LGB Train R2: {lgb_train_r2:.6f}")

oof_lgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m.predict(X_val)
    test_preds_lgb += m.predict(X_test) / 5
    print(f"  Fold {fold+1} R2: {r2_score(y_val, oof_lgb[val_idx]):.6f}")

oof_lgb_r2 = r2_score(y, oof_lgb)
print(f"\nOOF LGB R2: {oof_lgb_r2:.6f} -> Score: {100 * oof_lgb_r2:.2f}")

# ── CELL 8: XGBoost ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 8: XGBoost Optuna tuning (30 trials)...")

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 2000, 6000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': SEED,
        'tree_method': 'hist',
        'verbosity': 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scores.append(r2_score(y_val, model.predict(X_val)))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
print(f"Best XGB R2: {study_xgb.best_value:.6f}")

best_xgb_params = study_xgb.best_params.copy()
best_xgb_params.update({
    'objective': 'reg:squarederror',
    'random_state': SEED,
    'tree_method': 'hist',
    'verbosity': 0,
})

oof_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    m = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m.predict(X_val)
    test_preds_xgb += m.predict(X_test) / 5
    print(f"  Fold {fold+1} R2: {r2_score(y_val, oof_xgb[val_idx]):.6f}")

oof_xgb_r2 = r2_score(y, oof_xgb)
print(f"\nOOF XGB R2: {oof_xgb_r2:.6f} -> Score: {100 * oof_xgb_r2:.2f}")

# ── CELL 9: CatBoost ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 9: CatBoost training...")

cat_features_cb = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
                   'geo_prefix_4', 'geo_prefix_5']
DROP_COLS_CB = ['Index', 'demand', 'timestamp', 'geohash', 'demand_vs_geo_mean']

X_cb = train_fe.drop(columns=DROP_COLS_CB, errors='ignore').copy()
X_cb_test = test_fe.drop(columns=[c for c in DROP_COLS_CB if c != 'demand'], errors='ignore').copy()
X_cb_test = X_cb_test[X_cb.columns]

for c in cat_features_cb:
    if c in X_cb.columns:
        X_cb[c] = X_cb[c].fillna('Unknown').astype(str)
        X_cb_test[c] = X_cb_test[c].fillna('Unknown').astype(str)

cb_cat_indices = [X_cb.columns.tolist().index(c) for c in cat_features_cb if c in X_cb.columns]

oof_cb = np.zeros(len(X_cb))
test_preds_cb = np.zeros(len(X_cb_test))

cb_params = dict(
    iterations=5000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3, min_data_in_leaf=10, random_seed=SEED,
    eval_metric='R2', loss_function='RMSE',
    verbose=False, early_stopping_rounds=100,
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb)):
    X_tr, X_val = X_cb.iloc[tr_idx], X_cb.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    train_pool = Pool(X_tr, y_tr, cat_features=cb_cat_indices)
    val_pool   = Pool(X_val, y_val, cat_features=cb_cat_indices)
    m = CatBoostRegressor(**cb_params)
    m.fit(train_pool, eval_set=val_pool)
    oof_cb[val_idx] = m.predict(X_val)
    test_preds_cb += m.predict(X_cb_test) / 5
    print(f"  Fold {fold+1} R2: {r2_score(y_val, oof_cb[val_idx]):.6f}")

oof_cb_r2 = r2_score(y, oof_cb)
print(f"\nOOF CatBoost R2: {oof_cb_r2:.6f} -> Score: {100 * oof_cb_r2:.2f}")

# ── CELL 10: Ensemble ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 10: Ensemble with optimal weights...")

best_score = -np.inf
best_weights = [0.5, 0.25, 0.25]

for w0 in np.arange(0.2, 0.8, 0.1):
    for w1 in np.arange(0.1, 0.6, 0.1):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05:
            continue
        blend = w0 * oof_lgb + w1 * oof_xgb + w2 * oof_cb
        sc = r2_score(y, blend)
        if sc > best_score:
            best_score = sc
            best_weights = [w0, w1, w2]

W = np.array(best_weights)
W = W / W.sum()
print(f"Optimal weights: LGB={W[0]:.3f}, XGB={W[1]:.3f}, CB={W[2]:.3f}")
print(f"Ensemble OOF R2: {best_score:.6f} -> Score: {100 * best_score:.2f}")

final_test_preds = W[0] * test_preds_lgb + W[1] * test_preds_xgb + W[2] * test_preds_cb
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)
print(f"Prediction stats: min={final_test_preds.min():.4f}, max={final_test_preds.max():.4f}, mean={final_test_preds.mean():.4f}")

# ── CELL 11: Neighbor Correction ────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 11: Neighbor-based residual correction...")

lookup = train_fe.groupby(['geohash', 'time_slot'])['demand'].mean()
corrected_preds = final_test_preds.copy()
correction_weight = 0.15

for i, row in test_fe.iterrows():
    key = (row['geohash'], row['time_slot'])
    if key in lookup.index:
        corrected_preds[i] = (1 - correction_weight) * corrected_preds[i] + correction_weight * lookup[key]

corrected_preds = np.clip(corrected_preds, 0.0, 1.0)
print("Neighbor-corrected predictions computed.")

# ── CELL 12: Submission ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 12: Generating submission files...")

submission = pd.DataFrame({'Index': test['Index'], 'demand': final_test_preds})
submission.to_csv('submission.csv', index=False)
print(f"submission.csv saved: {submission.shape}")
print(submission.head())

submission_corrected = pd.DataFrame({'Index': test['Index'], 'demand': corrected_preds})
submission_corrected.to_csv('submission_corrected.csv', index=False)
print(f"\nsubmission_corrected.csv saved: {submission_corrected.shape}")

sample_sub = pd.read_csv('data/sample_submission.csv')
assert list(submission.columns) == ['Index', 'demand'], "Column mismatch!"
assert len(submission) == 41778, f"Row count mismatch: {len(submission)}"
print("\nSubmission validated successfully!")

# ── CELL 13: Feature Importance ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 13: Feature importance...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 features by LGB importance:")
print(feat_imp.head(20).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
feat_imp.head(25).plot(kind='barh', x='feature', y='importance', ax=ax)
ax.set_title('LightGBM Feature Importances (Top 25)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.close()
print("feature_importance.png saved.")

print("\n" + "=" * 60)
print("ALL DONE!")
print(f"OOF LGB R2:        {oof_lgb_r2:.6f} -> Score: {100 * oof_lgb_r2:.2f}")
print(f"OOF XGB R2:        {oof_xgb_r2:.6f} -> Score: {100 * oof_xgb_r2:.2f}")
print(f"OOF CatBoost R2:   {oof_cb_r2:.6f} -> Score: {100 * oof_cb_r2:.2f}")
print(f"Ensemble OOF R2:   {best_score:.6f} -> Score: {100 * best_score:.2f}")
