"""
Integrated approach from CURSOR_FINAL_BREAKTHROUGH_PROMPT.md

Core idea (valid for test rows):
- test.csv is Day 49, slots 9–55 (hours ~2:15–13:45).
- train.csv includes Day 49 hours 0–2 only; per-geohash early demand is a strong
  same-day anchor for later slots on Day 49.

Improvements over the raw prompt:
- Prefix-level (gp4) Day-49 early fallback when a test geohash never appears in
  train49 (reduces cold-start error for ~10% of geohashes).
- Scalar median for Temperature (prompt used a buggy column self-median).
- Optional blend with a simple scaled baseline for rows with weak early signal
  (reliability weight), to stabilize leaderboard variance.

Note on OOF R² ~0.999 when training on Day 48 with Day-49 features:
- Targets are Day-48 demand while features include Day-49 morning aggregates.
- Chronologically that uses "future" relative to the label day, so OOF can be
  optimistic vs predicting true future days. Leaderboard (same calendar Day 49)
  is the correct judge; early-Day-49 features are legitimate there.

Usage (from flipkart_traffic/):
  python3 run_solution_breakthrough.py
Outputs: submission_breakthrough.csv
"""
import sys
sys.path.insert(0, "/Users/kamakshipandoh/Desktop/grid/flipkart_traffic/lib")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

# ── Load ─────────────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

for df in (train, test):
    df[["hour", "minute"]] = df["timestamp"].str.split(":", expand=True).astype(int)
    df["slot"] = df["hour"] * 4 + df["minute"] // 15

train48 = train[train["day"] == 48].copy()
train49 = train[train["day"] == 49].copy()

# ── Lookups (Day 48 + Day 49 early) ─────────────────────────────────────────
slot_d48 = (
    train48.groupby(["geohash", "slot"])["demand"]
    .mean()
    .reset_index(name="d48_slot")
)

geo_d48_all = (
    train48.groupby("geohash")["demand"]
    .agg(
        d48_mean="mean",
        d48_std="std",
        d48_max="max",
        d48_q75=lambda x: x.quantile(0.75),
        d48_q25=lambda x: x.quantile(0.25),
    )
    .reset_index()
)

d48_early = (
    train48[train48["hour"] < 3]
    .groupby("geohash")["demand"]
    .agg(d48_early_mean="mean", d48_early_max="max")
    .reset_index()
)

d49_early = (
    train49.groupby("geohash")["demand"]
    .agg(d49_early_mean="mean", d49_early_max="max", d49_early_std="std")
    .reset_index()
)

# Prefix fallback: geohashes absent in train49 still get a Day-49-ish level
train49_gp = train49.copy()
train49_gp["gp4"] = train49_gp["geohash"].str[:4]
d49_early_gp4 = (
    train49_gp.groupby("gp4")["demand"].mean().reset_index(name="d49_early_gp4")
)

geo_scale = d48_early.merge(d49_early, on="geohash", how="outer")
geo_scale["day_scale_raw"] = geo_scale["d49_early_mean"] / (
    geo_scale["d48_early_mean"] + 1e-9
)
geo_scale["day_scale_robust"] = geo_scale["day_scale_raw"].clip(0.2, 10.0)
GLOBAL_SCALE = float(geo_scale["day_scale_robust"].median())
GLOBAL_D49_EARLY = float(train49["demand"].mean())

d49_slot_lookup = (
    train49.groupby(["geohash", "slot"])["demand"]
    .mean()
    .reset_index(name="d49_slot")
)

ROAD_MAP = {"Unknown": 0, "Residential": 1, "Street": 2, "Highway": 3}
WEATHER_MAP = {"Unknown": 0, "Snowy": 1, "Foggy": 2, "Rainy": 3, "Sunny": 4}
LV_MAP = {"Not Allowed": 0, "Allowed": 1}
LM_MAP = {"No": 0, "Yes": 1}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gp4"] = df["geohash"].str[:4]

    df = df.merge(slot_d48, on=["geohash", "slot"], how="left")
    df = df.merge(geo_d48_all, on="geohash", how="left")
    df = df.merge(d48_early, on="geohash", how="left")
    df = df.merge(d49_early, on="geohash", how="left")
    df = df.merge(geo_scale[["geohash", "day_scale_robust"]], on="geohash", how="left")
    df = df.merge(d49_slot_lookup, on=["geohash", "slot"], how="left")
    df = df.merge(d49_early_gp4, on="gp4", how="left")

    df["day_scale_robust"] = df["day_scale_robust"].fillna(GLOBAL_SCALE)
    df["d48_slot"] = df["d48_slot"].fillna(df["d48_mean"])

    # Early signal: geohash → prefix → global
    df["d49_early_mean"] = df["d49_early_mean"].fillna(df["d49_early_gp4"])
    df["d49_early_mean"] = df["d49_early_mean"].fillna(GLOBAL_D49_EARLY)

    df["d49_early_max"] = df["d49_early_max"].fillna(df["d48_early_max"])
    df["d49_early_std"] = df["d49_early_std"].fillna(0.0)

    med_temp = float(df["Temperature"].median()) if df["Temperature"].notna().any() else 20.0
    df["Temperature"] = df["Temperature"].fillna(med_temp)

    df["d48_slot_scaled"] = df["d48_slot"] * df["day_scale_robust"]
    df["d48_mean_scaled"] = df["d48_mean"] * df["day_scale_robust"]
    df["d49_early_x_d48slot"] = df["d49_early_mean"] * df["d48_slot"]
    df["d49_vs_d48_early"] = (
        df["d49_early_mean"] / (df["d48_early_mean"] + 1e-9)
    ).clip(0.1, 20.0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["slot"] / 96)
    df["is_peak"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)

    df["road_enc"] = df["RoadType"].fillna("Unknown").map(ROAD_MAP).fillna(0).astype(int)
    df["wx_enc"] = df["Weather"].fillna("Unknown").map(WEATHER_MAP).fillna(0).astype(int)
    df["lv_enc"] = df["LargeVehicles"].fillna("Not Allowed").map(LV_MAP).fillna(0).astype(int)
    df["lm_enc"] = df["Landmarks"].fillna("No").map(LM_MAP).fillna(0).astype(int)

    # Reliability of Day-49 early signal: direct geohash > prefix aggregate > global only
    gh_set = set(d49_early["geohash"].values)
    gp_set = set(d49_early_gp4["gp4"].values)
    gh_hit = df["geohash"].isin(gh_set)
    gp_hit = df["gp4"].isin(gp_set)
    df["early_reliability"] = np.where(gh_hit, 1.0, np.where(gp_hit, 0.6, 0.35))

    return df


FEATURES = [
    "slot",
    "hour",
    "minute",
    "hour_sin",
    "hour_cos",
    "slot_sin",
    "slot_cos",
    "is_peak",
    "d48_slot",
    "d48_slot_scaled",
    "d48_mean",
    "d48_mean_scaled",
    "d48_std",
    "d48_max",
    "d48_q75",
    "d48_q25",
    "d48_early_mean",
    "d48_early_max",
    "d49_early_mean",
    "d49_early_max",
    "d49_early_std",
    "d49_early_x_d48slot",
    "day_scale_robust",
    "d49_vs_d48_early",
    "early_reliability",
    "NumberofLanes",
    "road_enc",
    "wx_enc",
    "lv_enc",
    "lm_enc",
    "Temperature",
]

# ── Train matrix: Day 48 scoped to test slot window ─────────────────────────
train48_fe = build_features(train48)
test_fe = build_features(test)

train_scoped = train48_fe[(train48_fe["slot"] >= 9) & (train48_fe["slot"] <= 55)].copy()
X = train_scoped[FEATURES].fillna(0)
y = train_scoped["demand"].values
X_test = test_fe[FEATURES].fillna(0)

print(f"Train48 scoped rows: {len(X)} | Test: {len(X_test)} | Features: {len(FEATURES)}")
print(f"Global scale median: {GLOBAL_SCALE:.4f}")

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=191,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    verbosity=-1,
    n_jobs=-1,
)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(
        X.iloc[tr],
        y[tr],
        eval_set=[(X.iloc[va], y[va])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof[va] = m.predict(X.iloc[va])
    test_pred += m.predict(X_test) / kf.n_splits
    print(f"  Fold {fold+1} R²(val)={r2_score(y[va], oof[va]):.6f}")

print(f"OOF R² (scoped D48 + D49-early features): {r2_score(y, oof):.6f}")

# ── Scaled baseline + reliability blend (stabilizer) ─────────────────────────
baseline = np.clip(
    test_fe["d48_slot_scaled"].fillna(test_fe["d48_mean_scaled"]).fillna(0.0).values,
    0,
    1,
)
rel = test_fe["early_reliability"].values
test_pred = np.clip(test_pred, 0, 1)
# Where early signal weak, pull slightly toward scaled d48 anchor
test_blended = np.clip(
    rel * test_pred + (1 - rel) * (0.65 * test_pred + 0.35 * baseline),
    0,
    1,
)

out = pd.DataFrame({"Index": test["Index"], "demand": np.clip(test_blended, 0, 1)})
out.to_csv("submission_breakthrough.csv", index=False)
assert len(out) == 41778
assert out["demand"].between(0, 1, inclusive="both").all()
print("Saved submission_breakthrough.csv", out["demand"].describe().to_string())
