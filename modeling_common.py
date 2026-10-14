"""Shared pieces for the analysis scripts: data loading, date splits, features, models.

Design decisions (write these into your report):
  * FINAL TEST  = Tier A (Apr 3 to Oct 12, 2026), exact model runs. Touched once, at the very end.
  * DEVELOPMENT = Tier B up to Apr 2, 2026. All exploration and model selection happens here.
  * Validation  = 5 expanding-window folds inside development. Each fold trains on dates up to 2 days
    before the fold starts (a report is issued the morning after its day, so labels for the last
    two days are not yet known at the cutoff).
  * Trailing-error features for target date D use only days <= D-2, and they keep updating as new
    outcomes are verified (the same policy in validation and in the final test).
  * Every comparison uses the same station-days: rows where all three sources and every feature exist.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, STATIONS

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"

SRC = ["gfs", "hrrr", "ecmwf"]
SCOLS = [f"{k}_tmax_strict" for k in SRC]
DEV_END = pd.Timestamp("2026-04-02")
TEST_START = pd.Timestamp("2026-04-03")
FOLDS = [("2025-02-01", "2025-04-30"), ("2025-05-01", "2025-07-31"), ("2025-08-01", "2025-10-31"),
         ("2025-11-01", "2026-01-31"), ("2026-02-01", "2026-04-02")]
GAP_DAYS = 2
WINDOWS = [7, 14, 30]


def load_tier(tier):
    df = pd.read_csv(PROC / f"station_days_tier{tier}.csv", parse_dates=["date"])
    return df


def eligible(df):
    return df[df["eligible"]].copy()


# ---------- trailing-error features ----------
def trailing(hist, n):
    """Trailing signed bias and MAE per source, per station, using only outcomes dated <= D-2."""
    out = []
    mp = max(3, n // 3)
    for sid, g in hist.groupby("station"):
        g = g.set_index("date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max()))
        res = pd.DataFrame(index=g.index)
        for k in SRC:
            e = g[f"{k}_tmax_strict"] - g["y"]
            res[f"{k}_bias_tr"] = e.shift(2).rolling(n, min_periods=mp).mean()
            res[f"{k}_mae_tr"] = e.abs().shift(2).rolling(n, min_periods=mp).mean()
        eavg = g[SCOLS].mean(axis=1, skipna=False) - g["y"]
        res["avg_bias_tr"] = eavg.shift(2).rolling(n, min_periods=mp).mean()
        res["station"] = sid
        res.index.name = "date"
        out.append(res.reset_index())
    return pd.concat(out)


def make_features(df, hist, n):
    f = df.merge(trailing(hist, n), on=["station", "date"], how="left")
    f["src_mean"] = f[SCOLS].mean(axis=1)
    f["src_spread"] = f[SCOLS].std(axis=1, ddof=0)
    f["src_range"] = f[SCOLS].max(axis=1) - f[SCOLS].min(axis=1)
    doy = f["date"].dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for sid in STATIONS:
        f[f"st_{sid}"] = (f["station"] == sid).astype(float)
    for k in SRC:
        f[f"{k}_bc"] = f[f"{k}_tmax_strict"] - f[f"{k}_bias_tr"]
    f["bc_avg"] = f[[f"{k}_bc" for k in SRC]].mean(axis=1, skipna=False)
    f["resid"] = f["y"] - f["src_mean"]
    return f


G0 = list(SCOLS)
G1 = G0 + ["src_mean", "src_spread", "src_range"]
G2 = G1 + [f"st_{s}" for s in STATIONS] + ["doy_sin", "doy_cos"]
G3 = G2 + [f"{k}_bias_tr" for k in SRC] + [f"{k}_mae_tr" for k in SRC] + ["avg_bias_tr"]
GROUPS = {"G0 sources only": G0, "G1 + consensus & spread": G1,
          "G2 + station & season": G2, "G3 + past errors": G3}


def complete_rows(f, cols):
    need = list(set(cols + SCOLS + ["y", "resid", "bc_avg", "src_mean"]))
    return f.dropna(subset=need).copy()


# ---------- models ----------
def make_model(kind, **p):
    if kind == "ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=p.get("alpha", 10.0)))])
    return RandomForestRegressor(n_estimators=300, max_depth=p.get("max_depth", 6),
                                 min_samples_leaf=p.get("min_samples_leaf", 10), max_features=0.6,
                                 random_state=0, n_jobs=-1)


def predict(model, f, cols):
    """Final temperature prediction = three-source average + learned correction."""
    return f["src_mean"].to_numpy() + model.predict(f[cols].to_numpy())


def fold_splits(dev):
    """Yield (name, train_index, val_index) for expanding-window validation."""
    for i, (a, b) in enumerate(FOLDS, 1):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        val = dev.index[(dev["date"] >= a) & (dev["date"] <= b)]
        tr = dev.index[dev["date"] <= a - pd.Timedelta(days=GAP_DAYS)]
        yield f"fold{i} ({a.date()}..{b.date()})", tr, val


def cv_eval(dev, cols, kind, **params):
    """Per-fold MAE for the model and for the plain three-source average on the same rows."""
    rows, oof = [], []
    for name, tr, va in fold_splits(dev):
        m = make_model(kind, **params).fit(dev.loc[tr, cols].to_numpy(), dev.loc[tr, "resid"].to_numpy())
        p = predict(m, dev.loc[va], cols)
        y = dev.loc[va, "y"].to_numpy()
        base = dev.loc[va, "src_mean"].to_numpy()
        rows.append({"fold": name, "n_val": len(va), "mae": np.abs(p - y).mean(), "mae_avg3": np.abs(base - y).mean()})
        oof.append(pd.DataFrame({"idx": va, "pred": p}))
    return pd.DataFrame(rows), pd.concat(oof)


# ---------- uncertainty ----------
def block_bootstrap(dates, err_a, err_b=None, reps=2000, block=7, seed=0):
    """95% interval for MAE (or MAE difference a - b) by resampling 7-day blocks of dates.
    Stations on the same date always stay together, so nothing is double counted."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    ud, inv = np.unique(dates, return_inverse=True)
    n = len(ud)
    cnt = np.bincount(inv)
    sa = np.bincount(inv, weights=np.abs(err_a))
    sb = np.bincount(inv, weights=np.abs(err_b)) if err_b is not None else np.zeros(n)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    offs = np.arange(block)
    for r in range(reps):
        sel = (rng.integers(0, max(1, n - block + 1), nb)[:, None] + offs).ravel()[:n]
        out[r] = (sa[sel].sum() - sb[sel].sum()) / cnt[sel].sum()
    return np.percentile(out, [2.5, 97.5])
