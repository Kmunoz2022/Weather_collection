"""Step 7: build the station-day tables (one row = one station + one target date).

Reads data/raw/* and data/labels.csv, writes data/processed/:
  station_days_tierB.csv   extended history (Feb 2024 on), built from the Previous Runs
                           archive. APPROXIMATE availability: see notes below.
  station_days_tierA.csv   strict test period (Apr 2026 on), built from exact model runs
                           chosen with a documented availability lag.
  exclusions.csv           every station-day dropped from the shared set, with the reason.

Definitions (keep these in your report):
  * Target day = local STANDARD time midnight-to-midnight (how NWS climate reports and Kalshi
    daily highs are defined): KNYC, KMIA = UTC-5, KMDW = UTC-6, all year.
  * Cutoff = 8 p.m. local CIVIL time on the evening before the target date.
  * Forecast daily max = max of the hourly forecast temperatures in the target day. Hourly samples can
    miss the true peak, so these are labeled approximations.
  * *_tmax_full   uses every hour of the target day.
  * *_tmax_strict uses only hours up to cutoff + 24h. In Tier B, "previous_day1" means a forecast made
    about 24h before each valid hour, so later hours may come from a run issued after the cutoff.
    Use *_tmax_strict for Tier B features. Compare against *_tmax_full as a sensitivity check.
  * Tier A picks, for each product, the latest run with init_time + assumed_lag <= cutoff.
    Assumed publication lags (conservative): GFS 6h, ECMWF 6h, HRRR 3h.

Requires: pip install pandas numpy
"""
import json
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import RAW, ROOT, STATIONS

OUT = ROOT / "data" / "processed"
UTC = ZoneInfo("UTC")
LST_OFFSET = {"KNYC": -5, "KMDW": -6, "KMIA": -5}
PRODUCTS = {"gfs": "gfs_global", "hrrr": "ncep_hrrr_conus", "ecmwf": "ecmwf_ifs025", "nbm": "ncep_nbm_conus"}
CORE = ["gfs", "hrrr", "ecmwf"]            # the three sources every comparison must share
LAG_H = {"gfs": 6, "hrrr": 3, "ecmwf": 6}  # assumed publication lag after init time (Tier A)
MAX_RUN_AGE_H = 18                         # Tier A: drop a source if its newest allowed run is older than this
TIER_B_START = date(2024, 2, 6)
TIER_A_START = date(2026, 4, 3)


# ---------- time helpers ----------
def cutoff_utc(sid, d):
    """8 p.m. local civil time on the evening before d, as naive UTC."""
    p = d - timedelta(days=1)
    return datetime(p.year, p.month, p.day, 20, tzinfo=ZoneInfo(STATIONS[sid]["tz"])).astimezone(UTC).replace(tzinfo=None)


def day_windows(sid, d):
    start = datetime(d.year, d.month, d.day) - timedelta(hours=LST_OFFSET[sid])  # LST midnight, naive UTC
    end = start + timedelta(hours=24)
    strict_end = min(end, cutoff_utc(sid, d) + timedelta(hours=24))
    return start, end, strict_end


def window_values(s, start, stop):
    """Series values for hours in [start, stop). Returns (values, expected_count)."""
    lo = s.index.searchsorted(pd.Timestamp(start))
    hi = s.index.searchsorted(pd.Timestamp(stop))
    return s.iloc[lo:hi], int((stop - start).total_seconds() // 3600)


def safe_stat(s, start, stop, fn):
    """Apply fn over the window only if every hour is present; otherwise NaN."""
    vals, expected = window_values(s, start, stop)
    if len(vals) != expected or vals.isna().any() or expected == 0:
        return np.nan
    return float(fn(vals))


def parse_minutes(txt):
    """'237 PM' -> minutes after local midnight (14:37 = 877). 'MM' -> NaN."""
    m = re.fullmatch(r"\s*(\d{1,4})\s*([AP]M)\s*", str(txt))
    if not m:
        return np.nan
    hh, mm = divmod(int(m.group(1)), 100)
    hh = 0 if hh == 12 else hh
    hh += 12 if m.group(2) == "PM" else 0
    return hh * 60 + mm


# ---------- labels ----------
def load_labels():
    df = pd.read_csv(ROOT / "data" / "labels.csv", dtype={"issued_utc": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["y"] = pd.to_numeric(df["high_f"], errors="coerce")
    df["y_minute_lst"] = df["high_time"].map(parse_minutes)
    df = df.sort_values("issued_utc").drop_duplicates(["station", "date"], keep="last")
    # A climate report counts as final only if it was issued after the local-standard-time day ended.
    # Same-evening reports are preliminary and can still change.
    def day_end(r):
        end = datetime(r["date"].year, r["date"].month, r["date"].day) + timedelta(days=1, hours=-LST_OFFSET[r["station"]])
        return end.strftime("%Y%m%d%H%M")
    df["day_end_utc"] = df.apply(day_end, axis=1)
    df["label_state"] = np.where(df["y"].isna(), "no_value",
                                 np.where(df["issued_utc"] < df["day_end_utc"], "not_final", "ok"))
    df.loc[df["label_state"] != "ok", "y"] = np.nan
    return df[["station", "date", "y", "y_minute_lst", "issued_utc", "is_correction", "label_state"]]


# ---------- Tier B ----------
def load_prev(model):
    frames = {sid: [] for sid in STATIONS}
    for f in sorted((RAW / "previous_runs" / model).glob("*.json")):
        d = json.loads(f.read_text())
        for sid, loc in zip(d["station_order"], d["response"]):
            df = pd.DataFrame(loc["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            frames[sid].append(df.set_index("time"))
    out = {}
    for sid, parts in frames.items():
        if parts:
            df = pd.concat(parts).sort_index()
            out[sid] = df[~df.index.duplicated(keep="last")]
    return out


def build_tier_b(labels):
    prev = {k: load_prev(m) for k, m in PRODUCTS.items() if (RAW / "previous_runs" / m).exists()}
    lab = labels.set_index(["station", "date"])
    rows = []
    for sid in STATIONS:
        last = labels.loc[labels.station == sid, "date"].max()
        d = TIER_B_START
        while d <= last:
            start, end, s_end = day_windows(sid, d)
            row = {"station": sid, "date": d}
            if (sid, d) in lab.index:
                r = lab.loc[(sid, d)]
                row.update(y=r["y"], y_minute_lst=r["y_minute_lst"], label_issued_utc=r["issued_utc"], label_state=r["label_state"])
            else:
                row.update(y=np.nan, y_minute_lst=np.nan, label_issued_utc=None, label_state="missing")
            for k, per_station in prev.items():
                df = per_station.get(sid)
                if df is None:
                    continue
                t1, t2 = df.get("temperature_2m_previous_day1"), df.get("temperature_2m_previous_day2")
                row[f"{k}_tmax_strict"] = safe_stat(t1, start, s_end, np.max) if t1 is not None else np.nan
                row[f"{k}_tmax_full"] = safe_stat(t1, start, end, np.max) if t1 is not None else np.nan
                row[f"{k}_tmax_d2_strict"] = safe_stat(t2, start, s_end, np.max) if t2 is not None else np.nan
                for name, col, fn in (("cloud", "cloud_cover_previous_day1", np.mean),
                                      ("rh", "relative_humidity_2m_previous_day1", np.mean),
                                      ("wind", "wind_speed_10m_previous_day1", np.max)):
                    s = df.get(col)
                    row[f"{k}_{name}"] = safe_stat(s, start, s_end, fn) if s is not None else np.nan
            rows.append(row)
            d += timedelta(days=1)
    return pd.DataFrame(rows)


# ---------- Tier A ----------
def load_single(model):
    out = {sid: {} for sid in STATIONS}
    for f in (RAW / "single_runs" / model).glob("*.json"):
        d = json.loads(f.read_text())
        init = datetime.strptime(d["run"], "%Y-%m-%dT%H:%M")
        for sid, loc in zip(d["station_order"], d["response"]):
            h = loc["hourly"]
            out[sid][init] = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), dtype=float)
    return out


def pick_run(runs, sid, d, lag_h):
    cut = cutoff_utc(sid, d)
    for back in (1, 2):
        p = d - timedelta(days=back)
        for hh in (18, 12):
            init = datetime(p.year, p.month, p.day, hh)
            if init + timedelta(hours=lag_h) <= cut and init in runs:
                return init, (cut - init).total_seconds() / 3600
    return None, np.nan


def build_tier_a(labels):
    single = {k: load_single(PRODUCTS[k]) for k in CORE if (RAW / "single_runs" / PRODUCTS[k]).exists()}
    lab = labels.set_index(["station", "date"])
    rows = []
    for sid in STATIONS:
        last = labels.loc[labels.station == sid, "date"].max()
        d = TIER_A_START
        while d <= last:
            start, end, s_end = day_windows(sid, d)
            row = {"station": sid, "date": d}
            if (sid, d) in lab.index:
                r = lab.loc[(sid, d)]
                row.update(y=r["y"], y_minute_lst=r["y_minute_lst"], label_issued_utc=r["issued_utc"], label_state=r["label_state"])
            else:
                row.update(y=np.nan, y_minute_lst=np.nan, label_issued_utc=None, label_state="missing")
            for k, per_station in single.items():
                runs = per_station[sid]
                init, age = pick_run(runs, sid, d, LAG_H[k])
                row[f"{k}_run"] = init.strftime("%Y-%m-%dT%HZ") if init else None
                row[f"{k}_run_age_h"] = age
                if init is None or age > MAX_RUN_AGE_H:
                    row[f"{k}_tmax_full"] = row[f"{k}_tmax_strict"] = np.nan
                else:
                    s = runs[init]
                    row[f"{k}_tmax_full"] = safe_stat(s, start, end, np.max)
                    row[f"{k}_tmax_strict"] = safe_stat(s, start, s_end, np.max)
            rows.append(row)
            d += timedelta(days=1)
    return pd.DataFrame(rows)


# ---------- eligibility + reporting ----------
def mark_eligibility(df, tier):
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["label_state"] != "ok":
            why.append(f"label_{r['label_state']}")
        for k in CORE:
            if pd.isna(r.get(f"{k}_tmax_strict", np.nan)):
                why.append(f"missing_{k}_forecast")
        reasons.append(";".join(why))
    df = df.copy()
    df["exclusion_reason"] = reasons
    df["eligible"] = df["exclusion_reason"] == ""
    df.insert(0, "tier", tier)
    return df


def report(df, name):
    el = df[df.eligible]
    print(f"\n=== {name}: {len(df)} station-days, {len(el)} eligible ({len(df) - len(el)} excluded) ===")
    print(el.groupby("station").size().to_string())
    print(f"date range: {el.date.min()} to {el.date.max()}")
    if len(el):
        print("\nerror vs official high on eligible rows (deg F)   [MAE | bias = forecast - observed]")
        cols = {f"{k}": f"{k}_tmax_strict" for k in CORE}
        cols["avg3"] = None
        avg = el[[f"{k}_tmax_strict" for k in CORE]].mean(axis=1)
        for label, c in cols.items():
            x = avg if c is None else el[c]
            err = x - el["y"]
            print(f"  {label:6} MAE {err.abs().mean():5.2f} | bias {err.mean():+5.2f}")
        late = (el["y_minute_lst"] >= 20 * 60).mean() if el["y_minute_lst"].notna().any() else np.nan
        print(f"\nshare of days where the official high came at/after 20:00 LST: {late:.1%}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    b = mark_eligibility(build_tier_b(labels), "B")
    a = mark_eligibility(build_tier_a(labels), "A")
    b.to_csv(OUT / "station_days_tierB.csv", index=False)
    a.to_csv(OUT / "station_days_tierA.csv", index=False)
    ex = pd.concat([b[~b.eligible], a[~a.eligible]])[["tier", "station", "date", "exclusion_reason"]]
    ex.to_csv(OUT / "exclusions.csv", index=False)

    report(b, "Tier B (extended, approximate availability)")
    report(a, "Tier A (strict test period, exact runs)")

    # how close is the Tier B proxy to the exact-run result on the overlap?
    m = a[a.eligible].merge(b[b.eligible], on=["station", "date"], suffixes=("_A", "_B"))
    if len(m):
        print(f"\n=== Tier B proxy vs exact runs on {len(m)} overlapping station-days (mean abs difference, deg F) ===")
        for k in CORE:
            d_strict = (m[f"{k}_tmax_strict_B"] - m[f"{k}_tmax_strict_A"]).abs().mean()
            d_full = (m[f"{k}_tmax_full_B"] - m[f"{k}_tmax_full_A"]).abs().mean()
            print(f"  {k:6} strict-window {d_strict:4.2f} | full-window {d_full:4.2f}")

    print("\nexclusions by reason:")
    print(ex.groupby(["tier", "exclusion_reason"]).size().to_string())
    print(f"\nwrote {OUT}/station_days_tierB.csv, station_days_tierA.csv, exclusions.csv")


if __name__ == "__main__":
    main()