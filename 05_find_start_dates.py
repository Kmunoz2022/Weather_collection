"""Step 5: find the exact first date each model has Previous Runs data for KNYC.

Binary search between a date that was empty and a date that had data in the
probe. Then spot-checks later dates for gaps. Writes start_dates.json.

  python 05_find_start_dates.py
"""
import json
import random
from datetime import date, timedelta

from common import ROOT, STATIONS, get_json

S = STATIONS["KNYC"]
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"

# model: (known-empty date, known-has-data date) from the probe table
BOUNDS = {
    "gfs_global":      ("2021-01-01", "2022-07-15"),
    "ncep_hrrr_conus": ("2022-07-15", "2024-01-20"),
    "ecmwf_ifs025":    ("2024-01-20", "2024-07-15"),
    "ncep_nbm_conus":  ("2024-07-15", "2025-01-15"),
    "ecmwf_ifs":       ("2025-01-15", "2026-01-15"),
}


def has_data(model, d):
    params = dict(latitude=S["lat"], longitude=S["lon"], hourly="temperature_2m_previous_day1",
                  models=model, start_date=d.isoformat(), end_date=d.isoformat(), timezone="UTC")
    code, js = get_json(PREV, params)
    if isinstance(js, dict) and "hourly" in js:
        return any(v is not None for k, vals in js["hourly"].items()
                   if k.startswith("temperature_2m_previous_day1") for v in vals)
    return False


def main():
    result = {}
    for model, (lo_s, hi_s) in BOUNDS.items():
        lo, hi = date.fromisoformat(lo_s), date.fromisoformat(hi_s)
        if has_data(model, lo):
            result[model] = {"first_date": f"<= {lo_s}", "note": "data already at lower bound; widen bound"}
            print(f"{model}: has data at {lo_s}, start is earlier")
            continue
        while (hi - lo).days > 1:
            mid = lo + (hi - lo) // 2
            if has_data(model, mid):
                hi = mid
            else:
                lo = mid
        # spot-check for gaps after the start
        span = (date.today() - timedelta(days=2) - hi).days
        samples = sorted(hi + timedelta(days=random.randint(1, span)) for _ in range(8))
        gaps = [d.isoformat() for d in samples if not has_data(model, d)]
        result[model] = {"first_date": hi.isoformat(), "spot_check_gaps": gaps}
        print(f"{model}: first date with data {hi}  | spot-check gaps: {gaps or 'none'}")
    (ROOT / "start_dates.json").write_text(json.dumps(result, indent=1))
    print("\nwrote start_dates.json")


if __name__ == "__main__":
    main()
