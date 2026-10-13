"""Step 1: capability probe. Run this FIRST.

Asks Open-Meteo which models/dates actually return data for KNYC, so you can
fix the dataset design before building anything. Prints a table and writes
probe_results.json. Paste the table back to Claude if anything looks odd.
"""
import json
from common import ROOT, STATIONS, get_json

S = STATIONS["KNYC"]
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE = "https://single-runs-api.open-meteo.com/v1/forecast"

# Candidate model ids: the probe tells you which ones the API accepts.
MODELS = ["gfs_seamless", "gfs_global", "ncep_hrrr_conus", "ecmwf_ifs", "ecmwf_ifs025", "ncep_nbm_conus", "icon_seamless"]
PREV_DATES = ["2022-07-15", "2024-01-20", "2024-07-15", "2025-01-15", "2025-07-15", "2026-01-15", "2026-07-15", "2026-10-05"]
SINGLE_RUNS = ["2024-07-15T12:00", "2025-07-15T12:00", "2026-04-03T12:00", "2026-07-15T12:00", "2026-10-12T12:00"]


def nonnull(values):
    return sum(v is not None for v in values)


def probe_previous():
    out = []
    for model in MODELS:
        for d in PREV_DATES:
            params = dict(latitude=S["lat"], longitude=S["lon"], hourly="temperature_2m_previous_day1",
                          models=model, start_date=d, end_date=d, temperature_unit="fahrenheit", timezone=S["tz"])
            code, js = get_json(PREV, params)
            n, err = 0, None
            if isinstance(js, dict) and "hourly" in js:
                for k, v in js["hourly"].items():
                    if k.startswith("temperature_2m_previous_day1"):
                        n = max(n, nonnull(v))
            else:
                err = js.get("reason") if isinstance(js, dict) else str(js)[:80]
            out.append(dict(api="previous_runs", model=model, date=d, http=code, nonnull_hours=n, error=err))
    return out


def probe_single():
    out = []
    for model in MODELS:
        for run in SINGLE_RUNS:
            params = dict(latitude=S["lat"], longitude=S["lon"], hourly="temperature_2m", models=model,
                          run=run, forecast_days=2, temperature_unit="fahrenheit", timezone=S["tz"])
            code, js = get_json(SINGLE, params)
            n, err = 0, None
            if isinstance(js, dict) and "hourly" in js:
                for k, v in js["hourly"].items():
                    if k.startswith("temperature_2m"):
                        n = max(n, nonnull(v))
            else:
                err = js.get("reason") if isinstance(js, dict) else str(js)[:80]
            out.append(dict(api="single_runs", model=model, date=run, http=code, nonnull_hours=n, error=err))
    return out


if __name__ == "__main__":
    rows = probe_previous() + probe_single()
    (ROOT / "probe_results.json").write_text(json.dumps(rows, indent=1))
    print(f"{'api':14} {'model':16} {'date':18} {'http':>4} {'hours':>5}  error")
    for r in rows:
        print(f"{r['api']:14} {r['model']:16} {r['date']:18} {r['http']:>4} {r['nonnull_hours']:>5}  {r['error'] or ''}")
    print("\nRead it as: hours > 0 means that model has data on that date. Zero + an error means the model id or date is not supported.")
