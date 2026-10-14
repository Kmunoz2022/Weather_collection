"""Step 6: pull the extended (Tier B) forecast history and save raw responses.

For each model, fetches hourly forecasts made 1 and 2 days before the valid time
(temperature_2m_previous_day1 / _day2), plus a few optional weather variables,
for all three stations in one call per date chunk. Times are UTC. Resumable.

  python 06_pull_previous_runs.py --models gfs_global ncep_hrrr_conus ecmwf_ifs025 --start 2024-07-20

Choose --start from start_dates.json (use the LATEST first_date among your
models so all of them cover the same days).
"""
import argparse
import time
from datetime import date, datetime, timedelta

from common import RAW, STATIONS, get_json, save_json

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FULL = ",".join(f"{v}_previous_day{n}" for v in ("temperature_2m",) for n in (1, 2)) + \
    ",cloud_cover_previous_day1,relative_humidity_2m_previous_day1,wind_speed_10m_previous_day1"
TEMP_ONLY = "temperature_2m_previous_day1,temperature_2m_previous_day2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--chunk-days", type=int, default=120)
    ap.add_argument("--sleep", type=float, default=1.5)
    a = ap.parse_args()

    ids = list(STATIONS)
    lats = ",".join(str(STATIONS[i]["lat"]) for i in ids)
    lons = ",".join(str(STATIONS[i]["lon"]) for i in ids)
    start = datetime.fromisoformat(a.start).date()
    end = datetime.fromisoformat(a.end).date()

    for model in a.models:
        d = start
        while d <= end:
            c_end = min(d + timedelta(days=a.chunk_days - 1), end)
            path = RAW / "previous_runs" / model / f"{d.isoformat()}_{c_end.isoformat()}.json"
            if path.exists():
                d = c_end + timedelta(days=1)
                continue
            base = dict(latitude=lats, longitude=lons, models=model, start_date=d.isoformat(),
                        end_date=c_end.isoformat(), timezone="UTC", temperature_unit="fahrenheit")
            code, js = get_json(URL, {**base, "hourly": FULL})
            used = "full"
            if code == 400:  # an optional variable may be unsupported; fall back to temperature only
                code, js = get_json(URL, {**base, "hourly": TEMP_ONLY})
                used = "temp_only"
            if code == 200 and js:
                save_json(path, {"station_order": ids, "model": model, "variables": used, "response": js})
                print(f"saved {model} {d} to {c_end} ({used})")
            else:
                print(f"FAIL {model} {d} to {c_end}: {code} {str(js)[:150]}")
            time.sleep(a.sleep)
            d = c_end + timedelta(days=1)
    print("done")


if __name__ == "__main__":
    main()
