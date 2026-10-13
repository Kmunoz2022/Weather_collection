"""Step 4: save raw Open-Meteo Single Runs responses to disk.

Retention of this archive is unknown, so keep your own copy. Resumable: files
that already exist are skipped, so you can stop and restart freely.

  python 04_snapshot_single_runs.py --models gfs_seamless ncep_hrrr_conus ecmwf_ifs \
         --start 2026-04-02 --runs 12 18

Use the model ids that 01_probe.py showed working. One call covers all three
stations (comma-separated coordinates), so ~1,200 calls for the full period.
If you hit rate limits the script backs off and keeps going.
"""
import argparse
import time
from datetime import date, datetime, timedelta

from common import RAW, STATIONS, get_json, save_json

URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--start", default="2026-04-02")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--runs", nargs="+", type=int, default=[12, 18], help="UTC init hours to fetch")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    ids = list(STATIONS)
    lats = ",".join(str(STATIONS[i]["lat"]) for i in ids)
    lons = ",".join(str(STATIONS[i]["lon"]) for i in ids)
    d = datetime.fromisoformat(a.start).date()
    end = datetime.fromisoformat(a.end).date()
    done = skipped = failed = 0
    while d <= end:
        for model in a.models:
            for h in a.runs:
                run = f"{d.isoformat()}T{h:02d}:00"
                path = RAW / "single_runs" / model / f"{d.isoformat()}T{h:02d}.json"
                if path.exists():
                    skipped += 1
                    continue
                params = dict(latitude=lats, longitude=lons, hourly="temperature_2m", models=model,
                              run=run, forecast_days=3, temperature_unit="fahrenheit", timezone="UTC")
                code, js = get_json(URL, params)
                if code == 200 and js:
                    save_json(path, {"station_order": ids, "run": run, "model": model, "response": js})
                    done += 1
                else:
                    failed += 1
                    print(f"FAIL {model} {run}: {code} {str(js)[:120]}")
                time.sleep(a.sleep)
        d += timedelta(days=1)
        if d.day == 1 or d > end:
            print(f"through {d - timedelta(days=1)}: saved={done} skipped={skipped} failed={failed}")
    print(f"finished: saved={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
