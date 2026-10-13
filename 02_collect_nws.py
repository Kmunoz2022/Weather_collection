"""Step 2: NWS forecast collector. Run every 6 hours on a machine that stays on.

Saves the raw NWS forecast responses for each station, stamped with the time
you fetched them. These can never be recovered later, so keep running it.

  python 02_collect_nws.py            # one collection pass
  python 02_collect_nws.py --loop     # keep running, one pass every 6 hours

Cron alternative (every 6 hours):
  0 */6 * * * cd /path/to/weather-project && python 02_collect_nws.py >> collect.log 2>&1
"""
import sys
import time

from common import RAW, STATIONS, USER_AGENT, get_json, save_json, utc_stamp

HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}


def collect_once():
    stamp = utc_stamp()
    for sid, s in STATIONS.items():
        code, pts = get_json(f"https://api.weather.gov/points/{s['lat']},{s['lon']}", headers=HEADERS)
        if code != 200 or not isinstance(pts, dict):
            print(f"[{stamp}] {sid}: points lookup failed ({code}) {pts}")
            continue
        save_json(RAW / "nws" / sid / f"{stamp}_points.json", pts)
        props = pts["properties"]
        for kind in ("forecast", "forecastHourly", "forecastGridData"):
            url = props.get(kind)
            if not url:
                continue
            c, js = get_json(url, headers=HEADERS)
            if c == 200:
                save_json(RAW / "nws" / sid / f"{stamp}_{kind}.json", js)
                print(f"[{stamp}] {sid} {kind}: saved")
            else:
                print(f"[{stamp}] {sid} {kind}: failed ({c}) {js}")
            time.sleep(1)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            collect_once()
            time.sleep(6 * 3600)
    else:
        collect_once()
