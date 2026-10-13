"""Shared config and helpers. Stdlib + requests only."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

# Settlement stations (Kalshi daily-high markets). Coordinates are approximate;
# confirm each against the market's rules text and the station metadata.
STATIONS = {
    "KNYC": {"city": "New York (Central Park)", "lat": 40.7789, "lon": -73.9692, "tz": "America/New_York"},
    "KMDW": {"city": "Chicago (Midway)",        "lat": 41.7868, "lon": -87.7522, "tz": "America/Chicago"},
    "KMIA": {"city": "Miami",                   "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York"},
}

# NWS requires an identifying User-Agent. Put a real contact here.
USER_AGENT = "(weather-forecast-project, https://github.com/Kmunoz2022/Weather_collection)"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def get_json(url, params=None, headers=None, retries=4, timeout=60):
    """GET with exponential backoff. Returns (status_code, parsed_json_or_text)."""
    delay = 5
    last = (0, "no attempt")
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last = (r.status_code, r.text[:300])
                time.sleep(delay)
                delay *= 2
                continue
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, r.text[:300]
        except requests.RequestException as e:
            last = (0, str(e))
            time.sleep(delay)
            delay *= 2
    return last
