"""Step 3: official daily highs (the labels) from NWS Daily Climate Reports via IEM.

  python 03_fetch_labels.py            # 2024 through the current year
  python 03_fetch_labels.py 2022 2026  # custom year range

Writes raw JSON per station-year plus data/labels.csv. The `product` field
encodes the report issue time (UTC, yyyymmddhhmm); a trailing -CCA marks a
corrected report. Keep those columns: they are your evidence trail.
"""
import csv
import sys
from datetime import date

from common import RAW, ROOT, STATIONS, get_json, save_json

URL = "https://mesonet.agron.iastate.edu/json/cli.py"


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else date.today().year
    rows = []
    for sid in STATIONS:
        for year in range(y0, y1 + 1):
            code, js = get_json(URL, {"station": sid, "year": year})
            if code != 200 or not isinstance(js, dict):
                print(f"{sid} {year}: failed ({code}) {js}")
                continue
            save_json(RAW / "labels" / f"{sid}_{year}.json", js)
            for r in js.get("results", []):
                prod = r.get("product", "")
                rows.append({
                    "station": sid, "date": r.get("valid"), "high_f": r.get("high"),
                    "high_time": r.get("high_time"), "product": prod,
                    "issued_utc": prod[:12], "is_correction": prod.endswith("-CCA") or "-CC" in prod[-4:],
                })
            print(f"{sid} {year}: {len(js.get('results', []))} days")
    out = ROOT / "data" / "labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    missing = [r for r in rows if r["high_f"] in (None, "M", "")]
    print(f"\nwrote {out} ({len(rows)} rows, {len(missing)} with missing highs)")


if __name__ == "__main__":
    main()
