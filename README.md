# Weather forecast project: day-one scripts

Setup (once):

    pip install requests
    # edit USER_AGENT in common.py to include your real contact email

Run in this order:

1. `python 01_probe.py`  tells you which models and dates really return data. Read the table before anything else.
2. `python 02_collect_nws.py --loop`  start now, leave running. Needs a machine that stays on (always-on computer, small cloud VM, or a scheduled GitHub Action). A laptop that sleeps will leave gaps.
3. `python 03_fetch_labels.py`  official daily highs plus report issue times to `data/labels.csv`.
4. `python 04_snapshot_single_runs.py --models <ids from the probe>`  saves raw forecast runs to `data/raw/single_runs/`.

Notes:
- Raw responses go in `data/raw/` and are never edited. Build everything else from them.
- Station coordinates in `common.py` are approximate. Confirm them against each Kalshi market's rules text.
- Kalshi daily highs settle on local standard time, whole degrees F. Keep that in mind when you compute forecast daily maxima.
