"""Step 8: exploratory data analysis on the DEVELOPMENT period only (Tier B, up to Apr 2, 2026).

The final test period is not looked at here. Writes tables to results/ and figures to results/figures/.

  python 08_eda.py          (needs: pip install pandas numpy matplotlib scikit-learn)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling_common import DEV_END, FIG, PROC, RES, SCOLS, SRC, eligible, load_tier

COLORS = {"gfs": "#1f6fb2", "hrrr": "#d9822b", "ecmwf": "#2a9d8f", "avg3": "#444444"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25})


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    b = load_tier("B")
    dev_all = b[b["date"] <= DEV_END]
    dev = eligible(dev_all)
    dev["month"] = dev["date"].dt.to_period("M").dt.to_timestamp()
    dev["avg3"] = dev[SCOLS].mean(axis=1)
    for k in SRC:
        dev[f"{k}_err"] = dev[f"{k}_tmax_strict"] - dev["y"]
    dev["avg3_err"] = dev["avg3"] - dev["y"]
    dev["spread"] = dev[SCOLS].std(axis=1, ddof=0)
    dev["season"] = dev["date"].dt.month.map(lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5)
                                             else "JJA" if m in (6, 7, 8) else "SON")

    # 1. coverage / missingness
    cov = pd.DataFrame({"station_days": dev_all.groupby("station").size(),
                        "eligible": dev.groupby("station").size()})
    cov["excluded"] = cov["station_days"] - cov["eligible"]
    cov.to_csv(RES / "eda_coverage.csv")
    print("1. COVERAGE (development period)\n", cov.to_string(), "\n")

    # 2. error by source and station
    rows = []
    for st, g in [("ALL", dev)] + list(dev.groupby("station")):
        for k in SRC + ["avg3"]:
            e = g[f"{k}_err"]
            rows.append({"station": st, "source": k, "n": len(e), "MAE": e.abs().mean(), "bias": e.mean(), "RMSE": np.sqrt((e ** 2).mean())})
    err = pd.DataFrame(rows)
    err.to_csv(RES / "eda_error_by_source_station.csv", index=False)
    print("2. ERROR BY SOURCE AND STATION (deg F; bias = forecast - observed)\n", err.round(2).to_string(index=False), "\n")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    stations = ["ALL"] + sorted(dev["station"].unique())
    w = 0.2
    for i, k in enumerate(SRC + ["avg3"]):
        vals = [err[(err.station == s) & (err.source == k)]["MAE"].iloc[0] for s in stations]
        ax.bar(np.arange(len(stations)) + (i - 1.5) * w, vals, w, label=k, color=COLORS[k])
    ax.set_xticks(range(len(stations)), stations)
    ax.set_ylabel("Mean absolute error (°F)")
    ax.set_title("Forecast error by source and station (development period)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(FIG / "eda_mae_by_source_station.png")
    plt.close(fig)

    # 3. does bias vary by station and season?
    bias = dev.groupby(["station", "season"])[[f"{k}_err" for k in SRC]].mean().round(2)
    bias.columns = SRC
    bias.to_csv(RES / "eda_bias_station_season.csv")
    print("3. SIGNED BIAS BY STATION AND SEASON (deg F)\n", bias.to_string(), "\n")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (st, g) in zip(axes, dev.groupby("station")):
        mb = g.groupby("month")[[f"{k}_err" for k in SRC]].mean()
        for k in SRC:
            ax.plot(mb.index, mb[f"{k}_err"], marker="o", ms=3, color=COLORS[k], label=k)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(st)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Mean signed error (°F)")
    axes[0].legend(frameon=False)
    fig.suptitle("Monthly bias by source (positive = forecast too warm)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "eda_monthly_bias.png", bbox_inches="tight")
    plt.close(fig)

    # 4. does disagreement between sources predict larger errors?
    rho = dev[["spread"]].assign(abs_err=dev["avg3_err"].abs()).corr(method="spearman").iloc[0, 1]
    dev["spread_bin"] = pd.qcut(dev["spread"], 5, duplicates="drop")
    sp = dev.groupby("spread_bin", observed=True).agg(mean_spread=("spread", "mean"), mean_abs_err=("avg3_err", lambda x: x.abs().mean()), n=("spread", "size"))
    sp.to_csv(RES / "eda_spread_vs_error.csv")
    print(f"4. DISAGREEMENT VS ERROR: Spearman correlation between spread and |avg3 error| = {rho:.3f}\n", sp.round(2).to_string(), "\n")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(sp["mean_spread"], sp["mean_abs_err"], marker="o", color=COLORS["avg3"])
    ax.set_xlabel("Spread across the three sources (°F, std dev)")
    ax.set_ylabel("Mean |error| of the 3-source average (°F)")
    ax.set_title("Does disagreement signal larger error?")
    fig.tight_layout()
    fig.savefig(FIG / "eda_spread_vs_error.png")
    plt.close(fig)

    # 5. how much do the sources agree with each other, and how often is the official high late in the day?
    corr = dev[[f"{k}_err" for k in SRC]].corr().round(2)
    corr.index = corr.columns = SRC
    corr.to_csv(RES / "eda_error_correlation.csv")
    print("5. CORRELATION OF ERRORS BETWEEN SOURCES (high = they make the same mistakes)\n", corr.to_string(), "\n")
    late = (dev["y_minute_lst"] >= 20 * 60).mean()
    print(f"   Share of days where the official high came at/after 20:00 local standard time: {late:.1%}\n")
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
