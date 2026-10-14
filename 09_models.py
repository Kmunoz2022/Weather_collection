"""Step 9: baselines, Ridge, Random Forest, feature ablation, and the one-time final test.

  python 09_models.py          (needs: pip install pandas numpy scikit-learn matplotlib)

Order of operations (nothing on the final test is used for any choice):
  1. Choose settings with 5 expanding-window folds inside the development period (Tier B, to Apr 2 2026).
  2. Refit on all development data with the chosen settings.
  3. Score ONCE on the final test (Tier A, exact model runs, Apr 3 to Oct 12 2026).
  4. Robustness: score the same fitted models on Tier B days after Apr 3 (proxy forecasts).
"""
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling_common import (DEV_END, FIG, G3, GROUPS, RES, SCOLS, SRC, TEST_START, WINDOWS, block_bootstrap,
                             complete_rows, eligible, fold_splits, load_tier, make_features, make_model, predict)

KEY = ["station", "date"]
GRID = {"ridge": [dict(alpha=a) for a in (1.0, 10.0, 100.0)],
        "rf": [dict(max_depth=d, min_samples_leaf=l) for d, l in itertools.product((4, 8), (10, 30))]}


def cv_mae(f, cols, kind, params):
    errs = []
    for _, tr, va in fold_splits(f):
        m = make_model(kind, **params).fit(f.loc[tr, cols].to_numpy(), f.loc[tr, "resid"].to_numpy())
        errs.append(np.abs(predict(m, f.loc[va], cols) - f.loc[va, "y"].to_numpy()))
    return np.concatenate(errs)


def metrics(pred, y):
    e = np.asarray(pred) - np.asarray(y)
    return {"n": len(e), "MAE": np.abs(e).mean(), "bias": e.mean(), "within1F": (np.abs(e) <= 1).mean(),
            "exact_int": (np.round(pred) == y).mean()}


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    B, A = load_tier("B"), load_tier("A")
    dev0 = eligible(B[B["date"] <= DEV_END]).reset_index(drop=True)
    tierA = eligible(A[A["date"] >= TEST_START]).reset_index(drop=True)
    robB = eligible(B[B["date"] >= TEST_START]).reset_index(drop=True)

    # ---- identical rows for every development comparison ----
    keys = complete_rows(make_features(dev0, dev0, max(WINDOWS)), G3)[KEY]

    def dev_feats(n):
        return make_features(dev0, dev0, n).merge(keys, on=KEY).reset_index(drop=True)

    fm = {n: dev_feats(n) for n in WINDOWS}
    val_idx = np.concatenate([va for _, _, va in fold_splits(fm[WINDOWS[0]])])
    log = {}

    # ---- baselines on validation folds ----
    v = fm[WINDOWS[0]].loc[val_idx]
    base_cv = {f"{k}": np.abs(v[f"{k}_tmax_strict"] - v["y"]).mean() for k in SRC}
    base_cv["avg3"] = np.abs(v["src_mean"] - v["y"]).mean()
    best_src = min(SRC, key=lambda k: base_cv[k])
    bc_cv = {}
    for n in WINDOWS:
        vv = fm[n].loc[val_idx]
        bc_cv[n] = np.abs(vv["bc_avg"] - vv["y"]).mean()
    best_bc_n = min(bc_cv, key=bc_cv.get)
    base_cv["bias_corrected_avg3"] = bc_cv[best_bc_n]
    log.update(best_single_source=best_src, bias_corrected_window=best_bc_n, baseline_cv_mae=base_cv)
    print("VALIDATION MAE of baselines:", {k: round(x, 3) for k, x in base_cv.items()},
          f"| best single source = {best_src}, bias window = {best_bc_n}d\n")

    # ---- tune + ablation (validation only) ----
    abl, chosen = [], {}
    for kind, (gname, cols) in itertools.product(("ridge", "rf"), GROUPS.items()):
        uses_tr = gname.startswith("G3")
        best = None
        for n in (WINDOWS if uses_tr else [30]):
            for p in GRID[kind]:
                e = cv_mae(fm[n], cols, kind, p)
                if best is None or e.mean() < best[0]:
                    best = (e.mean(), n, p, e)
        chosen[(kind, gname)] = (best[1], best[2])
        abl.append({"model": kind, "features": gname, "n_features": len(cols), "cv_MAE": best[0],
                    "window_days": best[1] if uses_tr else None, "params": json.dumps(best[2]),
                    "cv_MAE_vs_avg3": best[0] - base_cv["avg3"]})
        print(f"{kind:5s} {gname:26s} cv MAE {best[0]:.3f}  (avg3 {base_cv['avg3']:.3f})  n={best[1]} {best[2]}")
    abl = pd.DataFrame(abl)
    abl.to_csv(RES / "ablation_validation.csv", index=False)

    # pick final feature set per model by validation only
    final = {}
    for kind in ("ridge", "rf"):
        r = abl[abl.model == kind].sort_values("cv_MAE").iloc[0]
        final[kind] = (r["features"], *chosen[(kind, r["features"])])
        print(f"\nChosen for {kind}: {final[kind]}")
    log["final_choices"] = {k: [v[0], v[1], v[2]] for k, v in final.items()}

    # ---- final fit on all development data ----
    def fit_predict(kind, test, hist):
        gname, n, p = final[kind]
        cols = GROUPS[gname]
        tr = fm[n]
        m = make_model(kind, **p).fit(tr[cols].to_numpy(), tr["resid"].to_numpy())
        ft = make_features(test, hist, n)
        ft = complete_rows(ft, G3)  # identical rows for every method
        return m, ft, predict(m, ft, cols), cols

    def evaluate(test, hist, label):
        out = {}
        mr, ft, pr, cr = fit_predict("ridge", test, hist)
        mf, ft2, pf, cf = fit_predict("rf", test, hist)
        assert len(ft) == len(ft2)
        n_bc = best_bc_n
        fb = make_features(test, hist, n_bc).merge(ft[KEY], on=KEY)
        preds = {k: ft[f"{k}_tmax_strict"].to_numpy() for k in SRC}
        preds["avg3"] = ft["src_mean"].to_numpy()
        preds["bias_corrected_avg3"] = fb.set_index(KEY).loc[list(zip(ft.station, ft.date))]["bc_avg"].to_numpy()
        preds["ridge"], preds["random_forest"] = pr, pf
        y = ft["y"].to_numpy()
        rows = []
        for st in ["ALL"] + sorted(ft.station.unique()):
            mk = np.ones(len(ft), bool) if st == "ALL" else (ft.station == st).to_numpy()
            for name, p in preds.items():
                rows.append({"set": label, "station": st, "method": name, **metrics(p[mk], y[mk])})
        res = pd.DataFrame(rows)
        # uncertainty: 7-day block bootstrap on all stations together
        ci = []
        d = ft["date"].to_numpy()
        for name, p in preds.items():
            ea = p - y
            lo, hi = block_bootstrap(d, ea)
            row = {"set": label, "method": name, "MAE_lo": lo, "MAE_hi": hi}
            for ref in ("avg3", best_src, "bias_corrected_avg3"):
                if name != ref:
                    lo2, hi2 = block_bootstrap(d, ea, preds[ref] - y)
                    row[f"diff_vs_{ref}_lo"], row[f"diff_vs_{ref}_hi"] = lo2, hi2
            ci.append(row)
        return res, pd.DataFrame(ci), (mr, cr, mf, cf), ft, preds

    hist_test = pd.concat([dev0, tierA]).reset_index(drop=True)
    resA, ciA, (mr, cr, mf, cf), ftA, predsA = evaluate(tierA, hist_test, "FINAL TEST (Tier A, exact runs)")
    hist_rob = pd.concat([dev0, robB]).reset_index(drop=True)
    resB, ciB, *_ = evaluate(robB, hist_rob, "Robustness (Tier B proxy, after Apr 3)")
    allres = pd.concat([resA, resB])
    allci = pd.concat([ciA, ciB])
    allres.to_csv(RES / "final_results.csv", index=False)
    allci.to_csv(RES / "final_uncertainty.csv", index=False)

    pd.set_option("display.width", 200)
    for lab, r, c in (("FINAL TEST (Tier A)", resA, ciA), ("ROBUSTNESS (Tier B proxy)", resB, ciB)):
        print(f"\n===== {lab} =====")
        print(r[r.station == "ALL"].drop(columns=["set", "station"]).round(3).to_string(index=False))
        print(c.drop(columns=["set"]).round(2).to_string(index=False))
        print("\nMAE by station:")
        print(r.pivot(index="method", columns="station", values="MAE").round(2).to_string())

    # ---- plain-language sentences generated from the numbers ----
    lines = []
    for lab, r, c in (("final test", resA, ciA), ("robustness set", resB, ciB)):
        a = r[r.station == "ALL"].set_index("method")["MAE"]
        cc = c.set_index("method")
        for m in ("ridge", "random_forest", "bias_corrected_avg3"):
            lo, hi = cc.loc[m, "diff_vs_avg3_lo"], cc.loc[m, "diff_vs_avg3_hi"]
            verdict = "better than" if hi < 0 else "worse than" if lo > 0 else "not distinguishable from"
            lines.append(f"{lab}: {m} MAE {a[m]:.2f} vs simple average {a['avg3']:.2f} (difference 95% CI {lo:+.2f} to {hi:+.2f}): {verdict} the simple average.")
    (RES / "final_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

    # ---- what the models learned ----
    imp = []
    gname, n, p = final["rf"]
    imp += [{"model": "random_forest", "feature": c, "importance": v} for c, v in zip(cf, mf.feature_importances_)]
    imp += [{"model": "ridge_std_coef", "feature": c, "importance": v} for c, v in zip(cr, mr.named_steps["m"].coef_)]
    pd.DataFrame(imp).to_csv(RES / "feature_importance.csv", index=False)
    print("\nRandom-forest importances:\n", pd.DataFrame(imp).query("model=='random_forest'").sort_values("importance", ascending=False).head(8).round(3).to_string(index=False))

    # ---- row-level audit file: every score traces to a forecast and an outcome ----
    aud = ftA[KEY + ["y"] + SCOLS].copy()
    for k, v in predsA.items():
        aud[f"pred_{k}"] = v
    aud.to_csv(RES / "final_test_predictions.csv", index=False)
    json.dump(log, open(RES / "model_selection_log.json", "w"), indent=1, default=str)

    # ---- figure ----
    a = resA[resA.station == "ALL"].set_index("method")["MAE"]
    ci = ciA.set_index("method")
    order = list(a.sort_values().index)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(order, a[order], xerr=[a[order] - ci.loc[order, "MAE_lo"], ci.loc[order, "MAE_hi"] - a[order]], color="#1f6fb2")
    ax.invert_yaxis()
    ax.set_xlabel("MAE on final test (°F), 95% interval")
    ax.set_title("Final test, unseen dates (Apr 3 – Oct 12, 2026)")
    fig.tight_layout()
    fig.savefig(FIG / "final_test_mae.png")


if __name__ == "__main__":
    main()
