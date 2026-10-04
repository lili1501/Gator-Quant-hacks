"""
Robustness check: does a fancier model predict the closing push better than our linear model?

This is a COMPARISON ONLY. It does not change H6 or any frozen rule. Every model:
  - is trained on year 1 (in-sample) only, with FIXED settings (no tuning on year 2)
  - is scored ONCE on year 2 (out-of-sample)
  - predicts the same target: 3:55 PM -> close move, market-adjusted (bps)

Fair design: every model family is run on the SAME input sets.
  Families:   Linear (OLS) | Random forest | Gradient boosting (scikit-learn) | XGBoost
  Input sets: "3 inputs"  = exactly the inputs of our frozen linear model
              "10 inputs" = those 3 + 7 more pre-3:55 PM features, given to every family
  Plus a zero-prediction "no skill" floor.
  -> Same-input rows answer "is a fancier model better?"
  -> 3-vs-10 rows answer "does more information help?" (for every family equally)
  The "Linear, 3 inputs" row is our frozen model exactly (from results/push_model.json).

Run (after models.py):  python src/model_comparison.py
Outputs: results/model_comparison.csv, results/fig_model_comparison.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import BLUE, GRAY, INK2
from models import add_model_inputs, predict, TARGET

warnings.filterwarnings("ignore")
OUT = Path("results")
SEED = 0

BASE = ["x_sqrt_imb", "x_imb_to_paired", "x_growth"]          # inputs of the frozen model
RICH = BASE + ["imb_pct_adv", "intraday_ret_bps", "move_with_imb", "prev_volume_ratio",
               "vol_20d", "imb_flipped", "is_special_day"]


def prep(df, cols, clip):
    X = df[cols].astype(float).copy()
    for c in cols:
        X[c] = X[c].fillna(0).clip(*clip[c])
    return X.values


def score(y, p, part, top_cut):
    """Accuracy + economic value on one sample."""
    y, p = np.asarray(y), np.asarray(p)
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    corr = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else 0.0
    big = part["imb_pct_adv"].abs().values >= top_cut
    direction = (np.sign(p[big]) == np.sign(y[big])).mean() if np.std(p) > 0 else 0.5
    # Economic value: trade the 10% with the largest predicted |push| in the predicted direction
    k = max(1, int(0.1 * len(p)))
    idx = np.argsort(-np.abs(p))[:k]
    captured = (np.sign(p[idx]) * y[idx]).mean() if np.std(p) > 0 else 0.0
    return {"corr": round(float(corr), 3), "r2_pct": round(100 * float(r2), 2),
            "direction_hit_top10_imb": round(float(direction), 3),
            "captured_bps_top10_pred": round(float(captured), 2)}


def run():
    df = pd.read_parquet("data/features.parquet")
    df = add_model_inputs(df[~df["bad_row"]])
    df = df[df[TARGET].notna()]
    train, test = df[df["split"] == "train"].sort_values("date"), df[df["split"] == "test"]
    top_cut = train["imb_pct_adv"].abs().quantile(0.9)

    clip = {c: (float(train[c].astype(float).quantile(0.01)), float(train[c].astype(float).quantile(0.99)))
            for c in RICH}
    y_lo, y_hi = train[TARGET].quantile([0.005, 0.995])
    y_tr = train[TARGET].clip(y_lo, y_hi).values

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

    # Fixed settings for every family (no tuning on year 2)
    families = {
        "Linear": lambda: LinearRegression(),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=SEED),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
            early_stopping=True, validation_fraction=0.2, random_state=SEED),
    }
    try:
        from xgboost import XGBRegressor
        families["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=200,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    except Exception:
        print("xgboost not installed: skipping it (pip install xgboost to include).")

    preds = {("No skill (predict 0)", "-"): (np.zeros(len(train)), np.zeros(len(test)))}
    model = json.loads((OUT / "push_model.json").read_text())
    for fam, make in families.items():
        for set_name, cols in [("3 inputs", BASE), ("10 inputs", RICH)]:
            if fam == "Linear" and set_name == "3 inputs":
                # exactly our frozen model, as trained in models.py
                preds[(fam, set_name)] = (predict(train, model), predict(test, model))
                continue
            m = make()
            m.fit(prep(train, cols, clip), y_tr)
            preds[(fam, set_name)] = (m.predict(prep(train, cols, clip)), m.predict(prep(test, cols, clip)))

    rows = []
    for (fam, set_name), (p_tr, p_te) in preds.items():
        tr = score(train[TARGET], p_tr, train, top_cut)
        te = score(test[TARGET], p_te, test, top_cut)
        rows.append({"model": fam, "inputs": set_name,
                     **{f"train_{k}": v for k, v in tr.items()},
                     **{f"test_{k}": v for k, v in te.items()}})
    tab = pd.DataFrame(rows)
    tab["overfit_gap_corr"] = (tab["train_corr"] - tab["test_corr"]).round(3)
    tab.to_csv(OUT / "model_comparison.csv", index=False)

    # Chart: test-year correlation, each family on the same two input sets
    ORANGE = "#eb6834"
    fam_names = [f for f in tab["model"].unique() if f != "No skill (predict 0)"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(fam_names))
    for off, set_name, color in [(-0.2, "3 inputs", BLUE), (0.2, "10 inputs", ORANGE)]:
        vals = [tab[(tab["model"] == f) & (tab["inputs"] == set_name)]["test_corr"].iloc[0] for f in fam_names]
        ax.barh(y + off, vals, height=0.38, color=color, label=f"Same {set_name} for every model")
        for yi, v in zip(y + off, vals):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f + (" (ours = 3 inputs)" if f == "Linear" else "") for f in fam_names])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Test-year correlation between predicted and actual push (trained on year 1)")
    ax.set_title("Fair comparison: same inputs for every model", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "fig_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    show = ["model", "inputs", "train_corr", "test_corr", "overfit_gap_corr", "test_r2_pct",
            "test_direction_hit_top10_imb", "test_captured_bps_top10_pred"]
    print("=== MODEL COMPARISON (all trained on year 1, scored once on year 2) ===\n")
    print(tab[show].to_string(index=False))
    ours = tab[(tab["model"] == "Linear") & (tab["inputs"] == "3 inputs")]["test_corr"].iloc[0]
    for set_name in ["3 inputs", "10 inputs"]:
        part = tab[tab["inputs"] == set_name].sort_values("test_corr", ascending=False).iloc[0]
        print(f"Best with {set_name}: {part['model']} (test corr {part['test_corr']}).")
    print(f"Our frozen model (Linear, 3 inputs): test corr {ours}.")
    print("This is a robustness check only; H6 still uses its frozen rule.")
    print("Saved results/model_comparison.csv, results/fig_model_comparison.png")


if __name__ == "__main__":
    run()
