"""
Step 4: The push model (the brain of the cost alert).

Question it answers:
    "Given the closing imbalance at 3:55 PM, how much will this stock's price
     get pushed by the 4:00 PM close (relative to the market)?"

Trained on year 1 (train) ONLY, then scored once on year 2 (test).

Run from the project folder (after build_table.py):
    python models.py

Outputs (in results/):
    push_model.json          model coefficients + test-year accuracy (used by cost_alert.py)
    push_by_group.png        push by imbalance size, train vs test  (does the effect hold?)
    push_calibration.png     predicted vs actual push on the test year
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results")
TARGET = "push_bps"             # 3:55 -> close, market-adjusted (bps)

BLUE, RED, GRAY, INK, INK2 = "#2a78d6", "#e34948", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
    "axes.grid": True, "grid.color": "#ebeae6", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
})


# --------------------------------------------------------------------------
# Features (all known at 3:55 PM)
# --------------------------------------------------------------------------
def add_model_inputs(df):
    df = df.copy()
    sign = np.sign(df["imb"])
    # Square-root of imbalance size: price impact grows less than proportionally
    # with order size (the well-known "square-root law" of market impact).
    df["x_sqrt_imb"] = sign * np.sqrt(df["imb_pct_adv"].abs())
    df["x_imb_to_paired"] = df["imb_to_paired"].replace([np.inf, -np.inf], np.nan)
    df["x_growth"] = sign * df["imb_growth_pct_adv"]
    # Nasdaq's own indicated closing price vs the current price (if the feed has it)
    if "ind_match_price" in df:
        ok = (df["ind_match_price"] > 0) & (df["ref_price"] > 0)
        move = 1e4 * (df["ind_match_price"] / df["ref_price"] - 1)
        df["x_indicated_move"] = move.where(ok & (move.abs() < 1000))
    return df


def choose_features(train):
    feats = ["x_sqrt_imb", "x_imb_to_paired", "x_growth"]
    if "x_indicated_move" in train and train["x_indicated_move"].notna().mean() > 0.5:
        feats.append("x_indicated_move")
    return feats


# --------------------------------------------------------------------------
# Model: ordinary least squares with outlier clipping (bounds from year 1 only)
# --------------------------------------------------------------------------
def fit(train, feats):
    clip = {f: [float(train[f].quantile(0.01)), float(train[f].quantile(0.99))] for f in feats}
    X = prep_X(train, feats, clip)
    y_lo, y_hi = train[TARGET].quantile([0.005, 0.995])
    y = train[TARGET].clip(y_lo, y_hi).values        # don't let a few crazy closes dominate
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"features": feats, "clip": clip,
            "intercept": float(coef[0]), "coef": dict(zip(feats, map(float, coef[1:])))}


def prep_X(df, feats, clip):
    return np.column_stack([df[f].fillna(0).clip(*clip[f]).values for f in feats])


def predict(df, model):
    X = prep_X(df, model["features"], model["clip"])
    w = np.array([model["coef"][f] for f in model["features"]])
    return model["intercept"] + X @ w


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def push_by_group_chart(train, test, path):
    cuts = np.quantile(train["imb_pct_adv"], np.linspace(0, 1, 11))   # groups from year 1
    cuts[0], cuts[-1] = -np.inf, np.inf
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, part, name in [(axes[0], train, "Train year"), (axes[1], test, "TEST year")]:
        grp = pd.cut(part["imb_pct_adv"], cuts, labels=range(1, 11))
        means = part.groupby(grp, observed=True)[TARGET].mean()
        ax.bar(means.index.astype(str), means.values, width=0.6,
               color=[BLUE if v >= 0 else RED for v in means.values])
        for x, v in zip(means.index.astype(str), means.values):
            ax.annotate(f"{v:+.1f}", (x, v), ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color=INK, xytext=(0, 3 if v >= 0 else -3),
                        textcoords="offset points")
        ax.axhline(0, color=INK2, lw=1)
        ax.margins(y=0.18)
        ax.set_title(f"{name}", loc="left")
        ax.set_xlabel("1 = biggest SELL imbalance  ...  10 = biggest BUY")
    axes[0].set_ylabel("Price move 3:55 PM -> close, vs market (bps)")
    fig.suptitle("Closing imbalances push the closing price, in both years",
                 x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_chart(test, path):
    q = pd.qcut(test["pred_push"], 10, labels=False, duplicates="drop")
    g = test.groupby(q).agg(pred=("pred_push", "mean"), actual=(TARGET, "mean"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    lim = max(g.abs().max().max() * 1.2, 1)
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=1, ls="--", label="Perfect prediction")
    ax.scatter(g["pred"], g["actual"], s=60, color=BLUE, zorder=3,
               edgecolor="#fcfcfb", linewidth=2, label="Test-year groups (10% each)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Predicted push (bps)")
    ax.set_ylabel("Actual push (bps)")
    ax.set_title("Push model on the TEST year", loc="left")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run():
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet("data/features.parquet")
    df = add_model_inputs(df[~df["bad_row"]])
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    # ---------- 1. Does the push hold up in the test year? ----------
    push_by_group_chart(train, test, OUT / "push_by_group.png")
    print("Step 1: push by imbalance size (bps, market-adjusted)")
    for name, part in [("train", train), ("test", test)]:
        big = part[part["imb_pct_adv"].abs() >= train["imb_pct_adv"].abs().quantile(0.9)]
        crowded = (np.sign(big["imb"]) * big[TARGET])
        t = crowded.mean() / (crowded.std() / np.sqrt(len(crowded)))
        print(f"  {name:5s}: biggest 10% of imbalances push the close {crowded.mean():+.2f} bps "
              f"in their direction (t-stat {t:.1f}, n={len(crowded):,})")

    # ---------- 2. Train the model on year 1 ----------
    feats = choose_features(train)
    model = fit(train, feats)
    test["pred_push"] = predict(test, model)
    train["pred_push"] = predict(train, model)

    # ---------- 3. Score it once on year 2 ----------
    def score(part):
        y, p = part[TARGET], part["pred_push"]
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        big = part[part["imb_pct_adv"].abs() >= train["imb_pct_adv"].abs().quantile(0.9)]
        direction = (np.sign(big["pred_push"]) == np.sign(big[TARGET])).mean()
        return {"corr": round(float(y.corr(p)), 3), "r2": round(float(r2), 4),
                "direction_hit_rate_top10": round(float(direction), 3),
                "avg_crowded_push_top10_bps": round(float((np.sign(big["imb"]) * big[TARGET]).mean()), 2)}

    model["train_score"] = score(train)
    model["test_score"] = score(test)
    model["notes"] = ("OLS on year 1 (Oct 2024 - Sep 2025). Target: 3:55 PM -> close move, "
                      "market-adjusted, in bps. Positive = price pushed up.")
    with open(OUT / "push_model.json", "w") as f:
        json.dump(model, f, indent=2)
    calibration_chart(test, OUT / "push_calibration.png")

    print(f"\nStep 2: push model features: {feats}")
    for f in feats:
        print(f"  {f:18s} coef {model['coef'][f]:+.3f}")
    print("\nStep 3: accuracy (a few % R^2 is normal for 5-minute price moves)")
    print(f"  train: {model['train_score']}")
    print(f"  TEST : {model['test_score']}")

    s = model["test_score"]
    usd = 10_000_000 * abs(s["avg_crowded_push_top10_bps"]) / 1e4
    print(f"\nIn plain words: on big-imbalance days in the TEST year, a closing order on the "
          f"crowded side paid about {s['avg_crowded_push_top10_bps']:+.1f} bps extra "
          f"(~${usd:,.0f} on a $10M order).")
    print("\nSaved results/push_model.json, results/push_by_group.png, results/push_calibration.png")


if __name__ == "__main__":
    run()
