"""
Step 3: Backtest the closing-auction fade strategy with FROZEN rules.

Run from the project folder (after build_table.py):
    python backtest.py

All rules below were chosen using year 1 (train) only, in 01_check_effect.ipynb.
Do NOT change them after looking at the test-year results; that would make the
test meaningless. If you do change something, add 1 to N_TRIALS and say so.

Outputs (in results/):
    summary.csv               all metrics, train and test
    equity_test.png           cumulative profit, test year
    equity_train.png          cumulative profit, train year
    trades_test.csv           every test-year trade
"""

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= FROZEN RULES (from year 1) =======================
IMB_THRESHOLD = 6.69        # |imbalance| >= 6.69% of 20-day avg dollar volume (year-1 95th pct)
REQUIRE_NOT_FLIPPED = True  # skip if the imbalance changed direction since 3:50
REQUIRE_GREW = True         # keep only if the imbalance grew from 3:50 to 3:55
COST_BPS = 3.0              # round-trip cost per trade (enter at close, exit at open)
N_TRIALS = 32               # ALL variants tried in the project: ~20 fade + 12 push (deflated Sharpe)
SPECIAL_TEST_DAYS = ["2026-09-18", "2026-09-30"]   # triple witching, quarter-end
# ==========================================================================

OUT = Path("results")
BLUE, GRAY, RED, INK2 = "#2a78d6", "#9a9993", "#e34948", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
    "axes.grid": True, "grid.color": "#ebeae6", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
})


# --------------------------------------------------------------------------
# Strategy definitions
# --------------------------------------------------------------------------
def select_trades(df, use_filter):
    """Which stock-days we trade. Uses only information known by 3:55 PM."""
    pick = df["imb_pct_adv"].abs() >= IMB_THRESHOLD
    if use_filter:
        if REQUIRE_NOT_FLIPPED:
            pick &= ~df["imb_flipped"]
        if REQUIRE_GREW:
            pick &= df["imb_growth_pct_adv"] > 0
    return df[pick].copy()


def daily_pnl(trades, all_days, col="fade_bounce_bps", cost=COST_BPS):
    """Equal-weight each day's trades. Days with no trade earn 0 (capital sits idle)."""
    trades = trades.assign(net_bps=trades[col] - cost)
    daily = trades.groupby("date")["net_bps"].mean()
    return daily.reindex(all_days, fill_value=0.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def deflated_sharpe(daily, n_trials):
    """Probability the true Sharpe is above what the best of n_trials random
    strategies would show by luck (Bailey & Lopez de Prado). > 0.95 is strong."""
    r = daily.values
    T = len(r)
    if T < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)                     # per-day Sharpe
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt() + 3                    # pandas gives excess kurtosis
    nd = NormalDist()
    gamma = 0.5772156649                              # Euler-Mascheroni constant
    sr_std = np.sqrt(1.0 / T)                         # spread of Sharpe under no skill
    sr0 = sr_std * ((1 - gamma) * nd.inv_cdf(1 - 1 / n_trials)
                    + gamma * nd.inv_cdf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return nd.cdf((sr - sr0) * np.sqrt(T - 1) / denom)


def perf_stats(daily, gross=None):
    """Standard performance numbers from a daily return series in bps of capital.
    gross: capital deployed each day as a fraction of AUM (default 1 on active days)."""
    if gross is None:
        gross = (daily != 0).astype(float)
    sd = daily.std(ddof=1)
    cum = daily.cumsum()
    years = len(daily) / 252
    return {
        "ann_return_pct": round(daily.mean() * 252 / 100, 2),
        "ann_vol_pct": round(sd * np.sqrt(252) / 100, 2),
        "sharpe": round(daily.mean() / sd * np.sqrt(252), 2) if sd > 0 else np.nan,
        "t_stat": round(daily.mean() / (sd / np.sqrt(len(daily))), 2) if sd > 0 else np.nan,
        "max_drawdown_bps": round((cum - cum.cummax()).min(), 1),
        # enter + exit every position each trading day: 2 x capital deployed, per year
        "turnover_x_per_year": round(2 * gross.sum() / years, 0),
        "skew": round(float(daily.skew()), 2),
        "deflated_sharpe_prob": round(deflated_sharpe(daily, N_TRIALS), 3),
    }


def metrics(trades, daily, label, col="fade_bounce_bps", cost=COST_BPS):
    cum = daily.cumsum()
    active = daily[daily != 0]
    return {
        "strategy": label,
        "trades": len(trades),
        "days_with_trades": int((daily != 0).sum()),
        "trading_days": len(daily),
        "avg_trades_per_active_day": round(len(trades) / max(len(active), 1), 1),
        "avg_net_bps_per_trade": round((trades[col] - cost).mean(), 2),
        "hit_rate_after_costs": round((trades[col] - cost > 0).mean(), 3),
        "total_net_bps": round(cum.iloc[-1], 1),
        **perf_stats(daily),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def equity_chart(curves, title, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (label, daily), color, lw in zip(curves.items(), [BLUE, GRAY, RED], [2.2, 1.5, 1.5]):
        cum = daily.cumsum()
        ax.plot(cum.index, cum.values, color=color, lw=lw, label=label)
        ax.annotate(f"{cum.iloc[-1]:+.0f}", (cum.index[-1], cum.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Cumulative net bps (after costs)")
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
    if "bounce_raw_bps" not in df.columns:
        raise SystemExit("Old features file. Re-run:  python build_table.py")
    df = df[~df["bad_row"]]

    rows, curves = [], {}
    for split in ["train", "test"]:
        part = df[df["split"] == split]
        days = pd.Index(sorted(part["date"].unique()))
        base = select_trades(part, use_filter=False)
        filt = select_trades(part, use_filter=True)

        d_filt = daily_pnl(filt, days)
        d_base = daily_pnl(base, days)
        d_2x = daily_pnl(filt, days, cost=2 * COST_BPS)
        d_raw = daily_pnl(filt, days, col="fade_bounce_raw_bps")

        rows += [
            {"split": split, **metrics(filt, d_filt, "Filtered (main strategy)")},
            {"split": split, **metrics(base, d_base, "Baseline: top 5%, no filter")},
            {"split": split, **metrics(filt, d_2x, f"Filtered, double costs ({2*COST_BPS:.0f} bps)",
                                             cost=2 * COST_BPS)},
            {"split": split, **metrics(filt, d_raw, "Filtered, NOT market-hedged",
                                             col="fade_bounce_raw_bps")},
        ]
        curves[split] = {"Filtered (main strategy)": d_filt,
                         "Baseline: top 5%, no filter": d_base,
                         f"Filtered, double costs": d_2x}
        pd.DataFrame({"date": days, "net_bps": d_filt.values}).to_csv(
            OUT / f"fade_daily_{split}.csv", index=False)
        if split == "test":
            filt.to_csv(OUT / "trades_test.csv", index=False)
            test_filt, test_daily = filt, d_filt

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    equity_chart(curves["train"], "Train year (Oct 2024 - Sep 2025): rules were chosen here",
                 OUT / "equity_train.png")
    equity_chart(curves["test"], "TEST year (Oct 2025 - Sep 2026): never seen while building",
                 OUT / "equity_test.png")

    # ---------------- Print ----------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    show = ["split", "strategy", "trades", "avg_net_bps_per_trade", "ann_return_pct", "ann_vol_pct",
            "sharpe", "t_stat", "max_drawdown_bps", "turnover_x_per_year", "deflated_sharpe_prob"]
    print("\n=== RESULTS (bps = 0.01%; all after costs) ===\n")
    print(summary[show].to_string(index=False))

    print("\n=== Test year by month (main strategy, net bps per day, summed) ===")
    by_month = test_daily.groupby(test_daily.index.to_period("M")).sum().round(1)
    print(by_month.to_string())
    print(f"Positive months: {(by_month > 0).sum()} of {len(by_month)}")

    print("\n=== Special test days ===")
    for day in SPECIAL_TEST_DAYS:
        t = test_filt[test_filt["date"] == pd.Timestamp(day)]
        if t.empty:
            print(f"  {day}: no trades")
        else:
            print(f"  {day}: {len(t)} trades, avg net {(t['fade_bounce_bps'] - COST_BPS).mean():+.1f} bps "
                  f"({', '.join(t['symbol'])})")

    print(f"\nSaved results/summary.csv, results/equity_test.png, results/equity_train.png, "
          f"results/trades_test.csv")
    print(f"Deflated Sharpe uses N_TRIALS = {N_TRIALS}. Raise it if you try more variations.")


if __name__ == "__main__":
    run()
