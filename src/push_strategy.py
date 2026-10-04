"""
Strategy 2: Closing Pressure (trade WITH the closing-auction imbalance into the close).

Economic idea (H1): a large closing imbalance must be absorbed in the 4:00 PM auction, so
the price is pushed in the imbalance's direction between 3:55 PM and the close. We take
that side at 3:55 PM in the continuous market and exit in the closing auction, where our
exit order OFFSETS the imbalance (we supply the liquidity the auction is short of).
Market-neutral: net long/short exposure is hedged with the equal-weight basket.
No overnight risk: flat by 4:00 PM, so after-hours earnings news cannot hit us.

Rule selection uses YEAR 1 ONLY: 12 variants declared below are run on year 1, the best
year-1 Sharpe is frozen to results/push_rule.json, then year 2 is run ONCE.

Run (after build_table.py, models.py; optionally download_quotes.py):
    python push_strategy.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import perf_stats, BLUE, GRAY, RED, INK2  # same metrics and style
from models import add_model_inputs, predict

OUT = Path("results")
QUOTES = Path("data/quotes_355.parquet")

# ===================== FIXED ASSUMPTIONS (set before testing) =====================
MIN_NAMES = 10                # per-name limit: no position larger than 1/10 of capital
FEE_BPS = 0.5                 # exchange/clearing fees, round trip (entry + MOC exit)
HEDGE_COST_BPS = 0.5          # cost of hedging net exposure with the basket, per unit
ASSUMED_HALF_SPREAD_BPS = 1.0 # used ONLY if real 3:55 quotes are not downloaded
# The 12 variants (declared up front; chosen on year 1 only)
GRID_QUANTILES = [0.80, 0.90, 0.95]
GRID_SIGNALS = ["imbalance", "model"]
GRID_FILTERS = [False, True]
# ==================================================================================


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_parquet("data/features.parquet")
    df = df[df["adv_dollars_20d"].notna() & (df["push_raw_bps"].abs() < 2000)
            & (df["ref_price"] > 0)].copy()

    if QUOTES.exists():
        q = pd.read_parquet(QUOTES)
        q["date"] = pd.to_datetime(q["date"])
        df = df.merge(q, on=["date", "symbol"], how="left")
        mid = (df["bid_355"] + df["ask_355"]) / 2
        ok = mid.notna() & ((mid / df["ref_price"] - 1).abs() < 0.02)   # drop bad quotes
        df = df[ok].copy()
        mid = mid[ok]
        df["half_spread_bps"] = 1e4 * (df["ask_355"] - df["bid_355"]) / 2 / mid
        df["push_mid_raw_bps"] = 1e4 * (df["close"] / mid - 1)            # 3:55:01 mid -> close
        df["cost_source"] = "real 3:55 quotes"
    else:
        df["half_spread_bps"] = ASSUMED_HALF_SPREAD_BPS
        df["push_mid_raw_bps"] = df["push_raw_bps"]
        df["cost_source"] = f"assumed {ASSUMED_HALF_SPREAD_BPS} bps half-spread"

    # Market move over the same 5 minutes (equal-weight, all stocks that day)
    df["mkt_push_bps"] = df.groupby("date")["push_mid_raw_bps"].transform("mean")
    df["push_adj_bps"] = df["push_mid_raw_bps"] - df["mkt_push_bps"]

    model = json.loads((OUT / "push_model.json").read_text())   # trained on year 1 only
    df = add_model_inputs(df)
    df["pred_push"] = predict(df, model)
    return df


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------
def score_and_direction(df, signal):
    if signal == "imbalance":
        return df["imb_pct_adv"].abs(), np.sign(df["imb"])
    return df["pred_push"].abs(), np.sign(df["pred_push"])


def select(df, rule):
    score, direction = score_and_direction(df, rule["signal"])
    pick = (score >= rule["threshold"]) & (direction != 0)
    if rule["filter"]:
        pick &= ~df["imb_flipped"] & (df["imb_growth_pct_adv"] > 0)
    t = df[pick].copy()
    t["direction"] = direction[pick]
    return t


def daily_returns(trades, days, cost_mult=1.0, cap=True, hedged=True):
    """Daily P&L in bps of capital. Returns (daily, per-day stats, trades with weights)."""
    t = trades.copy()
    n = t.groupby("date")["symbol"].transform("count")
    t["weight"] = 1.0 / (np.maximum(n, MIN_NAMES) if cap else n)
    move = t["push_adj_bps"] if hedged else t["push_mid_raw_bps"]
    t["gross_bps"] = t["direction"] * move
    t["cost_bps"] = cost_mult * (t["half_spread_bps"] + FEE_BPS)
    t["net_bps"] = t["gross_bps"] - t["cost_bps"]
    t["contrib_bps"] = t["weight"] * t["net_bps"]
    t["signed_weight"] = t["weight"] * t["direction"]
    g = t.groupby("date")
    per_day = pd.DataFrame({
        "pnl": g["contrib_bps"].sum(),
        "gross": g["weight"].sum(),
        "net_exposure": g["signed_weight"].sum(),
        "n_names": g.size(),
    }).reindex(days)
    per_day[["pnl", "gross", "net_exposure", "n_names"]] = per_day[
        ["pnl", "gross", "net_exposure", "n_names"]].fillna(0)
    if hedged:
        per_day["pnl"] -= cost_mult * HEDGE_COST_BPS * per_day["net_exposure"].abs()
    return per_day["pnl"], per_day, t


def summarize(trades, days, label, **kw):
    daily, per_day, t = daily_returns(trades, days, **kw)
    return {
        "strategy": label,
        "trades": len(t),
        "active_days": int((per_day["n_names"] > 0).sum()),
        "avg_names_per_active_day": round(per_day.loc[per_day["n_names"] > 0, "n_names"].mean(), 1),
        "avg_gross_used": round(per_day["gross"].mean(), 2),
        "avg_abs_net_exposure": round(per_day["net_exposure"].abs().mean(), 3),
        "avg_gross_edge_bps": round(t["gross_bps"].mean(), 2),
        "avg_cost_bps": round(t["cost_bps"].mean(), 2),
        "avg_net_bps_per_trade": round(t["net_bps"].mean(), 2),
        "hit_rate_after_costs": round((t["net_bps"] > 0).mean(), 3),
        "total_net_bps": round(daily.sum(), 1),
        **perf_stats(daily, per_day["gross"]),
    }, daily, per_day, t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def choose_rule(train, train_days):
    rows, best = [], None
    for q in GRID_QUANTILES:
        for signal in GRID_SIGNALS:
            score, _ = score_and_direction(train, signal)
            threshold = float(score.quantile(q))
            for filt in GRID_FILTERS:
                rule = {"signal": signal, "quantile": q, "threshold": threshold, "filter": filt}
                stats, *_ = summarize(select(train, rule), train_days, "variant")
                rows.append({**rule, **{k: stats[k] for k in
                             ["trades", "avg_net_bps_per_trade", "sharpe", "t_stat", "max_drawdown_bps"]}})
                if best is None or stats["sharpe"] > best[1]:
                    best = (rule, stats["sharpe"])
    return best[0], pd.DataFrame(rows)


def equity_chart(daily_train, daily_test, daily_2x, rule_text, path):
    d = pd.concat([daily_train, daily_test])
    d2 = pd.concat([daily_2x["train"], daily_2x["test"]])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(d.index, d.cumsum(), color=BLUE, lw=2.2, label="Closing Pressure strategy (after costs)")
    ax.plot(d2.index, d2.cumsum(), color=RED, lw=1.5, label="Same, double costs")
    split = daily_test.index[0]
    ax.axvspan(split, d.index[-1], color=BLUE, alpha=0.06, lw=0)
    ax.axvline(split, color=INK2, lw=1, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(split, ymax, "  TEST year (never seen)", va="top", fontsize=9, color=INK2)
    ax.text(d.index[0], ymax, "Train year: rule chosen here", va="top", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title("Closing Pressure: cumulative net return", loc="left")
    ax.set_ylabel("Cumulative net bps of capital")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.text(0.01, -0.02, rule_text, fontsize=8, color=INK2)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(f"Cost model: {df['cost_source'].iloc[0]}")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    days = {s: pd.Index(sorted(p["date"].unique())) for s, p in [("train", train), ("test", test)]}

    # ---------- 1. Choose the rule on year 1 only ----------
    rule, grid = choose_rule(train, days["train"])
    grid.to_csv(OUT / "push_grid_train.csv", index=False)
    rule["n_variants_tested"] = len(grid)
    rule["chosen_on"] = "train year only (best Sharpe after costs)"
    (OUT / "push_rule.json").write_text(json.dumps(rule, indent=2))
    print("\nYear-1 variants (train only):")
    print(grid.round(2).to_string(index=False))
    rule_text = (f"Rule (frozen from year 1): signal = {rule['signal']}, top "
                 f"{100 - rule['quantile'] * 100:.0f}% (threshold {rule['threshold']:.2f}), "
                 f"filter = {'grew & not flipped' if rule['filter'] else 'none'}, "
                 f"max {100 / MIN_NAMES:.0f}% per name, market-hedged.")
    print("\n" + rule_text)

    # ---------- 2. Run the frozen rule on both years ----------
    rows, dailies, d2x = [], {}, {}
    for split, part in [("train", train), ("test", test)]:
        trades = select(part, rule)
        main, daily, per_day, t = summarize(trades, days[split], "Closing Pressure (main)")
        rows.append({"split": split, **main})
        for label, kw in [("Double costs", {"cost_mult": 2.0}),
                          ("No per-name cap", {"cap": False}),
                          ("Not market-hedged", {"hedged": False})]:
            s, dd, *_ = summarize(trades, days[split], label, **kw)
            rows.append({"split": split, **s})
            if label == "Double costs":
                d2x[split] = dd
        dailies[split] = daily
        per_day.assign(pnl=daily).to_csv(OUT / f"push_daily_{split}.csv", index_label="date")
        t.to_csv(OUT / f"push_trades_{split}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "push_summary.csv", index=False)
    equity_chart(dailies["train"], dailies["test"], d2x, rule_text, OUT / "push_equity.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["split", "strategy", "trades", "avg_gross_edge_bps", "avg_cost_bps", "avg_net_bps_per_trade",
            "ann_return_pct", "ann_vol_pct", "sharpe", "t_stat", "max_drawdown_bps",
            "turnover_x_per_year", "deflated_sharpe_prob"]
    print("\n=== CLOSING PRESSURE RESULTS (bps = 0.01%; all after costs) ===\n")
    print(summary[show].to_string(index=False))
    test_daily = dailies["test"]
    by_month = test_daily.groupby(test_daily.index.to_period("M")).sum()
    top5 = test_daily.sort_values(ascending=False).head(5).sum()
    print(f"\nTest year: {(by_month > 0).sum()} of {len(by_month)} months positive; "
          f"best 5 days = {100 * top5 / test_daily.sum():.0f}% of total P&L")
    print("Saved results/push_summary.csv, push_equity.png, push_rule.json, push_daily_*.csv, "
          "push_trades_*.csv, push_grid_train.csv")


if __name__ == "__main__":
    run()
