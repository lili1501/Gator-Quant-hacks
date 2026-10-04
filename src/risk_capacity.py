"""
Risk management + liquidity & capacity analysis (judging criteria 3 and 4).

Run after push_strategy.py and backtest.py:
    python risk_capacity.py

Outputs (results/):
    risk_exposure.csv        names per day, gross/net exposure, worst single position
    risk_factors.csv         regression of daily returns on market, momentum, intraday factors
    risk_regimes.csv         performance by market-volatility regime + stress windows
    risk_derisk.csv          effect of the pre-set drawdown de-risking rule
    capacity.csv             net profit vs capital deployed (own measured auction impact)
    fig_capacity.png, fig_regimes.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import perf_stats, BLUE, GRAY, RED, INK2
import push_strategy as ps

OUT = Path("results")

# ============ Risk & capacity parameters (fixed, not tuned) ============
IMPACT_COEFS = {"standard (0.5)": 0.5, "conservative (1.0)": 1.0}   # square-root impact:
                           # cost = coef * daily_vol * sqrt(order / ADV); literature range ~0.5-1
PARTICIPATION_CAP = 0.25   # our closing-auction order may offset at most 25% of the imbalance
FADE_COST_BPS = 3.0        # same round-trip cost as backtest.py
ORANGE = "#eb6834"
CHART_MAX_AUM = 2.5e7
AUM_GRID = [1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 1e9]
DD_TRIGGER_SIGMAS = 2.0    # de-risk when drawdown > 2 x (train monthly volatility)
# =======================================================================


def ols(y, X, names):
    """OLS with intercept; returns coefficient table with t-stats."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=["alpha_daily_bps"] + names)
    out.loc["alpha_daily_bps", "annualized_pct"] = beta[0] * 252 / 100
    return out, r2


# --------------------------------------------------------------------------
# Factors, built only from our own Databento data
# --------------------------------------------------------------------------
def build_factors(df, window_col):
    """Long-short factor returns over the SAME window the strategy trades.
    MKT: equal-weight average of all stocks.
    MOM: past-20-day winners minus losers (top third minus bottom third, known in advance).
    INTRADAY: today's open->3:55 winners minus losers (end-of-day momentum check)."""
    d = df.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol")["close"]
    d["ret20"] = g.shift(1) / g.shift(21) - 1          # uses only past closes
    rows = {}
    for date, x in d.groupby("date"):
        r = x[window_col]
        f = {"MKT": r.mean()}
        for name, col in [("MOM", "ret20"), ("INTRADAY", "intraday_ret_bps")]:
            s = x[col]
            if s.notna().sum() >= 6:
                hi, lo = s.quantile(2 / 3), s.quantile(1 / 3)
                f[name] = r[s >= hi].mean() - r[s <= lo].mean()
        rows[date] = f
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def factor_table(daily, factors, label):
    j = pd.concat([daily.rename("ret"), factors], axis=1, join="inner").dropna()
    names = [c for c in ["MKT", "MOM", "INTRADAY"] if c in j.columns]
    tab, r2 = ols(j["ret"].values, j[names].values, names)
    tab["strategy"], tab["r2"], tab["days"] = label, round(r2, 3), len(j)
    return tab.reset_index().rename(columns={"index": "term"})


# --------------------------------------------------------------------------
# Regimes & tails
# --------------------------------------------------------------------------
def market_vol(df):
    m = (df["close"] / df["prev_close"] - 1).groupby(df["date"]).mean()
    return m.rolling(20, min_periods=10).std().shift(1), m      # known before the day starts


def regime_table(daily_by_split, vol, train_cut, label):
    rows = []
    for split, daily in daily_by_split.items():
        v = vol.reindex(daily.index)
        reg = pd.cut(v, [-np.inf, *train_cut, np.inf], labels=["Calm", "Normal", "Volatile"])
        for name, part in daily.groupby(reg, observed=True):
            sd = part.std(ddof=1)
            rows.append({"strategy": label, "split": split, "regime": name, "days": len(part),
                         "mean_daily_bps": round(part.mean(), 2),
                         "sharpe": round(part.mean() / sd * np.sqrt(252), 2) if sd > 0 else np.nan})
    return rows


def derisk(daily, trigger):
    """Pre-set rule: if drawdown exceeds `trigger`, trade half size; return to full size once
    the drawdown is back under trigger/2. Uses only past P&L."""
    size, out, peak, cum = 1.0, [], 0.0, 0.0
    for r in daily.values:
        out.append(size * r)
        cum += size * r
        peak = max(peak, cum)
        dd = peak - cum
        if dd > trigger:
            size = 0.5
        elif dd < trigger / 2:
            size = 1.0
    return pd.Series(out, index=daily.index)


# --------------------------------------------------------------------------
# Capacity: our own measured auction impact
# --------------------------------------------------------------------------
def impact_slope(train):
    """How much the close moves per sqrt(imbalance % of ADV), measured on year 1."""
    x = np.sqrt(train["imb_pct_adv"].abs())
    y = np.sign(train["imb"]) * train["push_adj_bps"]
    ok = x.notna() & y.notna() & (train["imb"] != 0)
    slope = np.polyfit(x[ok], y[ok], 1)[0]
    return max(slope, 0.0)


def capacity(trades, days, slope, coef):
    rows = []
    t = trades.copy()
    adv = t["adv_dollars_20d"]
    imb_usd = t["imb_dollars"].abs()
    for aum in AUM_GRID:
        q = np.minimum(aum * t["weight"], PARTICIPATION_CAP * imb_usd)        # $ per position
        q_pct = 100 * q / adv                                                  # % of ADV
        entry_impact = coef * t["vol_20d"] * 1e4 * np.sqrt(q / adv)     # bps
        imb_pct = t["imb_pct_adv"].abs()
        lost_push = slope * (np.sqrt(imb_pct) - np.sqrt(np.maximum(imb_pct - q_pct, 0)))
        net = t["gross_bps"] - t["cost_bps"] - entry_impact - lost_push
        pnl_usd = (q * net / 1e4).groupby(t["date"]).sum().reindex(days, fill_value=0)
        deployed = q.groupby(t["date"]).sum().reindex(days, fill_value=0)
        sd = pnl_usd.std(ddof=1)
        rows.append({
            "aum_usd": aum,
            "avg_deployed_usd": round(deployed[deployed > 0].mean(), 0),
            "avg_position_pct_adv": round(q_pct.mean(), 3),
            "avg_impact_bps": round((entry_impact + lost_push).mean(), 2),
            "avg_net_bps_per_trade": round(net.mean(), 2),
            "annual_profit_usd": round(pnl_usd.mean() * 252, 0),
            "annual_return_pct": round(100 * pnl_usd.mean() * 252 / aum, 2),
            "sharpe": round(pnl_usd.mean() / sd * np.sqrt(252), 2) if sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def capacity_fade(ft, days, slope, coef):
    """Overnight fade: enter in the closing auction (offsetting the imbalance), exit in the
    next opening auction. Impact paid on both legs; our order also removes part of the
    distortion we are trying to earn (approximated with the measured push curve)."""
    rows = []
    t = ft.copy()
    t["weight"] = 1.0 / t.groupby("date")["symbol"].transform("count")
    adv, imb_usd, imb_pct = t["adv_dollars_20d"], t["imb_dollars"].abs(), t["imb_pct_adv"].abs()
    for aum in AUM_GRID:
        q = np.minimum(aum * t["weight"], PARTICIPATION_CAP * imb_usd)
        q_pct = 100 * q / adv
        impact = 2 * coef * t["vol_20d"] * 1e4 * np.sqrt(q / adv)              # close + open
        lost = slope * (np.sqrt(imb_pct) - np.sqrt(np.maximum(imb_pct - q_pct, 0)))
        net = t["fade_bounce_bps"] - FADE_COST_BPS - impact - lost
        pnl_usd = (q * net / 1e4).groupby(t["date"]).sum().reindex(days, fill_value=0)
        deployed = q.groupby(t["date"]).sum().reindex(days, fill_value=0)
        sd = pnl_usd.std(ddof=1)
        rows.append({
            "aum_usd": aum,
            "avg_deployed_usd": round(deployed[deployed > 0].mean(), 0),
            "avg_position_pct_adv": round(q_pct.mean(), 3),
            "avg_impact_bps": round((impact + lost).mean(), 2),
            "avg_net_bps_per_trade": round(net.mean(), 2),
            "annual_profit_usd": round(pnl_usd.mean() * 252, 0),
            "annual_return_pct": round(100 * pnl_usd.mean() * 252 / aum, 2),
            "sharpe": round(pnl_usd.mean() / sd * np.sqrt(252), 2) if sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def spread_buckets(push_trades):
    """DIAGNOSTIC (exploratory, defined after H6 was rejected): split Closing Pressure trades
    by the half-spread paid at 3:55 PM. Bucket cut-offs come from year-1 trades only."""
    cuts = push_trades["train"]["half_spread_bps"].quantile([1 / 3, 2 / 3]).values
    rows = []
    for split, t in push_trades.items():
        b = pd.cut(t["half_spread_bps"], [-np.inf, *cuts, np.inf], labels=["Tight", "Medium", "Wide"])
        for name, g in t.groupby(b, observed=True):
            sd = g["net_bps"].std(ddof=1)
            rows.append({"split": split, "spread_bucket": name, "trades": len(g),
                         "cutoffs_bps_from_train": f"{cuts[0]:.2f} / {cuts[1]:.2f}",
                         "avg_half_spread_bps": round(g["half_spread_bps"].mean(), 2),
                         "avg_gross_push_bps": round(g["gross_bps"].mean(), 2),
                         "avg_cost_bps": round(g["cost_bps"].mean(), 2),
                         "avg_net_bps": round(g["net_bps"].mean(), 2),
                         "t_stat_net": round(g["net_bps"].mean() / (sd / np.sqrt(len(g))), 2) if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def spread_chart(sb, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, split in zip(axes, ["train", "test"]):
        part = sb[sb["split"] == split].set_index("spread_bucket").reindex(["Tight", "Medium", "Wide"])
        x = np.arange(3)
        for off, col, color, label in [(-0.2, "avg_gross_push_bps", BLUE, "Push captured (gross)"),
                                       (0.2, "avg_cost_bps", ORANGE, "Cost paid (spread + fees)")]:
            ax.bar(x + off, part[col], width=0.38, color=color, label=label)
            for xi, v in zip(x + off, part[col]):
                ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", xytext=(0, 3),
                            textcoords="offset points", fontsize=8, color=INK2)
        ax.set_xticks(x, [f"{b} spread\nnet {v:+.1f} bps" for b, v in zip(part.index, part["avg_net_bps"])])
        ax.set_title(f"{'Train' if split == 'train' else 'TEST'} year", loc="left")
        ax.margins(y=0.15)
    axes[0].set_ylabel("bps per trade")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Closing Pressure: the push is real, the spread decides who keeps it (exploratory)",
                 x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def capacity_chart(cap, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (strat, color) in [("Overnight fade", BLUE), ("Closing Pressure", ORANGE)]:
        for model, ls in [("standard (0.5)", "-"), ("conservative (1.0)", "--")]:
            part = cap[(cap["strategy"] == strat) & (cap["impact_model"] == model)
                       & (cap["aum_usd"] <= CHART_MAX_AUM)]
            ax.plot(part["aum_usd"], part["annual_return_pct"], color=color, lw=2 if ls == "-" else 1.4,
                    ls=ls, marker="o" if ls == "-" else None, ms=5, markeredgecolor="#fcfcfb",
                    markeredgewidth=1.5, label=f"{strat}, impact {model}")
    ax.set_xscale("log")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Capital (USD, log scale)")
    ax.set_ylabel("Annual net return on capital, test year (%)")
    ax.set_title("Capacity: return falls as our own trading moves prices", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(0.01, -0.03, "Shown up to $25M; beyond that, orders hit the 25%-of-imbalance cap and extra capital "
             "sits idle (full grid in capacity.csv).\nSquare-root impact (coef 0.5 standard, 1.0 conservative) + loss of edge from offsetting the "
             "imbalance (measured push curve); orders capped at 25% of the imbalance.", fontsize=8, color=INK2)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def regime_chart(rows, path):
    r = pd.DataFrame(rows)
    r = r[(r["strategy"] == "Closing Pressure")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    regimes = ["Calm", "Normal", "Volatile"]
    w = 0.38
    for i, (split, color) in enumerate([("train", GRAY), ("test", BLUE)]):
        part = r[r["split"] == split].set_index("regime").reindex(regimes)
        xs = np.arange(3) + (i - 0.5) * w
        ax.bar(xs, part["mean_daily_bps"], width=w - 0.04, color=color, label=f"{split.title()} year")
        for xi, v in zip(xs, part["mean_daily_bps"]):
            if pd.notna(v):
                ax.annotate(f"{v:+.2f}", (xi, v), ha="center", va="bottom" if v >= 0 else "top",
                            xytext=(0, 3 if v >= 0 else -3), textcoords="offset points", fontsize=8)
    ax.set_xticks(range(3), regimes)
    ax.axhline(0, color=INK2, lw=1)
    ax.margins(y=0.2)
    ax.set_ylabel("Mean daily net return (bps)")
    ax.set_title("Closing Pressure by market-volatility regime", loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run():
    OUT.mkdir(exist_ok=True)
    rule = json.loads((OUT / "push_rule.json").read_text())
    df = ps.load_data()
    feats = pd.read_parquet("data/features.parquet")

    push, push_days, push_trades = {}, {}, {}
    for split in ["train", "test"]:
        part = df[df["split"] == split]
        days = pd.Index(sorted(part["date"].unique()))
        daily, per_day, t = ps.daily_returns(ps.select(part, rule), days)
        push[split], push_days[split], push_trades[split] = daily, per_day, t
    fade = {s: pd.read_csv(OUT / f"fade_daily_{s}.csv", parse_dates=["date"]).set_index("date")["net_bps"]
            for s in ["train", "test"]}

    # ---------- 1. Exposure & limits ----------
    exp_rows = []
    for split in ["train", "test"]:
        pd_, t = push_days[split], push_trades[split]
        act = pd_[pd_["n_names"] > 0]
        worst = t.loc[t["contrib_bps"].idxmin()]
        exp_rows.append({
            "strategy": "Closing Pressure", "split": split,
            "active_day_pct": round(100 * len(act) / len(pd_), 1),
            "avg_names": round(act["n_names"].mean(), 1),
            "avg_gross_pct": round(100 * act["gross"].mean(), 1),
            "avg_abs_net_pct": round(100 * act["net_exposure"].abs().mean(), 1),
            "max_abs_net_pct": round(100 * act["net_exposure"].abs().max(), 1),
            "max_name_weight_pct": round(100 * t["weight"].max(), 1),
            "worst_position_loss_bps_of_capital": round(worst["contrib_bps"], 1),
            "worst_position": f"{worst['symbol']} {pd.Timestamp(worst['date']).date()}",
            "best5_days_share_pct": round(100 * push[split].nlargest(5).sum() / push[split].sum(), 0)
                                    if push[split].sum() > 0 else np.nan,
        })
    ft = pd.read_csv(OUT / "trades_test.csv", parse_dates=["date"])
    ft["w"] = 1 / ft.groupby("date")["symbol"].transform("count")
    ft["contrib"] = ft["w"] * (ft["fade_bounce_bps"] - 3.0)
    worst = ft.loc[ft["contrib"].idxmax()]
    ft["contrib_capped"] = (1 / np.maximum(ft.groupby("date")["symbol"].transform("count"), ps.MIN_NAMES)
                            ) * (ft["fade_bounce_bps"] - 3.0)
    exp_rows.append({
        "strategy": "Overnight fade (no cap, as frozen)", "split": "test",
        "max_name_weight_pct": round(100 * ft["w"].max(), 1),
        "worst_position_loss_bps_of_capital": round(ft["contrib"].min(), 1),
        "worst_position": f"largest single-name day: {worst['symbol']} {worst['date'].date()} "
                          f"{worst['contrib']:+.0f} bps; with a 10% cap it would be "
                          f"{ft.loc[worst.name, 'contrib_capped']:+.0f} bps",
    })
    exposure = pd.DataFrame(exp_rows)
    exposure.to_csv(OUT / "risk_exposure.csv", index=False)

    # ---------- 2. Factor exposure ----------
    f_push = build_factors(feats[feats["push_raw_bps"].abs() < 2000], "push_raw_bps")
    f_fade = build_factors(feats[~feats["bad_row"]], "bounce_raw_bps")
    ftab = pd.concat([
        factor_table(pd.concat([push["train"], push["test"]]), f_push, "Closing Pressure (both years)"),
        factor_table(push["test"], f_push, "Closing Pressure (test)"),
        factor_table(pd.concat([fade["train"], fade["test"]]), f_fade, "Overnight fade (both years)"),
    ])
    ftab.to_csv(OUT / "risk_factors.csv", index=False)

    # ---------- 3. Regimes, tails, stress ----------
    vol, mkt_daily = market_vol(feats)
    train_cut = vol.reindex(push["train"].index).quantile([1 / 3, 2 / 3]).values
    reg_rows = regime_table(push, vol, train_cut, "Closing Pressure")
    reg_rows += regime_table(fade, vol, train_cut, "Overnight fade")
    allp = pd.concat([push["train"], push["test"]])
    big_days = mkt_daily.abs().reindex(allp.index).nlargest(10).index
    for label, idx in [("Apr 2025 tariff shock", allp.loc["2025-04-01":"2025-04-30"].index),
                       ("10 largest market-move days", big_days)]:
        part = allp.reindex(idx)
        reg_rows.append({"strategy": "Closing Pressure", "split": "stress", "regime": label,
                         "days": len(part), "mean_daily_bps": round(part.mean(), 2),
                         "sharpe": np.nan})
    regimes = pd.DataFrame(reg_rows)
    regimes.to_csv(OUT / "risk_regimes.csv", index=False)
    regime_chart(reg_rows, OUT / "fig_regimes.png")

    # ---------- 4. De-risking rule (trigger set from year 1) ----------
    trigger = DD_TRIGGER_SIGMAS * push["train"].std(ddof=1) * np.sqrt(21)
    dr_rows = []
    for split in ["train", "test"]:
        for label, series in [("Without rule", push[split]), ("With de-risk rule", derisk(push[split], trigger))]:
            s = perf_stats(series)
            dr_rows.append({"split": split, "version": label, "trigger_bps": round(trigger, 1),
                            **{k: s[k] for k in ["ann_return_pct", "ann_vol_pct", "sharpe", "max_drawdown_bps"]}})
    derisk_tab = pd.DataFrame(dr_rows)
    derisk_tab.to_csv(OUT / "risk_derisk.csv", index=False)

    # ---------- 5. Capacity ----------
    slope = impact_slope(df[df["split"] == "train"])
    fade_days = fade["test"].index
    cap = pd.concat(
        [capacity(push_trades["test"], push_days["test"].index, slope, c).assign(
            strategy="Closing Pressure", impact_model=n) for n, c in IMPACT_COEFS.items()]
        + [capacity_fade(ft, fade_days, slope, c).assign(strategy="Overnight fade", impact_model=n)
           for n, c in IMPACT_COEFS.items()], ignore_index=True)
    cap["impact_slope_bps_per_sqrt_pct"] = round(slope, 3)
    cap.to_csv(OUT / "capacity.csv", index=False)
    capacity_chart(cap, OUT / "fig_capacity.png")

    # ---------- 6. Spread diagnostic (exploratory) ----------
    sb = spread_buckets(push_trades)
    sb.to_csv(OUT / "risk_spread_buckets.csv", index=False)
    spread_chart(sb, OUT / "fig_spread_edge.png")

    # ---------- Print ----------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== 1. EXPOSURE & LIMITS ===\n" + exposure.to_string(index=False))
    print("\n=== 2. FACTOR EXPOSURE (daily returns regressed on same-window factors) ===")
    print(ftab.round(3).to_string(index=False))
    print("\n=== 3. REGIMES & STRESS ===\n" + regimes.to_string(index=False))
    print(f"\n=== 4. DE-RISK RULE (halve size when drawdown > {trigger:.0f} bps) ===")
    print(derisk_tab.to_string(index=False))
    print(f"\n=== 5. CAPACITY (test year; measured auction impact slope {slope:.2f} bps per sqrt(%ADV)) ===")
    print(cap.drop(columns="impact_slope_bps_per_sqrt_pct").to_string(index=False))
    for (strat, name), part in cap.groupby(["strategy", "impact_model"], sort=False):
        best = part.loc[part["annual_profit_usd"].idxmax()]
        print(f"[{strat} | {name}] profit peaks at ~${best['aum_usd']:,.0f} of capital "
              f"(${best['annual_profit_usd']:,.0f}/yr, {best['annual_return_pct']}% on capital).")
    print("\n=== 6. SPREAD DIAGNOSTIC (exploratory; buckets from year-1 cut-offs) ===")
    print(sb.to_string(index=False))
    print("Saved risk_*.csv, capacity.csv, fig_capacity.png, fig_regimes.png, fig_spread_edge.png")


if __name__ == "__main__":
    run()
