"""
Step 6: Final charts for the slides and Devpost page.

Run from the project folder (after backtest.py and models.py):
    python make_charts.py

Saves to results/:
    fig_test_equity_explained.png   test-year profit, with the one big night marked
    fig_filter_drawdown.png         how much the filter reduces losses (test year)
(The push chart, results/push_by_group.png, comes from models.py.)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import select_trades, daily_pnl, OUT, BLUE, GRAY, RED, INK2  # same rules + style

df = pd.read_parquet("data/features.parquet")
df = df[~df["bad_row"]]
test = df[df["split"] == "test"]
days = pd.Index(sorted(test["date"].unique()))

filt = select_trades(test, use_filter=True)
base = select_trades(test, use_filter=False)
d_filt = daily_pnl(filt, days)
d_base = daily_pnl(base, days)

# ---------------- Chart 1: test equity, with the biggest night explained ----------------
best_day = d_filt.idxmax()
best = filt[filt["date"] == best_day]
without_best = d_filt.copy()
without_best[best_day] = 0.0

fig, ax = plt.subplots(figsize=(10, 4.5))
cum = d_filt.cumsum()
cum_wo = without_best.cumsum()
ax.plot(cum.index, cum.values, color=BLUE, lw=2.2, label="Strategy (test year, after costs)")
ax.plot(cum_wo.index, cum_wo.values, color=GRAY, lw=1.5,
        label=f"Same, without the single best night ({best_day.date()})")
ax.axhline(0, color=INK2, lw=1)
for series in (cum, cum_wo):
    ax.annotate(f"{series.iloc[-1]:+.0f}", (series.index[-1], series.iloc[-1]),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
names = ", ".join(best["symbol"].head(3))
ax.annotate(f"{best_day.date()}: {len(best)} trade(s): {names}\n"
            f"{d_filt[best_day]:+.0f} bps in one night",
            xy=(best_day, cum[best_day]), xytext=(-170, 10), textcoords="offset points",
            fontsize=9, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_title("Test year: one night made most of the profit", loc="left")
ax.set_ylabel("Cumulative net bps")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
fig.savefig(OUT / "fig_test_equity_explained.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- Chart 2: drawdown, filter vs no filter ----------------
fig, ax = plt.subplots(figsize=(10, 3.8))
for daily, color, label in [(d_base, GRAY, "No filter (top 5% only)"),
                            (d_filt, BLUE, "With filter (skip flipped, require growth)")]:
    c = daily.cumsum()
    dd = c - c.cummax()
    ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25, lw=0)
    ax.plot(dd.index, dd.values, color=color, lw=1.5, label=f"{label}: worst {dd.min():.0f} bps")
ax.axhline(0, color=INK2, lw=1)
ax.set_title("Test year: the filter cuts the worst loss", loc="left")
ax.set_ylabel("Drop from previous peak (bps)")
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
fig.savefig(OUT / "fig_filter_drawdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Best test night: {best_day.date()}, {len(best)} trade(s): "
      f"{', '.join(best['symbol'])}, {d_filt[best_day]:+.0f} bps")
print(f"Test total with it: {cum.iloc[-1]:+.0f} bps | without it: {cum_wo.iloc[-1]:+.0f} bps")
print("Saved results/fig_test_equity_explained.png and results/fig_filter_drawdown.png")
