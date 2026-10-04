# Hypotheses

**Data:** 100 Nasdaq stocks, Databento. **Year 1** (Oct 2024 – Sep 2025) = in-sample;
**year 2** (Oct 2025 – Sep 2026) = out-of-sample, run once per frozen rule.

**Status:** H1–H5 were first explored in an earlier working repo (Oct 2–3, 2026), so they are
not blind tests; this repo re-runs them from scratch. **H6 was written down here, with its rule
and success criteria, before it was ever run** (see the git history). Results are in the quant note.

| # | Hypothesis | Why |
|---|---|---|
| **H1** | Large closing imbalances push the price their way from 3:55 PM to the close. | The auction must clear; missing liquidity has to be paid for. |
| **H2** | The price partly reverses overnight. | Forced (index/ETF) flow pushes the close away from fair value. |
| **H3** | Imbalances that grew and never flipped reverse more. | Steady growth = mechanical orders; flips = informed or contested. |
| **H4** | Fading large imbalances overnight is profitable after costs. | Follows from H2 + H3. |
| **H5** | The push can be predicted at 3:55 PM. | Size, lopsidedness and growth of the imbalance drive the push. |
| **H6** | Trading *with* the imbalance at 3:55 PM and exiting in the closing auction is profitable after real costs. | Our exit supplies the liquidity the auction lacks; flat by 4 PM, so no overnight risk. |

## H6 frozen procedure (`src/push_strategy.py`)

- **Entry:** first Nasdaq bid/ask after 15:55:00 (pay the spread). **Exit:** the official close.
- **Costs:** half-spread + 0.5 bps fees; net exposure hedged with the 100-stock basket (0.5 bps).
- **Limit:** max 10% of capital per stock.
- **Rule:** best year-1 Sharpe of 12 variants (imbalance or model signal × top 20/10/5% ×
  filter on/off), frozen, then year 2 run once. Deflated Sharpe uses 32 trials.

**Success (set now):** year-2 Sharpe > 1, t-stat > 2, and still positive without its best day.
Partial: Sharpe > 0 and t > 1. Otherwise rejected.

## Amendments

- **Oct 3, 2026, before the H6 run:** added `src/model_comparison.py` (linear vs random forest vs gradient
  boosting vs XGBoost on identical inputs, trained on year 1, scored once on year 2). Robustness only; it does
  not select or change any rule.
- **Oct 3, 2026, after the H6 run:** added a capacity analysis for the overnight fade and a spread-tercile
  diagnostic of Closing Pressure trades (`src/risk_capacity.py`). Both are exploratory and change no frozen
  rule or H1–H6 verdict. The diagnostic motivates **H7**: Closing Pressure restricted to half-spreads below
  1.86 bps (year-1 cut-off), to be tested only on data after Sep 2026.