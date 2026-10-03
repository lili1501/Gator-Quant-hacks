# Hypotheses

**Data:** 100 Nasdaq stocks, Databento. **Year 1** (Oct 2024 – Sep 2025) = in-sample;
**year 2** (Oct 2025 – Sep 2026) = out-of-sample, run once per frozen rule.


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

_None._