# Who Gets Paid at the Close? Closing-Auction Imbalances, Two Strategies and a Closing Cost Alert

**Gator Quant Hacks 2026 · Systematic Trading track · Data: Databento**

Every day at 4:00 PM, US stocks set their official closing price in an auction. Index funds and
other benchmarked investors must trade there; when their orders don't balance, the exchange
publishes an **imbalance** from about 3:50 PM. We test whether that imbalance (1) pushes the
closing price, (2) reverses overnight, and (3) can be traded, or used to warn funds about the
cost of trading on the crowded side.

**Finding in one line:** the closing push is real and predictable, but it is smaller than the
cost of trading it. It is a cost, not a profit, so we built a tool for the funds that pay it.

- **Quant note (PDF):** [`quant_note.pdf`](quant_note.pdf)
- **Hypotheses and frozen test procedure:** [`HYPOTHESES.md`](HYPOTHESES.md). H6 (Closing Pressure),
  its selection rule and its success criteria were committed in `2f7ba89` (Oct 3, 17:36 ET), before
  the strategy was first run (18:34 ET). H1–H5 were explored in an earlier working repo and are
  re-run here from scratch; they are not blind tests.
- **Headline numbers:** `results/headline.json` (written by `run_all.py`)

## Reproduce everything (one command)

```bash
pip install -r requirements.txt
cp .env.example .env                 # then put your Databento, gemini,elevanlab key and  tiger database url in .env

python src/download_data.py --go     # imbalances + daily bars (~$30 of Databento credit)
python src/download_quotes.py --go   # 3:55 PM bid/ask (a few dollars around $7-8)

python run_all.py                    # reproduces every number and chart in the quant note
```

Each download script run **without** `--go` only prints the Databento cost.
All commands are run from the repository root. On macOS, if `import xgboost` fails, run
`brew install libomp`.

## Repository layout

```
├── README.md                 this file
├── HYPOTHESES.md             pre-registration (H1–H6, frozen procedure, success criteria, amendments)
├── quant_note.pdf            the 5-page quant note
├── requirements.txt          dependencies
├── .env.example              key template (copy to .env; .env is never committed)
├── run_all.py                ONE command that reproduces the note
├── 01_check_effect.ipynb     year-1 research (H1–H3), train data only
├── .streamlit/config.toml    colors for the demo app
├── src/
│   ├── download_data.py      Databento: closing-auction imbalances + daily bars
│   ├── download_quotes.py    Databento: Nasdaq bid/ask just after 3:55 PM
│   ├── build_table.py        ~500 MB of messages -> one row per stock per day
│   ├── backtest.py           Strategy 1: overnight fade (frozen rules)
│   ├── models.py             push model, trained on year 1, scored on year 2
│   ├── model_comparison.py   linear vs random forest vs gradient boosting vs XGBoost, same inputs
│   ├── push_strategy.py      Strategy 2: Closing Pressure (rule chosen on year 1)
│   ├── risk_capacity.py      limits, factor exposure, regimes, de-risking, capacity
│   ├── make_charts.py        remaining figures
│   ├── cost_alert.py         demo app (streamlit run src/cost_alert.py)
│   ├── ticker_ui.py          demo app: replay board, ticker tape, alert ticket
│   ├── mlh_integrations.py   optional: Gemini, ElevenLabs, Tiger Data (demo app only)
│   └── load_tigerdata.py     optional: load imbalances into Tiger Data / TimescaleDB
├── data/                     downloaded data (licensed; NOT committed)
└── results/                  every table and chart used in the note
```

## Data

- Nasdaq closing-auction imbalance messages, 3:45–4:01 PM ET (Databento `XNAS.ITCH`, `imbalance`)
- Nasdaq best bid/offer just after 3:55 PM (Databento `XNAS.ITCH`, `bbo-1s`)
- Official daily open/close/volume (Databento `EQUS.SUMMARY`, `ohlcv-1d`)
- 100 large Nasdaq-listed stocks, Oct 2024 – Sep 2026 (496 trading days, 49,560 stock-days)
- **Year 1 (Oct 2024 – Sep 2025) = in-sample. Year 2 (Oct 2025 – Sep 2026) = out-of-sample, run once.**

No raw data or trade-level files are committed; the download scripts recreate them.

## Results (net of costs; full detail in `quant_note.pdf`)

| | In-sample (year 1) | **Out-of-sample (year 2)** |
|---|---|---|
| **Overnight fade (H4)**, frozen rules | Sharpe 2.48, 9.75 bps/trade, 35.1%/yr | **Sharpe 0.84**, t = 0.83, 4.33 bps/trade, deflated-Sharpe prob. 0.07 |
| **Closing Pressure (H6)**, pre-registered | Sharpe −1.35 | **Sharpe −1.76**: captures 3.6 bps of push, pays 5.7 bps of spread + fees |
| **Push (H1)**, largest decile | +2.63 bps, t = 5.6 | **+2.07 bps, t = 3.2** |
| **Push model (H5)** | corr 0.126 | **corr 0.109**, direction right 56.9% on big imbalances |

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Large imbalances push the close their way | Supported in both years |
| H2 | The push reverses overnight |  Weak, unstable across cut-offs |
| H3 | Forced-looking imbalances reverse more | Directional (in year 2 the filter doubles the fade's Sharpe, 0.42 → 0.84, and cuts its worst drawdown) |
| H4 | Overnight fade is profitable | Not significant; one earnings night (MRVL, +1,329 bps) made most of the profit |
| H5 | The push is predictable at 3:55 PM |  Supported; the simple linear model beats XGBoost out of sample |
| H6 | Trading with the imbalance into the close is profitable | Rejected against its pre-set criteria; the spread eats the push |

**Variants tested:** 32 strategy variants and settings in total, including the 12 Closing Pressure
variants, all disclosed in the note; the deflated Sharpe ratio corrects for all of them.

**Model comparison** (`results/model_comparison.csv`): every model gets the same 3 inputs, then the
same 10. The most flexible model fits year 1 best (XGBoost, 10 inputs: corr 0.29) and year 2 worst
(0.085); the 3-input linear model holds up best (0.109).

**Risk and capacity** (`results/risk_*.csv`, `results/capacity.csv`):
- Limits: max 10% of capital per stock, market-hedged. The per-stock cap shrinks the fade's
  MRVL night from +1,329 to +133 bps.
- De-risking rule (halve size when drawdown exceeds 2× year-1 monthly volatility): Closing
  Pressure year-2 drawdown −189 → −117 bps.
- Capacity (square-root impact, at most 25% of the imbalance): the fade breaks even at ≈ $1.7M
  (≈ $0.45M with conservative impact) and peaks at ≈ $41k/yr profit at $1M. Closing Pressure is
  negative at every size.

**Exploratory, after H6 was rejected:** in the tightest-spread third of trades (half-spread
< 1.86 bps, cut-off from year 1) Closing Pressure nets +2.8 bps (year 1) and +3.6 bps (year 2) per
trade. This is **H7**, to be tested only on data after Sep 2026.

**Cost to funds:** a crowded-side closing order paid ≈ 2.07 bps in year 2 (≈ $2,070 per $10M).
That is what the Closing Cost Alert estimates.

## Demo app: Closing Cost Alert

```bash
streamlit run src/cost_alert.py
```

Pick a test-year date, a stock, your side and order size. The app replays that day's closing
auction from 3:50 to 4:00 PM like a trading-floor board:

- a **ticker tape** and **board** of the most imbalanced stocks, updating every 10 seconds of market time
- at **3:55 PM** the alert fires: predicted push per stock, and your expected extra cost in bps and dollars
- at **4:00 PM** the actual moves appear, with ✓ / ✗ for whether the predicted direction was right
- the alert printed as a **trade ticket**

It is a **replay of historical data**, not a live feed. The board only shows what was known at
each moment: predictions use the 3:55 PM snapshot, and actual moves appear only at 4:00 PM.

Optional sponsor integrations (switch on when the key is in `.env`; not used by `run_all.py` or
by any number in the quant note):

| Tool | What it does in the app |
|---|---|
| **Google Gemini API** | "Explain (Gemini)": a plain-English briefing for a portfolio manager |
| **ElevenLabs** | "Read aloud (ElevenLabs)": the alert as speech |
| **Tiger Data (TimescaleDB)** | Serves the replay from a hypertable + continuous aggregate (`time_bucket`, 10 s); load with `python src/load_tigerdata.py` (last 60 days) |

## Limitations

- Two years and 100 stocks; year-2 strategy results are not statistically significant.
- Costs use the quoted spread at 3:55 PM plus fixed fees and a square-root impact model, not real fills.
- Capacity is small (≈ $1–2M), so this is a cost signal for large funds more than a standalone strategy.
