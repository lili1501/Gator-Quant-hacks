"""
Download the Nasdaq best bid / best ask at 3:55 PM for every stock-day.

Why: the push strategy enters at 3:55 PM in the normal market, so it pays half the
bid-ask spread. Using the real spread (not an assumed one) makes the cost model honest.

Run from the project folder (same .env as download_data.py):
    python download_quotes.py          # cost check only
    python download_quotes.py --go     # download -> data/quotes_355.parquet (small, derived)

Uses Databento XNAS.ITCH, schema bbo-1s (falls back to mbp-1). Only the 10 seconds
after 3:55:00 PM are requested each day, so it is cheap.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import databento as db
from download_data import SYMBOLS, IMBALANCE_DATASET, trading_days

SIGNAL = "15:55:00"
WINDOW_END = "15:55:10"
SCHEMAS = ["bbo-1s", "mbp-1"]          # first one that works is used
FOLDER = Path("data/quotes")
OUT_FILE = Path("data/quotes_355.parquet")


def window_utc(day):
    tz = "America/New_York"
    s = pd.Timestamp(f"{day.date()} {SIGNAL}", tz=tz).tz_convert("UTC")
    e = pd.Timestamp(f"{day.date()} {WINDOW_END}", tz=tz).tz_convert("UTC")
    return s, e


def client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY not found in .env")
    return db.Historical(key)


def pick_schema(c):
    s, e = window_utc(trading_days()[0])
    for schema in SCHEMAS:
        try:
            cost = c.metadata.get_cost(dataset=IMBALANCE_DATASET, schema=schema,
                                       symbols=SYMBOLS, start=s, end=e)
            return schema, cost
        except Exception as err:
            print(f"  {schema}: not available ({str(err)[:80]})")
    sys.exit("No quote schema available on your key.")


def first_quote_after_signal(df, day):
    """The first bid/ask strictly AFTER 3:55:00 (no peeking before the decision)."""
    signal = pd.Timestamp(f"{day.date()} {SIGNAL}", tz="America/New_York").tz_convert("UTC")
    df = df[(df.index > signal) & (df["bid_px_00"] > 0) & (df["ask_px_00"] > df["bid_px_00"])]
    first = df.sort_index().groupby("symbol").head(1)
    return pd.DataFrame({
        "date": pd.Timestamp(day.date()),
        "symbol": first["symbol"].values,
        "bid_355": first["bid_px_00"].values,
        "ask_355": first["ask_px_00"].values,
        "bid_sz_355": first["bid_sz_00"].values,
        "ask_sz_355": first["ask_sz_00"].values,
    })


def one_day(c, schema, day):
    out = FOLDER / f"{day.date()}.parquet"
    if out.exists():
        return f"{day.date()}: already have"
    s, e = window_utc(day)
    try:
        raw = c.timeseries.get_range(dataset=IMBALANCE_DATASET, schema=schema,
                                     symbols=SYMBOLS, start=s, end=e).to_df()
    except Exception as err:
        return f"{day.date()}: skipped ({str(err)[:60]})"
    if raw.empty:
        return f"{day.date()}: no data (holiday?)"
    q = first_quote_after_signal(raw, day)
    q.to_parquet(out)
    return f"{day.date()}: {len(q)} stocks"


def consolidate():
    files = sorted(FOLDER.glob("*.parquet"))
    if not files:
        return
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    q.to_parquet(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} ({len(q):,} stock-days)")


if __name__ == "__main__":
    c = client()
    schema, one = pick_schema(c)
    days = trading_days()
    print(f"Schema {schema}: ${one:.4f}/day x {len(days)} days = ~${one * len(days):.2f}")
    if "--go" not in sys.argv:
        print("Nothing downloaded. If the cost looks fine:  python download_quotes.py --go")
        sys.exit()
    FOLDER.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(one_day, c, schema, d) for d in days]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  [{i}/{len(days)}] {f.result()}", flush=True)
    consolidate()
