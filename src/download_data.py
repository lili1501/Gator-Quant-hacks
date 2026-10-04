"""
Gator Quant Hacks - closing auction data download (Databento only).

Downloads:
  1. Closing-auction imbalance messages, 3:45-4:01 PM ET   (XNAS.ITCH, schema "imbalance")
  2. Official daily open/high/low/close/volume              (EQUS.SUMMARY, schema "ohlcv-1d")
  3. OPTIONAL: daily options volume per stock               (OPRA.PILLAR, schema "ohlcv-1d")
     -> used as the "informed trading" signal, since we have no news feed

Setup (once):
    pip install databento pandas pyarrow python-dotenv

Put your key in a file named .env in the same folder as this script:
    DATABENTO_API_KEY=db-xxxxxxxx
Add .env to your .gitignore so the key never reaches GitHub.

Usage:
    python download_data.py                  # only CHECKS costs, downloads nothing (add --options to price options too)
    python download_data.py --go             # downloads imbalances + daily bars into ./data/
    python download_data.py --go --options   # also downloads options volume
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Tip: run  pip install python-dotenv  so your .env file is read automatically.")

import databento as db

# ---------------- SETTINGS: edit these ----------------
START_DATE = "2024-10-01"   # 2 years: train on year 1, test on year 2
END_DATE = "2026-09-30"

# ~100 large Nasdaq-listed stocks (their closing auction is on Nasdaq, XNAS.ITCH).
# Note: this is TODAY's list, so it slightly favors stocks that survived. Mention it as a limitation.
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "QCOM", "TXN", "AMGN",
    "INTU", "ISRG", "AMAT", "BKNG", "HON", "SBUX", "GILD", "MDLZ", "ADP", "VRTX",
    "REGN", "LRCX", "PANW", "MU", "ADI", "KLAC", "SNPS", "CDNS", "MELI", "PYPL",
    "ASML", "PDD", "ABNB", "CRWD", "MAR", "ORLY", "CTAS", "CSX", "MRVL", "FTNT",
    "WDAY", "ADSK", "PCAR", "ROP", "NXPI", "CPRT", "MNST", "PAYX", "AEP", "KDP",
    "ROST", "CHTR", "ODFL", "FAST", "EXC", "IDXX", "BKR", "KHC", "EA", "VRSK",
    "CTSH", "XEL", "LULU", "GEHC", "DDOG", "TTWO", "CCEP", "ZS", "CSGP", "DXCM",
    "ON", "TEAM", "BIIB", "CDW", "MDB", "WBD", "GFS", "ARM", "DASH", "TTD",
    "PLTR", "APP", "MSTR", "AXON", "SMCI", "LIN", "AZN", "SHOP", "HOOD", "COIN",
]

IMBALANCE_DATASET = "XNAS.ITCH"   # Nasdaq closing cross. NYSE-listed stocks need "XNYS.PILLAR".
DAILY_DATASET = "EQUS.SUMMARY"    # consolidated official daily bars
DAILY_FALLBACK = "XNAS.ITCH"      # used if your key can't access EQUS.SUMMARY
OPTIONS_DATASET = "OPRA.PILLAR"
AUCTION_WINDOW_ET = ("15:45", "16:01")
OUT = Path("data")
# -------------------------------------------------------


def client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY not found. Check your .env file (see top of this script).")
    return db.Historical(key)


def trading_days():
    return pd.bdate_range(START_DATE, END_DATE)  # holidays simply return no data


def window_utc(day):
    """3:45-4:01 PM New York time for one day, in UTC."""
    tz = "America/New_York"
    s = pd.Timestamp(f"{day.date()} {AUCTION_WINDOW_ET[0]}", tz=tz).tz_convert("UTC")
    e = pd.Timestamp(f"{day.date()} {AUCTION_WINDOW_ET[1]}", tz=tz).tz_convert("UTC")
    return s, e


def daily_range(c, dataset):
    """START_DATE to one week past END_DATE (so the last day still has a
    'next morning' open), but never past what the dataset has available."""
    wanted_end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=8)
    try:
        available_end = pd.Timestamp(c.metadata.get_dataset_range(dataset)["end"])
        if available_end.tzinfo is None:
            available_end = available_end.tz_localize("UTC")
        end = min(wanted_end, available_end)
    except Exception:
        end = min(wanted_end, pd.Timestamp.now(tz="UTC").normalize())
    return START_DATE, end


def options_parents():
    return [f"{s}.OPT" for s in SYMBOLS]


# ======================= COST CHECK =======================
def safe_cost(c, **kw):
    try:
        return c.metadata.get_cost(**kw), None
    except Exception as err:
        return None, err


def check_costs(c):
    days = trading_days()
    s, e = window_utc(days[0])
    print("Checking Databento costs (nothing is downloaded)...\n")

    one, err = safe_cost(c, dataset=IMBALANCE_DATASET, schema="imbalance",
                         symbols=SYMBOLS, start=s, end=e)
    if err:
        print(f"  Imbalance data: NO ACCESS or error -> {err}")
        print("  This is the core dataset. Ask the organizers what your key covers.")
    else:
        print(f"  Imbalance data:   ${one:.4f}/day  x {len(days)} days  = ~${one * len(days):.2f}")

    ds, de = daily_range(c, DAILY_DATASET)
    cost, err = safe_cost(c, dataset=DAILY_DATASET, schema="ohlcv-1d",
                          symbols=SYMBOLS, start=ds, end=de)
    if err:
        print(f"  Daily bars ({DAILY_DATASET}): no access -> will fall back to {DAILY_FALLBACK}")
        ds, de = daily_range(c, DAILY_FALLBACK)
        cost, err = safe_cost(c, dataset=DAILY_FALLBACK, schema="ohlcv-1d",
                              symbols=SYMBOLS, start=ds, end=de)
    print(f"  Daily bars:       ~${cost:.2f}" if cost is not None else f"  Daily bars: error -> {err}")

    if "--options" in sys.argv:  # options pricing is slow to compute, so only when asked
        print("  Checking options cost (this one can take a minute)...", flush=True)
        ds, de = daily_range(c, OPTIONS_DATASET)
        cost, err = safe_cost(c, dataset=OPTIONS_DATASET, schema="ohlcv-1d", stype_in="parent",
                              symbols=options_parents(), start=ds, end=de)
        print(f"  Options: ~${cost:.2f}" if cost is not None else f"  Options: no access -> {err}")


# ======================= DOWNLOADS ========================
def _one_day(c, day, out_file):
    s, e = window_utc(day)
    try:
        df = c.timeseries.get_range(dataset=IMBALANCE_DATASET, schema="imbalance",
                                    symbols=SYMBOLS, start=s, end=e).to_df()
    except Exception as err:
        return f"{day.date()}: skipped ({err})"
    if df.empty:
        return f"{day.date()}: no data (holiday?)"
    df.to_parquet(out_file)
    return f"{day.date()}: {len(df):,} messages"


def download_imbalances(c):
    """Downloads 8 days at a time in parallel instead of one by one."""
    folder = OUT / "imbalance"
    folder.mkdir(parents=True, exist_ok=True)
    todo = [(d, folder / f"{d.date()}.parquet") for d in trading_days()]
    todo = [(d, f) for d, f in todo if not f.exists()]  # skip days already saved
    if not todo:
        print("  All days already downloaded.")
        return
    print(f"  {len(todo)} days to download...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one_day, c, d, f) for d, f in todo]
        for fut in as_completed(futures):
            done += 1
            print(f"  [{done}/{len(todo)}] {fut.result()}", flush=True)


def download_daily_bars(c):
    for dataset in (DAILY_DATASET, DAILY_FALLBACK):
        ds, de = daily_range(c, dataset)
        try:
            df = c.timeseries.get_range(dataset=dataset, schema="ohlcv-1d",
                                        symbols=SYMBOLS, start=ds, end=de).to_df()
            df["source"] = dataset
            df.to_parquet(OUT / "daily_bars.parquet")
            print(f"  Saved {len(df):,} daily bars from {dataset}")
            if dataset == DAILY_FALLBACK:
                print("  Note: these are Nasdaq-venue bars, not consolidated. Fine for a first pass.")
            return
        except Exception as err:
            print(f"  {dataset} failed ({err})")
    print("  Could not get daily bars.")


def download_options(c):
    ds, de = daily_range(c, OPTIONS_DATASET)
    try:
        df = c.timeseries.get_range(dataset=OPTIONS_DATASET, schema="ohlcv-1d",
                                    stype_in="parent", symbols=options_parents(),
                                    start=ds, end=de).to_df()
    except Exception as err:
        print(f"  Options failed ({err})")
        return
    # Option symbols look like "AAPL  261016C00230000": the stock ticker is the first word.
    df["underlying"] = df["symbol"].str.split().str[0]
    df["call_put"] = df["symbol"].str.strip().str[-9]
    daily = (df.reset_index()
               # daily bars are stamped midnight UTC, so the UTC date is the trading day
               .assign(date=lambda x: x["ts_event"].dt.date)
               .groupby(["underlying", "date", "call_put"])["volume"].sum()
               .unstack("call_put", fill_value=0)
               .rename(columns={"C": "call_volume", "P": "put_volume"})
               .reset_index())
    daily.to_parquet(OUT / "options_volume.parquet")
    print(f"  Saved options volume: {len(daily):,} stock-days")


# ========================= MAIN ==========================
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    c = client()
    check_costs(c)

    if "--go" not in sys.argv:
        print("\nNothing downloaded. If the costs look fine, run:  python download_data.py --go")
        sys.exit()

    t0 = time.time()
    print("\n[1] Imbalance messages")
    download_imbalances(c)
    print("\n[2] Daily bars")
    download_daily_bars(c)
    if "--options" in sys.argv:
        print("\n[3] Options volume")
        download_options(c)
    print(f"\nDone in {time.time() - t0:.0f}s. Everything is in the ./data folder.")
