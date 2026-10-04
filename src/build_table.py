"""
Step 1: Build the analysis table.

Turns the raw Databento files in data/ into ONE small table:
    data/features.parquet   (one row per stock per trading day)

Run from the same folder as download_data.py:
    python build_table.py

Every feature uses only information available by 3:55 PM that day (no peeking
at the future). Columns that are only known AFTER 3:55 are marked "OUTCOME" and
must never be used as model inputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
SIGNAL_TIME = "15:55:00"          # when we decide to trade
TRAIN_END = "2025-09-30"          # year 1 = train, year 2 = test

# Index reconstitution days the market treats as special (forced flow).
# Russell dates: please double-check against FTSE Russell's published calendar.
RUSSELL_RECON_DATES = ["2025-06-27", "2026-06-26"]

COLS = ["symbol", "auction_type", "side", "total_imbalance_qty", "paired_qty",
        "ref_price", "ind_match_price", "cont_book_clr_price", "auct_interest_clr_price"]


# --------------------------------------------------------------------------
# 1. Imbalance snapshots, one row per stock per day
# --------------------------------------------------------------------------
def snapshot_day(path):
    """For one day: the first closing-auction message, and the last one
    before SIGNAL_TIME, for every stock."""
    df = pd.read_parquet(path)
    df = df[[c for c in COLS if c in df.columns]]
    df = df[df["auction_type"] == "C"]
    if df.empty:
        return None

    df = df.copy()
    df["time_ny"] = df.index.tz_convert("America/New_York")
    df = df.sort_values("time_ny")
    sign = df["side"].map({"B": 1, "A": -1}).fillna(0)
    df["imb"] = sign * df["total_imbalance_qty"]          # + buyers, - sellers

    day = pd.Timestamp(path.stem)
    cutoff = pd.Timestamp(f"{path.stem} {SIGNAL_TIME}", tz="America/New_York")

    first = df.groupby("symbol").first()
    before = df[df["time_ny"] <= cutoff]
    at_signal = before.groupby("symbol").last()
    n_msgs = before.groupby("symbol").size()

    out = pd.DataFrame({
        "date": day,
        "imb_first": first["imb"],                         # first closing message (~3:50)
        "imb": at_signal["imb"],                           # imbalance at 3:55 (signed shares)
        "paired_qty": at_signal["paired_qty"],
        "ref_price": at_signal["ref_price"],               # price reference at 3:55
        "n_msgs_to_signal": n_msgs,
    })
    # Nasdaq's indicated closing price, if this column is filled in your data
    for col in ["ind_match_price", "cont_book_clr_price", "auct_interest_clr_price"]:
        if col in at_signal:
            out[col] = at_signal[col]
    return out.reset_index()


def load_snapshots():
    files = sorted((DATA / "imbalance").glob("*.parquet"))
    print(f"Reading {len(files)} days of imbalance files...")
    parts = []
    for i, f in enumerate(files, 1):
        s = snapshot_day(f)
        if s is not None:
            parts.append(s)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)
    snaps = pd.concat(parts, ignore_index=True)
    # Days where there is no imbalance at all yet by 3:55 have NaN; treat as 0
    snaps["imb"] = snaps["imb"].fillna(0)
    return snaps


# --------------------------------------------------------------------------
# 2. Daily prices with history-based features
# --------------------------------------------------------------------------
def load_bars():
    raw = pd.read_parquet(DATA / "daily_bars.parquet")
    bars = pd.DataFrame({
        # Daily bars are stamped at midnight UTC: the UTC date IS the trading day
        "date": pd.to_datetime(raw.index.date),
        "symbol": raw["symbol"].values,
        "open": raw["open"].values,
        "high": raw["high"].values,
        "low": raw["low"].values,
        "close": raw["close"].values,
        "volume": raw["volume"].values,
    })
    bars = bars.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    g = bars.groupby("symbol")

    bars["dollar_volume"] = bars["close"] * bars["volume"]
    # 20-day averages using ONLY previous days (shift(1) = no peeking at today)
    bars["adv_dollars_20d"] = g["dollar_volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean().shift(1))
    bars["adv_shares_20d"] = g["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean().shift(1))
    daily_ret = g["close"].pct_change()
    bars["vol_20d"] = daily_ret.groupby(bars["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=10).std().shift(1))
    # Yesterday's volume vs normal (today's full volume isn't known at 3:55)
    bars["prev_volume_ratio"] = g["volume"].shift(1) / bars["adv_shares_20d"]
    bars["prev_close"] = g["close"].shift(1)

    # OUTCOMES (known only after 3:55)
    bars["next_open"] = g["open"].shift(-1)
    bars["next_close"] = g["close"].shift(-1)
    bars["next_date"] = g["date"].shift(-1)
    return bars


# --------------------------------------------------------------------------
# 3. Calendar flags (forced-flow days)
# --------------------------------------------------------------------------
def add_calendar(df, trading_days):
    td = pd.Series(sorted(trading_days))
    month = td.dt.to_period("M")
    month_end_days = set(td.groupby(month).max())
    quarter_end_days = {d for d in month_end_days if d.month in (3, 6, 9, 12)}

    def third_friday(d):
        first = d.replace(day=1)
        offset = (4 - first.weekday()) % 7          # Friday = 4
        return first + pd.Timedelta(days=offset + 14)

    d = df["date"]
    df["is_month_end"] = d.isin(month_end_days)
    df["is_quarter_end"] = d.isin(quarter_end_days)
    df["is_opex"] = d == d.map(third_friday)                       # monthly options expiry
    df["is_triple_witching"] = df["is_opex"] & d.dt.month.isin([3, 6, 9, 12])
    # Nasdaq-100 annual reconstitution takes effect after the December triple witching
    df["is_ndx_recon"] = df["is_triple_witching"] & (d.dt.month == 12)
    df["is_russell_recon"] = d.isin(pd.to_datetime(RUSSELL_RECON_DATES))
    df["is_special_day"] = df[["is_quarter_end", "is_triple_witching",
                               "is_russell_recon", "is_month_end"]].any(axis=1)
    df["day_of_week"] = d.dt.dayofweek
    return df


# --------------------------------------------------------------------------
# 4. Put it together
# --------------------------------------------------------------------------
def build():
    snaps = load_snapshots()
    bars = load_bars()
    df = snaps.merge(bars, on=["symbol", "date"], how="inner")
    print(f"Matched {len(df):,} stock-days with daily prices "
          f"(of {len(snaps):,} imbalance stock-days)")

    sign = np.sign(df["imb"])

    # ---------------- FEATURES (known at 3:55) ----------------
    df["imb_dollars"] = df["imb"] * df["ref_price"]
    df["imb_pct_adv"] = 100 * df["imb_dollars"] / df["adv_dollars_20d"]     # main signal
    df["imb_to_paired"] = df["imb"] / df["paired_qty"].replace(0, np.nan)
    df["imb_growth_pct_adv"] = (100 * (df["imb"].abs() - df["imb_first"].abs().fillna(0))
                                * df["ref_price"] / df["adv_dollars_20d"])
    df["imb_flipped"] = np.sign(df["imb_first"].fillna(0)) * sign < 0       # direction changed
    df["intraday_ret_bps"] = 1e4 * (df["ref_price"] / df["open"] - 1)       # open -> 3:55
    df["overnight_gap_bps"] = 1e4 * (df["open"] / df["prev_close"] - 1)    # last night's gap
    # Does today's move go the same way as the imbalance? (chasing = more likely informed)
    df["move_with_imb"] = sign * df["intraday_ret_bps"]
    if "ind_match_price" in df:
        df["indicated_move_bps"] = 1e4 * (df["ind_match_price"] / df["ref_price"] - 1)

    # ---------------- OUTCOMES (never use as inputs) ----------------
    # Raw moves include the whole market's move that night
    df["push_raw_bps"] = 1e4 * (df["close"] / df["ref_price"] - 1)          # 3:55 -> close
    df["bounce_raw_bps"] = 1e4 * (df["next_open"] / df["close"] - 1)        # close -> next open
    df["next_day_raw_bps"] = 1e4 * (df["next_close"] / df["close"] - 1)     # close -> next close

    # ---------------- Data quality ----------------
    # Unadjusted prices: a stock split overnight looks like a -50% "bounce". Flag those.
    df["bad_row"] = (
        df["next_open"].isna()
        | df["adv_dollars_20d"].isna()
        | (df["bounce_raw_bps"].abs() > 2500)        # >25% overnight: split or data error
        | (df["push_raw_bps"].abs() > 2000)
        | (df["next_date"] - df["date"] > pd.Timedelta(days=5))   # gap in data
    )

    # ---------------- Remove the market's move ----------------
    # The average move of all (good) stocks that night = "the market". Subtracting it
    # leaves only the stock's OWN move, so a market-wide rally isn't counted as profit.
    good_rows = df.loc[~df["bad_row"]]
    for col in ["push", "bounce", "next_day"]:
        mkt = good_rows.groupby("date")[f"{col}_raw_bps"].mean().rename(f"mkt_{col}_bps")
        df = df.merge(mkt, left_on="date", right_index=True, how="left")
        df[f"{col}_bps"] = df[f"{col}_raw_bps"] - df[f"mkt_{col}_bps"]      # market-adjusted

    # P&L of FADING the imbalance (selling into buy imbalances, buying into sell ones),
    # measured relative to the market
    df["fade_bounce_bps"] = -sign * df["bounce_bps"]
    df["fade_bounce_raw_bps"] = -sign * df["bounce_raw_bps"]
    df["reversed"] = (df["fade_bounce_bps"] > 0) & (sign != 0)

    df = add_calendar(df, bars["date"].unique())
    df["split"] = np.where(df["date"] <= pd.Timestamp(TRAIN_END), "train", "test")

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    out = DATA / "features.parquet"
    df.to_parquet(out, index=False)

    # ---------------- Summary ----------------
    good = df[~df["bad_row"]]
    print(f"\nSaved {out}  ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
    print(f"Dates: {df['date'].min().date()} to {df['date'].max().date()}, "
          f"{df['symbol'].nunique()} stocks")
    print(f"Rows flagged bad_row (excluded from analysis): {df['bad_row'].sum():,}")
    print(good["split"].value_counts().rename("rows").to_string())

    tr = good[good["split"] == "train"]
    print("\nFirst look (TRAIN year only):")
    print(f"  corr(imbalance % ADV, overnight bounce): "
          f"{tr['imb_pct_adv'].corr(tr['bounce_bps']):+.3f}   (negative = reversal)")
    print(f"  corr(imbalance % ADV, 3:55->close push): "
          f"{tr['imb_pct_adv'].corr(tr['push_bps']):+.3f}   (positive = imbalance pushes the close)")
    big = tr[tr["imb_pct_adv"].abs() >= tr["imb_pct_adv"].abs().quantile(0.9)]
    print(f"  Fading the top 10% biggest imbalances: avg {big['fade_bounce_bps'].mean():+.1f} bps "
          f"per trade before costs, reversal rate {big['reversed'].mean():.0%}")
    print(f"     (raw, before removing the market's move: {big['fade_bounce_raw_bps'].mean():+.1f} bps)")
    print("  All bounce/push numbers above are MARKET-ADJUSTED (stock move minus the average stock).")
    print("\nNext: open 01_check_effect.ipynb (the 10-group chart).")


if __name__ == "__main__":
    build()
