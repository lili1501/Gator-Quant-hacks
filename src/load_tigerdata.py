"""
Load closing-auction imbalances into Tiger Data (TimescaleDB) for the Cost Alert app.

Optional, for the MLH "Best Use of Tiger Data" prize. NOT used by run_all.py.

Setup:
    1. Create a free service at https://console.cloud.timescale.com (Tiger Cloud)
    2. Copy its connection string into .env:
           TIGER_DATABASE_URL=postgres://tsdbadmin:...@....tsdb.cloud.timescale.com:3xxxx/tsdb?sslmode=require
    3. pip install "psycopg[binary]"
    4. python load_tigerdata.py              # last 60 trading days (fits the free tier)
       python load_tigerdata.py --days 250   # more history

What it creates:
    auction_imbalance   hypertable, one row per closing-auction message (time, symbol,
                        signed imbalance, paired shares, reference price), columnstore
                        compressed and segmented by symbol
    imbalance_10s       continuous aggregate: last imbalance per stock per 10 seconds
                        (time_bucket), which the app queries for its charts
"""

import os
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TABLE_MODERN = """
CREATE TABLE IF NOT EXISTS auction_imbalance (
    time       TIMESTAMPTZ      NOT NULL,
    symbol     TEXT             NOT NULL,
    imbalance  BIGINT           NOT NULL,   -- + buy / - sell, shares
    paired     BIGINT,
    ref_price  DOUBLE PRECISION
) WITH (
    tsdb.hypertable,
    tsdb.partition_column = 'time',
    tsdb.segmentby = 'symbol',
    tsdb.orderby = 'time DESC'
)"""

TABLE_CLASSIC = """
CREATE TABLE IF NOT EXISTS auction_imbalance (
    time       TIMESTAMPTZ      NOT NULL,
    symbol     TEXT             NOT NULL,
    imbalance  BIGINT           NOT NULL,
    paired     BIGINT,
    ref_price  DOUBLE PRECISION
)"""

CAGG = """
CREATE MATERIALIZED VIEW IF NOT EXISTS imbalance_10s
WITH (timescaledb.continuous) AS
SELECT time_bucket(INTERVAL '10 seconds', time) AS bucket,
       symbol,
       last(imbalance, time) AS imbalance,
       last(ref_price, time) AS ref_price
FROM auction_imbalance
GROUP BY bucket, symbol
WITH NO DATA"""


def connect():
    url = os.environ.get("TIGER_DATABASE_URL")
    if not url:
        sys.exit("Set TIGER_DATABASE_URL in .env (Tiger Cloud service connection string).")
    import psycopg
    return psycopg.connect(url, autocommit=True)


def create_schema(conn):
    with conn.cursor() as cur:
        try:
            cur.execute(TABLE_MODERN)
            print("Created hypertable (modern syntax, columnstore segmented by symbol).")
        except Exception as err:
            print(f"Modern syntax not supported ({str(err)[:80]}); using create_hypertable().")
            cur.execute(TABLE_CLASSIC)
            cur.execute("SELECT create_hypertable('auction_imbalance', 'time', if_not_exists => TRUE)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_imb_symbol_time ON auction_imbalance (symbol, time DESC)")
        cur.execute(CAGG)


def load_day(conn, path):
    raw = pd.read_parquet(path, columns=["symbol", "auction_type", "side",
                                         "total_imbalance_qty", "paired_qty", "ref_price"])
    raw = raw[raw["auction_type"] == "C"]
    if raw.empty:
        return 0
    sign = raw["side"].map({"B": 1, "A": -1}).fillna(0).astype(int)
    rows = zip(raw.index.to_pydatetime(), raw["symbol"].astype(str),
               (sign * raw["total_imbalance_qty"]).astype("int64"),
               raw["paired_qty"].astype("int64"), raw["ref_price"].astype(float))
    day = pd.Timestamp(path.stem, tz="America/New_York")
    with conn.cursor() as cur:
        # idempotent: replace the day if it was loaded before
        cur.execute("DELETE FROM auction_imbalance WHERE time >= %s AND time < %s",
                    (day.tz_convert("UTC").to_pydatetime(),
                     (day + pd.Timedelta(days=1)).tz_convert("UTC").to_pydatetime()))
        with cur.copy("COPY auction_imbalance (time, symbol, imbalance, paired, ref_price) FROM STDIN") as cp:
            n = 0
            for r in rows:
                cp.write_row(r)
                n += 1
    return n


def main():
    days = 60
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])
    files = sorted(Path("data/imbalance").glob("*.parquet"))[-days:]
    if not files:
        sys.exit("No files in data/imbalance. Run download_data.py --go first.")
    conn = connect()
    create_schema(conn)
    total = 0
    for i, f in enumerate(files, 1):
        n = load_day(conn, f)
        total += n
        print(f"  [{i}/{len(files)}] {f.stem}: {n:,} rows", flush=True)
    with conn.cursor() as cur:
        cur.execute("CALL refresh_continuous_aggregate('imbalance_10s', NULL, NULL)")
        for stmt in ["CALL add_columnstore_policy('auction_imbalance', after => INTERVAL '7 days')",
                     "SELECT add_compression_policy('auction_imbalance', INTERVAL '7 days')"]:
            try:
                cur.execute(stmt)
                break
            except Exception:
                continue
        cur.execute("SELECT count(*), count(DISTINCT symbol) FROM imbalance_10s")
        n_buckets, n_sym = cur.fetchone()
    print(f"\nLoaded {total:,} rows. Continuous aggregate imbalance_10s: {n_buckets:,} buckets, "
          f"{n_sym} stocks. Start the app: streamlit run cost_alert.py")


if __name__ == "__main__":
    main()
