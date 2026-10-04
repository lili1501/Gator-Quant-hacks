"""
ONE COMMAND that reproduces every number in the quant note. Run from the repo root:

    python run_all.py

All code lives in src/; data is read from data/ and results are written to results/.

Order:
    1. build_table.py      (only if raw Databento files are present; otherwise uses
                            data/features.parquet)
    2. backtest.py         Strategy 1: overnight fade (frozen rules)      -> results/summary.csv
    3. models.py           push model, trained on year 1                  -> results/push_model.json
       model_comparison.py robustness: zero / linear / ridge / forest / boosting / XGBoost
    4. push_strategy.py    Strategy 2: Closing Pressure (rule from year 1) -> results/push_summary.csv
    5. risk_capacity.py    risk management + capacity                     -> results/risk_*.csv
    6. make_charts.py      remaining figures
Then writes results/headline.json with the headline numbers.

First-time setup (needs a Databento key in .env):
    python src/download_data.py --go
    python src/download_quotes.py --go  # optional but recommended (real 3:55 PM spreads)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = ["backtest.py", "models.py", "model_comparison.py", "push_strategy.py",
         "risk_capacity.py", "make_charts.py"]


def run(script):
    t0 = time.time()
    print(f"\n{'=' * 70}\n>>> python src/{script}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed (exit code {result.returncode}). Fix it and re-run run_all.py.")
    print(f"[{script} done in {time.time() - t0:.0f}s]")


def headline():
    r = ROOT / "results"
    fade = pd.read_csv(r / "summary.csv")
    push = pd.read_csv(r / "push_summary.csv")
    model = json.loads((r / "push_model.json").read_text())
    cap = pd.read_csv(r / "capacity.csv")
    keep = ["trades", "avg_net_bps_per_trade", "ann_return_pct", "ann_vol_pct", "sharpe", "t_stat",
            "max_drawdown_bps", "turnover_x_per_year", "deflated_sharpe_prob"]

    def pick(tab, name):
        return {s: tab[(tab["split"] == s) & (tab["strategy"] == name)][keep].iloc[0].to_dict()
                for s in ["train", "test"]}

    cap = cap[cap["impact_model"].str.startswith("standard")]
    peaks = {strat: part.loc[part["annual_profit_usd"].idxmax(),
                             ["aum_usd", "annual_profit_usd", "annual_return_pct"]].to_dict()
             for strat, part in cap.groupby("strategy")}
    out = {
        "closing_pressure": pick(push, "Closing Pressure (main)"),
        "closing_pressure_rule": json.loads((r / "push_rule.json").read_text()),
        "overnight_fade": pick(fade, "Filtered (main strategy)"),
        "push_model_test": model["test_score"],
        "capacity_peak_standard_impact": peaks,
    }
    (r / "headline.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n" + "=" * 70 + "\nHEADLINE NUMBERS (results/headline.json)\n" + "=" * 70)
    for strat in ["closing_pressure", "overnight_fade"]:
        for s in ["train", "test"]:
            m = out[strat][s]
            print(f"{strat:17s} {s:5s}: Sharpe {m['sharpe']:5.2f} | ann. return {m['ann_return_pct']:6.2f}% "
                  f"| vol {m['ann_vol_pct']:5.2f}% | max DD {m['max_drawdown_bps']:7.1f} bps "
                  f"| turnover {m['turnover_x_per_year']:.0f}x/yr")


if __name__ == "__main__":
    start = time.time()
    if not SRC.exists():
        sys.exit("src/ folder not found. Run this from the repo root.")
    raw = ROOT / "data" / "imbalance"
    if raw.exists() and any(raw.glob("*.parquet")):
        run("build_table.py")
    elif (ROOT / "data" / "features.parquet").exists():
        print("Raw imbalance files not found; using existing data/features.parquet")
    else:
        sys.exit("No data. Run:  python src/download_data.py --go   (needs DATABENTO_API_KEY in .env)")
    for step in STEPS:
        run(step)
    headline()
    print(f"\nAll done in {time.time() - start:.0f}s. Every number in the quant note comes from results/.")
