"""
Step 5: Closing Cost Alert (demo app).

A fund that must trade at the 4:00 PM close sees, at 3:55 PM, how crowded the
closing auction is and how much extra it will likely pay.

The screen replays a historical day from 3:50 to 4:00 PM like a trading-floor board:
ticker tape, imbalances building up, the 3:55 alert, then what actually happened.

Run from the project folder (after run_all.py):
    streamlit run src/cost_alert.py

Optional sponsor integrations (switch on automatically when the key is in .env):
    GEMINI_API_KEY       "Explain this close" in plain English (Google Gemini)
    ELEVENLABS_API_KEY   "Read the alert aloud" (ElevenLabs)
    TIGER_DATABASE_URL   replay data served from Tiger Data / TimescaleDB
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from models import add_model_inputs, predict
import mlh_integrations as mlh
import ticker_ui as ui

st.set_page_config(page_title="Closing Cost Alert", layout="wide")
ui.apply()


@st.cache_data
def load():
    df = pd.read_parquet("data/features.parquet")
    df = add_model_inputs(df[~df["bad_row"]])
    model = json.loads(Path("results/push_model.json").read_text())
    df["pred_push"] = predict(df, model)
    return df, model


@st.cache_data
def local_messages(date_str):
    """All stocks' closing-auction messages for one day, if the raw file is on this computer."""
    path = Path("data/imbalance") / f"{date_str}.parquet"
    if not path.exists():
        return None
    raw = pd.read_parquet(path, columns=["symbol", "auction_type", "side",
                                         "total_imbalance_qty", "ref_price"])
    raw = raw[raw["auction_type"] == "C"]
    if raw.empty:
        return None
    sign = raw["side"].map({"B": 1, "A": -1}).fillna(0)
    return pd.DataFrame({
        "time": raw.index.tz_convert("America/New_York").tz_localize(None),
        "symbol": raw["symbol"].astype(str).values,
        "imbalance": (sign * raw["total_imbalance_qty"]).astype(float).values,
        "ref_price": raw["ref_price"].astype(float).values,
    })


@st.cache_data(ttl=600)
def replay_frames(date_str):
    buckets = mlh.tiger_day_buckets(date_str)
    if buckets is not None:
        return ui.tiger_frames(buckets), "served from Tiger Data · TimescaleDB continuous aggregate"
    return ui.frames_from_messages(local_messages(date_str)), "Databento, local files"


df, model = load()

# ---------------- Sidebar: the order ----------------
st.sidebar.header("Your closing order")
dates = sorted(df.loc[df["split"] == "test", "date"].dt.date.unique())
date = st.sidebar.selectbox("Date (test year)", dates, index=len(dates) - 1)
day = df[df["date"].dt.date == date]
symbols = sorted(day["symbol"].unique())
default = int(day["pred_push"].abs().values.argmax())
symbol = st.sidebar.selectbox("Stock", symbols, index=symbols.index(day.iloc[default]["symbol"]))
side = st.sidebar.radio("You need to", ["BUY at the close", "SELL at the close"])
size = st.sidebar.number_input("Order size ($)", min_value=100_000, value=10_000_000,
                               step=1_000_000, format="%d")
alert_bps = st.sidebar.slider("Alert me above (bps)", 0.5, 10.0, 2.0, 0.5)

st.sidebar.divider()
st.sidebar.caption("Integrations")
for name, ok in [("Gemini explanation", mlh.gemini_available()),
                 ("ElevenLabs voice", mlh.elevenlabs_available()),
                 ("Tiger Data replay", mlh.tiger_available())]:
    st.sidebar.caption(f"{'✓' if ok else '–'} {name}{'' if ok else ' (add key to .env)'}")

row = day[day["symbol"] == symbol].iloc[0]
my_sign = 1 if side.startswith("BUY") else -1
imb_sign = int(np.sign(row["imb"]))
pred = float(row["pred_push"])                     # + = price pushed up
cost_bps = my_sign * pred                          # + = you pay more / receive less
cost_usd = cost_bps / 1e4 * size
crowded = imb_sign != 0 and imb_sign == my_sign
verb = side.split()[0].lower()

if crowded and cost_bps >= alert_bps:
    kind, verdict = "crowded", "Crowded close"
    note = "Your order is on the crowded side. Consider executing part of it before the close."
    spoken = (f"Crowded close in {symbol}. Your {verb} order is on the crowded side. "
              f"Expected extra cost about {cost_bps:.1f} basis points, or {cost_usd:,.0f} dollars. "
              f"Consider trading part of it before the close.")
elif cost_bps < 0:
    kind, verdict = "helpful", "Helpful side"
    note = "The imbalance runs against your order, so the close is likely to move in your favor."
    spoken = (f"{symbol}: you're on the helpful side of the closing imbalance. "
              f"Expected benefit about {-cost_bps:.1f} basis points.")
else:
    kind, verdict = "normal", "Normal close"
    note = f"Expected impact is below your {alert_bps:.1f} bps alert level."
    spoken = f"{symbol}: normal close. Expected impact about {cost_bps:.1f} basis points."

# ---------------- Header + replay board ----------------
ui.title("Closing Cost Alert", f"Nasdaq closing auction · {date} · model trained on Oct 2024 – Sep 2025 only")
frames, source = replay_frames(str(date))
if frames is None:
    source = "no second-by-second file for this date; showing the 15:55 snapshot"
ui.board(day, frames, symbol, my_sign, alert_bps, source)

# ---------------- Ticket + explanations ----------------
left, right = st.columns([1, 1], gap="large")
with left:
    imb_txt = (f"{'BUY' if imb_sign > 0 else 'SELL' if imb_sign < 0 else 'NONE'} "
               f"{abs(row['imb']):,.0f} sh")
    ui.ticket(kind,
              rows=[("Stock", symbol), ("Date", str(date)), ("Order", f"{side.split()[0]} ${size:,.0f}"),
                    ("Imbalance", imb_txt), ("Size vs normal volume", f"{abs(row['imb_pct_adv']):.1f}%"),
                    ("Predicted push", f"{pred:+.1f} bps"),
                    ("Expected cost" if cost_bps >= 0 else "Expected benefit",
                     f"{abs(cost_bps):.1f} bps  ${abs(cost_usd):,.0f}")],
              note=note)

with right:
    key = (str(date), symbol, side, int(size))
    b1, b2 = st.columns(2)
    if mlh.gemini_available() and b1.button("Explain (Gemini)", use_container_width=True):
        with st.spinner("Asking Gemini..."):
            st.session_state[("explain", key)] = mlh.explain_close({
                "stock": symbol, "date": str(date), "time": "3:55 PM ET",
                "order": f"{side} for ${size:,.0f}",
                "imbalance side": "buy" if imb_sign > 0 else "sell" if imb_sign < 0 else "none",
                "imbalance size": f"{abs(row['imb_pct_adv']):.1f}% of normal daily volume",
                "predicted move to the close (vs market)": f"{pred:+.1f} bps",
                "expected cost for this order": f"{cost_bps:+.1f} bps (${cost_usd:,.0f})",
                "verdict": verdict,
                "model accuracy on big imbalances (unseen year)":
                    f"direction right {model['test_score']['direction_hit_rate_top10']:.0%}",
            })
    if mlh.elevenlabs_available() and b2.button("Read aloud (ElevenLabs)", use_container_width=True):
        with st.spinner("Generating voice..."):
            audio, err = mlh.speak(spoken)
        if audio:
            st.session_state[("voice", key)] = audio
        else:
            st.warning(f"ElevenLabs call failed: {err}")
    if ("voice", key) in st.session_state:
        st.audio(st.session_state[("voice", key)], format="audio/mp3", autoplay=True)
    if ("explain", key) in st.session_state:
        ui.brief("Gemini briefing", ui.md_inline(st.session_state[("explain", key)]))

    s = model["test_score"]
    with st.expander("How accurate is this? (test year, never seen in training)", expanded=True):
        st.write(f"- On the biggest 10% of imbalances, the model got the **direction** of the push right "
                 f"**{s['direction_hit_rate_top10']:.0%}** of the time.")
        st.write(f"- On those days, orders on the crowded side paid **{s['avg_crowded_push_top10_bps']:+.1f} bps** "
                 f"extra on average.")
        st.write(f"- Correlation between predicted and actual push: **{s['corr']:.2f}**. Five-minute price "
                 f"moves are noisy, so the value is in the average, not any single day.")
        st.write("- Moves are measured relative to the market (stock move minus the average stock).")
