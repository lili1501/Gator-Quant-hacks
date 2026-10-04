"""
None of this is used by run_all.py or by any number in the quant note. Every function
returns None (or False) when its key is missing, so the app works without them.

    Gemini API   GEMINI_API_KEY        -> plain-English explanation of a closing alert
    ElevenLabs   ELEVENLABS_API_KEY    -> the alert read aloud
    Tiger Data   TIGER_DATABASE_URL    -> second-by-second imbalances served from a
                                          TimescaleDB hypertable + continuous aggregate
                                          (load it with: python load_tigerdata.py)
"""

import os

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TIMEOUT = 30


def _key(name):
    v = os.environ.get(name, "").strip()
    return v or None


# ======================================================================
# Gemini: explain the alert in plain English
# ======================================================================
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
                             "gemini-3-flash-preview", "gemini-2.5-flash"] if m]


def gemini_available():
    return _key("GEMINI_API_KEY") is not None


def explain_close(ctx):
    """ctx: dict of the numbers shown in the app. Returns text, or an error message."""
    import requests
    key = _key("GEMINI_API_KEY")
    if not key:
        return None
    prompt = (
        "You are a concise execution analyst at an asset manager. In 3-4 short sentences, "
        "explain to a portfolio manager what this closing-auction alert means and what they "
        "could do. Use only the numbers given. Mention that the model is right about the "
        "direction only about 57% of the time on big imbalances, so this is an average "
        "expected cost, not a guarantee. No investment advice, no hype.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in ctx.items())
    )
    # Newer Gemini models "think" before answering and those thinking tokens count toward
    # maxOutputTokens, so leave plenty of room or the answer gets cut off mid-sentence.
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096}}
    last_err = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, json=body, headers={"x-goog-api-key": key}, timeout=TIMEOUT)
            if r.status_code == 404:          # model name not available: try the next one
                last_err = f"{model}: not found"
                continue
            r.raise_for_status()
            cand = r.json()["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if not text:
                last_err = f"{model}: empty answer ({cand.get('finishReason')})"
                continue
            if cand.get("finishReason") == "MAX_TOKENS":
                text += " …"
            return text + f"\n\n_(Gemini · {model})_"
        except Exception as err:
            last_err = f"{model}: {err}"
    return f"Gemini call failed ({last_err})"


# ======================================================================
# ElevenLabs: read the alert aloud
# ======================================================================
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVEN_MODELS = [m for m in [os.environ.get("ELEVENLABS_MODEL"), "eleven_multilingual_v2",
                             "eleven_flash_v2_5"] if m]


def elevenlabs_available():
    return _key("ELEVENLABS_API_KEY") is not None


def speak(text):
    """Returns (mp3_bytes, None) or (None, error_message)."""
    import requests
    key = _key("ELEVENLABS_API_KEY")
    if not key:
        return None, "no key"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}?output_format=mp3_44100_128"
    last_err = None
    for model in ELEVEN_MODELS:
        try:
            r = requests.post(url, json={"text": text, "model_id": model},
                              headers={"xi-api-key": key, "accept": "audio/mpeg"}, timeout=TIMEOUT)
            if r.ok and r.content:
                return r.content, None
            last_err = f"{model}: HTTP {r.status_code} {r.text[:120]}"
        except Exception as err:
            last_err = f"{model}: {err}"
    return None, last_err


# ======================================================================
# Tiger Data (TimescaleDB): serve the second-by-second imbalances
# ======================================================================
def tiger_available():
    return _key("TIGER_DATABASE_URL") is not None


def _tiger_query(sql, params):
    import psycopg
    with psycopg.connect(_key("TIGER_DATABASE_URL"), autocommit=True, connect_timeout=10) as conn:
        return _fetch(conn, sql, params)


def _fetch(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _window(date_str):
    tz = "America/New_York"
    return (pd.Timestamp(f"{date_str} 15:45", tz=tz).tz_convert("UTC").to_pydatetime(),
            pd.Timestamp(f"{date_str} 16:00", tz=tz).tz_convert("UTC").to_pydatetime())


def tiger_imbalance_path(date_str, symbol):
    """10-second imbalance path from the continuous aggregate. None if unavailable."""
    if not tiger_available():
        return None
    start, end = _window(date_str)
    try:
        df = _tiger_query(
            "SELECT bucket, imbalance FROM imbalance_10s "
            "WHERE symbol = %s AND bucket >= %s AND bucket < %s ORDER BY bucket",
            (symbol, start, end))
    except Exception:
        return None
    if df.empty:
        return None
    idx = pd.to_datetime(df["bucket"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.Series(df["imbalance"].astype(float).values / 1e3, index=idx,
                     name="Imbalance (thousand shares, + buy / - sell)")


def tiger_day_buckets(date_str):
    """Every stock's 10-second imbalance buckets from 15:50 to 16:00 (for the replay board).
    Columns: bucket (ET, naive), symbol, imbalance, ref_price. None if unavailable."""
    if not tiger_available():
        return None
    tz = "America/New_York"
    start = pd.Timestamp(f"{date_str} 15:50", tz=tz).tz_convert("UTC").to_pydatetime()
    end = pd.Timestamp(f"{date_str} 16:00", tz=tz).tz_convert("UTC").to_pydatetime()
    try:
        df = _tiger_query(
            "SELECT bucket, symbol, imbalance, ref_price FROM imbalance_10s "
            "WHERE bucket >= %s AND bucket < %s ORDER BY bucket", (start, end))
    except Exception:
        return None
    if df.empty:
        return None
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df["imbalance"] = df["imbalance"].astype(float)
    df["ref_price"] = df["ref_price"].astype(float)
    return df


def tiger_market_pressure(date_str):
    """Net closing imbalance across ALL stocks, in $ millions, per 10-second bucket."""
    if not tiger_available():
        return None
    start, end = _window(date_str)
    try:
        df = _tiger_query(
            "SELECT bucket AS t, SUM(imbalance * ref_price) / 1e6 AS net_imbalance_musd "
            "FROM imbalance_10s WHERE bucket >= %s AND bucket < %s "
            "GROUP BY bucket ORDER BY bucket", (start, end))
    except Exception:
        return None
    if df.empty:
        return None
    idx = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.Series(df["net_imbalance_musd"].astype(float).values, index=idx,
                     name="Net closing imbalance, all stocks ($M, + buy / - sell)")
