"""
Trading-floor look for the Closing Cost Alert app: a scrolling ticker tape, an announcement
board that replays the closing auction from 3:50 to 4:00 PM, and the alert printed as a
trade ticket.

The replay uses historical Databento messages (served from Tiger Data when connected).
It is labelled REPLAY on screen: nothing here is live market data. The board only shows
what was known at each moment: predictions appear at 3:55, actual moves only at 4:00.
"""

import html
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BG, PANEL, RULE = "#0D1015", "#141922", "#232B38"
AMBER, UP, DOWN = "#FFB547", "#3DDC97", "#FF5C5C"
TEXT, DIM = "#C9D1DD", "#7C8798"

FONTS = ("https://fonts.googleapis.com/css2?family=VT323&"
         "family=IBM+Plex+Sans+Condensed:wght@400;600;700&display=swap")

N_FRAMES = 60          # 10-second steps from 15:50:00 to 16:00:00
ALERT_FRAME = 30       # 15:55:00
TIMES = [f"15:{50 + i // 6:02d}:{(i % 6) * 10:02d}" for i in range(N_FRAMES)] + ["16:00:00"]


# ======================================================================
# Page chrome (Streamlit's own widgets)
# ======================================================================
_CSS = f"""
<style>
@import url('{FONTS}');
.stApp {{ background: {BG}; color: {TEXT}; font-family: 'IBM Plex Sans Condensed', sans-serif; }}
.stApp p, .stApp li, .stApp label {{ font-family: 'IBM Plex Sans Condensed', sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 1.6rem; max-width: 1240px; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1rem; letter-spacing: .02em; color: {AMBER}; font-weight: 700; }}
.stApp h1, .stApp h2, .stApp h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; }}
[data-testid="stCaptionContainer"] {{ color: {DIM} !important; }}

.tk-title {{ display: flex; align-items: baseline; gap: .9rem; margin: 0 0 .9rem; flex-wrap: wrap; }}
.tk-title h1 {{ font: 700 1.9rem 'IBM Plex Sans Condensed', sans-serif; color: #fff; margin: 0; padding: 0; }}
.tk-title span {{ color: {DIM}; font-size: .95rem; }}

.stButton > button {{
  font: 600 .95rem 'IBM Plex Sans Condensed', sans-serif; border-radius: 3px;
  background: transparent; color: {AMBER}; border: 1px solid {AMBER};
}}
.stButton > button:hover {{ background: {AMBER}; color: {BG}; border-color: {AMBER}; }}
.stButton > button:focus-visible {{ outline: 2px solid {AMBER}; outline-offset: 2px; }}
[data-testid="stExpander"] details {{ border: 1px solid {RULE}; border-radius: 3px; background: {PANEL}; }}

/* ---------- the trade ticket ---------- */
.ticket {{
  position: relative; background: #EEF0F3; color: #15181D; font-family: 'VT323', monospace;
  font-size: 1.32rem; line-height: 1.25; padding: 1.3rem 1.4rem 1.1rem; margin: .2rem 0 1rem;
  -webkit-mask: radial-gradient(circle 6px at 50% 0, #0000 98%, #000) 50% 0 / 18px 51% repeat-x,
                radial-gradient(circle 6px at 50% 100%, #0000 98%, #000) 50% 100% / 18px 51% repeat-x;
          mask: radial-gradient(circle 6px at 50% 0, #0000 98%, #000) 50% 0 / 18px 51% repeat-x,
                radial-gradient(circle 6px at 50% 100%, #0000 98%, #000) 50% 100% / 18px 51% repeat-x;
}}
.ticket .head {{ display: flex; justify-content: space-between; border-bottom: 2px dashed #9AA1AC;
  padding-bottom: .4rem; margin-bottom: .55rem; }}
.ticket .row {{ display: flex; justify-content: space-between; gap: 1rem; }}
.ticket .row b {{ font-weight: 400; color: #000; }}
.ticket .sep {{ border-top: 2px dashed #9AA1AC; margin: .55rem 0; }}
.ticket .note {{ font-size: 1.15rem; color: #3A404A; margin-top: .4rem; }}
.ticket .stamp {{
  display: inline-block; transform: rotate(-3deg); margin: .1rem 0 .7rem;
  border: 3px solid var(--stamp); color: var(--stamp); padding: .1rem .55rem;
  font-size: 1.7rem; letter-spacing: .06em; background: rgba(238,240,243,.85);
}}
.brief {{ background: {PANEL}; border-left: 3px solid {AMBER}; padding: .8rem 1rem; margin: .4rem 0 1rem;
  line-height: 1.55; }}
.brief .who {{ color: {AMBER}; font-weight: 700; margin-bottom: .3rem; }}
</style>
"""


def apply():
    st.markdown(_CSS, unsafe_allow_html=True)


def title(text, sub):
    st.markdown(f'<div class="tk-title"><h1>{html.escape(text)}</h1>'
                f'<span>{html.escape(sub)}</span></div>', unsafe_allow_html=True)


def md_inline(text):
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", t)
    return t.replace("\n\n", "<br><br>").replace("\n", "<br>")


def brief(who, body_html):
    st.markdown(f'<div class="brief"><div class="who">{html.escape(who)}</div>{body_html}</div>',
                unsafe_allow_html=True)


def ticket(kind, rows, note):
    """kind: crowded | helpful | normal. rows: list of (label, value)."""
    stamp, color = {"crowded": ("CROWDED CLOSE", "#C2410C"), "helpful": ("HELPFUL SIDE", "#15803D"),
                    "normal": ("NORMAL CLOSE", "#475569")}[kind]
    def lines(rs):
        return "".join(f'<div class="row"><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>'
                       for a, b in rs)
    st.markdown(f"""<div class="ticket" style="--stamp:{color}">
  <div class="head"><span>CLOSING COST ALERT</span><span>15:55:00 ET</span></div>
  <div class="stamp">{stamp}</div>
  {lines(rows)}
  <div class="sep"></div>
  <div class="note">{html.escape(note)}</div>
</div>""", unsafe_allow_html=True)


# ======================================================================
# Replay data: imbalance of every stock at 15:50:00, 15:50:10, ..., 16:00:00
# ======================================================================
def frames_from_messages(msgs):
    """msgs: DataFrame with time (ET, naive), symbol, imbalance (signed shares), ref_price.
    Frame i holds the last message at or before TIMES[i] (nothing from the future)."""
    if msgs is None or msgs.empty:
        return None
    t0 = pd.Timestamp(msgs["time"].iloc[0].date()) + pd.Timedelta("15:50:00")
    secs = (msgs["time"] - t0).dt.total_seconds()
    f = np.ceil(secs / 10).clip(0, N_FRAMES).astype(int)
    m = msgs.assign(frame=f).sort_values("time").groupby(["frame", "symbol"]).last()
    full = range(N_FRAMES + 1)
    shares = m["imbalance"].unstack().reindex(full).ffill().fillna(0)
    price = m["ref_price"].unstack().reindex(full).ffill().bfill()
    return shares, price


def tiger_frames(path_df):
    """Tiger's imbalance_10s rows (bucket, symbol, imbalance, ref_price): bucket b holds the
    last message in [b, b+10s), i.e. the state at b+10s."""
    if path_df is None or path_df.empty:
        return None
    d = path_df.rename(columns={"bucket": "time"}).copy()
    d["time"] = d["time"] + pd.Timedelta(seconds=10) - pd.Timedelta(microseconds=1)
    return frames_from_messages(d)


def board(day, frames, selected, my_sign, alert_bps, source, height=580):
    """Render the replay board. day: features rows for the date (with pred_push)."""
    shares, price = frames if frames is not None else (None, None)
    syms = []
    for _, r in day.iterrows():
        s = r["symbol"]
        if shares is not None and s in shares:
            sh = shares[s].values
            px = price[s].fillna(r["ref_price"]).values
        else:                                      # no messages: flat at the 3:55 snapshot
            sh = np.r_[np.zeros(ALERT_FRAME), np.full(N_FRAMES + 1 - ALERT_FRAME, r["imb"])]
            px = np.full(N_FRAMES + 1, r["ref_price"])
        adv = r["adv_dollars_20d"] if r.get("adv_dollars_20d", 0) else np.nan
        pct = 100 * sh * px / adv
        syms.append({"s": s, "sh": np.round(sh).astype(int).tolist(),
                     "pct": np.nan_to_num(np.round(pct, 2)).tolist(),
                     "usd": np.nan_to_num(np.round(sh * px / 1e6, 2)).tolist(),
                     "pred": round(float(r["pred_push"]), 2), "act": round(float(r["push_bps"]), 2)})
    payload = {"date": str(day["date"].iloc[0].date()), "times": TIMES, "alertFrame": ALERT_FRAME,
               "syms": syms, "sel": selected, "mySign": my_sign, "alert": alert_bps,
               "source": source}
    components.html(_BOARD_HTML.replace("__DATA__", json.dumps(payload)), height=height,
                    scrolling=False)


_BOARD_HTML = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="__FONTS__" rel="stylesheet">
<style>
:root{--bg:#0D1015;--panel:#141922;--rule:#232B38;--amber:#FFB547;--up:#3DDC97;--down:#FF5C5C;
      --text:#C9D1DD;--dim:#7C8798}
*{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--text);
  font-family:'IBM Plex Sans Condensed',sans-serif}
.lcd{font-family:'VT323',monospace}
.bar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;row-gap:6px;background:var(--panel);
  border:1px solid var(--rule);border-bottom:0;padding:8px 12px}
.tag{font:700 13px 'IBM Plex Sans Condensed',sans-serif;color:var(--bg);background:var(--amber);padding:2px 7px;border-radius:2px}
.clock{font-size:38px;line-height:1;color:var(--amber);letter-spacing:1px}
.phase{color:var(--text);font-size:15px}
.sp{flex:1}
.net{font-size:14px;color:var(--dim)} .net b{font-family:'VT323',monospace;font-size:24px;margin-left:6px}
button,select{font:600 14px 'IBM Plex Sans Condensed',sans-serif;background:transparent;color:var(--amber);
  border:1px solid var(--amber);border-radius:3px;padding:4px 10px;cursor:pointer}
select option{background:var(--panel)} .scrub select{padding:1px 6px;font-size:12px}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
input[type=range]{width:100%;margin:0;display:block;accent-color:var(--amber);background:var(--panel)}
.scrub{background:var(--panel);border:1px solid var(--rule);border-top:0;border-bottom:0;padding:0 12px 6px;
  display:flex;gap:10px;align-items:center;font-size:12px;color:var(--dim);white-space:nowrap}
.tape{overflow:hidden;white-space:nowrap;border:1px solid var(--rule);background:#07090C;height:38px;
  display:flex;align-items:center}
.track{display:inline-block;padding-left:0;animation:roll 45s linear infinite;font-size:24px}
.track span{margin-right:34px}
@keyframes roll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.up{color:var(--up)} .down{color:var(--down)} .am{color:var(--amber)} .dim{color:var(--dim)}
.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:12px;margin-top:12px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--rule)}
th{font:600 12px 'IBM Plex Sans Condensed',sans-serif;color:var(--dim);text-align:left;padding:7px 8px;border-bottom:1px solid var(--rule)}
th.r,td.r{text-align:right}
td{font-family:'VT323',monospace;font-size:22px;padding:3px 8px;border-bottom:1px solid #1A202B;white-space:nowrap}
tr.sel td{background:#1E2532} tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--amber)}
.barw{display:inline-block;width:80px;height:8px;background:#1E2430;margin-right:8px;vertical-align:middle}
.barw i{display:block;height:100%}
.st{font:600 12px 'IBM Plex Sans Condensed',sans-serif;padding:1px 6px;border-radius:2px;border:1px solid currentColor}
.panel{background:var(--panel);border:1px solid var(--rule);padding:10px 12px}
.panel h3{margin:0 0 4px;font:600 14px 'IBM Plex Sans Condensed',sans-serif;color:var(--text)}
.panel .big{font-family:'VT323',monospace;font-size:26px}
.foot{font-size:12px;color:var(--dim);margin-top:8px}
.flash{animation:flash 1.2s ease-out 1}
@keyframes flash{0%{background:rgba(255,181,71,.35)}100%{background:transparent}}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){.track{animation:none}.flash{animation:none}}
</style></head><body>
<div class="bar">
  <span class="tag">REPLAY</span>
  <span class="lcd clock" id="clock">15:50:00</span>
  <span class="phase" id="phase"></span>
  <span class="sp"></span>
  <span class="net">Net imbalance, all stocks<b class="lcd" id="net">$0M</b></span>
  <button id="play" aria-label="Pause replay">Pause</button>
</div>
<div class="scrub"><span id="d"></span>
  <select id="speed" aria-label="Replay speed">
    <option value="10">10×</option><option value="30" selected>30×</option><option value="60">60×</option>
  </select><input type="range" id="scrub" min="0" max="60" value="0" aria-label="Replay time"></div>
<div class="tape"><div class="track lcd" id="track"></div></div>
<div class="grid">
  <table id="tbl"><thead><tr>
    <th>Stock</th><th>Side</th><th>Imbalance, % of daily volume</th>
    <th class="r">Predicted push</th><th class="r" id="lastcol">Status</th>
  </tr></thead><tbody id="rows"></tbody></table>
  <div>
    <div class="panel"><h3 id="ctitle"></h3><svg id="svg" viewBox="0 0 420 230" width="100%" role="img"></svg></div>
    <div class="panel" style="margin-top:12px" id="you"></div>
  </div>
</div>
<div class="foot" id="foot"></div>
<script>
const D = __DATA__;
const N = D.times.length - 1, A = D.alertFrame;
let f = 0, playing = true, speed = 30, timer = null, flashed = false;
const $ = id => document.getElementById(id);
const fmtSh = v => { const a = Math.abs(v); return a >= 1e6 ? (a/1e6).toFixed(2)+'M' : (a/1e3).toFixed(0)+'K'; };
const sgn = v => v > 0 ? 1 : v < 0 ? -1 : 0;
const bps = v => (v >= 0 ? '+' : '') + v.toFixed(1) + ' bps';
$('d').textContent = D.date;
$('foot').textContent = 'Replay of historical Nasdaq closing-auction messages for ' + D.date +
  ' (' + D.source + '). Predictions use only what was known at 3:55 PM; actual moves appear at 4:00 PM.';

function ranked(){
  const key = f < A ? (s => Math.abs(s.pct[f])) : (s => Math.abs(s.pred));
  return D.syms.slice().sort((a,b) => key(b) - key(a));
}
function render(){
  $('clock').textContent = D.times[f];
  $('scrub').value = f;
  $('phase').textContent = f < A ? 'Imbalances building · prediction at 15:55:00'
                         : f < N ? '15:55 alert issued · waiting for the close'
                         : 'Closed · actual moves shown';
  const net = D.syms.reduce((t,s) => t + s.usd[f], 0);
  $('net').textContent = (net >= 0 ? '+$' : '−$') + Math.abs(net).toFixed(0) + 'M';
  $('net').className = 'lcd ' + (net >= 0 ? 'up' : 'down');

  const R = ranked();
  // ticker tape (two copies for a seamless loop)
  const items = R.slice(0, 16).filter(s => s.sh[f] !== 0).map(s => {
    const c = s.sh[f] > 0 ? 'up' : 'down', arr = s.sh[f] > 0 ? '▲' : '▼';
    const extra = f >= A ? ' <span class="am">' + bps(s.pred) + '</span>' : '';
    return '<span><b>' + s.s + '</b> <span class="' + c + '">' + arr + ' ' + (s.sh[f] > 0 ? 'BUY ' : 'SELL ') +
           fmtSh(s.sh[f]) + '</span> <span class="dim">' + Math.abs(s.pct[f]).toFixed(1) + '%</span>' + extra + '</span>';
  }).join('');
  $('track').innerHTML = items + items;

  // board: top 10 plus the selected stock
  let rows = R.slice(0, 10);
  if (!rows.some(s => s.s === D.sel)) { const me = D.syms.find(s => s.s === D.sel); if (me) rows.push(me); }
  const maxPct = Math.max(1, ...rows.map(s => Math.abs(s.pct[f])));
  $('lastcol').textContent = f < N ? 'Status' : 'Actual push';
  $('rows').innerHTML = rows.map(s => {
    const v = s.sh[f], side = v > 0 ? '<span class="up">▲ BUY</span>' : v < 0 ? '<span class="down">▼ SELL</span>' : '<span class="dim">—</span>';
    const w = Math.min(100, 100 * Math.abs(s.pct[f]) / maxPct), col = v >= 0 ? 'var(--up)' : 'var(--down)';
    const pred = f >= A ? '<span class="' + (s.pred >= 0 ? 'up' : 'down') + '">' + bps(s.pred) + '</span>' : '<span class="dim">at 15:55</span>';
    let st;
    if (f < A) st = '<span class="st dim">Building</span>';
    else if (f < N) st = Math.abs(s.pred) >= D.alert ? '<span class="st am">Alert</span>' : '<span class="st dim">Normal</span>';
    else { const hit = sgn(s.act) === sgn(s.pred); st = '<span class="' + (s.act >= 0 ? 'up' : 'down') + '">' + bps(s.act) + '</span> ' +
           '<span class="dim" title="' + (hit ? 'direction right' : 'direction wrong') + '">' + (hit ? '✓' : '✗') + '</span>'; }
    return '<tr class="' + (s.s === D.sel ? 'sel' : '') + '"><td>' + s.s + '</td><td>' + side + '</td><td>' +
           '<span class="barw"><i style="width:' + w + '%;background:' + col + '"></i></span>' + Math.abs(s.pct[f]).toFixed(1) + '%</td>' +
           '<td class="r">' + pred + '</td><td class="r">' + st + '</td></tr>';
  }).join('');
  if (f === A && !flashed) { flashed = true; const t = $('tbl'); t.classList.remove('flash'); void t.offsetWidth; t.classList.add('flash'); }
  if (f < A) flashed = false;

  drawChart(); drawYou();
}
function drawChart(){
  const s = D.syms.find(x => x.s === D.sel), svg = $('svg');
  $('ctitle').textContent = D.sel + ' imbalance, thousand shares (+ buy / − sell)';
  if (!s) { svg.innerHTML = ''; return; }
  const W = 420, H = 230, L = 46, Rr = 10, T = 12, B = 26;
  const vals = s.sh.map(v => v / 1e3), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = (hi - lo) || 1;
  const x = i => L + (W - L - Rr) * i / N, y = v => T + (H - T - B) * (hi - v) / span;
  let p = ''; for (let i = 0; i <= f; i++) p += (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(vals[i]).toFixed(1);
  const col = vals[f] >= 0 ? '#3DDC97' : '#FF5C5C';
  const ticks = [hi, (hi + lo) / 2, lo].map(v => '<text x="' + (L - 6) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + v.toFixed(0) + '</text>').join('');
  svg.innerHTML =
    '<style>text{font:11px "IBM Plex Sans Condensed",sans-serif;fill:#7C8798}</style>' +
    '<line x1="' + L + '" x2="' + (W - Rr) + '" y1="' + y(0) + '" y2="' + y(0) + '" stroke="#2E3746"/>' +
    '<line x1="' + x(A) + '" x2="' + x(A) + '" y1="' + T + '" y2="' + (H - B) + '" stroke="#FFB547" stroke-dasharray="3 3"/>' +
    '<text x="' + (x(A) + 4) + '" y="' + (T + 10) + '" style="fill:#FFB547">15:55 alert</text>' + ticks +
    '<text x="' + L + '" y="' + (H - 8) + '">15:50</text><text x="' + (W - Rr) + '" y="' + (H - 8) + '" text-anchor="end">16:00</text>' +
    '<path d="' + p + '" fill="none" stroke="' + col + '" stroke-width="2"/>' +
    '<circle cx="' + x(f) + '" cy="' + y(vals[f]) + '" r="3.5" fill="' + col + '"/>';
}
function drawYou(){
  const s = D.syms.find(x => x.s === D.sel); if (!s) { $('you').innerHTML = ''; return; }
  const side = D.mySign > 0 ? 'buy' : 'sell';
  let h = '<h3>Your ' + side + ' order in ' + D.sel + '</h3>';
  if (f < A) h += '<div class="big dim lcd">Waiting for 15:55</div>';
  else {
    const cost = D.mySign * s.pred;
    h += '<div class="big lcd ' + (cost > 0 ? 'down' : 'up') + '">' + (cost > 0 ? 'Expected extra cost ' : 'Expected benefit ') + Math.abs(cost).toFixed(1) + ' bps</div>';
    if (f === N) { const real = D.mySign * s.act; h += '<div class="dim">What happened: ' + (real > 0 ? 'paid ' : 'saved ') + Math.abs(real).toFixed(1) + ' bps</div>'; }
  }
  $('you').innerHTML = h;
}
function tick(){ if (f < N) { f++; render(); } else { pause(); } }
function play(){ if (f >= N) f = 0; playing = true; $('play').textContent = 'Pause'; $('play').setAttribute('aria-label','Pause replay');
  clearInterval(timer); timer = setInterval(tick, 10000 / speed); }
function pause(){ playing = false; $('play').textContent = f >= N ? 'Replay' : 'Play'; $('play').setAttribute('aria-label','Play replay'); clearInterval(timer); }
$('play').onclick = () => playing ? pause() : play();
$('speed').onchange = e => { speed = +e.target.value; if (playing) play(); };
$('scrub').oninput = e => { f = +e.target.value; render(); };
render(); play();
</script></body></html>
""".replace("__FONTS__", FONTS)
