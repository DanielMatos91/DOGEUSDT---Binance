import os, time, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone

# ===== Parâmetros da estratégia (ATUALIZADOS) =====
# EMAf=8 | EMAl=20 | RSI=21 | Slope=5 | TP=6% | SL=4% | 1h
EMAF       = 8
EMAL       = 20
RSI_PERIOD = 21
SLOPE_N    = 5
TP_PCT     = 0.06  # 6%
SL_PCT     = 0.04  # 4%

# ===== Config via env =====
SYMBOL       = os.getenv("SYMBOL", "DOGEUSDT")
INTERVAL     = os.getenv("INTERVAL", "1h")
DAYS         = int(os.getenv("DAYS", "10"))
PROXY_BASE   = os.getenv("PROXY_BASE", "").rstrip("/")  # ex.: https://seu-worker.workers.dev
PROVIDER     = os.getenv("PROVIDER", "okx").lower()     # okx | bybit
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ===== HTTP session =====
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (CryptoSignalBot/1.3)",
    "Accept": "application/json"
})

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram não configurado (faltam envs).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("Falha ao enviar Telegram:", e)

# ===== utils =====
INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,"1d":1440}
def to_ms(dt): return int(dt.timestamp()*1000)

# ===== fetch via Worker (OKX/BYBIT normalizados) =====
def fetch_via_worker(symbol, interval, days, limit=1500):
    assert PROXY_BASE, "Defina PROXY_BASE (URL do Cloudflare Worker)."
    step  = INTERVAL_TO_MIN[interval]
    total = int(days*24*60/step)
    iters = total // limit + 1
    frames = []
    for _ in range(iters):
        url = f"{PROXY_BASE}/k/{PROVIDER}?symbol={symbol}&interval={interval}&limit={min(limit, total)}"
        r = SESSION.get(url, timeout=30); r.raise_for_status()
        data = r.json()
        if not data: break
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        frames.append(df)
        total -= len(df)
        if total <= 0: break
        time.sleep(0.2)
    if not frames: raise RuntimeError("Sem dados do Worker.")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        out[c] = out[c].astype(float)
    out = out[["timestamp","open","high","low","close","volume"]].sort_values("timestamp")
    out = out.drop_duplicates("timestamp").reset_index(drop=True)
    return out

# ===== indicadores =====
def compute_indicators(df, ema_fast=8, ema_slow=20, rsi_period=21, slope_window=5, vol_window=14):
    dd = df.copy()
    dd["EMA_FAST"] = dd["close"].ewm(span=ema_fast, adjust=False).mean()
    dd["EMA_SLOW"] = dd["close"].ewm(span=ema_slow, adjust=False).mean()

    delta = dd["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    dd["RSI"] = 100 - (100/(1+rs))
    dd["RSI"] = dd["RSI"].fillna(method="bfill")

    dd["Volatility"] = dd["close"].rolling(vol_window).std()

    def _slope(x):
        x = np.asarray(x); t = np.arange(len(x))
        if len(x) < 2: return np.nan
        m, _ = np.polyfit(t, x, 1)
        return m

    dd["Slope"] = dd["close"].rolling(slope_window).apply(_slope, raw=False)
    dd = dd.dropna().reset_index(drop=True)
    return dd

def generate_signal_row(row):
    if row["EMA_FAST"] > row["EMA_SLOW"] and row["RSI"] > 50 and row["Slope"] > 0:
        return "BUY"
    if row["EMA_FAST"] < row["EMA_SLOW"] and row["RSI"] < 50 and row["Slope"] < 0:
        return "SELL"
    return "HOLD"

# ===== principal =====
def latest_signal(symbol=SYMBOL, interval=INTERVAL, days=DAYS):
    df = fetch_via_worker(symbol, interval, days)
    # usa os parâmetros atualizados
    df = compute_indicators(df, EMAF, EMAL, RSI_PERIOD, SLOPE_N, 14)
    last = df.iloc[-1]
    close = float(last["close"])
    tp_px = float(close * (1 + TP_PCT))  # +6%
    sl_px = float(close * (1 - SL_PCT))  # -4%
    return {
        "symbol": symbol, "interval": interval, "timestamp": str(last["timestamp"]),
        "close": close, "signal": generate_signal_row(last),
        "rsi": float(last["RSI"]), "ema_fast": float(last["EMA_FAST"]), "ema_slow": float(last["EMA_SLOW"]),
        "slope": float(last["Slope"]),
        "volatility": float(last["Volatility"]) if not np.isnan(last["Volatility"]) else None,
        "sl": sl_px, "tp": tp_px,
        "_source": PROVIDER.upper(), "_generated_at_utc": datetime.now(timezone.utc).isoformat()
    }

def main():
    out = latest_signal(SYMBOL, INTERVAL, DAYS)
    msg = (
        f"🪙 *{out['symbol']}* ({out['interval']})\n"
        f"🕒 {out['timestamp']}\n"
        f"📈 *Sinal:* {out['signal']}\n"
        f"💰 Preço: {out['close']}\n"
        f"RSI: {out['rsi']:.2f} | EMAf: {out['ema_fast']:.6f} | EMAl: {out['ema_slow']:.6f}\n"
        f"Slope: {out['slope']:.6f} | Vol: {out['volatility']}\n"
        f"🛑 SL (−{SL_PCT*100:.1f}%): {out['sl']:.6f} | 🎯 TP (+{TP_PCT*100:.1f}%): {out['tp']:.6f}\n"
        f"Fonte: {out['_source']}"
    )
    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"❌ Erro no sinal: {e}"
        print(err)
        send_telegram(err)
        rais
