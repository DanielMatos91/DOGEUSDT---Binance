import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ========= CONFIG =========
SYMBOL   = os.getenv("SYMBOL", "DOGEUSDT")
INTERVAL = os.getenv("INTERVAL", "1h")
DAYS     = int(os.getenv("DAYS", "10"))   # histórico suficiente p/ indicadores

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")  # defina no Render (Environment)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ========= TELEGRAM =========
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram não configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

# ========= BINANCE =========
INTERVAL_TO_MIN = {
    "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
    "1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,
    "1d":1440,"3d":4320,"1w":10080,"1M":43200
}

def to_ms(dt: datetime)->int:
    return int(dt.timestamp()*1000)

def get_futures_klines(symbol="DOGEUSDT", interval="1h", days=10):
    limit = 1500
    step_min = INTERVAL_TO_MIN[interval]
    total = int(days*24*60/step_min)
    iters = total//limit + 1
    end_time = datetime.now(tz=timezone.utc)
    frames = []
    for _ in range(iters):
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval={interval}&limit={limit}"
               f"&endTime={to_ms(end_time)}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data: break
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        frames.append(df)
        first_open = int(df.iloc[0]["open_time"])
        end_time = datetime.fromtimestamp(first_open/1000, tz=timezone.utc)
        end_time = end_time.replace(microsecond=0)  # retrocede para paginação
        # pequena pausa
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("Sem dados da Binance.")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        out[c] = out[c].astype(float)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out[["timestamp","open","high","low","close","volume"]]

# ========= INDICADORES & SINAL =========
def compute_indicators(df, ema_fast=8, ema_slow=20, rsi_period=21, slope_window=5, vol_window=14):
    dd = df.copy()
    dd["EMA_FAST"] = dd["close"].ewm(span=ema_fast, adjust=False).mean()
    dd["EMA_SLOW"] = dd["close"].ewm(span=ema_slow, adjust=False).mean()

    delta = dd["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    dd["RSI"] = 100 - (100/(1+rs))

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

def latest_signal(symbol=SYMBOL, interval=INTERVAL, days=DAYS):
    df = get_futures_klines(symbol, interval, days)
    df = compute_indicators(df, 8, 20, 21, 5, 14)
    row = df.iloc[-1]
    signal = generate_signal_row(row)
    sl = row["close"] * (1 - 0.02)  # SL -2% (opcional, informativo)
    tp = row["close"] * (1 + 0.04)  # TP +4% (opcional, informativo)
    return {
        "symbol": symbol,
        "interval": interval,
        "timestamp": str(row["timestamp"]),
        "close": float(row["close"]),
        "signal": signal,
        "rsi": float(row["RSI"]),
        "ema_fast": float(row["EMA_FAST"]),
        "ema_slow": float(row["EMA_SLOW"]),
        "slope": float(row["Slope"]),
        "sl": float(sl),
        "tp": float(tp),
    }

def main():
    out = latest_signal(SYMBOL, INTERVAL, DAYS)
    msg = (
        f"🪙 *{out['symbol']}* ({out['interval']})\n"
        f"🕒 {out['timestamp']}\n"
        f"📈 *Sinal:* {out['signal']}\n"
        f"💰 Preço: {out['close']}\n"
        f"RSI: {out['rsi']:.2f} | EMAf: {out['ema_fast']:.6f} | EMAl: {out['ema_slow']:.6f}\n"
        f"Slope: {out['slope']:.6f}\n"
        f"🛑 SL: {out['sl']:.6f} | 🎯 TP: {out['tp']:.6f}"
    )
    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_telegram(f"❌ Erro no sinal: {e}")
        raise
