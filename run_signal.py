import os, time, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone

# ========== Config via env ==========
SYMBOL       = os.getenv("SYMBOL", "DOGEUSDT")
INTERVAL     = os.getenv("INTERVAL", "1h")
DAYS         = int(os.getenv("DAYS", "10"))

# Proxy (Cloudflare Worker) + fonte (okx/bybit)
USE_PROXY    = True  # deixe True para rodar no GitHub Actions sem bloqueio
PROXY_BASE   = os.getenv("PROXY_BASE", "").rstrip("/")
PROVIDER     = os.getenv("PROVIDER", "okx").lower()     # "okx" ou "bybit"

# Telegram
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Parâmetros da estratégia (melhor combinação encontrada para os sinais)
EMA_FAST    = int(os.getenv("EMA_FAST", "8"))
EMA_SLOW    = int(os.getenv("EMA_SLOW", "20"))
RSI_PERIOD  = int(os.getenv("RSI_PERIOD", "21"))
SLOPE_WIN   = int(os.getenv("SLOPE_WIN", "5"))

# >>> TP/SL otimizado (mapeamento): TP = 2%, SL = 5%
TP_PCT      = float(os.getenv("TP_PCT", "0.02"))  # 0.02 = 2%
SL_PCT      = float(os.getenv("SL_PCT", "0.05"))  # 0.05 = 5%

# ========== HTTP session ==========
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (CryptoSignalBot/1.4)",
    "Accept": "application/json"
})

# ========== Utils ==========
INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,"1d":1440}
def to_ms(dt): return int(dt.timestamp()*1000)

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

# ========== Coleta via Worker (OKX/BYBIT normalizado p/ formato Binance) ==========
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

def get_data(symbol, interval, days):
    if not USE_PROXY:
        raise RuntimeError("Para Actions, USE_PROXY precisa ser True. (Local: você pode trocar.)")
    return fetch_via_worker(symbol, interval, days)

# ========== Indicadores & Sinal ==========
def compute_indicators(df, ema_fast, ema_slow, rsi_period, slope_window, vol_window=14):
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

    def _slope(x):
        x = np.asarray(x); t = np.arange(len(x))
        if len(x) < 2: return np.nan
        m, _ = np.polyfit(t, x, 1)
        return m
    dd["Slope"] = dd["close"].rolling(slope_window).apply(_slope, raw=False)

    dd["Volatility"] = dd["close"].rolling(vol_window).std()
    dd = dd.dropna().reset_index(drop=True)
    return dd

def rule_signal(r):
    if r["EMA_FAST"] > r["EMA_SLOW"] and r["RSI"] > 50 and r["Slope"] > 0:
        return "BUY"
    if r["EMA_FAST"] < r["EMA_SLOW"] and r["RSI"] < 50 and r["Slope"] < 0:
        return "SELL"
    return "HOLD"

def latest_signal(symbol, interval, days):
    df = get_data(symbol, interval, days)
    df = compute_indicators(df, EMA_FAST, EMA_SLOW, RSI_PERIOD, SLOPE_WIN)
    last = df.iloc[-1]
    sig  = rule_signal(last)

    price = float(last["close"])

    # TP/SL definidos pela direção do sinal
    tp_price = None
    sl_price = None
    if sig == "BUY":
        tp_price = price * (1 + TP_PCT)
        sl_price = price * (1 - SL_PCT)
    elif sig == "SELL":
        tp_price = price * (1 - TP_PCT)
        sl_price = price * (1 + SL_PCT)

    payload = {
        "symbol": symbol,
        "interval": interval,
        "timestamp": str(last["timestamp"]),
        "close": price,
        "signal": sig,
        "ema_fast": float(last["EMA_FAST"]),
        "ema_slow": float(last["EMA_SLOW"]),
        "rsi": float(last["RSI"]),
        "slope": float(last["Slope"]),
        "volatility": float(last["Volatility"]) if not np.isnan(last["Volatility"]) else None,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tp_price": float(tp_price) if tp_price is not None else None,
        "sl_price": float(sl_price) if sl_price is not None else None,
        "_source": PROVIDER.upper(),
        "_generated_at_utc": datetime.now(timezone.utc).isoformat()
    }
    return payload

def main():
    out = latest_signal(SYMBOL, INTERVAL, DAYS)

    # Monta mensagem
    header = f"🪙 *{out['symbol']}* ({out['interval']})\n🕒 {out['timestamp']}\n📈 *Sinal:* {out['signal']}\n💰 Preço: {out['close']}"
    indics = f"RSI: {out['rsi']:.2f} | EMAf: {out['ema_fast']:.6f} | EMAl: {out['ema_slow']:.6f}\nSlope: {out['slope']:.6f}"
    tp_sl_line = ""
    if out["signal"] in ("BUY", "SELL"):
        tp_sl_line = (
            f"🎯 TP ({out['tp_pct']*100:.1f}%): {out['tp_price']:.6f}\n"
            f"🛑 SL ({out['sl_pct']*100:.1f}%): {out['sl_price']:.6f}"
        )
    else:
        tp_sl_line = f"ℹ️ TP/SL aguardando direção (sinal = HOLD)\n" \
                     f"· TP({out['tp_pct']*100:.1f}%) / SL({out['sl_pct']*100:.1f}%) serão calculados quando BUY/SELL surgir."

    footer = f"Fonte: {out['_source']}"
    msg = f"{header}\n{indics}\n{tp_sl_line}\n{footer}"

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"❌ Erro no sinal: {e}"
        print(err)
        send_telegram(err)
        raise
