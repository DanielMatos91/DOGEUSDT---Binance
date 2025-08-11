import os, time, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from requests.exceptions import HTTPError

# ===== Config por ENV =====
SYMBOL   = os.getenv("SYMBOL", "DOGEUSDT")
INTERVAL = os.getenv("INTERVAL", "1h")
DAYS     = int(os.getenv("DAYS", "10"))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ===== HTTP session (User-Agent + retry) =====
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (CryptoSignalBot/1.0)"})

def http_get(url, timeout=30, retries=3, backoff=2.0):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            print(f"[HTTP] {e.response.status_code if e.response else '??'} for {url}")
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))
        except Exception as e:
            print(f"[HTTP] Error: {e} for {url}")
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))

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

# ===== Coleta com Fallback (Futures -> Spot) =====
INTERVAL_TO_MIN = {
    "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
    "1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,
    "1d":1440,"3d":4320,"1w":10080,"1M":43200
}
def to_ms(dt): return int(dt.timestamp()*1000)

def _paginate_klines(base_url, symbol, interval, days, limit=1500):
    step  = INTERVAL_TO_MIN[interval]
    total = int(days*24*60/step)
    iters = total//limit + 1

    end = datetime.now(timezone.utc)
    frames = []
    for _ in range(iters):
        url = f"{base_url}?symbol={symbol}&interval={interval}&limit={limit}&endTime={to_ms(end)}"
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
        except HTTPError as e:
            e.status = getattr(e.response, "status_code", None)
            raise
        data = resp.json()
        if not data:
            break
        frames.append(pd.DataFrame(data))
        first_open = int(data[0][0])
        end = datetime.fromtimestamp(first_open/1000, tz=timezone.utc) - timedelta(milliseconds=1)
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("Sem dados retornados.")
    return pd.concat(frames, ignore_index=True)

def get_klines_with_fallback(symbol="DOGEUSDT", interval="1h", days=10):
    # força SPOT-only se definido no ambiente (recomendado no GitHub Actions)
    if os.getenv("USE_SPOT_ONLY") == "1":
        print("[Data] SPOT only (USE_SPOT_ONLY=1).")
        df = _paginate_klines("https://api.binance.com/api/v3/klines", symbol, interval, days)
        source = "SPOT_ONLY"
    else:
        try:
            print("[Data] Tentando FUTURES (fapi)...")
            df = _paginate_klines("https://fapi.binance.com/fapi/v1/klines", symbol, interval, days)
            source = "FUTURES"
        except HTTPError as e:
            if getattr(e, "status", None) == 451:
                print("[Data] 451 em FUTURES -> fallback para SPOT.")
                df = _paginate_klines("https://api.binance.com/api/v3/klines", symbol, interval, days)
                source = "SPOT_FALLBACK"
            else:
                print(f"[Data] HTTP {getattr(e,'status',None)} em FUTURES -> reerguendo.")
                raise

    df.columns = [
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ]
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["open_time"], unit="ms", utc=True),
        "open":  df["open"].astype(float),
        "high":  df["high"].astype(float),
        "low":   df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume":df["volume"].astype(float)
    }).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"[Data] Fonte: {source} | Candles: {len(out)}")
    return out, source

# ===== Indicadores =====
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

# ===== Execução =====
def latest_signal(symbol=SYMBOL, interval=INTERVAL, days=DAYS):
    df, source = get_klines_with_fallback(symbol, interval, days)
    df = compute_indicators(df, 8, 20, 21, 5, 14)
    last = df.iloc[-1]
    signal = generate_signal_row(last)
    payload = {
        "symbol": symbol,
        "interval": interval,
        "timestamp": str(last["timestamp"]),
        "close": float(last["close"]),
        "signal": signal,
        "rsi": float(last["RSI"]),
        "ema_fast": float(last["EMA_FAST"]),
        "ema_slow": float(last["EMA_SLOW"]),
        "slope": float(last["Slope"]),
        "volatility": float(last["Volatility"]) if not np.isnan(last["Volatility"]) else None,
        "sl": float(last["close"]*0.98),
        "tp": float(last["close"]*1.04),
        "_source": source,
        "_generated_at_utc": datetime.now(timezone.utc).isoformat()
    }
    return payload

def main():
    out = latest_signal(SYMBOL, INTERVAL, DAYS)
    msg = (
        f"🪙 *{out['symbol']}* ({out['interval']})\n"
        f"🕒 {out['timestamp']}\n"
        f"📈 *Sinal:* {out['signal']}\n"
        f"💰 Preço: {out['close']}\n"
        f"RSI: {out['rsi']:.2f} | EMAf: {out['ema_fast']:.6f} | EMAl: {out['ema_slow']:.6f}\n"
        f"Slope: {out['slope']:.6f} | Vol: {out['volatility']}\n"
        f"🛑 SL: {out['sl']:.6f} | 🎯 TP: {out['tp']:.6f}\n"
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
        raise
