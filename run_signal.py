import os, time, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from requests.exceptions import HTTPError

# ===== Config por ENV =====
SYMBOL   = os.getenv("SYMBOL", "DOGEUSDT")   # ex.: DOGEUSDT
INTERVAL = os.getenv("INTERVAL", "1h")       # "1h", "15m", "4h"
DAYS     = int(os.getenv("DAYS", "10"))
USE_PROVIDER = os.getenv("USE_PROVIDER", "").upper()  # BINANCE_FUTURES|BINANCE_SPOT|BYBIT|OKX|""(auto)

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ===== HTTP session (User-Agent + retry) =====
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (CryptoSignalBot/1.1)"})

def http_get(url, timeout=30, retries=3, backoff=2.0):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            print(f"[HTTP] {code} for {url}")
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

# ===== util =====
INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,"1d":1440}
def to_ms(dt): return int(dt.timestamp()*1000)

# ===== Coletor: Binance Futures (paginado) =====
def fetch_binance_futures(symbol, interval, days, limit=1500):
    step = INTERVAL_TO_MIN[interval]
    total = int(days*24*60/step)
    iters = total//limit + 1
    end = datetime.now(timezone.utc)
    frames = []
    for _ in range(iters):
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={to_ms(end)}"
        data = http_get(url).json()
        if not data: break
        frames.append(pd.DataFrame(data))
        first_open = int(data[0][0])
        end = datetime.fromtimestamp(first_open/1000, tz=timezone.utc) - timedelta(milliseconds=1)
        time.sleep(0.2)
    if not frames: raise RuntimeError("Sem dados Futures")
    df = pd.concat(frames, ignore_index=True)
    df.columns = ['open_time','open','high','low','close','volume','close_time','qav','trades','tbbav','tbqav','ignore']
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["open_time"], unit="ms", utc=True),
        "open":df["open"].astype(float),"high":df["high"].astype(float),
        "low":df["low"].astype(float),"close":df["close"].astype(float),
        "volume":df["volume"].astype(float),
    }).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out

# ===== Coletor: Binance Spot (paginado) =====
def fetch_binance_spot(symbol, interval, days, limit=1500):
    step = INTERVAL_TO_MIN[interval]
    total = int(days*24*60/step)
    iters = total//limit + 1
    end = datetime.now(timezone.utc)
    frames = []
    for _ in range(iters):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={to_ms(end)}"
        data = http_get(url).json()
        if not data: break
        frames.append(pd.DataFrame(data))
        first_open = int(data[0][0])
        end = datetime.fromtimestamp(first_open/1000, tz=timezone.utc) - timedelta(milliseconds=1)
        time.sleep(0.2)
    if not frames: raise RuntimeError("Sem dados Spot")
    df = pd.concat(frames, ignore_index=True)
    df.columns = ['open_time','open','high','low','close','volume','close_time','qav','trades','tbbav','tbqav','ignore']
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["open_time"], unit="ms", utc=True),
        "open":df["open"].astype(float),"high":df["high"].astype(float),
        "low":df["low"].astype(float),"close":df["close"].astype(float),
        "volume":df["volume"].astype(float),
    }).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out

# ===== Coletor: BYBIT (v5 market/kline) =====
# Doc: https://bybit-exchange.github.io/docs/v5/market/kline
BYBIT_INTERVAL_MAP = {
    "1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
    "1h":"60","2h":"120","4h":"240","6h":"360","12h":"720","1d":"D"
}
def fetch_bybit(symbol, interval, days, limit=1000):
    iv = BYBIT_INTERVAL_MAP[interval]
    # Bybit retorna até `limit` candles por chamada; sem `end` pega os mais recentes.
    # Faremos múltiplas chamadas recuando o tempo pelo open time retornado.
    frames = []
    next_end = None
    step = INTERVAL_TO_MIN[interval]
    total_need = int(days*24*60/step)
    while total_need > 0:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={iv}&limit={min(limit, total_need)}"
        if next_end is not None:
            url += f"&end={next_end}"
        data = http_get(url).json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
        rows = data["result"]["list"]
        if not rows: break
        # Bybit retorna [start, open, high, low, close, volume, turnover] em strings
        rows_df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
        rows_df["start"] = pd.to_datetime(rows_df["start"].astype("int64"), unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            rows_df[c] = rows_df[c].astype(float)
        frames.append(rows_df)
        # prepara próxima página (antes do primeiro registro atual)
        first_start = int(rows[-1][0])  # menor timestamp desta página
        next_end = first_start - 1
        total_need -= len(rows)
        time.sleep(0.2)
    if not frames: raise RuntimeError("Sem dados Bybit")
    df = pd.concat(frames, ignore_index=True)[["start","open","high","low","close","volume"]]
    df = df.rename(columns={"start":"timestamp"}).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df

# ===== Coletor: OKX (candles) =====
# Doc: https://www.okx.com/docs-v5/en/#market-data-market-candles
OKX_INTERVAL_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H","1d":"1D"
}
def okx_symbol(symbol:str)->str:
    # BTCUSDT -> BTC-USDT
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USDT"
    return symbol

def fetch_okx(symbol, interval, days, limit=300):
    inst = okx_symbol(symbol)
    bar  = OKX_INTERVAL_MAP[interval]
    frames = []
    step = INTERVAL_TO_MIN[interval]
    total_need = int(days*24*60/step)
    before = None
    while total_need > 0:
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar={bar}&limit={min(limit, total_need)}"
        if before:
            url += f"&before={before}"
        data = http_get(url).json()
        rows = data.get("data", [])
        if not rows: break
        # OKX retorna [[ts, o,h,l,c, vol, ...], ...] em ordem DESC
        rows_df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","volccy","volccyquote","confirm"])
        rows_df["ts"] = pd.to_datetime(rows_df["ts"].astype("int64"), unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            rows_df[c] = rows_df[c].astype(float)
        frames.append(rows_df)
        # próximo page: before = último ts (menor) desta página
        before = int(rows[-1][0])
        total_need -= len(rows)
        time.sleep(0.2)
    if not frames: raise RuntimeError("Sem dados OKX")
    df = pd.concat(frames, ignore_index=True)[["ts","open","high","low","close","volume"]]
    df = df.rename(columns={"ts":"timestamp"}).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df

# ===== Orquestrador de provedores =====
def get_klines_multi(symbol, interval, days):
    chain = []
    if USE_PROVIDER:
        chain = [USE_PROVIDER]
    else:
        chain = ["BINANCE_FUTURES", "BINANCE_SPOT", "BYBIT", "OKX"]

    last_err = None
    for prov in chain:
        try:
            if prov == "BINANCE_FUTURES":
                print("[Data] BINANCE_FUTURES…")
                return fetch_binance_futures(symbol, interval, days), prov
            elif prov == "BINANCE_SPOT":
                print("[Data] BINANCE_SPOT…")
                return fetch_binance_spot(symbol, interval, days), prov
            elif prov == "BYBIT":
                print("[Data] BYBIT…")
                return fetch_bybit(symbol, interval, days), prov
            elif prov == "OKX":
                print("[Data] OKX…")
                return fetch_okx(symbol, interval, days), prov
        except Exception as e:
            last_err = e
            print(f"[Data] {prov} falhou: {e}. Tentando próximo…")
            continue
    raise RuntimeError(f"Todas as fontes falharam. Último erro: {last_err}")

# ===== Indicadores & sinal =====
def compute_indicators(df, ema_fast=8, ema_slow=20, rsi_period=21, slope_window=5, vol_window=14):
    dd = df.copy()
    dd["EMA_FAST"] = dd["close"].ewm(span=ema_fast, adjust=False).mean()
    dd["EMA_SLOW"] = dd["close"].ewm(span=ema_slow, adjust=False).mean()
    delta = dd["close"].diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_period).mean(); avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    dd["RSI"] = 100 - (100/(1+rs))
    dd["RSI"] = dd["RSI"].fillna(method="bfill")
    dd["Volatility"] = dd["close"].rolling(vol_window).std()

    def _slope(x):
        x = np.asarray(x); t = np.arange(len(x))
        if len(x) < 2: return np.nan
        m, _ = np.polyfit(t, x, 1); return m
    dd["Slope"] = dd["close"].rolling(slope_window).apply(_slope, raw=False)

    return dd.dropna().reset_index(drop=True)

def generate_signal_row(row):
    if row["EMA_FAST"] > row["EMA_SLOW"] and row["RSI"] > 50 and row["Slope"] > 0:
        return "BUY"
    if row["EMA_FAST"] < row["EMA_SLOW"] and row["RSI"] < 50 and row["Slope"] < 0:
        return "SELL"
    return "HOLD"

# ===== Execução =====
def latest_signal(symbol=SYMBOL, interval=INTERVAL, days=DAYS):
    df, source = get_klines_multi(symbol, interval, days)
    print(f"[Data] Fonte usada: {source} | candles: {len(df)}")
    df = compute_indicators(df, 8, 20, 21, 5, 14)
    last = df.iloc[-1]
    signal = generate_signal_row(last)
    payload = {
        "symbol": symbol, "interval": interval,
        "timestamp": str(last["timestamp"]), "close": float(last["close"]),
        "signal": signal, "rsi": float(last["RSI"]),
        "ema_fast": float(last["EMA_FAST"]), "ema_slow": float(last["EMA_SLOW"]),
        "slope": float(last["Slope"]),
        "volatility": float(last["Volatility"]) if not np.isnan(last["Volatility"]) else None,
        "sl": float(last["close"]*0.98), "tp": float(last["close"]*1.04),
        "_source": source, "_generated_at_utc": datetime.now(timezone.utc).isoformat()
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
