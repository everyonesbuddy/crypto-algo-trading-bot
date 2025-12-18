import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import datetime
import os
import csv
import time
import boto3
import schedule
from botocore.client import Config
from dotenv import load_dotenv

# ==========================================================
# ENVIRONMENT SETUP
# ==========================================================
load_dotenv()

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_REGION = os.getenv("AWS_REGION")

# ==========================================================
# STRATEGY SETTINGS
# ==========================================================
LIVE_TRADING = True
TIMEFRAME = "4h"

WATCHLIST = ["BTC/USD", "ETH/USD"]
MAX_POSITIONS = 2

RISK_PER_TRADE = 0.01
MIN_TRADE_USD = 10
MIN_POSITION_USD = 10     # 🔑 DUST FILTER (IMPORTANT)

ATR_MULTIPLIER_SL = 2.5
ATR_MULTIPLIER_TRAIL = 2.0

TRADES_CSV = "crypto_trades_log.csv"
DAILY_BALANCE_CSV = "daily_balance_log.csv"

# ==========================================================
# EXCHANGE CONNECTION
# ==========================================================
exchange = ccxt.kraken({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True
})

# ==========================================================
# AWS S3 CONNECTION
# ==========================================================
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION
)

s3 = session.client("s3", config=Config(s3={'addressing_style': 'virtual'}))

# ==========================================================
# UTILITIES
# ==========================================================
def send_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception:
        pass

def upload_csv(local, remote):
    try:
        s3.upload_file(local, AWS_S3_BUCKET, remote)
    except Exception:
        pass

def download_csv(remote, local, headers):
    try:
        s3.download_file(AWS_S3_BUCKET, remote, local)
    except Exception:
        with open(local, "w", newline="") as f:
            csv.writer(f).writerow(headers)

# ==========================================================
# MARKET DATA & INDICATORS
# ==========================================================
def fetch_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=300)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)

    df["EMA_50"] = ta.ema(df["close"], 50)
    df["EMA_200"] = ta.ema(df["close"], 200)
    df["RSI"] = ta.rsi(df["close"], 14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], 14)

    df.dropna(inplace=True)
    return df

# ==========================================================
# PORTFOLIO HELPERS (DUST-SAFE)
# ==========================================================
def get_balances():
    bal = exchange.fetch_balance()["free"]
    usd = bal.get("USD", 0)

    positions = {}

    for asset, amount in bal.items():
        if asset in ["USD", "ZUSD"] or amount <= 0:
            continue

        pair = asset.replace("X","").replace("Z","") + "/USD"

        try:
            price = exchange.fetch_ticker(pair)["last"]
            value_usd = amount * price
        except Exception:
            continue

        # 🔑 Ignore dust
        if value_usd >= MIN_POSITION_USD:
            positions[pair] = amount

    return usd, positions

def portfolio_value(positions):
    total = 0.0
    for pair, amt in positions.items():
        try:
            total += amt * exchange.fetch_ticker(pair)["last"]
        except Exception:
            pass
    return total

# ==========================================================
# LOGGING
# ==========================================================
def log_trade(row):
    exists = os.path.isfile(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(row.keys())
        w.writerow(row.values())

def log_daily(date, start, current):
    exists = os.path.isfile(DAILY_BALANCE_CSV)
    with open(DAILY_BALANCE_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date","starting_balance","current_balance"])
        w.writerow([date, f"{start:.2f}", f"{current:.2f}"])

# ==========================================================
# STRATEGY LOGIC (WITH DIAGNOSTICS)
# ==========================================================
def generate_signal(df):
    latest = df.iloc[-1]
    reasons = []

    if latest["close"] < latest["EMA_200"]:
        reasons.append("price < EMA200")
    if latest["close"] < latest["EMA_50"]:
        reasons.append("price < EMA50")
    if latest["RSI"] <= 50:
        reasons.append(f"RSI {latest['RSI']:.1f} ≤ 50")

    if reasons:
        return "HOLD", reasons

    return "BUY", []

# ==========================================================
# POSITION SIZING
# ==========================================================
def calc_position_size(usd_balance, atr, price):
    risk_usd = usd_balance * RISK_PER_TRADE
    stop_distance = atr * ATR_MULTIPLIER_SL
    if stop_distance <= 0:
        return 0
    qty = risk_usd / stop_distance
    return max(qty * price, MIN_TRADE_USD)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================
def compute_performance():
    if not os.path.isfile(TRADES_CSV):
        return None

    df = pd.read_csv(TRADES_CSV)
    df = df[df["side"].isin(["BUY","SELL"])]

    results = []
    for _, g in df.groupby("symbol"):
        buys = g[g.side=="BUY"].reset_index()
        sells = g[g.side=="SELL"].reset_index()
        for i in range(min(len(buys), len(sells))):
            pnl = (sells.loc[i].price - buys.loc[i].price) / buys.loc[i].price
            results.append(pnl)

    if not results:
        return None

    s = pd.Series(results)
    return {
        "trades": len(s),
        "win_rate": (s > 0).mean(),
        "expectancy": s.mean()
    }

# ==========================================================
# MAIN BOT
# ==========================================================
LAST_STATUS = {}

def run_bot():
    print("\n⏱️ Running bot", datetime.datetime.now())

    download_csv(TRADES_CSV, TRADES_CSV,
        ["timestamp","symbol","side","price","qty","sl","trail_sl"])
    download_csv(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV,
        ["date","starting_balance","current_balance"])

    usd, positions = get_balances()
    total = usd + portfolio_value(positions)
    log_daily(datetime.date.today().isoformat(), total, total)

    for pair in WATCHLIST:
        if pair in positions:
            continue

        df = fetch_data(pair)
        signal, reasons = generate_signal(df)

        if signal == "HOLD":
            reason_text = " | ".join(reasons)
            if LAST_STATUS.get(pair) != reason_text:
                send_discord(f"⏸️ {pair} — No trade\n• {reason_text}")
                LAST_STATUS[pair] = reason_text
            continue

        atr = df.iloc[-1]["ATR"]
        price = df.iloc[-1]["close"]
        alloc = calc_position_size(usd, atr, price)
        qty = alloc / price
        sl = price - atr * ATR_MULTIPLIER_SL

        if LIVE_TRADING:
            exchange.create_market_buy_order(pair, qty)

        log_trade({
            "timestamp": datetime.datetime.now(),
            "symbol": pair,
            "side": "BUY",
            "price": price,
            "qty": qty,
            "sl": sl,
            "trail_sl": sl
        })

        send_discord(f"🟢 BUY {pair} @ ${price:.2f}")
        LAST_STATUS[pair] = "BOUGHT"

    perf = compute_performance()
    if perf:
        send_discord(
            f"📊 Performance\n"
            f"Trades: {perf['trades']} | "
            f"WinRate: {perf['win_rate']*100:.2f}% | "
            f"Expectancy: {perf['expectancy']*100:.2f}%"
        )

    upload_csv(TRADES_CSV, TRADES_CSV)
    upload_csv(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)

    print("✅ Cycle complete")
# ==========================================================
# SCHEDULER
# ==========================================================
schedule.every(4).hours.do(run_bot)
print("🟢 Bot running (4H swing strategy)")
while True:
    schedule.run_pending()
    time.sleep(30)

# ==========================================================
# RUN ONCE
# ==========================================================
# run_bot()
