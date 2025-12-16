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
LIVE_TRADING = False
TIMEFRAME = "4h"

WATCHLIST = ["BTC/USD", "ETH/USD"]
MAX_POSITIONS = 2

RISK_PER_TRADE = 0.01
MIN_TRADE_USD = 10

ATR_MULTIPLIER_SL = 2.5
ATR_MULTIPLIER_TRAIL = 2.0

TRADES_CSV = "crypto_trades_log.csv"
DAILY_BALANCE_CSV = "daily_balance_log.csv"

# ==========================================================
# EXCHANGE
# ==========================================================
exchange = ccxt.kraken({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True
})

# ==========================================================
# AWS S3
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
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except:
        pass

def upload_csv(local, remote):
    try:
        s3.upload_file(local, AWS_S3_BUCKET, remote)
    except:
        pass

def download_csv(remote, local, headers):
    try:
        s3.download_file(AWS_S3_BUCKET, remote, local)
    except:
        with open(local, "w", newline="") as f:
            csv.writer(f).writerow(headers)

# ==========================================================
# MARKET DATA
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
# PORTFOLIO
# ==========================================================
def get_balances():
    bal = exchange.fetch_balance()["free"]
    usd = bal.get("USD", 0)

    positions = {
        k.replace("X","").replace("Z","") + "/USD": v
        for k, v in bal.items()
        if v > 0 and k not in ["USD","ZUSD"]
    }
    return usd, positions

def portfolio_value(positions):
    total = 0.0
    for pair, amt in positions.items():
        try:
            price = exchange.fetch_ticker(pair)["last"]
            total += amt * price
        except:
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

# ==========================================================
# >>> PERFORMANCE METRICS
# ==========================================================
def calculate_performance():
    if not os.path.isfile(TRADES_CSV):
        return None

    df = pd.read_csv(TRADES_CSV)

    buys = df[df["side"] == "BUY"]
    sells = df[df["side"] == "SELL"]

    if sells.empty or buys.empty:
        return None

    trade_results = []

    for _, sell in sells.iterrows():
        prior_buys = buys[
            (buys["symbol"] == sell["symbol"]) &
            (buys["timestamp"] < sell["timestamp"])
        ]

        if prior_buys.empty:
            continue

        buy = prior_buys.iloc[-1]
        pnl = (sell["price"] - buy["price"]) * sell["qty"]
        trade_results.append(pnl)

    if not trade_results:
        return None

    wins = [p for p in trade_results if p > 0]
    losses = [p for p in trade_results if p < 0]

    win_rate = len(wins) / len(trade_results)
    avg_win = sum(wins)/len(wins) if wins else 0
    avg_loss = abs(sum(losses)/len(losses)) if losses else 0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        "trades": len(trade_results),
        "win_rate": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy
    }

# ==========================================================
# STRATEGY
# ==========================================================
def generate_signal(df):
    latest = df.iloc[-1]

    if latest["close"] < latest["EMA_200"]:
        return "HOLD"

    if latest["RSI"] > 50 and latest["close"] > latest["EMA_50"]:
        return "BUY"

    return "HOLD"

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
# MAIN BOT
# ==========================================================
def run_bot():
    print("\n⏱️ Running bot", datetime.datetime.now())

    download_csv(
        TRADES_CSV, TRADES_CSV,
        ["timestamp","symbol","side","price","qty","sl","trail_sl"]
    )

    trades = pd.read_csv(TRADES_CSV)

    usd, positions = get_balances()

    # ----- SELL MANAGEMENT -----
    for pair, amt in positions.items():
        df = fetch_data(pair)
        price = df.iloc[-1]["close"]
        atr = df.iloc[-1]["ATR"]

        pair_buys = trades[
            (trades["symbol"] == pair) &
            (trades["side"] == "BUY")
        ]

        if pair_buys.empty:
            continue

        last_buy = pair_buys.iloc[-1]
        sl = float(last_buy["sl"])
        trail = float(last_buy["trail_sl"])

        new_trail = max(trail, price - atr * ATR_MULTIPLIER_TRAIL)

        if price <= sl or price <= new_trail:
            if LIVE_TRADING:
                exchange.create_market_sell_order(pair, amt)

            log_trade({
                "timestamp": datetime.datetime.now(),
                "symbol": pair,
                "side": "SELL",
                "price": price,
                "qty": amt,
                "sl": "",
                "trail_sl": ""
            })

            send_discord(f"🔻 SELL {pair} @ ${price:.2f}")
        else:
            trades.loc[last_buy.name, "trail_sl"] = new_trail
            trades.to_csv(TRADES_CSV, index=False)

    # ----- BUY LOGIC -----
    for pair in WATCHLIST:
        if pair in positions:
            continue

        df = fetch_data(pair)
        if generate_signal(df) != "BUY":
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

    # >>> METRICS OUTPUT
    metrics = calculate_performance()
    if metrics:
        msg = (
            f"📊 Performance\n"
            f"Trades: {metrics['trades']}\n"
            f"Win Rate: {metrics['win_rate']:.2f}%\n"
            f"Avg Win: ${metrics['avg_win']:.2f}\n"
            f"Avg Loss: ${metrics['avg_loss']:.2f}\n"
            f"Expectancy: ${metrics['expectancy']:.2f}"
        )
        print(msg)
        send_discord(msg)

    upload_csv(TRADES_CSV, TRADES_CSV)
    print("✅ Cycle complete")

# ==========================================================
# RUN ONCE (MANUAL)
# ==========================================================
# run_bot()

# ==========================================================
# SCHEDULER (ENABLE LATER)
# ==========================================================
schedule.every(4).hours.do(run_bot)
print("🟢 Bot running (4H swing strategy)")
while True:
    schedule.run_pending()
    time.sleep(30)
