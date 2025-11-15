import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import datetime
import os
import time
import csv
import boto3
import schedule
from botocore.client import Config
from dotenv import load_dotenv

# ==========================================================
# CONFIGURATION & SETUP
# ==========================================================
load_dotenv()
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_REGION = os.getenv("AWS_REGION")

# --- Trading Settings ---
LIVE_TRADING = True                     # Set False for dry-run testing
USD_ALLOCATION_PER_TRADE = 10           # Max $ per trade
MAX_POSITIONS = 5                       # How many assets we can hold
MIN_TRADE_USD = 1.0                     # Smallest order size
DAILY_TARGET = 0.005                    # Daily target = +0.5%

# --- Risk Management Settings ---
STOP_LOSS_PCT = 0.02                    # 2% below buy price
TAKE_PROFIT_PCT = 0.04                  # 4% above buy price

WATCHLIST = [
    "BTC/USD","ETH/USD","SOL/USD","XRP/USD","ADA/USD",
    "DOGE/USD","AVAX/USD","LINK/USD","DOT/USD","LTC/USD"
]

TRADES_CSV = "crypto_trades_log.csv"
DAILY_BALANCE_CSV = "daily_balance_log.csv"

# ==========================================================
# AWS S3 CONNECTION
# ==========================================================
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION
)
s3_client = session.client("s3", config=Config(s3={'addressing_style': 'virtual'}))

# ==========================================================
# EXCHANGE CONNECTION (KRAKEN)
# ==========================================================
exchange = ccxt.kraken({
    "apiKey": API_KEY,
    "secret": API_SECRET
})

# ==========================================================
# S3 HELPERS
# ==========================================================
def upload_csv_to_s3(local_csv_path, s3_key):
    """Upload local CSV to S3 for persistence."""
    try:
        s3_client.upload_file(local_csv_path, AWS_S3_BUCKET, s3_key)
        print(f"✅ Uploaded {local_csv_path} to S3:{s3_key}")
    except Exception as e:
        print(f"❌ S3 upload error: {e}")

def download_csv_from_s3(s3_key, local_path):
    """Download CSV from S3 if it exists, otherwise create a new one."""
    try:
        s3_client.download_file(AWS_S3_BUCKET, s3_key, local_path)
        print(f"✅ Downloaded S3:{s3_key} to {local_path}")
    except s3_client.exceptions.ClientError:
        print(f"⚠️ {s3_key} not found, creating new file locally")
        with open(local_path, "w", newline="") as f:
            writer = csv.writer(f)
            if s3_key == TRADES_CSV:
                writer.writerow(["timestamp","symbol","action","price","amount","usd_allocation","stop_loss","take_profit"])
            elif s3_key == DAILY_BALANCE_CSV:
                writer.writerow(["date","starting_balance","current_balance"])

# ==========================================================
# DISCORD HELPERS
# ==========================================================
def send_discord_alert(message):
    """Send a simple alert to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}).raise_for_status()
    except Exception as e:
        print(f"❌ Discord error: {e}")

def send_signal_discord(symbol, action, latest=None, score=None, allocation=None, note=""):
    """Send formatted trade signal to Discord."""
    parts = [f"📊 {symbol} | Action: {action}"]
    if latest is not None:
        parts.append(f"Price: ${latest['close']:.2f}")
    if score is not None:
        parts.append(f"Signal Score: {score}")
    if allocation is not None:
        parts.append(f"Allocation: ${allocation:.2f}")
    if note:
        parts.append(note)
    send_discord_alert("\n".join(parts))

# ==========================================================
# MARKET DATA & INDICATORS
# ==========================================================
def get_crypto_data(symbol, timeframe="1h", limit=200):
    """Fetch OHLCV data for a symbol from Kraken."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate RSI, MACD, EMA, Bollinger Bands, and StochRSI."""
    if df.empty: return df
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    if macd is not None:
        df["MACD_Hist"] = macd["MACDh_12_26_9"]
    df["EMA_50"] = ta.ema(df["close"], length=50)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["BBL"], df["BBU"] = bb["BBL_20_2.0"], bb["BBU_20_2.0"]
    st = ta.stochrsi(df["close"], length=14)
    if st is not None:
        df["StochRSI_K"], df["StochRSI_D"] = st.iloc[:,0], st.iloc[:,1]
    df.dropna(inplace=True)
    return df

def check_signals(df):
    """Return BUY/SELL/HOLD and confidence score."""
    if df.empty: return "HOLD", None, 0
    latest = df.iloc[-1]
    sell_conditions = [
        latest["RSI"] > 60,
        latest["MACD_Hist"] < 0,
        latest["close"] < latest["EMA_50"],
        latest["StochRSI_K"] < latest["StochRSI_D"],
        latest["close"] > latest["BBU"] * 0.97
    ]
    buy_conditions = [
        latest["RSI"] < 45,
        latest["MACD_Hist"] > 0,
        latest["close"] > latest["EMA_50"],
        latest["StochRSI_K"] > latest["StochRSI_D"],
        latest["close"] < latest["BBL"] * 1.03
    ]
    buy_score = sum(buy_conditions)
    sell_score = sum(sell_conditions)
    if buy_score >= 3: return "BUY", latest, buy_score
    elif sell_score >= 2: return "SELL", latest, sell_score
    else: return "HOLD", latest, max(buy_score, sell_score)

# ==========================================================
# PORTFOLIO HELPERS
# ==========================================================
def normalize_asset(asset):
    """Normalize Kraken asset codes to standard symbols."""
    mapping = {"XXBT":"BTC","XBT":"BTC","XETH":"ETH","XSOL":"SOL","XXRP":"XRP","XADA":"ADA"}
    if asset in mapping: return mapping[asset]
    s = asset.replace(".F","")
    if s.startswith(("X","Z")) and len(s) > 3 and s[1:].isupper():
        s = s[1:]
    return s

def get_portfolio():
    """Return open crypto positions (non-USD)."""
    try:
        balances = exchange.fetch_balance()
        free = balances.get("free", {})
        portfolio = {}
        for asset, amount in free.items():
            if amount and amount > 0 and asset not in ("USD", "ZUSD"):
                pair = f"{normalize_asset(asset)}/USD"
                portfolio[pair] = portfolio.get(pair, 0) + amount
        return portfolio
    except Exception as e:
        print(f"❌ Portfolio fetch error: {e}")
        return {}

def get_tradable_usd():
    """Return available USD balance."""
    try:
        balances = exchange.fetch_balance()
        return float(balances.get("free", {}).get("USD", 0) or 0)
    except Exception:
        return 0.0

def get_portfolio_value(portfolio):
    """Calculate total USD value of portfolio positions."""
    total = 0.0
    for pair, amt in portfolio.items():
        try:
            price = exchange.fetch_ticker(pair).get("last") or 0
            total += amt * price
        except Exception:
            pass
    return total

# ==========================================================
# LOGGING HELPERS
# ==========================================================
def log_trade(symbol, action, price, amount, usd_alloc, stop_loss=None, take_profit=None):
    """Log trades with SL/TP if applicable."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.isfile(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","symbol","action","price","amount","usd_allocation","stop_loss","take_profit"])
        w.writerow([ts, symbol, action, f"{price:.8f}", f"{amount:.8f}", f"{usd_alloc:.2f}", stop_loss, take_profit])

def log_daily_balance(date, start_balance, current_balance):
    """Log the daily balance progression."""
    exists = os.path.isfile(DAILY_BALANCE_CSV)
    with open(DAILY_BALANCE_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date","starting_balance","current_balance"])
        w.writerow([date, f"{start_balance:.2f}", f"{current_balance:.2f}"])

# ==========================================================
# TRADING LOGIC
# ==========================================================
def place_trade(symbol, side, usd_amount):
    """Execute live or test trade."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or ticker.get("close") or 0)
        qty = usd_amount / price if price > 0 else 0
        if usd_amount < MIN_TRADE_USD: return None, 0.0, price
        if not LIVE_TRADING:
            print(f"[TEST] {side} {qty:.8f} {symbol} @ ${price:.2f}")
            return None, qty, price
        if side.lower() == "buy":
            order = exchange.create_market_buy_order(symbol, qty)
        else:
            order = exchange.create_market_sell_order(symbol, qty)
        print(f"✅ {side.upper()} executed: {qty:.8f} {symbol} @ ${price:.2f}")
        return order, qty, price
    except Exception as e:
        print(f"❌ Trade error {symbol}: {e}")
        return None, 0.0, 0.0

# ==========================================================
# MAIN BOT FUNCTION
# ==========================================================
def run_crypto_bot():
    print(f"\n⏱️ Running bot {datetime.datetime.now()}")
    print("AWS_S3_BUCKET:", AWS_S3_BUCKET)

    # --- S3 Sync ---
    download_csv_from_s3(TRADES_CSV, TRADES_CSV)
    download_csv_from_s3(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)

    # --- Portfolio & Balances ---
    portfolio = get_portfolio()
    tradable_usd = get_tradable_usd()
    total_value = tradable_usd + get_portfolio_value(portfolio)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if os.path.isfile(DAILY_BALANCE_CSV):
        df_bal = pd.read_csv(DAILY_BALANCE_CSV)
    else:
        df_bal = pd.DataFrame(columns=["date","starting_balance","current_balance"])

    if today not in df_bal["date"].values:
        starting_balance = total_value
        log_daily_balance(today, starting_balance, total_value)
    else:
        starting_balance = float(df_bal[df_bal["date"] == today]["starting_balance"].iloc[0])
        log_daily_balance(today, starting_balance, total_value)

    daily_return = (total_value - starting_balance) / starting_balance if starting_balance > 0 else 0
    print(f"📈 Daily P/L: {daily_return*100:.2f}% | Tradable USD: ${tradable_usd:.2f}")

    # --- Global daily take-profit check ---
    if daily_return >= DAILY_TARGET:
        print("🎯 DAILY TARGET REACHED — closing all positions")
        for pair, amt in portfolio.items():
            if amt > 0:
                price = exchange.fetch_ticker(pair)["last"]
                place_trade(pair, "sell", amt * price)
                log_trade(pair, "SELL (DAILY TP)", price, amt, amt*price)
        upload_csv_to_s3(TRADES_CSV, TRADES_CSV)
        upload_csv_to_s3(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)
        return

    # --- SELL PASS (Stop-loss / Take-profit / Technical) ---
    trade_log = pd.read_csv(TRADES_CSV) if os.path.isfile(TRADES_CSV) else pd.DataFrame()
    for pair, amt in list(portfolio.items()):
        df = calculate_indicators(get_crypto_data(pair))
        if df.empty: continue
        latest_price = df.iloc[-1]["close"]

        # find last BUY trade
        open_trades = trade_log[(trade_log["symbol"] == pair) & (trade_log["action"].str.contains("BUY"))]
        if not open_trades.empty:
            last_trade = open_trades.iloc[-1]
            stop_loss = float(last_trade["stop_loss"]) if pd.notna(last_trade["stop_loss"]) else 0
            take_profit = float(last_trade["take_profit"]) if pd.notna(last_trade["take_profit"]) else 0

            # --- Stop Loss / Take Profit Enforcement ---
            if latest_price <= stop_loss:
                print(f"🚨 STOP LOSS hit for {pair}")
                place_trade(pair, "sell", amt * latest_price)
                log_trade(pair, "SELL (SL)", latest_price, amt, amt*latest_price)
                continue
            elif latest_price >= take_profit:
                print(f"🎯 TAKE PROFIT hit for {pair}")
                place_trade(pair, "sell", amt * latest_price)
                log_trade(pair, "SELL (TP)", latest_price, amt, amt*latest_price)
                continue

        # --- Technical SELL signal ---
        action, latest, score = check_signals(df)
        if action == "SELL":
            place_trade(pair, "sell", amt * latest_price)
            log_trade(pair, "SELL", latest_price, amt, amt*latest_price)

    # --- BUY PASS ---
    signals = []
    for pair in WATCHLIST:
        df = calculate_indicators(get_crypto_data(pair))
        if df.empty: continue
        action, latest, score = check_signals(df)
        if action == "BUY":
            signals.append((pair, score, latest))

    if signals:
        signals = sorted(signals, key=lambda x: x[1], reverse=True)[:MAX_POSITIONS]
        per_alloc = min(USD_ALLOCATION_PER_TRADE, tradable_usd / len(signals))
        for pair, score, latest in signals:
            order, qty, price = place_trade(pair, "buy", per_alloc)
            if qty > 0:
                sl_price = price * (1 - STOP_LOSS_PCT)
                tp_price = price * (1 + TAKE_PROFIT_PCT)
                log_trade(pair, "BUY", price, qty, per_alloc, sl_price, tp_price)
                send_signal_discord(pair, "BUY", latest, score, per_alloc,
                    note=f"SL={sl_price:.2f}, TP={tp_price:.2f}")
    else:
        print("ℹ️ No BUY signals this run")

    # --- Final sync to S3 ---
    upload_csv_to_s3(TRADES_CSV, TRADES_CSV)
    upload_csv_to_s3(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)
    print("🔁 Run complete")

# ==========================================================
# RUN BOT Manually
# ==========================================================
# run_crypto_bot()

# === Scheduler ===
schedule.every(3).hours.do(run_crypto_bot)


# # === Run Scheduler ===
print("🟢 Bot scheduler started")
while True:
    schedule.run_pending()
    time.sleep(1)
