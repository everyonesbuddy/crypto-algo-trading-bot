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
# STRATEGY SETTINGS (FIXED FOR SLOW BLEED MARKETS)
# ==========================================================
LIVE_TRADING = True

# Primary execution timeframe - 1H for better responsiveness
TIMEFRAME = "1h"
LOOKBACK_BARS = 500  # ~20 days of hourly data

# Regime detection timeframe
REGIME_TIMEFRAME = "4h"
REGIME_LOOKBACK = 200  # ~33 days of 4H data

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LTC/USD"]
MAX_POSITIONS = 5

# Risk & sizing - REDUCED for bear market
RISK_PER_TRADE = 0.01  # Back to 1% in bear market
MIN_TRADE_USD = 10
MIN_POSITION_USD = 10

# Dynamic stops based on recent volatility
ATR_MULTIPLIER_SL = 1.5  # Tighter stop in bear
ATR_MULTIPLIER_TRAIL = 1.2  # Very tight trail

# Dynamic take-profit - MUCH MORE CONSERVATIVE
TP_BULL_MULTIPLIER = 2.0
TP_BEAR_MULTIPLIER = 1.2  # Take fast profits
TP_RANGE_MULTIPLIER = 1.5

# Mean reversion settings - LOOSENED FOR SLOW BLEEDS
ENABLE_MEAN_REVERSION = True
MR_RSI_OVERSOLD = 40  # Loosened from 30 - catch earlier
MR_RSI_OVERBOUGHT = 62  # Tightened from 70 - exit faster
MR_VOLUME_THRESHOLD = 0.7  # Way looser - don't require volume in slow bleeds
MR_ALLOW_BELOW_EMA = True  # NEW: allow entries even if below EMA20

TRADES_CSV = "crypto_trades_log.csv"
DAILY_BALANCE_CSV = "daily_balance_log.csv"

# ==========================================================
# EXCHANGE CONNECTION (KRAKEN)
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
def send_discord(msg: str) -> None:
    """Send a message to Discord (best-effort)."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(f"Discord send failed: {e}")

def upload_csv(local: str, remote: str) -> None:
    """Upload a CSV to S3 with error handling."""
    try:
        s3.upload_file(local, AWS_S3_BUCKET, remote)
        print(f"✅ Uploaded {local} to S3")
    except Exception as e:
        print(f"⚠️ S3 upload failed for {local}: {e}")
        send_discord(f"⚠️ S3 upload failed for {local}: {e}")

def download_csv(remote: str, local: str, headers: list[str]) -> None:
    """Download CSV from S3 if it exists; otherwise create local CSV with headers."""
    try:
        s3.download_file(AWS_S3_BUCKET, remote, local)
        print(f"✅ Downloaded {remote} from S3")
    except Exception as e:
        print(f"⚠️ S3 download failed for {remote}, creating new file: {e}")
        with open(local, "w", newline="") as f:
            csv.writer(f).writerow(headers)

def safe_float(x, default=0.0) -> float:
    """Convert to float safely."""
    try:
        return float(x)
    except Exception:
        return float(default)

# ==========================================================
# MARKET DATA
# ==========================================================
def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV and return a standardized dataframe."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for 1H timeframe."""
    # Trend indicators
    df["EMA_20"] = ta.ema(df["close"], 20)
    df["EMA_50"] = ta.ema(df["close"], 50)
    df["EMA_100"] = ta.ema(df["close"], 100)

    # Momentum
    df["RSI"] = ta.rsi(df["close"], 14)

    # Stochastic for better oversold detection
    stoch = ta.stoch(df["high"], df["low"], df["close"], 14, 3, 3)
    df["Stoch_K"] = stoch["STOCHk_14_3_3"]
    df["Stoch_D"] = stoch["STOCHd_14_3_3"]

    # Volatility
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], 14)

    # Volume
    df["Volume_MA"] = df["volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["volume"] / df["Volume_MA"]

    # Support/Resistance zones
    df["Swing_High"] = df["high"].rolling(10, center=True).max()
    df["Swing_Low"] = df["low"].rolling(10, center=True).min()

    df.dropna(inplace=True)
    return df

def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators for regime detection on 4H timeframe."""
    df["EMA_50"] = ta.ema(df["close"], 50)
    df["EMA_100"] = ta.ema(df["close"], 100)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], 14)["ADX_14"]
    df.dropna(inplace=True)
    return df

def determine_regime(symbol: str) -> tuple[str, dict]:
    """
    Determine market regime using 4H timeframe.

    Returns:
      regime: "BULL", "BEAR", or "RANGE"
      info: dict with regime values for debugging
    """
    try:
        df = fetch_ohlcv(symbol, REGIME_TIMEFRAME, REGIME_LOOKBACK)
        df = add_regime_indicators(df)
        latest = df.iloc[-1]

        close = float(latest["close"])
        ema50 = float(latest["EMA_50"])
        ema100 = float(latest["EMA_100"])
        adx = float(latest["ADX"])

        # Trend strength threshold
        trending = adx > 25

        if trending:
            if close > ema50 and ema50 > ema100:
                regime = "BULL"
            elif close < ema50 and ema50 < ema100:
                regime = "BEAR"
            else:
                regime = "RANGE"
        else:
            regime = "RANGE"

        return regime, {
            "close": close,
            "ema50": ema50,
            "ema100": ema100,
            "adx": adx,
            "trending": trending
        }
    except Exception as e:
        return "RANGE", {"error": str(e)}

# ==========================================================
# PORTFOLIO HELPERS - FIXED WITH DEBUGGING
# ==========================================================
def get_balances() -> tuple[float, dict]:
    """
    Return USD balance and positions dict.

    CRITICAL FIX: Properly handle Kraken asset names
    """
    try:
        bal = exchange.fetch_balance().get("free", {})

        # DEBUG: Log what Kraken actually returns
        print(f"\n🔍 DEBUG: Raw Kraken balance keys: {list(bal.keys())}")

        usd = safe_float(bal.get("USD", 0.0))
        if usd == 0:
            usd = safe_float(bal.get("ZUSD", 0.0))

        positions = {}

        # CRITICAL: Kraken asset name mapping
        # Explicit mapping for known assets
        KRAKEN_ASSET_MAP = {
            "XXBT": "BTC",
            "XBT": "BTC",
            "XETH": "ETH",
            "ZUSD": "USD",
            "XXRP": "XRP",
        }

        for asset, amount in bal.items():
            amount = safe_float(amount)

            # Skip USD and zero balances
            if asset in ["USD", "ZUSD"] or amount <= 0:
                continue

            # Map asset name using explicit mapping first
            if asset in KRAKEN_ASSET_MAP:
                clean_asset = KRAKEN_ASSET_MAP[asset]
            else:
                # For unmapped assets, use smart logic:
                if asset.startswith(("X", "Z")) and len(asset) > 3:
                    # Try stripping first character
                    test_asset = asset[1:]
                    # Only use stripped version if it looks like a real ticker (3-4 chars)
                    if 3 <= len(test_asset) <= 4:
                        clean_asset = test_asset
                    else:
                        clean_asset = asset
                else:
                    # Asset is already in correct format (SOL, XRP, ADA, etc.)
                    clean_asset = asset

            # Create pair symbol
            pair = f"{clean_asset}/USD"

            # DEBUG: Log the mapping with more detail
            print(f"🔍 DEBUG: '{asset}' → '{clean_asset}' → '{pair}' | Amount: {amount:.8f}")

            # Verify pair exists in watchlist before checking price
            if pair not in WATCHLIST:
                print(f"⚠️ WARNING: {pair} not in watchlist, skipping")
                continue

            try:
                ticker = exchange.fetch_ticker(pair)
                price = safe_float(ticker.get("last", 0.0))

                if price <= 0:
                    print(f"⚠️ WARNING: Invalid price for {pair}: {price}")
                    continue

            except Exception as e:
                print(f"⚠️ WARNING: Failed to fetch ticker for {pair}: {e}")
                continue

            value_usd = amount * price

            # Filter out dust positions
            if value_usd >= MIN_POSITION_USD:
                positions[pair] = amount
                print(f"✅ Position: {pair} = {amount:.8f} (${value_usd:.2f})")
            else:
                print(f"⏭️ Skipping dust: {pair} = {amount:.8f} (${value_usd:.2f})")

        print(f"\n💰 Final USD: ${usd:.2f}")
        print(f"📊 Final Positions: {positions}")

        return usd, positions

    except Exception as e:
        print(f"🚨 CRITICAL: get_balances() failed: {e}")
        send_discord(f"🚨 CRITICAL: get_balances() failed: {e}")
        return 0.0, {}

def portfolio_value(positions: dict) -> float:
    """Estimate USD value of current positions."""
    total = 0.0
    for pair, amt in positions.items():
        try:
            price = safe_float(exchange.fetch_ticker(pair).get("last", 0.0))
            total += safe_float(amt) * price
        except Exception:
            pass
    return total

# ==========================================================
# LOGGING - FIXED WITH VERIFICATION
# ==========================================================
def ensure_trade_log_headers() -> None:
    """Ensure trade log has proper headers."""
    if os.path.isfile(TRADES_CSV):
        return
    with open(TRADES_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "symbol", "side", "price", "qty",
            "sl", "trail_sl", "tp", "regime", "signal_type", "exit_reason"
        ])

def log_trade(row: dict) -> None:
    """
    Append a trade row to CSV with verification and immediate S3 backup.

    CRITICAL FIX: Verify write succeeded and upload immediately
    """
    try:
        # Write to local file
        exists = os.path.isfile(TRADES_CSV)
        with open(TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(row.keys())
            w.writerow(row.values())

        print(f"✅ Logged trade: {row['side']} {row['symbol']} @ ${row['price']}")

        # IMMEDIATELY upload to S3 after EVERY trade
        upload_csv(TRADES_CSV, TRADES_CSV)

        # Verify the write succeeded by reading back
        verify_df = pd.read_csv(TRADES_CSV)
        if verify_df.empty:
            raise Exception("CSV appears empty after write!")

        # Verify the last row matches what we just wrote
        last_row = verify_df.iloc[-1]
        if last_row['symbol'] != row['symbol'] or last_row['side'] != row['side']:
            raise Exception(f"CSV verification failed! Expected {row['symbol']} {row['side']}, got {last_row['symbol']} {last_row['side']}")

        print(f"✅ Trade log verified")

    except Exception as e:
        error_msg = f"🚨 CRITICAL: log_trade() failed: {e}"
        print(error_msg)
        send_discord(error_msg)

        # Try to log to a backup file
        try:
            with open("trade_log_backup.txt", "a") as f:
                f.write(f"{datetime.datetime.now()}: {row}\n")
        except:
            pass

def log_daily(date: str, start: float, current: float) -> None:
    """Append daily balance snapshot."""
    exists = os.path.isfile(DAILY_BALANCE_CSV)
    with open(DAILY_BALANCE_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date", "starting_balance", "current_balance"])
        w.writerow([date, f"{start:.2f}", f"{current:.2f}"])

# ==========================================================
# STRATEGY LOGIC - FIXED FOR SLOW BLEEDS
# ==========================================================
def generate_entry_signal(df: pd.DataFrame, regime: str) -> tuple[str, str, list[str]]:
    """
    Generate entry signals based on regime and market conditions.

    Returns:
      signal: "BUY", "HOLD"
      signal_type: "MOMENTUM", "MEAN_REVERSION", or ""
      reasons: list of reason strings
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest["close"])
    ema20 = float(latest["EMA_20"])
    ema50 = float(latest["EMA_50"])
    ema100 = float(latest["EMA_100"])
    rsi = float(latest["RSI"])
    stoch_k = float(latest["Stoch_K"])
    vol_ratio = float(latest["Volume_Ratio"])

    reasons = []

    # ==========================================================
    # BULL REGIME: Momentum Trading
    # ==========================================================
    if regime == "BULL":
        if close <= ema20:
            reasons.append("price below EMA20")
        if ema20 <= ema50:
            reasons.append("EMA20 below EMA50")
        if rsi >= 65:
            reasons.append(f"RSI too high ({rsi:.1f})")
        if rsi <= 40:
            reasons.append(f"RSI too low ({rsi:.1f})")
        if vol_ratio < 1.0:
            reasons.append(f"low volume ({vol_ratio:.2f}x)")

        if not reasons:
            return "BUY", "MOMENTUM", []
        return "HOLD", "", reasons

    # ==========================================================
    # BEAR REGIME: Mean Reversion - LOOSENED SIGNIFICANTLY
    # ==========================================================
    elif regime == "BEAR":
        if not ENABLE_MEAN_REVERSION:
            return "HOLD", "", ["bear market - no mean reversion enabled"]

        # NEW LOGIC for slow-bleed bears:
        # Either: RSI < 40 OR Stochastic oversold (< 20)
        # Plus: Not in free-fall (some sign of stabilization)

        is_oversold = rsi < MR_RSI_OVERSOLD or stoch_k < 20

        # Check if price is stabilizing (not free-falling)
        # Allow entry even if below EMA20, as long as RSI/Stoch show oversold
        stabilizing = True  # Default to allowing

        if not MR_ALLOW_BELOW_EMA:
            # If strict mode, require price near EMA20
            if close < ema20 * 0.95:  # More than 5% below
                stabilizing = False
                reasons.append("price too far below EMA20")

        # Volume check - very loose now
        if vol_ratio < MR_VOLUME_THRESHOLD:
            reasons.append(f"insufficient volume ({vol_ratio:.2f}x)")

        # Main oversold check
        if not is_oversold:
            reasons.append(f"not oversold (RSI {rsi:.1f}, Stoch {stoch_k:.1f})")

        if not reasons and is_oversold:
            return "BUY", "MEAN_REVERSION", [f"bear bounce (RSI {rsi:.1f}, Stoch {stoch_k:.1f})"]
        return "HOLD", "", reasons

    # ==========================================================
    # RANGE REGIME: Mean Reversion
    # ==========================================================
    else:  # RANGE
        if not ENABLE_MEAN_REVERSION:
            return "HOLD", "", ["range market - no mean reversion enabled"]

        swing_low = float(latest["Swing_Low"])
        near_support = close <= swing_low * 1.02
        is_oversold = rsi < MR_RSI_OVERSOLD or stoch_k < 20

        if not is_oversold and not near_support:
            reasons.append(f"not oversold (RSI {rsi:.1f}) and not at support")
        if close < ema50:
            reasons.append("price below EMA50")
        if vol_ratio < 0.8:  # Very loose
            reasons.append(f"low volume ({vol_ratio:.2f}x)")

        if not reasons:
            return "BUY", "MEAN_REVERSION", ["range bounce"]
        return "HOLD", "", reasons

def calc_position_size(usd_balance: float, atr: float, price: float) -> float:
    """Risk-based position sizing."""
    usd_balance = safe_float(usd_balance)
    atr = safe_float(atr)
    price = safe_float(price)

    risk_usd = usd_balance * RISK_PER_TRADE
    stop_distance = atr * ATR_MULTIPLIER_SL

    if stop_distance <= 0 or price <= 0:
        return 0.0

    qty = risk_usd / stop_distance
    usd_alloc = qty * price
    return max(usd_alloc, MIN_TRADE_USD)

def calc_targets(price: float, atr: float, regime: str, signal_type: str) -> tuple[float, float]:
    """
    Calculate stop-loss and take-profit.

    Returns:
      sl: stop loss price
      tp: take profit price
    """
    sl = price - atr * ATR_MULTIPLIER_SL

    # Dynamic TP based on regime
    if regime == "BULL" and signal_type == "MOMENTUM":
        multiplier = TP_BULL_MULTIPLIER
    elif regime == "BEAR" or signal_type == "MEAN_REVERSION":
        multiplier = TP_BEAR_MULTIPLIER
    else:  # RANGE
        multiplier = TP_RANGE_MULTIPLIER

    R = price - sl
    tp = price + multiplier * R

    return sl, tp

def place_market_order(symbol: str, side: str, qty: float) -> bool:
    """
    Place a market order with error handling.

    Returns: True if successful, False if failed
    """
    if qty <= 0:
        print(f"⚠️ Invalid quantity: {qty}")
        return False

    try:
        if not LIVE_TRADING:
            print(f"[TEST] {side.upper()} {symbol} qty={qty:.8f}")
            return True

        if side.lower() == "buy":
            order = exchange.create_market_buy_order(symbol, qty)
        else:
            order = exchange.create_market_sell_order(symbol, qty)

        print(f"✅ Order executed: {side.upper()} {symbol} qty={qty:.8f}")
        print(f"🔍 Order details: {order}")
        return True

    except Exception as e:
        error_msg = f"🚨 CRITICAL: Order failed! {side.upper()} {symbol} qty={qty:.8f}: {e}"
        print(error_msg)
        send_discord(error_msg)
        return False

# ==========================================================
# TRADE MANAGEMENT
# ==========================================================
def load_trades_df() -> pd.DataFrame:
    """Load trades CSV."""
    if not os.path.isfile(TRADES_CSV):
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "side", "price", "qty",
            "sl", "trail_sl", "tp", "regime", "signal_type", "exit_reason"
        ])
    df = pd.read_csv(TRADES_CSV)
    return df

def get_last_open_trade(trades_df: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Find the most recent open BUY for a symbol."""
    g = trades_df[trades_df["symbol"] == symbol].copy()
    if g.empty:
        return None

    g = g[g["side"].isin(["BUY", "SELL"])].copy()
    if g.empty:
        return None

    buys = g[g["side"] == "BUY"]
    sells = g[g["side"] == "SELL"]

    # DEBUG
    print(f"🔍 DEBUG: {symbol} - Buys: {len(buys)}, Sells: {len(sells)}")

    if len(buys) <= len(sells):
        return None

    return buys.iloc[-1]

def update_trailing_stop(trades_df: pd.DataFrame, buy_row_index: int, new_trail: float) -> None:
    """Update trailing stop in trades CSV."""
    trades_df.loc[buy_row_index, "trail_sl"] = new_trail
    trades_df.to_csv(TRADES_CSV, index=False)
    upload_csv(TRADES_CSV, TRADES_CSV)  # Immediate backup

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================
def compute_performance() -> dict | None:
    """Compute win-rate and expectancy from completed trades."""
    if not os.path.isfile(TRADES_CSV):
        return None

    df = pd.read_csv(TRADES_CSV)
    if df.empty or "side" not in df.columns:
        return None

    df = df[df["side"].isin(["BUY", "SELL"])].copy()
    if df.empty:
        return None

    results = []
    for symbol, g in df.groupby("symbol"):
        buys = g[g.side == "BUY"].reset_index(drop=True)
        sells = g[g.side == "SELL"].reset_index(drop=True)

        n = min(len(buys), len(sells))
        for i in range(n):
            entry = float(buys.loc[i, "price"])
            exitp = float(sells.loc[i, "price"])
            if entry > 0:
                results.append((exitp - entry) / entry)

    if not results:
        return None

    s = pd.Series(results)
    win_rate = float((s > 0).mean())
    expectancy = float(s.mean())
    avg_win = float(s[s > 0].mean()) if (s > 0).any() else 0
    avg_loss = float(s[s < 0].mean()) if (s < 0).any() else 0

    return {
        "trades": int(len(s)),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }

# ==========================================================
# DISCORD STATUS DEDUPING
# ==========================================================
LAST_STATUS = {}

def send_no_trade_once(symbol: str, reason_text: str) -> None:
    """Avoid spamming Discord."""
    if LAST_STATUS.get(symbol) != reason_text:
        send_discord(f"⏸️ {symbol} — No trade\n• {reason_text}")
        LAST_STATUS[symbol] = reason_text

# ==========================================================
# MAIN BOT - WITH COMPREHENSIVE DEBUGGING
# ==========================================================
def run_bot():
    now = datetime.datetime.now()
    print(f"\n{'='*60}")
    print(f"⏱️ Running bot at {now}")
    print(f"{'='*60}")

    # Setup files
    download_csv(
        TRADES_CSV, TRADES_CSV,
        ["timestamp", "symbol", "side", "price", "qty", "sl", "trail_sl", "tp", "regime", "signal_type", "exit_reason"]
    )
    download_csv(
        DAILY_BALANCE_CSV, DAILY_BALANCE_CSV,
        ["date", "starting_balance", "current_balance"]
    )
    ensure_trade_log_headers()

    # Load state
    trades_df = load_trades_df()
    print(f"\n📊 Loaded {len(trades_df)} historical trades")

    usd, positions = get_balances()
    print(f"\n💰 Account Status:")
    print(f"   USD: ${usd:.2f}")
    print(f"   Positions: {len(positions)}/{MAX_POSITIONS}")
    for pair, qty in positions.items():
        print(f"   - {pair}: {qty:.8f}")

    # Balance snapshot
    total = usd + portfolio_value(positions)
    log_daily(datetime.date.today().isoformat(), total, total)

    # ======================================================
    # 1) MANAGE EXISTING POSITIONS (EXIT LOGIC)
    # ======================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: Managing {len(positions)} existing positions")
    print(f"{'='*60}")

    for symbol, qty_held in positions.items():
        print(f"\n📍 Checking {symbol} (holding {qty_held:.8f})")

        if symbol not in WATCHLIST:
            print(f"⏭️ Skipping {symbol} (not in watchlist)")
            continue

        try:
            df = fetch_ohlcv(symbol, TIMEFRAME, LOOKBACK_BARS)
            df = add_indicators(df)
            latest = df.iloc[-1]
            price = float(latest["close"])
            atr = float(latest["ATR"])
            rsi = float(latest["RSI"])

            print(f"   Current price: ${price:.2f}, RSI: {rsi:.1f}, ATR: ${atr:.2f}")

        except Exception as e:
            send_discord(f"⚠️ {symbol} — data fetch failed: {e}")
            continue

        last_buy = get_last_open_trade(trades_df, symbol)
        if last_buy is None:
            send_discord(f"⚠️ {symbol} — position exists but no open BUY in log")
            print(f"⚠️ WARNING: No BUY record found in log!")
            continue

        sl = safe_float(last_buy.get("sl", 0.0))
        trail_sl = safe_float(last_buy.get("trail_sl", sl))
        tp = safe_float(last_buy.get("tp", 0.0))
        signal_type = last_buy.get("signal_type", "")
        entry_price = safe_float(last_buy.get("price", 0.0))

        print(f"   Entry: ${entry_price:.2f}, SL: ${sl:.2f}, Trail: ${trail_sl:.2f}, TP: ${tp:.2f}")
        print(f"   P/L: {((price - entry_price) / entry_price * 100):.2f}%")

        # Update trailing stop
        new_trail = max(trail_sl, price - atr * ATR_MULTIPLIER_TRAIL)

        if new_trail != trail_sl:
            print(f"   📈 Updating trailing stop: ${trail_sl:.2f} → ${new_trail:.2f}")

        buy_row_index = int(last_buy.name) if hasattr(last_buy, "name") else None
        if buy_row_index is not None and new_trail != trail_sl:
            try:
                update_trailing_stop(trades_df, buy_row_index, new_trail)
                trades_df = load_trades_df()
                trail_sl = new_trail
            except Exception as e:
                print(f"⚠️ Failed to update trailing stop: {e}")
                trail_sl = new_trail

        # Exit logic
        exit_reason = None

        # Hard stop loss
        if sl > 0 and price <= sl:
            exit_reason = "SL"
            print(f"   🚨 Stop-loss hit! Price ${price:.2f} <= SL ${sl:.2f}")
        # Trailing stop
        elif trail_sl > 0 and price <= trail_sl:
            exit_reason = "TRAIL"
            print(f"   📉 Trailing stop hit! Price ${price:.2f} <= Trail ${trail_sl:.2f}")
        # Take profit
        elif tp > 0 and price >= tp:
            exit_reason = "TP"
            print(f"   🎯 Take-profit hit! Price ${price:.2f} >= TP ${tp:.2f}")
        # Mean-reversion: exit at RSI 62 now
        elif signal_type == "MEAN_REVERSION" and rsi >= MR_RSI_OVERBOUGHT:
            exit_reason = "MR_OVERBOUGHT"
            print(f"   🔄 Mean-reversion overbought! RSI {rsi:.1f} >= {MR_RSI_OVERBOUGHT}")

        if exit_reason:
            print(f"   🔻 SELLING {symbol} @ ${price:.2f} ({exit_reason})")

            # Place SELL order
            success = place_market_order(symbol, "sell", float(qty_held))

            if success:
                # Log the trade
                log_trade({
                    "timestamp": datetime.datetime.now(),
                    "symbol": symbol,
                    "side": "SELL",
                    "price": price,
                    "qty": float(qty_held),
                    "sl": "",
                    "trail_sl": "",
                    "tp": "",
                    "regime": last_buy.get("regime", ""),
                    "signal_type": "",
                    "exit_reason": exit_reason
                })

                send_discord(f"🔻 SELL {symbol} @ ${price:.2f} ({exit_reason})")
                LAST_STATUS[symbol] = f"SOLD({exit_reason})"
            else:
                send_discord(f"🚨 FAILED to sell {symbol}!")
        else:
            print(f"   ✅ Holding {symbol} (no exit condition met)")

    # Refresh after sells
    print(f"\n🔄 Refreshing balances after exits...")
    usd, positions = get_balances()

    # ======================================================
    # 2) FIND NEW ENTRIES (BUY LOGIC)
    # ======================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Looking for new entries")
    print(f"{'='*60}")
    print(f"USD available: ${usd:.2f}")
    print(f"Current positions: {len(positions)}/{MAX_POSITIONS}")

    if len(positions) >= MAX_POSITIONS:
        send_discord(f"ℹ️ Max positions reached ({len(positions)}/{MAX_POSITIONS})")
        print(f"⏭️ Max positions reached, skipping new entries")
    else:
        for symbol in WATCHLIST:
            print(f"\n🔍 Evaluating {symbol}")

            # CRITICAL CHECK: Skip if we already own this symbol
            if symbol in positions:
                print(f"   ⏭️ Already holding {symbol}, skipping")
                continue

            if len(positions) >= MAX_POSITIONS:
                print(f"   ⏭️ Max positions reached, stopping search")
                break

            # Determine regime
            regime, regime_info = determine_regime(symbol)
            print(f"   Regime: {regime} (ADX: {regime_info.get('adx', 0):.1f})")

            # Fetch data and evaluate entry
            try:
                df = fetch_ohlcv(symbol, TIMEFRAME, LOOKBACK_BARS)
                df = add_indicators(df)
            except Exception as e:
                send_discord(f"⚠️ {symbol} — data fetch failed: {e}")
                print(f"   ⚠️ Data fetch failed: {e}")
                continue

            signal, signal_type, reasons = generate_entry_signal(df, regime)

            if signal != "BUY":
                regime_str = f"{regime} (ADX: {regime_info.get('adx', 0):.1f})"
                latest = df.iloc[-1]
                rsi = float(latest["RSI"])
                stoch = float(latest["Stoch_K"])
                vol = float(latest["Volume_Ratio"])

                print(f"   ⏸️ No entry: {', '.join(reasons)}")
                print(f"   📊 RSI:{rsi:.1f} Stoch:{stoch:.1f} Vol:{vol:.2f}x")

                # Enhanced logging to see WHY we're not trading
                reason_text = " | ".join(reasons) + f" | {regime_str} | RSI:{rsi:.1f} Stoch:{stoch:.1f} Vol:{vol:.2f}x"
                send_no_trade_once(symbol, reason_text)
                continue

            latest = df.iloc[-1]
            price = float(latest["close"])
            atr = float(latest["ATR"])
            rsi = float(latest["RSI"])
            stoch = float(latest["Stoch_K"])

            print(f"   🟢 BUY SIGNAL! {signal_type}")
            print(f"   📊 Price: ${price:.2f}, RSI: {rsi:.1f}, Stoch: {stoch:.1f}, ATR: ${atr:.2f}")

            # Position sizing
            usd_alloc = calc_position_size(usd, atr, price)
            qty = usd_alloc / price if price > 0 else 0.0

            print(f"   💵 Position size: ${usd_alloc:.2f} ({qty:.8f} units)")

            # Calculate targets
            sl, tp = calc_targets(price, atr, regime, signal_type)

            print(f"   🎯 SL: ${sl:.2f}, TP: ${tp:.2f}")
            print(f"   📉 Risk: ${(price - sl) * qty:.2f} ({RISK_PER_TRADE*100}%)")
            print(f"   📈 Reward: ${(tp - price) * qty:.2f}")

            # Place BUY order
            print(f"   🔵 Placing BUY order...")
            success = place_market_order(symbol, "buy", qty)

            if success:
                # Log the trade
                log_trade({
                    "timestamp": datetime.datetime.now(),
                    "symbol": symbol,
                    "side": "BUY",
                    "price": price,
                    "qty": qty,
                    "sl": sl,
                    "trail_sl": sl,
                    "tp": tp,
                    "regime": regime,
                    "signal_type": signal_type,
                    "exit_reason": ""
                })

                send_discord(
                    f"🟢 BUY {symbol} @ ${price:.2f}\n"
                    f"Type: {signal_type} | Regime: {regime}\n"
                    f"SL: ${sl:.2f} | TP: ${tp:.2f} | Risk: {RISK_PER_TRADE*100:.1f}%"
                )
                LAST_STATUS[symbol] = f"BOUGHT({signal_type})"

                print(f"   ✅ BUY order successful!")

                # Refresh balances
                usd, positions = get_balances()
            else:
                send_discord(f"🚨 FAILED to buy {symbol}!")
                print(f"   🚨 BUY order FAILED!")

    # ======================================================
    # 3) PERFORMANCE METRICS
    # ======================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Performance metrics")
    print(f"{'='*60}")

    perf = compute_performance()
    if perf and perf['trades'] >= 5:
        print(f"📊 Trades: {perf['trades']}")
        print(f"   Win Rate: {perf['win_rate']*100:.1f}%")
        print(f"   Avg Win: {perf['avg_win']*100:.1f}%")
        print(f"   Avg Loss: {perf['avg_loss']*100:.1f}%")
        print(f"   Expectancy: {perf['expectancy']*100:.2f}%")

        send_discord(
            f"📊 Performance ({perf['trades']} trades)\n"
            f"Win Rate: {perf['win_rate']*100:.1f}% | "
            f"Avg Win: {perf['avg_win']*100:.1f}% | "
            f"Avg Loss: {perf['avg_loss']*100:.1f}% | "
            f"Expectancy: {perf['expectancy']*100:.2f}%"
        )

    # ======================================================
    # 4) SYNC TO S3
    # ======================================================
    print(f"\n{'='*60}")
    print(f"STEP 4: Syncing to S3")
    print(f"{'='*60}")

    upload_csv(TRADES_CSV, TRADES_CSV)
    upload_csv(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)

    send_discord(
        f"🔁 Run complete ({now.strftime('%H:%M:%S')})\n"
        f"USD: ${usd:.2f} | Positions: {len(positions)}/{MAX_POSITIONS}"
    )

    print(f"\n{'='*60}")
    print(f"✅ Cycle complete at {datetime.datetime.now()}")
    print(f"{'='*60}\n")

# ==========================================================
# SCHEDULER - RUN EVERY HOUR
# ==========================================================
schedule.every(1).hours.do(run_bot)

print("🟢 Bot starting (DEBUGGED VERSION)")
print(f"Strategy: {TIMEFRAME} timeframe | Regime detection: {REGIME_TIMEFRAME}")
print(f"Risk per trade: {RISK_PER_TRADE*100}% | Max positions: {MAX_POSITIONS}")
print(f"Mean-reversion: RSI < {MR_RSI_OVERSOLD} | Volume > {MR_VOLUME_THRESHOLD}x")
print("\nDEBUG MODE ENABLED - All operations will be logged verbosely\n")

while True:
    schedule.run_pending()
    time.sleep(60)

# ==========================================================
# RUN ONCE (for testing)
# ==========================================================
# run_bot()