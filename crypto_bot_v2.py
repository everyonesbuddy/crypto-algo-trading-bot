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
# STRATEGY SETTINGS (REGIME-AWARE SWING)
# ==========================================================
LIVE_TRADING = True

# Micro execution timeframe (entries/exits)
TIMEFRAME_MICRO = "4h"

# Macro regime timeframe (trend context)
TIMEFRAME_MACRO = "1d"
MACRO_LOOKBACK_BARS = 240  # ~8 months of daily candles (enough for EMA200)

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD","XRP/USD","LTC/USD"]
MAX_POSITIONS = 5

# Risk & sizing
RISK_PER_TRADE = 0.01
MIN_TRADE_USD = 10
MIN_POSITION_USD = 10  # dust filter for existing holdings

# Volatility stops
ATR_MULTIPLIER_SL = 2.5
ATR_MULTIPLIER_TRAIL = 2.0

# Generous take profit: TP = entry + TP_R_MULTIPLE * (entry - SL)
TP_R_MULTIPLE = 3.0

# Bear-mode entry config (optional mean-reversion)
ENABLE_BEAR_ENTRIES = True
BEAR_RSI_OVERSOLD = 35  # example: allow bounce entries when RSI is very low

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
    except Exception:
        pass

def upload_csv(local: str, remote: str) -> None:
    """Upload a CSV to S3 (best-effort)."""
    try:
        s3.upload_file(local, AWS_S3_BUCKET, remote)
    except Exception:
        pass

def download_csv(remote: str, local: str, headers: list[str]) -> None:
    """
    Download CSV from S3 if it exists; otherwise create local CSV with headers.
    This ensures every run has a file to read/write.
    """
    try:
        s3.download_file(AWS_S3_BUCKET, remote, local)
    except Exception:
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

def add_micro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators used for entries/exits on the micro (4h) timeframe."""
    df["EMA_50"] = ta.ema(df["close"], 50)
    df["RSI"] = ta.rsi(df["close"], 14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], 14)
    df.dropna(inplace=True)
    return df

def add_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators used for regime detection on the macro (daily) timeframe."""
    df["EMA_200"] = ta.ema(df["close"], 200)
    df.dropna(inplace=True)
    return df

def determine_regime(symbol: str) -> tuple[str, dict]:
    """
    Determine regime per symbol using daily EMA200.

    Returns:
      regime: "BULL" or "BEAR"
      info: dict with macro values for debugging/discord
    """
    try:
        df = fetch_ohlcv(symbol, TIMEFRAME_MACRO, MACRO_LOOKBACK_BARS)
        df = add_macro_indicators(df)
        latest = df.iloc[-1]
        close = float(latest["close"])
        ema200 = float(latest["EMA_200"])
        regime = "BULL" if close > ema200 else "BEAR"
        return regime, {"macro_close": close, "macro_ema200": ema200}
    except Exception as e:
        # If regime can't be determined, default conservative (BEAR) and explain why.
        return "BEAR", {"error": str(e)}

# ==========================================================
# PORTFOLIO HELPERS (DUST-SAFE)
# ==========================================================
def get_balances() -> tuple[float, dict]:
    """
    Return:
      usd_balance: available USD
      positions: dict of { "BTC/USD": qty, ... } excluding dust positions
    """
    bal = exchange.fetch_balance().get("free", {})
    usd = safe_float(bal.get("USD", 0.0))

    positions = {}
    for asset, amount in bal.items():
        amount = safe_float(amount)
        if asset in ["USD", "ZUSD"] or amount <= 0:
            continue

        pair = asset.replace("X", "").replace("Z", "") + "/USD"

        # If ticker not available for a dust asset, skip.
        try:
            price = safe_float(exchange.fetch_ticker(pair).get("last", 0.0))
        except Exception:
            continue

        value_usd = amount * price

        # Ignore dust to avoid blocking MAX_POSITIONS.
        if value_usd >= MIN_POSITION_USD:
            positions[pair] = amount

    return usd, positions

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
# LOGGING
# ==========================================================
def ensure_trade_log_headers() -> None:
    """
    Ensure the trade log schema supports our new strategy.
    We store: sl, trail_sl, tp, and exit_reason for sells.
    """
    if os.path.isfile(TRADES_CSV):
        return
    with open(TRADES_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "symbol", "side", "price", "qty",
            "sl", "trail_sl", "tp", "regime", "exit_reason"
        ])

def log_trade(row: dict) -> None:
    """Append a trade row to CSV."""
    exists = os.path.isfile(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(row.keys())
        w.writerow(row.values())

def log_daily(date: str, start: float, current: float) -> None:
    """Append daily balance snapshot."""
    exists = os.path.isfile(DAILY_BALANCE_CSV)
    with open(DAILY_BALANCE_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date", "starting_balance", "current_balance"])
        w.writerow([date, f"{start:.2f}", f"{current:.2f}"])

# ==========================================================
# STRATEGY LOGIC
# ==========================================================
def generate_entry_signal(micro_df: pd.DataFrame, regime: str) -> tuple[str, list[str]]:
    """
    Regime-aware entry logic.

    Bull regime: trend-following
      - close > EMA50 AND RSI > 50

    Bear regime: optional conservative mean-reversion
      - RSI < BEAR_RSI_OVERSOLD AND close > EMA50 (reclaim)  [safer]
      (You can loosen/tighten later.)
    """
    latest = micro_df.iloc[-1]
    close = float(latest["close"])
    ema50 = float(latest["EMA_50"])
    rsi = float(latest["RSI"])

    reasons = []

    if regime == "BULL":
        if close <= ema50:
            reasons.append("close <= EMA50 (bull entry needs micro uptrend)")
        if rsi <= 50:
            reasons.append(f"RSI {rsi:.1f} <= 50 (bull entry needs momentum)")
        if reasons:
            return "HOLD", reasons
        return "BUY", []

    # regime == BEAR
    if not ENABLE_BEAR_ENTRIES:
        return "HOLD", ["bear regime (bear entries disabled)"]

    # Conservative bear entry: oversold + reclaim micro trend
    if rsi >= BEAR_RSI_OVERSOLD:
        reasons.append(f"RSI {rsi:.1f} >= {BEAR_RSI_OVERSOLD} (need oversold)")
    if close <= ema50:
        reasons.append("close <= EMA50 (need reclaim for bounce)")

    if reasons:
        return "HOLD", reasons
    return "BUY", ["bear bounce entry"]

def calc_position_usd(usd_balance: float, atr: float, price: float) -> float:
    """
    Risk-based sizing:
      risk_usd = account_usd * RISK_PER_TRADE
      stop_distance = ATR * ATR_MULTIPLIER_SL
      qty = risk_usd / stop_distance
      usd_alloc = qty * price
    """
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

def place_market_order(symbol: str, side: str, qty: float) -> None:
    """Place a market order (or simulate if LIVE_TRADING is False)."""
    if qty <= 0:
        return
    if not LIVE_TRADING:
        print(f"[TEST] {side.upper()} {symbol} qty={qty:.8f}")
        return
    if side.lower() == "buy":
        exchange.create_market_buy_order(symbol, qty)
    else:
        exchange.create_market_sell_order(symbol, qty)

# ==========================================================
# TRADE MANAGEMENT (SL / TRAIL / TP)
# ==========================================================
def load_trades_df() -> pd.DataFrame:
    """Load trades CSV safely with expected columns."""
    if not os.path.isfile(TRADES_CSV):
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "side", "price", "qty",
            "sl", "trail_sl", "tp", "regime", "exit_reason"
        ])
    df = pd.read_csv(TRADES_CSV)
    return df

def get_last_open_trade(trades_df: pd.DataFrame, symbol: str) -> pd.Series | None:
    """
    Find the most recent BUY that has not been "closed" by a SELL.
    We approximate this by counting buys vs sells per symbol.
    """
    g = trades_df[trades_df["symbol"] == symbol].copy()
    if g.empty:
        return None

    g = g[g["side"].isin(["BUY", "SELL"])].copy()
    if g.empty:
        return None

    # If #buys == #sells, no open trade in our log.
    buys = g[g["side"] == "BUY"]
    sells = g[g["side"] == "SELL"]
    if len(buys) <= len(sells):
        return None

    # Open trade = last BUY row
    return buys.iloc[-1]

def update_trailing_stop(trades_df: pd.DataFrame, buy_row_index: int, new_trail: float) -> None:
    """Persist updated trailing stop into trades CSV by editing the specific BUY row."""
    trades_df.loc[buy_row_index, "trail_sl"] = new_trail
    trades_df.to_csv(TRADES_CSV, index=False)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================
def compute_performance() -> dict | None:
    """
    Compute win-rate + expectancy from completed Buy->Sell pairs.
    Expectancy here = mean return per trade (simple, percent).
    """
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

        # Pair in order (assumes you do not pyramid multiple entries per symbol).
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
    return {"trades": int(len(s)), "win_rate": win_rate, "expectancy": expectancy}

# ==========================================================
# DISCORD STATUS DEDUPING
# ==========================================================
LAST_STATUS = {}

def send_no_trade_once(symbol: str, reason_text: str) -> None:
    """Avoid spamming Discord with the same hold reason repeatedly."""
    if LAST_STATUS.get(symbol) != reason_text:
        send_discord(f"⏸️ {symbol} — No trade\n• {reason_text}")
        LAST_STATUS[symbol] = reason_text

# ==========================================================
# MAIN BOT
# ==========================================================
def run_bot():
    now = datetime.datetime.now()
    print("\n⏱️ Running bot", now)

    # Ensure local files exist and have the right headers
    download_csv(
        TRADES_CSV, TRADES_CSV,
        ["timestamp", "symbol", "side", "price", "qty", "sl", "trail_sl", "tp", "regime", "exit_reason"]
    )
    download_csv(
        DAILY_BALANCE_CSV, DAILY_BALANCE_CSV,
        ["date", "starting_balance", "current_balance"]
    )
    ensure_trade_log_headers()

    # Load state
    trades_df = load_trades_df()
    usd, positions = get_balances()

    # Balance snapshot (simple running log; we keep it minimal here)
    total = usd + portfolio_value(positions)
    log_daily(datetime.date.today().isoformat(), total, total)

    # ======================================================
    # 1) MANAGE EXISTING POSITIONS FIRST (SELL LOGIC)
    # ======================================================
    for symbol, qty_held in positions.items():
        # Only manage symbols we are watching (optional).
        # If you want to manage ANY held symbol, remove this if-check.
        if symbol not in WATCHLIST:
            continue

        # Pull micro data for management
        try:
            micro = fetch_ohlcv(symbol, TIMEFRAME_MICRO, 300)
            micro = add_micro_indicators(micro)
            latest = micro.iloc[-1]
            price = float(latest["close"])
            atr = float(latest["ATR"])
        except Exception as e:
            send_discord(f"⚠️ {symbol} — data fetch failed for management: {e}")
            continue

        last_buy = get_last_open_trade(trades_df, symbol)
        if last_buy is None:
            # We have a position on exchange but no matching BUY in our log.
            # Don't trade blindly; just notify.
            send_discord(f"⚠️ {symbol} — position exists but no open BUY in log; skipping management.")
            continue

        sl = safe_float(last_buy.get("sl", 0.0))
        trail_sl = safe_float(last_buy.get("trail_sl", sl))
        tp = safe_float(last_buy.get("tp", 0.0))

        # --- Update trailing stop (always)
        new_trail = max(trail_sl, price - atr * ATR_MULTIPLIER_TRAIL)

        # Persist trailing stop update into the BUY row
        buy_row_index = int(last_buy.name) if hasattr(last_buy, "name") else None
        if buy_row_index is not None and new_trail != trail_sl:
            try:
                update_trailing_stop(trades_df, buy_row_index, new_trail)
                # reload to keep indices aligned
                trades_df = load_trades_df()
                trail_sl = new_trail
            except Exception:
                # If update fails, we still continue (best-effort)
                trail_sl = new_trail

        # --- Exit priority: SL -> TRAIL -> TP
        exit_reason = None
        if sl > 0 and price <= sl:
            exit_reason = "SL"
        elif trail_sl > 0 and price <= trail_sl:
            exit_reason = "TRAIL"
        elif tp > 0 and price >= tp:
            exit_reason = "TP"

        if exit_reason:
            # Place SELL
            place_market_order(symbol, "sell", float(qty_held))

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
                "exit_reason": exit_reason
            })

            send_discord(f"🔻 SELL {symbol} @ ${price:.2f} ({exit_reason})")
            LAST_STATUS[symbol] = f"SOLD({exit_reason})"

    # Refresh balances after possible sells
    usd, positions = get_balances()

    # ======================================================
    # 2) FIND NEW ENTRIES (BUY LOGIC)
    # ======================================================
    # Enforce MAX_POSITIONS (dust-safe): count only meaningful positions
    if len(positions) >= MAX_POSITIONS:
        send_discord(f"ℹ️ Max positions reached ({len(positions)}/{MAX_POSITIONS}). No new entries this run.")
    else:
        for symbol in WATCHLIST:
            if symbol in positions:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            # Determine regime using macro timeframe
            regime, macro_info = determine_regime(symbol)

            # Fetch micro data (4h) and evaluate entry
            try:
                micro = fetch_ohlcv(symbol, TIMEFRAME_MICRO, 300)
                micro = add_micro_indicators(micro)
            except Exception as e:
                send_discord(f"⚠️ {symbol} — micro data fetch failed: {e}")
                continue

            signal, reasons = generate_entry_signal(micro, regime)

            if signal != "BUY":
                # Add macro info to help you understand "why no trades"
                macro_note = ""
                if "macro_close" in macro_info and "macro_ema200" in macro_info:
                    macro_note = f" | macro close {macro_info['macro_close']:.2f} vs EMA200 {macro_info['macro_ema200']:.2f} ({regime})"
                elif "error" in macro_info:
                    macro_note = f" | macro regime error: {macro_info['error']}"

                reason_text = " | ".join(reasons) + macro_note
                send_no_trade_once(symbol, reason_text)
                continue

            latest = micro.iloc[-1]
            price = float(latest["close"])
            atr = float(latest["ATR"])

            # Position sizing based on risk
            usd_alloc = calc_position_usd(usd, atr, price)
            qty = usd_alloc / price if price > 0 else 0.0

            # Compute SL / TP based on ATR risk model
            sl = price - atr * ATR_MULTIPLIER_SL
            R = price - sl  # initial risk per unit
            tp = price + TP_R_MULTIPLE * R

            # Place BUY
            place_market_order(symbol, "buy", qty)

            # Log BUY with full context
            log_trade({
                "timestamp": datetime.datetime.now(),
                "symbol": symbol,
                "side": "BUY",
                "price": price,
                "qty": qty,
                "sl": sl,
                "trail_sl": sl,   # trail starts at SL (can start higher later)
                "tp": tp,
                "regime": regime,
                "exit_reason": ""
            })

            send_discord(
                f"🟢 BUY {symbol} @ ${price:.2f} ({regime})\n"
                f"SL {sl:.2f} | TP {tp:.2f} | ATR {atr:.2f}"
            )
            LAST_STATUS[symbol] = "BOUGHT"

            # Refresh balances after buy
            usd, positions = get_balances()

    # ======================================================
    # 3) PERFORMANCE METRICS (from completed trades)
    # ======================================================
    perf = compute_performance()
    if perf:
        send_discord(
            f"📊 Performance\n"
            f"Trades: {perf['trades']} | "
            f"WinRate: {perf['win_rate']*100:.2f}% | "
            f"Expectancy: {perf['expectancy']*100:.2f}%"
        )

    # ======================================================
    # 4) SYNC TO S3 + RUN SUMMARY
    # ======================================================
    upload_csv(TRADES_CSV, TRADES_CSV)
    upload_csv(DAILY_BALANCE_CSV, DAILY_BALANCE_CSV)

    send_discord(
        f"🔁 Run complete ({now.strftime('%Y-%m-%d %H:%M:%S')})\n"
        f"USD: ${usd:.2f} | Positions: {len(positions)}/{MAX_POSITIONS}"
    )

    print("✅ Cycle complete")

# ==========================================================
# SCHEDULER
# ==========================================================
schedule.every(4).hours.do(run_bot)
print("🟢 Bot running (Regime-aware 4H swing strategy)")
while True:
    schedule.run_pending()
    time.sleep(30)

# ==========================================================
# RUN ONCE (optional)
# ==========================================================
# run_bot()
