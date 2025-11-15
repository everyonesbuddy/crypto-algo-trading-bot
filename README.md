# 🚀 Automated Crypto Trading Bot (Kraken + AWS S3 + Discord)

This project is a fully automated algorithmic crypto trading bot that:

### ✔ Trades live on **Kraken**

### ✔ Stores logs on **AWS S3**

### ✔ Sends alerts to **Discord**

### ✔ Runs automatically on **Heroku Scheduler**

### ✔ Uses a hybrid strategy of:

- Technical indicators (RSI, MACD, EMA50, Bollinger Bands, StochRSI)
- Stop-loss (-2%) + Take-profit (+4%)
- Daily global target (+0.5%)
- Dynamic position sizing
- Portfolio risk limits

---

## 🔥 **Daily Goal**

Our objective is:

### → **+0.5% net portfolio growth per day**

If the portfolio hits **+0.5% daily performance**, the bot:

### → Takes profit

### → Closes all open positions

### → Stops trading until next day

This protects gains and prevents overtrading.

---

## 📊 **Starting Capital**

We are beginning with:

### 💰 **$234 USD total account balance**

The bot automatically splits between:

- **Tradable USD balance**
- **Open crypto positions**

Portfolio values sync each run.

---

## 🛠 **What the Bot Does Each Run**

Every 3 hours:

### 1. Syncs logs with AWS S3

### 2. Fetches watchlist crypto OHLCV

### 3. Calculates indicators

### 4. Applies trading logic

- Detect buy signals
- Detect sell signals
- Enforce stop-loss & take-profit
- Enforce global daily target

### 5. Executes buys/sells on Kraken

### 6. Logs results

### 7. Updates Discord

### 8. Uploads logs back to S3

---

## 🧠 **Trading Strategy Summary**

### BUY when (score ≥ 3):

- RSI < 45
- MACD histogram > 0
- Price above EMA50
- StochRSI bullish
- Near lower Bollinger Band

### SELL when (score ≥ 2):

- RSI > 60
- MACD histogram < 0
- Price below EMA50
- StochRSI bearish
- Near upper Bollinger Band

---

## 🛑 **Risk Controls**

- **Stop loss:** -2% from buy price
- **Take profit:** +4% from buy price
- **Daily kill-switch:** +0.5%
- **Max positions:** 5
- **Per-trade allocation:** $10 (dynamic if low USD)

---

## ☁ AWS S3 Files

The bot stores:

### `crypto_trades_log.csv`

Every trade executed.

### `daily_balance_log.csv`

Tracks start-of-day and current balance.

These files survive Heroku dyno resets.

---

## 🔧 Environment Variables

In Heroku you will set:

KRAKEN_API_KEY=
KRAKEN_API_SECRET=
DISCORD_WEBHOOK_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-2
AWS_S3_BUCKET=
