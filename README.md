# CryptoSignalSweep
🚀 A Dockerized signal-generator for Binance Spot markets.

Scans multiple pairs and timeframes using RSI, MACD, EMA, ADX and optional filters to emit **LONG/SHORT entry signals** with SL/TP levels. Alerts are pushed to Telegram and stored in PostgreSQL for full backtesting & performance tracking.

---

## ✅ Features
- 📈 Multi-pair (`PAIRS`) & multi-timeframe (`TIMEFRAMES`)
- 🧠 Smart entry logic using:
  - RSI (regime-aware via ADX)
  - MACD (momentum + signal cross + histogram diff)
  - EMA cross with optional minimum separation
  - Optional: SMA trend filter
  - Optional: Higher-Timeframe confirmation
- 🧮 SL/TP sizing via ATR with configurable multipliers
- 🧠 Signal scoring logic (requires **min N/5 gates**, dynamic via ADX)
- 📬 Real-time Telegram alerts (LONG / SHORT entries)
- 🧾 PostgreSQL tracking of all signals & outcomes
- 🔧 Adaptive ATR filter: skip low-volatility pairs

## Setup
1. Prepare docker-compose and fill values.
2. Ensure Docker network `all_dockers` exists: `docker network create all_dockers`
3. `docker-compose up --build`

## Configuration
All behavior is controlled via app/config.py (or overridden by environment variables):

* Signal Logic: Configure RSI, MACD, EMA periods and thresholds.
* Scoring System: Adaptive gate score based on ADX (e.g., 4/5 in trending, 3/5 in ranging).
* Volatility Filter: Minimum ATR relative to price ensures only meaningful signals (e.g., no $0.001 stop on a $100 asset).
* SL/TP Calculation: Automatically sized using ATR × multipliers.
* Higher Timeframe Confirmation: Optional MACD+RSI check on higher TF.
* Trend Filter: Optional SMA filter to only trade in strong trends.
* Each gate and score is logged and sent to Telegram per signal, along with confirmation status and outcome tracking.

##  Backtesting & Stats

All signals (and whether they hit SL/TP) are stored in PostgreSQL. You can spin up pgAdmin on port 5050 (persisted data!) and run ad-hoc queries to see your win/loss ratios, average RR, etc.

    `SELECT
      COUNT(*) FILTER (WHERE hit = 'SUCCESS') AS success_count,
      COUNT(*) FILTER (WHERE hit = 'FAILURE') AS failure_count,
      ROUND(
        100.0 * COUNT(*) FILTER (WHERE hit = 'SUCCESS') /
        NULLIF(COUNT(*) FILTER (WHERE hit IN ('SUCCESS', 'FAILURE')), 0),
        2
      ) AS success_percentage
    FROM public.signals;`

## License
GPL-3.0
