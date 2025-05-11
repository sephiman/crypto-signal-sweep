# CryptoSignalSweep

A Dockerized signal-generator for Binance Spot that:
* Scans dozens of trading pairs (e.g. BTC/USDT, ETH/USDT, SUI/USDT)
* Runs on multiple custom intervals (15m, 1h, etc.)
* Combines RSI, MACD, EMA (and optional trend/Higher-TF filters) to emit LONG & SHORT entry signals
* Automatically calculates SL/TP via ATR-based sizing (configurable multipliers)
* Sends real-time alerts to your Telegram chat
* Logs every signal (and its eventual outcome) to PostgreSQL for backtesting & performance tracking

## Features
* Multi-pair via `PAIRS`
* Multi-timeframe via `TIMEFRAMES`
* Optional higher timeframe confirmation
* LONG and SHORT telegram signals
* Signal tracking in PostgreSQL

## Setup
1. Prepare docker-compose and fill values.
2. Ensure Docker network `all_dockers` exists: `docker network create all_dockers`
3. `docker-compose up --build`

## Configuration
See app/config.py—you can tweak:
* Signal logic: RSI_PERIOD, RSI_OVERSOLD, MACD_FAST/SLOW/SIGNAL, EMA_FAST/SLOW
* Stop/Target sizing: ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, or legacy SL_MULTIPLIER, TP_MULTIPLIER
* Pairs / Timeframes: PAIRS, TIMEFRAMES, optional USE_HIGHER_TF_CONFIRM, USE_TREND_FILTER

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
