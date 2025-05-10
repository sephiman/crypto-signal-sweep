# CryptoSignalSweep

A Dockerized bot scanning multiple Binance Spot pairs every multiple timeframes,
creating LONG and SHORT signals (RSI + MACD + EMA trend), and sending Telegram alerts
with SL/TP. Signals logged in PostgreSQL for backtesting.

## Features
- Multi-pair via `PAIRS`
- Multi-timeframe via `TIMEFRAMES`
- Optional higher timeframe confirmation
- LONG and SHORT signals
- Signal tracking in PostgreSQL

## Setup
1. Prepare docker-compose and fill values.
2. Ensure Docker network `all_dockers` exists: `docker network create all_dockers`
3. `docker-compose up --build`
4. Initialize DB: `docker exec -it css_db python db/init_db.py`

## Configuration
See `config.py` for env vars and defaults.

## License
GPL-3.0
