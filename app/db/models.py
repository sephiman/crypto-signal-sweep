from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    side = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    hit = Column(String, default='PENDING', nullable=False)
    hit_timestamp = Column(DateTime, nullable=True)
    sl_moved_to_be = Column(Boolean, default=False, nullable=False)
    momentum_ok = Column(Boolean, nullable=False)
    trend_confirmed = Column(Boolean, nullable=False)
    higher_tf_confirmed = Column(Boolean, nullable=False)
    confirmed = Column(Boolean, nullable=False, default=False)
    score = Column(Integer, nullable=False, default=0)
    required_score = Column(Integer, nullable=False, default=0)
    rsi_ok = Column(Boolean, nullable=False, default=False)
    ema_ok = Column(Boolean, nullable=False, default=False)
    macd_ok = Column(Boolean, nullable=False, default=False)
    macd_momentum_ok = Column(Boolean, nullable=False, default=False)
    stoch_ok = Column(Boolean, nullable=False, default=False)
    rsi = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_diff = Column(Float, nullable=True)
    ema_fast = Column(Float, nullable=True)
    ema_slow = Column(Float, nullable=True)
    ema_diff = Column(Float, nullable=True)
    stoch_k = Column(Float, nullable=True)
    stoch_d = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    atr_pct = Column(Float, nullable=True)
    regime = Column(String, nullable=True)
    htf_used = Column(Boolean, nullable=False, default=False)
    volume_ratio = Column(Float, nullable=True, default=1.0)
    confidence = Column(String, nullable=True, default='MEDIUM')


class MarketAnalysis(Base):
    __tablename__ = 'market_analyses'
    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    
    # Technical indicators
    rsi = Column(Float, nullable=False)
    adx = Column(Float, nullable=False)
    macd = Column(Float, nullable=False)
    macd_signal = Column(Float, nullable=False)
    macd_diff = Column(Float, nullable=False)
    ema_fast = Column(Float, nullable=False)
    ema_slow = Column(Float, nullable=False)
    ema_diff = Column(Float, nullable=False)
    stoch_k = Column(Float, nullable=False)
    stoch_d = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    atr_pct = Column(Float, nullable=False)
    volume_ratio = Column(Float, nullable=False)
    
    # Gate conditions (boolean checks)
    rsi_ok_long = Column(Boolean, nullable=False)
    rsi_ok_short = Column(Boolean, nullable=False)
    macd_ok_long = Column(Boolean, nullable=False)
    macd_ok_short = Column(Boolean, nullable=False)
    momentum_ok_long = Column(Boolean, nullable=False)
    momentum_ok_short = Column(Boolean, nullable=False)
    ema_ok_long = Column(Boolean, nullable=False)
    ema_ok_short = Column(Boolean, nullable=False)
    trend_ok_long = Column(Boolean, nullable=False)
    trend_ok_short = Column(Boolean, nullable=False)
    stoch_ok_long = Column(Boolean, nullable=False)
    stoch_ok_short = Column(Boolean, nullable=False)
    htf_confirm_long = Column(Boolean, nullable=False)
    htf_confirm_short = Column(Boolean, nullable=False)
    
    # Filter conditions
    volume_pass = Column(Boolean, nullable=False)
    atr_pass = Column(Boolean, nullable=False)
    time_pass = Column(Boolean, nullable=False)
    
    # Scores and regime
    long_score = Column(Integer, nullable=False)
    short_score = Column(Integer, nullable=False)
    min_score_required = Column(Integer, nullable=False)
    regime = Column(String, nullable=False)  # 'TREND' or 'RANGE'
    is_trending = Column(Boolean, nullable=False)
    
    # Final result
    signal_generated = Column(Boolean, nullable=False, default=False)
    signal_side = Column(String, nullable=True)  # 'LONG', 'SHORT', or None
    skip_reason = Column(String, nullable=True)  # 'LOW_VOL', 'LOW_ATR', 'NO_SIGNAL', etc.
