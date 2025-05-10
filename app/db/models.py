from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, DateTime, Integer

Base = declarative_base()

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    pair = Column(String)
    timeframe = Column(String)
    side = Column(String)
    price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    timestamp = Column(DateTime)
    hit = Column(String)
    hit_timestamp = Column(DateTime, nullable=True)
