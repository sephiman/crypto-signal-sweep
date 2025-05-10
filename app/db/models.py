from sqlalchemy import Column, Integer, String, Float, DateTime
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
    take_profit = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    hit = Column(String, default='PENDING', nullable=False)
    hit_timestamp = Column(DateTime, nullable=True)
