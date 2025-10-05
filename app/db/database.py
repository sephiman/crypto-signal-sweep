"""Database session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DB_URL, DB_ENABLED

if DB_ENABLED:
    engine = create_engine(DB_URL)
    SessionLocal = sessionmaker(bind=engine)
else:
    engine = None
    SessionLocal = None
