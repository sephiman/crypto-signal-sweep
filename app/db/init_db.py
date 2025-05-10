from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DB_URL, DB_ENABLED
from app.db.models import Base

if DB_ENABLED:
    engine = create_engine(DB_URL)
    SessionLocal = sessionmaker(bind=engine)


    def init_db():
        Base.metadata.create_all(bind=engine)
else:
    engine = None
    SessionLocal = None


    def init_db():
        return
