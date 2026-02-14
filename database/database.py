from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.app.config import get_settings
import os

settings = get_settings()

# Create required directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)

# Resolve database path to absolute (anchored to project root)
# This prevents pathrad.db from appearing in wrong directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(PROJECT_ROOT, "database")
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, "pathrad.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Database setup
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if "sqlite" in DATABASE_URL
    else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
