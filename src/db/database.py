"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from src.config import get_settings


def get_engine():
    """Get database engine."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_session_maker():
    """Get session maker."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.

    Yields:
        Database session
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
