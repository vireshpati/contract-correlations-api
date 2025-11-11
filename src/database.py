"""Database connection and session management."""
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .config import get_settings


# Create engine lazily
def get_engine():
    """Get SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope for database operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
