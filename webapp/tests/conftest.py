"""
Pytest configuration and fixtures for Stock Toolbox Web tests.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_db():
    """
    Create a test database for the session.
    Uses SQLite in-memory database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from webapp.database import Base
    from webapp import models
    
    # Create in-memory database
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestingSessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_db):
    """
    Create a new database session for a test.
    Rolls back after each test.
    """
    session = test_db()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
