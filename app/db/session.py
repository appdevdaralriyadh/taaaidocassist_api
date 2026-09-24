"""
SQLAlchemy engine/session setup for SQL Server, via the pyodbc driver.

This module only opens connections — it does not create or manage schema.
The DarAI_* tables (see spec §4) are created and owned outside the app.
"""

from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

_driver = quote_plus(settings.DB_DRIVER)
_password = quote_plus(settings.DB_PASSWORD)

# TrustServerCertificate=yes: ODBC Driver 18 for SQL Server (unlike 17)
# defaults to Encrypt=yes -- without this, a SQL Server that only has a
# self-signed/non-CA-trusted certificate (the normal case for an internal
# on-prem instance) fails the connection with a certificate-trust error
# rather than a driver problem. This tells the client to skip that
# validation, which is appropriate for a connection that stays inside a
# trusted internal network -- it should NOT be relied on for a connection
# that crosses the open internet. Harmless with ODBC Driver 17 too, since
# 17 doesn't force encryption on and simply ignores this if unused.
CONNECTION_STRING = (
    f"mssql+pyodbc://{settings.DB_USER}:{_password}"
    f"@{settings.DB_SERVER}/{settings.DB_NAME}?driver={_driver}"
    "&TrustServerCertificate=yes"
)

engine = create_engine(CONNECTION_STRING, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
