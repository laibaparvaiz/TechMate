"""
File: TechMate/app/config.py
Description: Centralized configuration, environment variables, and file paths with corrected FAISS directory routing.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Base Directory Setup (Points to TechMate/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# 2. API Keys & Models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY is not set in the .env file.")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY is not set in the .env file.")

# Set the primary model (using flash for speed and token limits)
PRIMARY_MODEL = "gemini-3.6-flash"

# 3. Data & Memory Paths
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Legacy SQLite checkpoint path (kept for reference only; long-term
# memory runs on PostgreSQL, see section 4 below).
CHECKPOINTS_DB_PATH = str(DATA_DIR / "checkpoints.sqlite")

# FAISS Vector Database Directory (Study Agent RAG) - Updated to folder path
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")

# ============================================================
# 4. PostgreSQL Configuration (Long-term memory)
#
# TWO ways to configure this - pick ONE:
#
# Option 1 (Neon / any managed provider) - RECOMMENDED:
#   Paste the FULL connection string your provider gives you
#   into .env as a single DATABASE_URL. For Neon specifically,
#   this comes from: Project Dashboard -> "Connect" button ->
#   copy the string shown (it already includes the host,
#   password, database name, and required params like
#   sslmode=require&channel_binding=require - don't strip
#   those, Neon requires them).
#
#     .env:
#     DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require&channel_binding=require
#
# Option 2 (local PostgreSQL):
#   Set the individual pieces instead - DATABASE_URL stays
#   unset, and the fallback below builds the connection string
#   from these.
#
#     .env:
#     POSTGRES_USER=postgres
#     POSTGRES_PASSWORD=postgres
#     POSTGRES_HOST=localhost
#     POSTGRES_PORT=5432
#     POSTGRES_DB=techmate_db
# ============================================================

_env_database_url = os.getenv("DATABASE_URL", "").strip()

if _env_database_url:

    # Option 1: use the provider's full string exactly as given.
    POSTGRES_URL = _env_database_url

else:

    # Option 2: build from individual pieces (local PostgreSQL).
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "techmate_db")

    # Empty by default (a local database usually doesn't need SSL).
    # Set to "require" in .env if your local setup needs it too.
    POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "")

    _ssl_query = (
        f"?sslmode={POSTGRES_SSLMODE}" if POSTGRES_SSLMODE else ""
    )

    POSTGRES_URL = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        f"{_ssl_query}"
    )

# ------------------------------------------------------------
# SQLAlchemy-specific variant - for create_engine() usage in
# app/database/models.py. The "+psycopg" suffix tells
# SQLAlchemy to use the psycopg v3 driver instead of defaulting
# to psycopg2. Derived from POSTGRES_URL above by inserting the
# suffix into whichever scheme was actually used ("postgresql://"
# or the "postgres://" shorthand some providers use), so
# whatever parameters Option 1 or Option 2 produced (sslmode,
# channel_binding, etc.) are preserved unchanged.
#
# NOTE: do NOT use this "+psycopg" variant with a direct
# psycopg.connect() call (see app/memory/memory.py) - psycopg's
# own connect() does not understand SQLAlchemy's dialect-suffix
# syntax. memory.py correctly uses the plain POSTGRES_URL above.
# ------------------------------------------------------------

if POSTGRES_URL.startswith("postgresql://"):

    POSTGRES_URL_SQLALCHEMY = POSTGRES_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

elif POSTGRES_URL.startswith("postgres://"):

    POSTGRES_URL_SQLALCHEMY = POSTGRES_URL.replace(
        "postgres://", "postgresql+psycopg://", 1
    )

else:

    POSTGRES_URL_SQLALCHEMY = POSTGRES_URL