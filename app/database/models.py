"""
File: TechMate/app/database/models.py
Description: SQLAlchemy schema models for persistent memory in PostgreSQL.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base

from app.config import POSTGRES_URL_SQLALCHEMY

Base = declarative_base()

# Initialize PostgreSQL Engine
#
# Uses POSTGRES_URL_SQLALCHEMY (the "+psycopg" variant), not
# the plain POSTGRES_URL — see app/config.py for why: it makes
# SQLAlchemy use the psycopg v3 driver you already have
# installed (app/memory/memory.py depends on it directly)
# instead of defaulting to psycopg2, which raised
# "ModuleNotFoundError: No module named 'psycopg2'" when this
# used the plain URL.
#
# pool_pre_ping=True and pool_recycle are important specifically
# because you're on Neon (or any serverless/managed Postgres
# that auto-suspends idle compute to save resources). When Neon
# suspends and you come back after a while, SQLAlchemy's pooled
# connections are silently dead — without pre_ping, the next
# query fails with "SSL connection has been closed unexpectedly"
# instead of transparently reconnecting. pre_ping tests each
# connection with a lightweight check before using it and
# reconnects if needed; pool_recycle proactively refreshes
# connections older than 280 seconds so they rarely go stale
# long enough to hit this in the first place.
if POSTGRES_URL_SQLALCHEMY:
    engine = create_engine(
        POSTGRES_URL_SQLALCHEMY,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=280,
    )
else:
    engine = None


class UserSkill(Base):
    """Tracks recurring user weak spots, conceptual errors, and bad coding habits."""
    __tablename__ = "user_skills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    topic = Column(String(200), nullable=False)
    error_count = Column(Integer, default=1)
    last_error_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SessionLog(Base):
    """Stores full conversation history summaries for analytics and debugging."""
    __tablename__ = "session_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(50), nullable=False)
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """
    Maps a LangGraph thread_id to a display title for the
    Streamlit sidebar's chat history list.

    Previously this lived in a separate local SQLite file
    (CHECKPOINTS_DB_PATH) — a leftover from before the LangGraph
    checkpointer itself moved to PostgreSQL. That left chat
    titles disconnected from where the actual conversation
    content lives. Now stored here, in the same database as
    everything else, so it persists exactly as durably.
    """
    __tablename__ = "chat_sessions"

    thread_id = Column(String(100), primary_key=True)
    title = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Creates tables if they do not already exist in the database."""
    if engine:
        Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("PostgreSQL tables initialized successfully.")