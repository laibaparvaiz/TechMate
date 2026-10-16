"""
File: TechMate/app/memory/memory.py
Description: PostgreSQL-backed LangGraph checkpointer factory
             (long-term persistent conversation state, e.g. on Neon).
"""

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


def get_checkpointer(db_url: str):
    """
    Creates and returns a (checkpointer, connection) pair backed
    by PostgreSQL. Sets up required checkpoint tables automatically.

    Returns:
        (checkpointer, conn) - a PostgresSaver instance and the raw
        psycopg connection it wraps.

    IMPORTANT - connection lifecycle:
        The returned connection is left OPEN and must be closed by
        the CALLER when the app shuts down, or left open for the
        lifetime of a cached Streamlit resource, which closes when
        the process exits. It is returned alongside the
        checkpointer specifically so the caller can manage this -
        do not assume PostgresSaver manages it for you.

    Raises:
        ConnectionError with an actionable message if PostgreSQL
        isn't reachable.
    """

    try:

        conn = psycopg.connect(
            db_url,
            autocommit=True,
        )

    except Exception as e:

        raise ConnectionError(
            "Could not connect to PostgreSQL. Check that DATABASE_URL "
            "in your .env is correct and the database is reachable. "
            "(A hosted database such as Neon may be waking up from "
            "idle - wait a few seconds and try again.) "
            f"Original error: {e}"
        ) from e

    checkpointer = PostgresSaver(
        conn
    )

    # Creates checkpoint tables if they don't already exist.
    checkpointer.setup()

    return checkpointer, conn