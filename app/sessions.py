"""
File: TechMate/app/sessions.py

Description:
Chat session titles and thread mapping.

Stored in PostgreSQL (the same database as the LangGraph
checkpointer and skill tracking) — see app/database/models.py's
ChatSession model for why this used to live in a separate,
now-orphaned local SQLite file, and why that was a real bug:
every chat's title was being saved somewhere DIFFERENT from
its actual conversation content since the checkpointer itself
moved to Postgres.
"""

from sqlalchemy.orm import Session

from app.database.models import Base, engine, ChatSession


def init_chat_db():
    """
    Ensures the chat_sessions table exists.

    Safe to call even if app.database.models.init_db() already
    ran elsewhere in this process — create_all() only creates
    tables that don't exist yet, it never touches existing ones.
    """

    if engine:

        Base.metadata.create_all(
            bind=engine
        )


def save_chat_session(
    thread_id: str,
    title: str,
) -> bool:
    """
    Saves or updates a chat session title.

    Returns True on success, False if PostgreSQL isn't
    configured/reachable — mirrors app/memory/skills.py's
    graceful-degradation pattern. A failure here should never
    crash the chat flow just because a sidebar title couldn't
    be saved.
    """

    if not engine:

        return False

    try:

        with Session(engine) as session:

            existing = session.get(
                ChatSession,
                thread_id,
            )

            if existing:

                existing.title = title

            else:

                session.add(
                    ChatSession(
                        thread_id=thread_id,
                        title=title,
                    )
                )

            session.commit()

            return True

    except Exception as e:

        print(
            "Error saving chat session:",
            str(e)
        )

        return False


def get_all_chat_sessions():
    """
    Retrieves all chat sessions, most recent first.

    Returns an empty list if PostgreSQL isn't configured/
    reachable, rather than crashing the sidebar.
    """

    if not engine:

        return []

    try:

        with Session(engine) as session:

            sessions = (
                session.query(ChatSession)
                .order_by(
                    ChatSession.created_at.desc()
                )
                .all()
            )

            return [
                {
                    "thread_id": s.thread_id,
                    "title": s.title,
                }
                for s in sessions
            ]

    except Exception as e:

        print(
            "Error retrieving chat sessions:",
            str(e)
        )

        return []