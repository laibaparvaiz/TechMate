"""
File: TechMate/app/memory/skills.py
Description: Manages long-term skill tracking and user weakness logs in PostgreSQL.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.database.models import UserSkill, SessionLog, engine

logger = logging.getLogger("techmate.skills")


def log_user_weakness(user_id: str, topic: str, error_details: str) -> bool:
    """
    Saves or increments a user weakness/bad practice in PostgreSQL.
    Increments error_count if the topic exists; creates a new record if not.
    """
    if not engine:
        logger.warning("PostgreSQL engine not initialized. Skipping skill persistence.")
        return False

    try:
        with Session(engine) as session:
            skill = (
                session.query(UserSkill)
                .filter_by(user_id=user_id, topic=topic)
                .first()
            )

            if skill:
                skill.error_count += 1
                skill.last_error_notes = error_details
            else:
                skill = UserSkill(
                    user_id=user_id,
                    topic=topic,
                    error_count=1,
                    last_error_notes=error_details
                )
                session.add(skill)

            session.commit()
            return True
    except Exception as e:
        logger.error(f"Error persisting user skill weakness: {str(e)}")
        return False


def get_user_weaknesses(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves top recurring weaknesses for a user sorted by highest error count.
    Useful for Code Mentor and Study Agent to tailor explanations.
    """
    if not engine:
        return []

    try:
        with Session(engine) as session:
            skills = (
                session.query(UserSkill)
                .filter_by(user_id=user_id)
                .order_by(UserSkill.error_count.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "topic": s.topic,
                    "error_count": s.error_count,
                    "last_notes": s.last_error_notes,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None
                }
                for s in skills
            ]
    except Exception as e:
        logger.error(f"Error querying user skills: {str(e)}")
        return []


def log_session_activity(user_id: str, agent_type: str, query_text: str, response_summary: str) -> bool:
    """
    Logs an agent interaction turn to PostgreSQL session history.
    """
    if not engine:
        return False

    try:
        with Session(engine) as session:
            log_entry = SessionLog(
                user_id=user_id,
                agent_type=agent_type,
                user_query=query_text,
                agent_response=response_summary
            )
            session.add(log_entry)
            session.commit()
            return True
    except Exception as e:
        logger.error(f"Error logging session activity: {str(e)}")
        return False