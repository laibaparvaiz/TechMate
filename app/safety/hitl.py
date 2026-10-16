"""
File: TechMate/app/safety/hitl.py

Description:
Lightweight Human-In-The-Loop (HITL) pending-action registry.

Design (Option B — simple two-step confirm, NOT a full
LangGraph-native interrupt):

    Turn 1:
        The user asks for something that would write/change
        data (edit a file, save flashcards, save notes). The
        relevant tool VALIDATES the request but does NOT
        perform the write. Instead it registers the proposed
        action here (keyed by thread_id) and returns a
        human-readable description, which the agent relays to
        the user asking for confirmation.

    Turn 2:
        The user replies (e.g. "yes" / "no"). The supervisor's
        confirmation_node (see app/agents/supervisor.py) checks
        this registry / the graph's persisted pending_action
        field for the same thread, and either executes the
        action for real or discards it.

This registry is intentionally a simple in-process dict, not
part of LangGraph's checkpointed state. It exists ONLY as a
relay: tools running inside a nested create_react_agent call
(inside coding_node / mentor_node) have no direct way to write
into the OUTER supervisor graph's state, so they register here,
and the calling node pops the registration right after the
nested agent finishes and persists it into the real (Postgres-
checkpointed) graph state as "pending_action".

Known limitation: if the app restarts between proposal and
confirmation, anything sitting only in THIS registry (not yet
popped into graph state) is lost. In practice this window is
extremely short (same turn), so it's an accepted tradeoff of
the simpler Option B design.
"""

from typing import Any, Dict, Optional

_pending_registry: Dict[str, Dict[str, Any]] = {}


def register_pending_action(
    thread_id: str,
    action: Dict[str, Any],
) -> None:
    """
    Register a proposed write action for the given thread,
    awaiting user confirmation on the next turn.
    """

    _pending_registry[thread_id] = action


def pop_pending_action(
    thread_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve and remove any action registered for this thread
    during the CURRENT turn's tool execution.

    Called by coding_node right after running its nested
    tool-calling agent, to check whether a write tool proposed
    an action this turn, so it can be persisted into the
    outer graph's state.
    """

    return _pending_registry.pop(thread_id, None)