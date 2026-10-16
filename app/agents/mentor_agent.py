"""
File: TechMate/app/agents/mentor_agent.py

Description:
The Code Mentor Agent for reviews, explanations,
debugging guidance, and programming mentorship.

This version uses the same create_react_agent()
prompt= style as the Study and Coding agents.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from app.prompts.mentor import (
    MENTOR_AGENT_SYSTEM_PROMPT
)

# Only read-only tools are provided.
from app.tools.code_tools import (
    read_project_file,
)

from app.tools.system_tools import (
    list_directory,
)

from app.utils.llm_router import (
    get_resilient_llm,
)


# ============================================================
# MENTOR AGENT
# ============================================================

def get_mentor_agent(llm=None, extra_context: str = ""):
    """
    Create and return the Code Mentor Agent.

    The Mentor Agent is intentionally read-only.

    It can:
    - inspect project files
    - list directories
    - explain code
    - diagnose problems
    - perform code reviews

    It cannot directly edit project files.

    Args:
        llm: Optional pre-built chat model to use instead of
            the default OpenRouter-backed resilient LLM. This
            lets the supervisor rebuild the agent on a
            fallback provider (e.g. Gemini) if OpenRouter's
            free-tier quota is exhausted.
        extra_context: Optional text appended to the system
            prompt — used by the supervisor to inject a
            summary of the user's known recurring weaknesses
            (see app/memory/skills.py), so explanations can be
            tailored without needing a separate LLM call.
    """

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    if llm is None:

        llm = get_resilient_llm(
            temperature=0.4
        )


    # --------------------------------------------------------
    # READ-ONLY TOOLS
    # --------------------------------------------------------

    tools = [
        read_project_file,
        list_directory,
    ]


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_content = MENTOR_AGENT_SYSTEM_PROMPT

    if extra_context:

        system_content = (
            system_content
            + "\n\n"
            + extra_context
        )

    system_message = SystemMessage(
        content=system_content
    )


    # --------------------------------------------------------
    # CREATE REACT AGENT
    # --------------------------------------------------------

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
    )


    # --------------------------------------------------------
    # RETURN AGENT
    # --------------------------------------------------------

    return agent_executor