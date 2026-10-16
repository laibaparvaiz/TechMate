"""
File: TechMate/app/agents/coding_agent.py

Description:
The Autonomous Coding Agent using LangGraph.

This agent can inspect files, modify files, run tests,
inspect directories, and check dependencies.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from app.prompts.coding import CODING_AGENT_SYSTEM_PROMPT
from app.tools.code_tools import (
    read_project_file,
    edit_project_file,
    run_pytest_suite,
)
from app.tools.system_tools import (
    list_directory,
    check_python_dependencies,
)
from app.tools.pdf_tools import (
    extract_pdf_text,
    merge_pdfs,
    split_pdf,
    add_watermark,
    add_text_annotation,
    fill_pdf_form,
    generate_pdf_document,
)
from app.utils.llm_router import get_resilient_llm


# ============================================================
# BUILD CODING AGENT
# ============================================================

def get_coding_agent(llm=None):
    """
    Create and return the Autonomous Coding Agent.

    Args:
        llm: Optional pre-built chat model to use instead of
            the default OpenRouter-backed resilient LLM. This
            lets the supervisor rebuild the agent on a
            fallback provider (e.g. Gemini) if OpenRouter's
            free-tier quota is exhausted.
    """

    if llm is None:

        llm = get_resilient_llm(
            temperature=0.2
        )

    tools = [
        read_project_file,
        edit_project_file,
        run_pytest_suite,
        list_directory,
        check_python_dependencies,
        extract_pdf_text,
        merge_pdfs,
        split_pdf,
        add_watermark,
        add_text_annotation,
        fill_pdf_form,
        generate_pdf_document,
    ]

    system_message = SystemMessage(
        content=CODING_AGENT_SYSTEM_PROMPT
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
    )

    return agent_executor
