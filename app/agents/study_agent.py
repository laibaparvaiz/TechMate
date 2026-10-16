"""
File: TechMate/app/agents/study_agent.py

Description:
Study & Academic Agent with RAG, CRAG (3-way grading + web fallback)
and Self-RAG grounding verification.

Flow:

    Question
       |
    FAISS retrieval
       |
    CRAG grade each chunk
       |
    correct   (>= 2 relevant chunks) -> answer from uploaded material
    ambiguous (exactly 1 relevant)   -> uploaded material + web results
    incorrect (0 relevant / none)    -> web results only
    error     (AI call failed)       -> "service busy" message
       |
    Generation (streamed)
       |
    Self-RAG grounding check (against the SAME combined context)
       |
    verified answer  /  safe refusal  /  "service busy" message

The result dict includes "source_mode": "kb" | "web" | "mixed" | "none"
so the supervisor can tell the user when web sources were used.
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.prompts.study import STUDY_AGENT_SYSTEM_PROMPT
from app.prompts.verification import (
    SELF_RAG_HALLUCINATION_PROMPT,
)
from app.tools.study_tools import (
    save_flashcards,
    save_study_notes,
)
from app.tools.web_search import web_search_fallback
from app.rag.retrieval import retrieve_relevant_chunks
from app.utils.context_budget import (
    truncate_to_budget,
    DEFAULT_MAX_CHARS,
)
from app.utils.llm_router import (
    get_resilient_llm,
    get_fallback_llm,
    invoke_with_fallback,
    stream_with_fallback,
)


# How many relevant chunks (out of the retrieved ones) count as
# "correct" (answer fully from uploads). Exactly 1 -> ambiguous.
CRAG_CORRECT_MIN_RELEVANT = 2

SERVICE_BUSY_MESSAGE = (
    "⚠️ The AI service is busy right now. "
    "Please try again in a moment."
)

QUOTA_MESSAGE = (
    "⚠️ The free daily limit has been reached on the configured AI "
    "providers (OpenRouter and/or Gemini free tiers). Please try again "
    "later, or add credits / another API key."
)

# True  = web/mixed answers stream live (if verification fails, the
#         text is replaced by the rewritten version).
# False = web/mixed answers are shown only after verification.
STREAM_WEB_ANSWERS = True


def _busy_message(notes: str) -> str:
    """Pick the right message: daily quota exhausted vs temporary glitch."""

    text = (notes or "").lower()

    if any(
        marker in text
        for marker in (
            "per-day", "perday", "free_tier", "free-tier",
            "quota", "resource_exhausted", "429",
        )
    ):
        return QUOTA_MESSAGE

    return SERVICE_BUSY_MESSAGE


# ============================================================
# VERIFICATION SCHEMAS
# ============================================================

class CRAGDocumentDecision(BaseModel):
    """Relevance decision for one retrieved document."""

    document_index: int = Field(
        description="The 1-based index of the retrieved document being evaluated."
    )

    relevant: bool = Field(
        description=(
            "Whether this document contains useful facts, definitions, "
            "explanations, examples, or other information that can "
            "directly help answer the user's question."
        )
    )

    reason: str = Field(
        description="Brief factual explanation of why the document is relevant or not."
    )


class CRAGBatchDecision(BaseModel):
    """Batch CRAG decision for all retrieved documents."""

    documents: List[CRAGDocumentDecision] = Field(
        description="Relevance evaluations for every retrieved document."
    )


class SelfRAGDecision(BaseModel):
    """Structured Self-RAG grounding evaluation."""

    is_grounded: bool = Field(
        description="Whether the generated response is supported by the provided context."
    )

    hallucination_detected: bool = Field(
        description="Whether the response contains unsupported information."
    )

    notes: str = Field(
        description="Brief explanation of the verification result."
    )


# ============================================================
# CONTENT NORMALIZATION
# ============================================================

def _stream_chunk_to_text(content) -> str:
    """
    Convert ONE streamed chunk's content into text WITHOUT
    stripping whitespace (stripping would destroy the spaces
    between words in token streams).
    """

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):

        parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text", "")

                if text:
                    parts.append(str(text))

            elif isinstance(block, str):

                parts.append(block)

        return "".join(parts)

    if isinstance(content, dict):

        text = content.get("text", "")

        return str(text) if text else ""

    if content is None:
        return ""

    return str(content)


# ============================================================
# STREAMING HELPERS
# ============================================================

def _get_stream_writer():
    """Safely obtain the LangGraph stream writer (None outside streaming)."""

    try:
        return get_stream_writer()
    except Exception:
        return None


def _emit_stream_event(writer, event_type: str, **data):
    """Send a controlled custom event to the UI."""

    if writer is None:
        print(f"Study streaming writer unavailable: {event_type}")
        return

    writer({"type": event_type, **data})


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_study_context(
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """Retrieve potentially relevant chunks from the local FAISS index."""

    try:

        return retrieve_relevant_chunks(query, top_k=top_k)

    except Exception as e:

        print("Study retrieval error:", str(e))

        return []


# ============================================================
# CRAG RELEVANCE CHECK (3-way label + relevant docs)
# ============================================================

def check_crag_relevance(
    query: str,
    docs: List[Document],
    conversation_context: str = "",
) -> Tuple[str, List[Document], str]:
    """
    Grade every retrieved document with ONE structured LLM call.

    Returns (status, relevant_docs, reason) where status is:

        'correct'    - enough relevant chunks, answer from uploads
        'ambiguous'  - only one relevant chunk, add web results
        'incorrect'  - nothing relevant, use web results
        'error'      - the AI call failed (NOT the same as "not found")

    Only the relevant docs are returned, so irrelevant chunks never
    reach answer generation.
    """

    if not docs:

        return "incorrect", [], "No documents were retrieved."

    document_sections = []

    per_doc_budget = max(
        1000,
        DEFAULT_MAX_CHARS // max(len(docs), 1),
    )

    for index, document in enumerate(docs):

        capped_content = truncate_to_budget(
            document.page_content,
            per_doc_budget,
            label=f"CRAG document {index + 1}",
        )

        document_sections.append(
            f"[Document {index + 1}]\n{capped_content}"
        )

    combined_documents = "\n\n---\n\n".join(document_sections)

    system_instruction = """
You are the CRAG relevance evaluator for TechMate.

Your job is to evaluate retrieved study documents against
the user's question.

Evaluate EVERY retrieved document independently.

A document is relevant when it contains useful facts,
definitions, explanations, examples, or information that
could directly help answer the user's question.

A document is not relevant when it is unrelated, only
superficially similar, or does not provide useful information
for answering the question.

If recent conversation context is provided, use it ONLY to
understand what the user's question refers to (e.g. pronouns
like "it", or references like "the second point"). Do not use
the conversation context itself as a source of facts - only
the retrieved documents count as evidence.

Return exactly one structured evaluation for every document.

Do not invent information.
Do not use outside knowledge when judging the documents.
""".strip()

    context_block = (
        "Recent Conversation Context "
        "(for resolving references only, not a source of "
        f"facts):\n\n{conversation_context}\n\n"
        if conversation_context
        else ""
    )

    user_content = f"""
User Question:

{query}

{context_block}Retrieved Documents:

{combined_documents}

Evaluate each document independently.

For each document return:

- document_index: its 1-based document number
- relevant: true or false
- reason: brief explanation
""".strip()

    try:

        llm = get_resilient_llm(temperature=0.0)

        fallback_llm = get_fallback_llm(temperature=0.0)

        structured_llm = llm.with_structured_output(CRAGBatchDecision)

        structured_fallback = (
            fallback_llm.with_structured_output(CRAGBatchDecision)
            if fallback_llm is not None
            else None
        )

        decision = invoke_with_fallback(
            structured_llm,
            structured_fallback,
            [
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_content),
            ],
        )

    except Exception as e:

        print("CRAG verification error:", str(e))

        return "error", [], f"CRAG verification could not be completed: {e}"

    if not decision or not decision.documents:

        return "error", [], "CRAG returned no document evaluations."

    relevant_indexes = {
        d.document_index
        for d in decision.documents
        if d.relevant
    }

    relevant_docs = [
        doc
        for position, doc in enumerate(docs, start=1)
        if position in relevant_indexes
    ]

    reason = " ".join(
        f"Document {d.document_index}: {d.reason}"
        for d in decision.documents
        if d.reason
    )

    if len(relevant_docs) >= CRAG_CORRECT_MIN_RELEVANT:
        return "correct", relevant_docs, reason

    if relevant_docs:
        return "ambiguous", relevant_docs, reason

    return "incorrect", [], reason


# ============================================================
# CONTEXT FORMATTER
# ============================================================

def format_context(docs: List[Document], label: str = "Document") -> str:
    """
    Convert documents into a clean context block, including
    source/page metadata when available (works for uploaded
    chunks and web results alike).

    `label` is the prefix of each entry ("Document" for uploads,
    "Web source" for web results) so the two numberings never clash.
    """

    if not docs:
        return ""

    context_parts = []

    per_doc_budget = max(
        1000,
        DEFAULT_MAX_CHARS // max(len(docs), 1),
    )

    for index, document in enumerate(docs):

        metadata = document.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
        )

        page = (
            metadata.get("page")
            if metadata.get("page") is not None
            else metadata.get("page_number")
        )

        source_info = ""

        if source:
            source_info += f"Source: {source}"

        if page is not None:
            source_info += (
                f", Page: {page}" if source_info else f"Page: {page}"
            )

        header = (
            f"[{label} {index + 1}]\n{source_info}"
            if source_info
            else f"[{label} {index + 1}]"
        )

        capped_content = truncate_to_budget(
            document.page_content,
            per_doc_budget,
            label=f"context document {index + 1}",
        )

        context_parts.append(f"{header}\n{capped_content}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# SEARCH TOOL (used by get_study_agent only)
# ============================================================

@tool
def search_study_materials(query: str) -> str:
    """
    Search uploaded study documents.

    Performs FAISS retrieval followed by batch CRAG relevance
    checking before returning verified context.
    """

    docs = retrieve_study_context(query, top_k=5)

    if not docs:

        return (
            "NO_RELEVANT_CONTEXT\n"
            "No study documents were retrieved. "
            "Do not invent an answer from the uploaded materials."
        )

    status, relevant_docs, reason = check_crag_relevance(query, docs)

    if status in ("incorrect", "error") or not relevant_docs:

        return (
            "NO_RELEVANT_CONTEXT\n"
            "CRAG verification determined that the retrieved "
            "documents are not sufficiently relevant.\n"
            f"Reason: {reason}\n\n"
            "Do not pretend that the documents contain an answer."
        )

    return (
        "CRAG_VERIFIED_CONTEXT\n"
        "The retrieved material was judged relevant.\n\n"
        f"{format_context(relevant_docs)}"
    )


def get_study_agent():
    """
    Create the ReAct Study agent. Retained for compatibility; the
    supervisor currently uses run_verified_study_agent() directly.
    """

    llm = get_resilient_llm(temperature=0.3)

    tools = [
        search_study_materials,
        save_flashcards,
        save_study_notes,
    ]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=STUDY_AGENT_SYSTEM_PROMPT,
    )


# ============================================================
# SELF-RAG VERIFICATION
# ============================================================

def verify_study_response(
    query: str,
    context: str,
    response: str,
) -> Tuple[str, str]:
    """
    Self-RAG: verify the answer against the context it was given.

    Returns (status, notes), status one of:
        'pass'  - grounded, no hallucination
        'fail'  - NOT supported by the context
        'error' - the verification call itself failed
    """

    if not context:

        return "fail", "There was no verified context."

    if not response:

        return "fail", "The Study Agent returned an empty response."

    try:

        llm = get_resilient_llm(temperature=0.0)

        fallback_llm = get_fallback_llm(temperature=0.0)

        structured_llm = llm.with_structured_output(SelfRAGDecision)

        structured_fallback = (
            fallback_llm.with_structured_output(SelfRAGDecision)
            if fallback_llm is not None
            else None
        )

        prompt = SELF_RAG_HALLUCINATION_PROMPT.format(
            context=context,
            response=response,
        )

        decision = invoke_with_fallback(
            structured_llm,
            structured_fallback,
            [
                SystemMessage(
                    content=(
                        "You are the TechMate Self-RAG grounding "
                        "evaluator. Determine whether the answer is "
                        "fully supported by the supplied context."
                    )
                ),
                HumanMessage(content=prompt),
            ],
        )

        if decision is None:

            return "error", "Self-RAG verification returned no result."

        if decision.is_grounded and not decision.hallucination_detected:

            return "pass", decision.notes

        return "fail", decision.notes

    except Exception as e:

        print("Self-RAG verification error:", str(e))

        return "error", f"Self-RAG verification could not be completed: {e}"


# ============================================================
# CITATION CLEANUP
# ============================================================

def _doc_source(doc: Document, with_page: bool) -> str:
    """Readable source for a document: URL for web, file name + page for uploads."""

    metadata = doc.metadata or {}

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or ""
    )

    if not source:
        return ""

    if not source.startswith("http"):
        source = re.split(r"[\\/]", source)[-1]  # file name only, no path

    page = (
        metadata.get("page")
        if metadata.get("page") is not None
        else metadata.get("page_number")
    )

    if with_page and page is not None and not source.startswith("http"):
        return f"{source}, page {page}"

    return source


def _resolve_citations(
    answer: str,
    kb_docs: List[Document],
    web_docs: List[Document],
) -> str:
    """
    Replace leftover "Web source N" / "Document N" citations (which mean
    nothing to the reader) with the real URL or file name + page.
    """

    def web_sub(match):

        index = int(match.group(1))

        if 1 <= index <= len(web_docs):
            source = _doc_source(web_docs[index - 1], with_page=False)
            if source:
                return f" [{source}]"

        return ""

    def kb_sub(match):

        index = int(match.group(1))

        if 1 <= index <= len(kb_docs):
            source = _doc_source(kb_docs[index - 1], with_page=True)
            if source:
                return f" [{source}]"

        return ""

    answer = re.sub(
        r"[ \t]*[\(\[]?[ \t]*Web source (\d+)(?:[ \t]*[\)\]])?",
        web_sub,
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(
        r"[ \t]*[\(\[]?[ \t]*Document (\d+)(?:[ \t]*[\)\]])?",
        kb_sub,
        answer,
        flags=re.IGNORECASE,
    )

    return answer


# ============================================================
# FULL PIPELINE
# ============================================================

def run_verified_study_agent(
    query: str,
    conversation_context: str = "",
) -> dict:
    """
    Run: Retrieve -> CRAG -> (web fallback) -> Generate -> Self-RAG.

    conversation_context (optional) is folded into the retrieval
    query, passed to CRAG for reference resolution only, and passed
    to generation so follow-ups ("the second point") resolve.

    Returned dict keys: response, context, is_grounded, crag_passed,
    self_rag_passed, source_mode, verification_notes (optional).
    """

    writer = _get_stream_writer()

    def result(response, *, context="", grounded=False, crag=False,
               self_rag=False, mode="none", notes=""):
        return {
            "response": response,
            "context": context,
            "is_grounded": grounded,
            "crag_passed": crag,
            "self_rag_passed": self_rag,
            "source_mode": mode,
            "verification_notes": notes,
        }

    # ========================================================
    # 1. RETRIEVE
    # ========================================================

    retrieval_query = (
        f"{conversation_context}\n\nCurrent question: {query}"
        if conversation_context
        else query
    )

    _emit_stream_event(
        writer, "status", message="Retrieving study material..."
    )

    docs = retrieve_study_context(retrieval_query, top_k=5)

    # ========================================================
    # 2. CRAG GRADE
    # ========================================================

    _emit_stream_event(
        writer, "status", message="Checking document relevance..."
    )

    crag_status, relevant_docs, crag_reason = check_crag_relevance(
        query,
        docs,
        conversation_context=conversation_context,
    )

    print(f"CRAG status: {crag_status} ({len(relevant_docs)} relevant chunk(s))")

    if crag_status == "error":

        return result(_busy_message(crag_reason), notes=crag_reason)

    # ========================================================
    # 3. WEB FALLBACK + CONTEXT
    # ========================================================

    web_docs: List[Document] = []

    if crag_status in ("incorrect", "ambiguous"):

        _emit_stream_event(
            writer, "status", message="Searching the web..."
        )

        # Raw question, not the conversation context.
        web_docs = web_search_fallback(query)

        print(f"Web fallback returned {len(web_docs)} result(s)")

    if crag_status == "incorrect" and not web_docs:

        return result(
            (
                "I couldn't find this in your uploaded materials, "
                "and the web search returned nothing usable."
            ),
            notes=crag_reason,
        )

    context_parts = []

    if relevant_docs:

        context_parts.append(
            "=== UPLOADED STUDY MATERIAL ===\n\n"
            + format_context(relevant_docs)
        )

    if web_docs:

        context_parts.append(
            "=== WEB SOURCES (not from the user's uploads) ===\n\n"
            + format_context(web_docs, label="Web source")
        )

    context = "\n\n".join(context_parts)

    if relevant_docs and web_docs:
        source_mode = "mixed"
    elif web_docs:
        source_mode = "web"
    else:
        source_mode = "kb"

    # ========================================================
    # 4. GENERATE (streamed, with provider fallback)
    # ========================================================

    _emit_stream_event(
        writer,
        "status",
        message=(
            "Generating answer..."
            if (source_mode == "kb" or STREAM_WEB_ANSWERS)
            else "Writing and checking the answer (this takes a bit longer "
            "for web sources)..."
        ),
    )

    source_rules = {
        "kb": (
            "Use ONLY the uploaded study material below."
        ),
        "web": (
            "The uploaded materials did not cover this question. "
            "Use ONLY the web sources below, and cite the URL for "
            "each key point."
        ),
        "mixed": (
            "Use the uploaded study material first; use the web "
            "sources only to fill gaps. Mark web-sourced points "
            "with (Web) and cite the URL."
        ),
    }[source_mode]

    try:

        llm = get_resilient_llm(temperature=0.3)

        fallback_llm = get_fallback_llm(temperature=0.3)

        context_section = (
            "Recent Conversation "
            "(use ONLY to understand what the user's question "
            "refers to, e.g. pronouns or 'the second point' - "
            f"never as a source of facts):\n\n{conversation_context}\n\n"
            if conversation_context
            else ""
        )

        generation_prompt = (
            "You are the TechMate Study Agent.\n\n"
            "Answer the user's question. "
            f"{source_rules}\n\n"
            "Rules:\n"
            "1. Do not invent information.\n"
            "2. Do not use outside knowledge beyond the context below.\n"
            "3. If the context does not contain enough information, "
            "explicitly say so.\n"
            "4. Give a clear and useful answer.\n"
            "5. Preserve important wording from the source when "
            "defining concepts.\n"
            "6. Do not claim that information came from the uploaded "
            "documents unless it is actually present there.\n"
            "7. When source/page information is available in the "
            "context, cite it inline: file name and page for uploaded "
            "material, the URL for web sources. Never write 'Document "
            "N' as a citation.\n"
            "8. Explain in plain, approachable language, as if to a "
            "student learning this material for the first time - not "
            "to an expert. Define any technical term from the source "
            "the first time you use it. Prefer a concrete example over "
            "an abstract definition when the material supports one.\n"
            "9. Do not offer to save or export anything, ask the user "
            "to confirm a save, claim that a file was saved, or give "
            "instructions for saving/exporting the answer. The "
            "supervisor handles all save offers, confirmation prompts, "
            "and file creation. If the user asks to save the answer, "
            "answer the study question normally and leave all "
            "save-related messaging to the supervisor. Never mention "
            "PDF files, file paths, or earlier saves, and never add a "
            "separate 'notes you can save' section: write ONE answer.\n"
            "10. Never use Markdown tables. Use bullet lists instead.\n"
            "11. Do not end with statements about where the information "
            "came from or what was/wasn't added (for example 'all "
            "points come directly from...'). Inline citations are "
            "enough; do not add a 'Sources used' list.\n\n"
            f"{context_section}"
            "Verified Context:\n\n"
            f"{context}\n\n"
            "User Question:\n"
            f"{query}"
        )

        generation_prompt = truncate_to_budget(
            generation_prompt,
            max_chars=DEFAULT_MAX_CHARS * 2,
            label="study generation prompt",
        )

        messages = [
            SystemMessage(
                content=(
                    "You are the TechMate Study Agent. "
                    "Follow the supplied context strictly."
                )
            ),
            HumanMessage(content=generation_prompt),
        ]

        answer_parts = []

        for chunk in stream_with_fallback(llm, fallback_llm, messages):

            text = _stream_chunk_to_text(chunk.content)

            if not text:
                continue

            answer_parts.append(text)

            # See STREAM_WEB_ANSWERS at the top of this file.
            if source_mode == "kb" or STREAM_WEB_ANSWERS:
                _emit_stream_event(writer, "token", text=text)

        answer = "".join(answer_parts).strip()

    except Exception as e:

        print("Study generation error:", str(e))

        raise

    if not answer:

        return result(
            (
                "I couldn't generate a reliable answer from the "
                "available material."
            ),
            context=context,
            crag=True,
            mode=source_mode,
        )

    # ========================================================
    # 5. SELF-RAG
    # ========================================================

    _emit_stream_event(
        writer, "status", message="Verifying answer against the material..."
    )

    verify_status, verification_notes = verify_study_response(
        query,
        context,
        answer,
    )

    if verify_status == "fail":

        print(f"Self-RAG failed. Verifier notes: {verification_notes}")

        # One rewrite attempt: tell the model what was unsupported.
        _emit_stream_event(
            writer, "status", message="Rewriting answer to match the sources..."
        )

        try:

            retry_prompt = (
                "Your previous answer contained claims that are NOT "
                "supported by the context below.\n"
                f"Verifier notes: {verification_notes}\n\n"
                "Rewrite the answer using ONLY facts stated in the "
                "context. Remove or soften anything unsupported. If the "
                "context is thin, give a shorter answer instead of "
                "adding outside knowledge. Keep the same rules: no "
                "Markdown tables, cite web sources by URL (never 'Document "
                "N'), no 'sources used' list, no save offers.\n\n"
                f"Context:\n\n{context}\n\n"
                f"User Question:\n{query}\n\n"
                f"Previous answer:\n{answer}"
            )

            retry_response = invoke_with_fallback(
                llm,
                fallback_llm,
                [
                    SystemMessage(
                        content=(
                            "You are the TechMate Study Agent. "
                            "Follow the supplied context strictly."
                        )
                    ),
                    HumanMessage(
                        content=truncate_to_budget(
                            retry_prompt,
                            max_chars=DEFAULT_MAX_CHARS * 2,
                            label="study retry prompt",
                        )
                    ),
                ],
            )

            retry_answer = (
                _stream_chunk_to_text(retry_response.content).strip()
                if retry_response is not None
                else ""
            )

            if retry_answer:

                retry_status, retry_notes = verify_study_response(
                    query,
                    context,
                    retry_answer,
                )

                print(f"Self-RAG retry: {retry_status}. {retry_notes}")

                if retry_status == "pass":

                    answer = retry_answer
                    verify_status = "pass"
                    verification_notes = retry_notes

        except Exception as e:

            print("Study retry error (non-fatal):", str(e))

    if verify_status == "error":

        return result(
            _busy_message(verification_notes),
            context=context,
            crag=True,
            mode=source_mode,
            notes=verification_notes,
        )

    if verify_status == "fail":

        _emit_stream_event(
            writer,
            "verification_failed",
            message=(
                "The generated answer could not be fully verified "
                "against the material."
            ),
        )

        return result(
            (
                "I found relevant material, but I couldn't verify "
                "that my generated answer was fully supported by it. "
                "I don't want to guess."
            ),
            context=context,
            crag=True,
            mode=source_mode,
            notes=verification_notes,
        )

    # ========================================================
    # 6. VERIFIED ANSWER
    # ========================================================

    _emit_stream_event(writer, "complete", message="Answer verified.")

    answer = _resolve_citations(answer, relevant_docs, web_docs)

    return result(
        answer,
        context=context,
        grounded=True,
        crag=True,
        self_rag=True,
        mode=source_mode,
        notes=verification_notes,
    )