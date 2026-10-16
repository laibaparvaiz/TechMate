"""
File: TechMate/app/tools/pdf_tools.py
Description: PDF reading and editing tools for the Autonomous Coding Agent.

Uses PyMuPDF (imported as fitz) — already a dependency via
app/rag/ingestion.py's PyMuPDFLoader, so no new package is
needed. Deliberately not using any external PDF API: that would
reintroduce the exact free-tier-quota fragility this whole
project has spent a lot of effort working around, and would
send document content to a third party unnecessarily.

HITL (Human-In-The-Loop):
All write tools (everything except extract_pdf_text) follow the
same propose-then-confirm pattern as edit_project_file in
code_tools.py: they validate the request and register a
pending action (see app/safety/hitl.py), but do NOT write
anything until the user confirms. The actual writes happen via
the execute_*() functions, called only by
app/agents/supervisor.py's confirmation_node dispatcher.
"""

import html
import io
import json
import re
import math

import pymupdf as fitz  # PyMuPDF

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.safety.permissions import (
    is_command_safe,
    resolve_within_base,
    sanitize_filename_component,
)
from app.safety.hitl import register_pending_action
from app.config import BASE_DIR
from app.utils.context_budget import truncate_to_budget

PDF_OUTPUT_DIR = BASE_DIR / "data" / "pdf_outputs"


def _markdown_to_pdf_bytes(markdown_content: str) -> bytes:
    """Render basic Markdown to a paginated PDF using PyMuPDF."""

    blocks = []
    paragraph = []
    list_items = []
    code_lines = []
    in_code = False

    def inline_markdown_to_html(text: str) -> str:
        """Escape text, then convert common inline Markdown emphasis."""

        value = html.escape(text)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        value = re.sub(
            r"\*\*(.+?)\*\*|__(.+?)__",
            lambda match: (
                "<strong>"
                + (match.group(1) or match.group(2))
                + "</strong>"
            ),
            value,
        )
        value = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|"
            r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)",
            lambda match: "<em>" + (match.group(1) or match.group(2)) + "</em>",
            value,
        )
        return value

    def flush_paragraph():
        if paragraph:
            value = inline_markdown_to_html(" ".join(paragraph))
            blocks.append(f"<p>{value}</p>")
            paragraph.clear()

    def flush_list():
        if list_items:
            list_tag = list_items[0][0]
            items_html = "".join(
                f"<li>{item}</li>"
                for item_kind, item in list_items
                if item_kind == list_tag
            )
            blocks.append(
                f"<{list_tag}>{items_html}</{list_tag}>"
            )
            list_items.clear()

    def split_table_row(row: str) -> list[str]:
        return [
            cell.strip()
            for cell in row.strip().strip("|").split("|")
        ]

    def is_table_separator(row: str) -> bool:
        cells = split_table_row(row)
        return len(cells) >= 2 and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
            for cell in cells
        )

    lines = markdown_content.splitlines()
    line_index = 0

    while line_index < len(lines):
        raw_line = lines[line_index]
        line = raw_line.strip()

        # Render standard Markdown tables as HTML tables instead
        # of collapsing their pipe-delimited rows into paragraph text.
        if (
            "|" in line
            and line_index + 1 < len(lines)
            and is_table_separator(lines[line_index + 1])
        ):
            flush_paragraph()
            flush_list()

            headers = split_table_row(line)
            line_index += 2  # Skip the Markdown alignment row.
            table_rows = []

            while line_index < len(lines) and "|" in lines[line_index]:
                row = lines[line_index].strip()
                if not is_table_separator(row):
                    table_rows.append(split_table_row(row))
                line_index += 1

            header_html = "".join(
                f"<th>{inline_markdown_to_html(cell)}</th>"
                for cell in headers
            )
            body_rows_html = []

            for row in table_rows:
                row = (row + [""] * len(headers))[:len(headers)]
                cells_html = "".join(
                    f"<td>{inline_markdown_to_html(cell)}</td>"
                    for cell in row
                )
                body_rows_html.append(f"<tr>{cells_html}</tr>")

            blocks.append(
                "<table><thead><tr>"
                + header_html
                + "</tr></thead><tbody>"
                + "".join(body_rows_html)
                + "</tbody></table>"
            )
            # The table loop has already advanced to the next
            # non-table line; process it on the next iteration.
            continue

        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            if in_code:
                blocks.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")
                code_lines.clear()
            in_code = not in_code
            line_index += 1
            continue
        if in_code:
            code_lines.append(raw_line)
            line_index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = min(len(heading.group(1)), 3)
            blocks.append(
                f"<h{level}>{html.escape(heading.group(2))}</h{level}>"
            )
        elif re.match(r"^[-*+]\s+", line) or re.match(
            r"^\d+[.)]\s+",
            line,
        ):
            flush_paragraph()
            is_ordered = bool(
                re.match(r"^\d+[.)]\s+", line)
            )
            list_tag = "ol" if is_ordered else "ul"

            if list_items and list_items[0][0] != list_tag:
                flush_list()

            item = re.sub(
                r"^(?:[-*+]\s+|\d+[.)]\s+)",
                "",
                line,
            )
            list_items.append(
                (list_tag, inline_markdown_to_html(item))
            )
        elif not line:
            flush_paragraph()
            flush_list()
        else:
            flush_list()
            paragraph.append(line)
        line_index += 1
    flush_paragraph()
    flush_list()
    if code_lines:
        blocks.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")

    body = "".join(blocks) or "<p></p>"
    document_html = (
        "<html><head><style>"
        "body { font-family: sans-serif; font-size: 11pt; line-height: 1.4; }"
        "h1 { font-size: 22pt; } h2 { font-size: 17pt; } h3 { font-size: 13pt; }"
        "table { border-collapse: collapse; width: 100%; }"
        "th, td { border: 1px solid #777; padding: 5px; vertical-align: top; }"
        "th { background-color: #e8e8e8; }"
        "pre { font-family: monospace; background-color: #f2f2f2; }"
        "</style></head><body>" + body + "</body></html>"
    )

    output = io.BytesIO()
    writer = fitz.DocumentWriter(output)
    story = fitz.Story(document_html)
    page_rect = fitz.Rect(0, 0, 595, 842)
    content_rect = fitz.Rect(54, 54, 541, 788)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    return output.getvalue()


# ============================================================
# SHARED HELPERS
# ============================================================

def _get_thread_id(config: RunnableConfig) -> str:
    """
    Defensive thread_id extraction, matching the pattern in
    code_tools.py / study_tools.py.
    """

    try:
        return config["configurable"]["thread_id"]
    except Exception:
        return "unknown_thread"


def _resolve_pdf_output_path(output_filename: str):
    """
    Resolves an output filename into PDF_OUTPUT_DIR, sanitizing
    it first. sanitize_filename_component() strips path
    separators, so the result can never escape this directory
    regardless of what the LLM/user supplies as output_filename.
    """

    safe_name = sanitize_filename_component(
        output_filename
    )

    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    PDF_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return PDF_OUTPUT_DIR / safe_name


def _validate_pdf_input(file_path: str):
    """
    Shared input validation for all PDF tools: path safety,
    existence, and extension check.

    Returns (path, error_message). error_message is None on
    success, otherwise path is None and error_message explains
    what failed.
    """

    path = resolve_within_base(
        file_path
    )

    if not is_command_safe(str(path)):
        return None, (
            "Error: Permission denied. Access restricted "
            "by safety policy."
        )

    if not path.exists():
        return None, f"Error: File '{file_path}' does not exist."

    if path.suffix.lower() != ".pdf":
        return None, f"Error: '{file_path}' is not a PDF file."

    return path, None


def _parse_page_ranges(page_ranges: str):
    """
    Parses a string like "1-3,5,7-9" (1-based, inclusive) into
    a sorted list of 0-based page indices.
    """

    pages = set()

    for part in page_ranges.split(","):

        part = part.strip()

        if not part:
            continue

        if "-" in part:

            start_str, end_str = part.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())

            for p in range(start, end + 1):
                pages.add(p - 1)

        else:

            pages.add(int(part) - 1)

    return sorted(pages)


# ============================================================
# READ-ONLY TOOL — no HITL needed
# ============================================================

@tool
def extract_pdf_text(file_path: str) -> str:
    """Extracts and returns the text content of a PDF file. Read-only, does not modify the file.
    Use this to inspect a PDF's contents before deciding what edit to propose.
    """

    path, error = _validate_pdf_input(file_path)

    if error:
        return error

    try:

        doc = fitz.open(str(path))

        text_parts = [
            f"[Page {page_num + 1}]\n{page.get_text()}"
            for page_num, page in enumerate(doc)
        ]

        doc.close()

        full_text = "\n\n".join(text_parts)

        return truncate_to_budget(
            full_text,
            max_chars=20000,
            label=f"PDF text ({file_path})",
        )

    except Exception as e:

        return f"Error reading PDF '{file_path}': {str(e)}"


# ============================================================
# WRITE TOOLS — propose only, never write directly
# ============================================================

@tool
def merge_pdfs(
    file_paths_json: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes merging multiple PDF files into one, in the given order.
    file_paths_json: a JSON array of PDF file paths, e.g. '["a.pdf", "b.pdf"]'.
    This does NOT write immediately — it asks the user to confirm first.
    """

    try:
        file_paths = json.loads(file_paths_json)
    except json.JSONDecodeError:
        return "Error: file_paths_json is not valid JSON."

    if not isinstance(file_paths, list) or not file_paths:
        return (
            "Error: file_paths_json must be a non-empty "
            "JSON array of file paths."
        )

    for fp in file_paths:

        _, error = _validate_pdf_input(fp)

        if error:
            return f"Error with '{fp}': {error}"

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "merge_pdfs",
            "file_paths": file_paths,
            "output_filename": output_filename,
        },
    )

    return (
        f"I'm ready to merge {len(file_paths)} PDF(s) "
        f"({', '.join(file_paths)}) into "
        f"'{output_filename}'. Reply 'yes' to confirm, "
        "or 'no' to cancel."
    )


@tool
def split_pdf(
    file_path: str,
    page_ranges: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes extracting specific pages from a PDF into a new file.
    page_ranges format: "1-3,5,7-9" (1-based page numbers).
    This does NOT write immediately — it asks the user to confirm first.
    """

    _, error = _validate_pdf_input(file_path)

    if error:
        return error

    try:

        pages = _parse_page_ranges(
            page_ranges
        )

        if not pages:
            raise ValueError("no pages parsed")

    except Exception:

        return (
            "Error: Could not parse page_ranges. Use a "
            "format like '1-3,5,7-9'."
        )

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "split_pdf",
            "file_path": file_path,
            "page_ranges": page_ranges,
            "output_filename": output_filename,
        },
    )

    return (
        f"I'm ready to extract pages {page_ranges} from "
        f"'{file_path}' into '{output_filename}'. "
        "Reply 'yes' to confirm, or 'no' to cancel."
    )


@tool
def add_watermark(
    file_path: str,
    watermark_text: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes stamping watermark_text diagonally across every page of a PDF.
    This does NOT write immediately — it asks the user to confirm first.
    """

    _, error = _validate_pdf_input(file_path)

    if error:
        return error

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "add_watermark",
            "file_path": file_path,
            "watermark_text": watermark_text,
            "output_filename": output_filename,
        },
    )

    return (
        f"I'm ready to add the watermark '{watermark_text}' "
        f"to every page of '{file_path}', saving as "
        f"'{output_filename}'. Reply 'yes' to confirm, or "
        "'no' to cancel."
    )


@tool
def add_text_annotation(
    file_path: str,
    page_number: int,
    text: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes adding a text annotation (sticky note) to a specific page of a PDF.
    page_number is 1-based.
    This does NOT write immediately — it asks the user to confirm first.
    """

    path, error = _validate_pdf_input(file_path)

    if error:
        return error

    try:

        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()

    except Exception as e:

        return f"Error opening '{file_path}': {str(e)}"

    if not (1 <= page_number <= page_count):

        return (
            f"Error: page_number {page_number} is out of "
            f"range (PDF has {page_count} page(s))."
        )

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "add_text_annotation",
            "file_path": file_path,
            "page_number": page_number,
            "text": text,
            "output_filename": output_filename,
        },
    )

    return (
        f"I'm ready to add a note on page {page_number} of "
        f"'{file_path}' saying: \"{text}\". Saving as "
        f"'{output_filename}'. Reply 'yes' to confirm, or "
        "'no' to cancel."
    )


@tool
def fill_pdf_form(
    file_path: str,
    field_values_json: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes filling in an existing fillable PDF form's fields.
    field_values_json: a JSON object mapping field name to value,
    e.g. '{"name": "Alice", "date": "2026-09-19"}'.
    This does NOT write immediately — it asks the user to confirm first.
    """

    _, error = _validate_pdf_input(file_path)

    if error:
        return error

    try:
        field_values = json.loads(field_values_json)
    except json.JSONDecodeError:
        return "Error: field_values_json is not valid JSON."

    if not isinstance(field_values, dict) or not field_values:
        return (
            "Error: field_values_json must be a non-empty "
            "JSON object."
        )

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "fill_pdf_form",
            "file_path": file_path,
            "field_values_json": field_values_json,
            "output_filename": output_filename,
        },
    )

    preview_items = list(field_values.items())[:5]
    fields_preview = ", ".join(
        f"{k}={v}" for k, v in preview_items
    )

    if len(field_values) > 5:
        fields_preview += ", ..."

    return (
        f"I'm ready to fill {len(field_values)} field(s) "
        f"in '{file_path}' ({fields_preview}), saving as "
        f"'{output_filename}'. Reply 'yes' to confirm, or "
        "'no' to cancel."
    )


# ============================================================
# ACTUAL WRITE LOGIC — called ONLY after user confirmation
# ============================================================

@tool
def generate_pdf_document(
    title: str,
    markdown_content: str,
    output_filename: str,
    config: RunnableConfig,
) -> str:
    """Proposes creating a PDF document from Markdown content.

    The PDF is only written after the user confirms the proposal.
    """

    if not markdown_content or not markdown_content.strip():
        return "Error: markdown_content must not be empty."

    register_pending_action(
        _get_thread_id(config),
        {
            "type": "generate_pdf_document",
            "title": title,
            "markdown_content": markdown_content,
            "output_filename": output_filename,
        },
    )

    return (
        f"I'm ready to create '{output_filename}' from the "
        f"document '{title}'. Reply 'yes' to confirm, or "
        "'no' to cancel."
    )


def execute_generate_pdf_document(
    title: str,
    markdown_content: str,
    output_filename: str,
) -> str:
    """Create the proposed Markdown document PDF after confirmation."""

    try:
        if not markdown_content or not markdown_content.strip():
            return "Error: markdown_content must not be empty."

        first_content_line = next(
            (
                line.strip().rstrip("\\").strip()
                for line in markdown_content.splitlines()
                if line.strip()
            ),
            "",
        )
        existing_heading = re.match(
            r"^(?:#{1,6}\s+|\*\*)(.+?)(?:\*\*)?$",
            first_content_line,
        )
        if (
            existing_heading
            and existing_heading.group(1).strip().casefold()
            == title.strip().casefold()
        ):
            pdf_markdown = markdown_content
        else:
            pdf_markdown = f"# {title}\n\n{markdown_content}"

        output_path = _resolve_pdf_output_path(output_filename)
        output_path.write_bytes(
            _markdown_to_pdf_bytes(
                pdf_markdown
            )
        )
        return f"Successfully created '{title}' as '{output_path}'."
    except Exception as e:
        return f"Error creating PDF document: {str(e)}"

def execute_merge_pdfs(
    file_paths,
    output_filename: str,
) -> str:

    try:

        merged = fitz.open()

        for fp in file_paths:

            path, error = _validate_pdf_input(fp)

            if error:
                merged.close()
                return error

            src = fitz.open(str(path))
            merged.insert_pdf(src)
            src.close()

        output_path = _resolve_pdf_output_path(
            output_filename
        )

        merged.save(str(output_path))
        merged.close()

        return (
            f"Successfully merged {len(file_paths)} "
            f"PDF(s) into '{output_path}'."
        )

    except Exception as e:

        return f"Error merging PDFs: {str(e)}"


def execute_split_pdf(
    file_path: str,
    page_ranges: str,
    output_filename: str,
) -> str:

    try:

        path, error = _validate_pdf_input(file_path)

        if error:
            return error

        pages = _parse_page_ranges(page_ranges)

        src = fitz.open(str(path))
        out = fitz.open()

        for p in pages:

            if 0 <= p < len(src):
                out.insert_pdf(src, from_page=p, to_page=p)

        output_path = _resolve_pdf_output_path(
            output_filename
        )

        out.save(str(output_path))
        out.close()
        src.close()

        return (
            f"Successfully extracted {len(pages)} "
            f"page(s) into '{output_path}'."
        )

    except Exception as e:

        return f"Error splitting PDF: {str(e)}"


def execute_add_watermark(
    file_path: str,
    watermark_text: str,
    output_filename: str,
) -> str:

    try:

        path, error = _validate_pdf_input(file_path)

        if error:
            return error

        doc = fitz.open(str(path))

        angle = 45

        for page in doc:

            rect = page.rect

            fontsize = 40

            length = fitz.get_text_length(
                watermark_text, fontname="helv", fontsize=fontsize
            )

            # Shrink very long watermark text so it fits the diagonal.
            diagonal = math.hypot(rect.width, rect.height)

            if length > diagonal * 0.8:
                fontsize = fontsize * (diagonal * 0.8) / length
                length = diagonal * 0.8

            # Start point so the rotated text is centred on the page.
            dx = (length / 2) * math.cos(math.radians(angle))
            dy = (length / 2) * math.sin(math.radians(angle))
            start = fitz.Point(rect.width / 2 - dx, rect.height / 2 + dy)

            # insert_text(rotate=...) only allows 0/90/180/270, so a
            # diagonal watermark has to use a morph (rotation matrix).
            page.insert_text(
                start,
                watermark_text,
                fontsize=fontsize,
                fontname="helv",
                color=(0.7, 0.7, 0.7),
                morph=(start, fitz.Matrix(angle)),
                overlay=True,
            )

        output_path = _resolve_pdf_output_path(output_filename)

        doc.save(str(output_path))
        doc.close()

        return f"Successfully added watermark, saved as '{output_path}'."

    except Exception as e:

        return f"Error adding watermark: {str(e)}"


def execute_add_text_annotation(
    file_path: str,
    page_number: int,
    text: str,
    output_filename: str,
) -> str:

    try:

        path, error = _validate_pdf_input(file_path)

        if error:
            return error

        doc = fitz.open(str(path))

        idx = page_number - 1

        if not (0 <= idx < len(doc)):
            doc.close()
            return (
                f"Error: page_number {page_number} is "
                f"out of range (PDF has {len(doc)} "
                "page(s))."
            )

        page = doc[idx]
        page.add_text_annot((72, 72), text)

        output_path = _resolve_pdf_output_path(
            output_filename
        )

        doc.save(str(output_path))
        doc.close()

        return (
            "Successfully added annotation, saved as "
            f"'{output_path}'."
        )

    except Exception as e:

        return f"Error adding annotation: {str(e)}"


def execute_fill_pdf_form(
    file_path: str,
    field_values_json: str,
    output_filename: str,
) -> str:

    try:

        path, error = _validate_pdf_input(file_path)

        if error:
            return error

        field_values = json.loads(field_values_json)

        doc = fitz.open(str(path))
        filled_count = 0

        for page in doc:

            for widget in (page.widgets() or []):

                if widget.field_name in field_values:

                    widget.field_value = str(
                        field_values[widget.field_name]
                    )
                    widget.update()
                    filled_count += 1

        output_path = _resolve_pdf_output_path(
            output_filename
        )

        doc.save(str(output_path))
        doc.close()

        return (
            f"Successfully filled {filled_count} "
            f"field(s), saved as '{output_path}'."
        )

    except Exception as e:

        return f"Error filling PDF form: {str(e)}"
