"""
File: TechMate/tests/test_pdf_tools.py
Description: Tests for app/tools/pdf_tools.py's pure-logic
helpers (page-range parsing, input validation). Does NOT
exercise actual PDF reading/writing (fitz/PyMuPDF I/O) — those
need real PDF fixture files and are better covered by manual/
integration testing, which has already been done for
watermarking, merging, and splitting against real uploaded PDFs.
"""

import pytest

from app.tools.pdf_tools import _parse_page_ranges, _validate_pdf_input


class TestParsePageRanges:

    def test_single_range(self):
        assert _parse_page_ranges("1-3") == [0, 1, 2]

    def test_single_page(self):
        assert _parse_page_ranges("5") == [4]

    def test_mixed_ranges_and_single_pages(self):
        assert _parse_page_ranges("1-3,5,7-9") == [0, 1, 2, 4, 6, 7, 8]

    def test_result_is_sorted_and_deduplicated(self):
        # Overlapping ranges shouldn't produce duplicate indices.
        assert _parse_page_ranges("1-3,2-4") == [0, 1, 2, 3]

    def test_whitespace_is_tolerated(self):
        assert _parse_page_ranges("1 - 3, 5") == [0, 1, 2, 4]


class TestValidatePdfInput:

    def test_nonexistent_file_returns_error(self):
        path, error = _validate_pdf_input("definitely_not_a_real_file_xyz123.pdf")
        assert path is None
        assert error is not None
        assert "does not exist" in error.lower()

    def test_non_pdf_extension_returns_error(self):
        # This file exists (it's this test file) but isn't a PDF.
        path, error = _validate_pdf_input(__file__)
        assert path is None
        assert error is not None
        assert "not a pdf" in error.lower()

    def test_traversal_attempt_is_blocked(self):
        path, error = _validate_pdf_input("../../../etc/passwd.pdf")
        assert path is None
        assert error is not None