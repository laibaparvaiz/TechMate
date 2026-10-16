"""
File: TechMate/tests/conftest.py
Description: Shared pytest configuration.

Ensures the project root is on sys.path so `from app...` imports
work regardless of how pytest is invoked (from the project root,
from within tests/, via an IDE's test runner, etc.).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))