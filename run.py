
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Make the project root available to Python imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run the existing Streamlit application.
import streamlit.web.cli as stcli

sys.argv = [
    "streamlit",
    "run",
    str(ROOT / "app" / "ui" / "streaming.py"),
]

stcli.main()
