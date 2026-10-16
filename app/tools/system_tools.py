"""
File: TechMate/app/tools/systems_tools.py
Description: Safe environment inspection tools for folder navigation and dependency checking.
"""

import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool

from app.safety.permissions import is_command_safe
from app.config import BASE_DIR


@tool
def list_directory(relative_path: str = ".") -> str:
    """Lists all files and folders in a specified directory within the project workspace.
    Use this to understand the project structure before editing files.
    """
    try:
        target_path = (BASE_DIR / relative_path).resolve()
        
        if not is_command_safe(str(target_path)):
            return "Error: Cannot inspect directories outside the secure workspace."
            
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory '{relative_path}' does not exist."

        items = os.listdir(target_path)
        # Filter out hidden or blocked system files automatically
        safe_items = [f for f in items if not f.startswith(".") and f != "__pycache__"]
        
        if not safe_items:
            return f"Directory '{relative_path}' is empty."
            
        return "\n".join(safe_items)
    except Exception as e:
        return f"Error reading directory: {str(e)}"


@tool
def check_python_dependencies(package_name: str = "") -> str:
    """Checks if specific Python packages are installed in the current environment via pip freeze.
    Leave package_name empty to list all installed packages.
    """
    try:
        if package_name and not is_command_safe(package_name):
            return "Error: Invalid package name."

        result = subprocess.run(
            ["pip", "freeze"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode != 0:
            return f"Error checking dependencies: {result.stderr}"

        packages = result.stdout.split('\n')
        
        if package_name:
            matches = [p for p in packages if package_name.lower() in p.lower()]
            return "\n".join(matches) if matches else f"Package '{package_name}' not found."
            
        return "\n".join(packages)
    except Exception as e:
        return f"System error checking dependencies: {str(e)}"