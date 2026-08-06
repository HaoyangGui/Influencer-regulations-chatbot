#!/usr/bin/env python3
"""Startup script for the RAG server that ensures proper module paths."""
import sys
from pathlib import Path

def main():
    # Add project root to sys.path
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    
    # Check if --reload was passed
    if "--reload" in sys.argv:
        import subprocess
        # Use uvicorn with reload from project root
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.server:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        subprocess.run(cmd, cwd=str(project_root))
    else:
        # Direct execution without reload
        from src.server import _run_server
        _run_server()

if __name__ == "__main__":
    main()
