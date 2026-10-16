"""
backend/app/__init__.py
AnimeRecSys AI backend application package.
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent.parent)
_workspace_root = str(Path(__file__).resolve().parent.parent.parent)

if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
