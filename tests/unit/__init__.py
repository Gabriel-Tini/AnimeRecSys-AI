"""
tests/unit/__init__.py
Unit test package initialization for AnimeRecSys AI.
"""

import sys
from pathlib import Path

_root_dir = Path(__file__).resolve().parent.parent.parent
_backend_dir = _root_dir / "backend"

for p in [str(_root_dir), str(_backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
