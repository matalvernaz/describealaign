"""
Test infrastructure: ensure the in-tree describealaign module is on the path
so tests run against the working copy, not whatever happens to be installed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
