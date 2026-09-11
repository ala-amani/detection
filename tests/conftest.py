"""Ensure tests import the extension package beside this test directory."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
package_root = str(PACKAGE_ROOT)
if package_root in sys.path:
    sys.path.remove(package_root)
sys.path.insert(0, package_root)
