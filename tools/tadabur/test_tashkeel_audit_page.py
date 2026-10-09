"""Runs the listening page's client tests (``test_tashkeel_audit_session.mjs``) under Node.

The page's logic is a DOM-free module so it can be tested on its own: rapid taps, saves
completing out of order, a failed save and a resumed session. Skipped where Node is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).parent / "test_tashkeel_audit_session.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_logic_passes_its_client_tests():
    result = subprocess.run(["node", "--test", str(TESTS)], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_loads_its_logic_from_the_tested_module():
    page = (Path(__file__).parent / "tashkeel_audit_page.html").read_text(encoding="utf-8")
    assert 'import { createSession } from "/session.mjs";' in page
    assert "event.repeat" in page
