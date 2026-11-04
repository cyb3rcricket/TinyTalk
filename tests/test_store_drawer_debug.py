"""Failed fact saves stay silent unless TINYTALK_DEBUG is on.

The user-facing not_saved message is fixed. These tests stub add_drawer
and never open a real MemPalace store.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk


NOT_SAVED = {
    "status": "not_saved",
    "message": "Memory: not saved. MemPalace could not store this fact.",
}


class _Memory(object):
    """Just enough of a store for remember_fact to reach add_drawer."""

    def __init__(self, add_drawer):
        self.palace = object()
        self.kg = None
        self.add_drawer = add_drawer


class StoreDrawerDebugTests(unittest.TestCase):
    def setUp(self):
        self._saved_debug = os.environ.get("TINYTALK_DEBUG")

    def tearDown(self):
        if self._saved_debug is None:
            os.environ.pop("TINYTALK_DEBUG", None)
        else:
            os.environ["TINYTALK_DEBUG"] = self._saved_debug

    def _remember(self, add_drawer):
        out = io.StringIO()
        with redirect_stdout(out):
            result = tinytalk.remember_fact(
                "Remember this: my test spaceship is named Enterprise",
                _Memory(add_drawer),
                None,
            )
        return result, out.getvalue()

    def test_rejected_drawer_error_is_debugged_and_not_saved(self):
        os.environ["TINYTALK_DEBUG"] = "1"

        def add_drawer(**kwargs):
            return {"success": False, "error": "boom-marker"}

        result, output = self._remember(add_drawer)

        self.assertEqual(result, NOT_SAVED)
        self.assertIn("boom-marker", output)
        self.assertNotIn("boom-marker", result["message"])

    def test_add_drawer_exception_is_debugged_and_not_saved(self):
        os.environ["TINYTALK_DEBUG"] = "1"

        def add_drawer(**kwargs):
            raise RuntimeError("boom-marker")

        result, output = self._remember(add_drawer)

        self.assertEqual(result, NOT_SAVED)
        self.assertIn("boom-marker", output)
        self.assertNotIn("boom-marker", result["message"])

    def test_drawer_errors_stay_quiet_when_debug_is_off(self):
        os.environ.pop("TINYTALK_DEBUG", None)

        def add_drawer(**kwargs):
            return {"success": False, "error": "boom-marker"}

        result, output = self._remember(add_drawer)

        self.assertEqual(result, NOT_SAVED)
        self.assertNotIn("boom-marker", output)
