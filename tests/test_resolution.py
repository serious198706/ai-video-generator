from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
sys.path.insert(0, str(ROOT))

from wan22.infer.generate import dims_for_resolution, generate_dims  # noqa: E402


class ResolutionDimsTests(unittest.TestCase):
    def test_480p_matches_current_canvas(self):
        self.assertEqual(dims_for_resolution("480p"), (480, 832, 640))

    def test_720p_and_1080p_scale_the_480p_canvas(self):
        self.assertEqual(dims_for_resolution("720p"), (720, 1248, 960))
        self.assertEqual(dims_for_resolution("1080p"), (1080, 1872, 1440))

    def test_1080p_generates_on_720p_canvas(self):
        self.assertEqual(generate_dims("1080p"), dims_for_resolution("720p"))
        self.assertEqual(generate_dims("720p"), dims_for_resolution("720p"))
        self.assertEqual(generate_dims("480p"), dims_for_resolution("480p"))

    def test_missing_resolution_keeps_480p(self):
        self.assertEqual(dims_for_resolution(None), (480, 832, 640))
        self.assertEqual(dims_for_resolution("540p"), (480, 832, 640))


if __name__ == "__main__":
    unittest.main()
