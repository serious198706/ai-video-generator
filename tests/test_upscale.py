from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
os.environ.setdefault("WAN22_UPSCALE_ENABLE", "0")
sys.path.insert(0, str(ROOT))

from wan22.infer.upscale import snap_4n1, target_max_edge, target_short_side  # noqa: E402


class UpscaleHelperTests(unittest.TestCase):
    def test_snap_4n1(self):
        self.assertEqual(snap_4n1(1), 1)
        self.assertEqual(snap_4n1(5), 5)
        self.assertEqual(snap_4n1(20), 17)
        self.assertEqual(snap_4n1(21), 21)

    def test_720p_portrait_scales_to_1080_short_side(self):
        self.assertEqual(target_short_side(720, 1248), 1080)
        self.assertEqual(target_max_edge(720, 1248), 1872)

    def test_square_720p_scales_to_1440(self):
        self.assertEqual(target_short_side(960, 960), 1440)
        self.assertEqual(target_max_edge(960, 960), 1440)


if __name__ == "__main__":
    unittest.main()
