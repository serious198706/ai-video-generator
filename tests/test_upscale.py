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

from wan22.infer.upscale import (  # noqa: E402
    decode_json_line,
    even,
    target_max_edge,
    target_short_side,
    target_size,
)


class UpscaleHelperTests(unittest.TestCase):
    def test_720p_portrait_scales_to_1080_short_side(self):
        self.assertEqual(target_short_side(720, 1248), 1080)
        self.assertEqual(target_max_edge(720, 1248), 1872)
        self.assertEqual(target_size(720, 1248, 1.5), (1080, 1872))
        self.assertEqual(even(1873), 1872)

    def test_square_720p_scales_to_1440(self):
        self.assertEqual(target_short_side(960, 960), 1440)
        self.assertEqual(target_max_edge(960, 960), 1440)
        self.assertEqual(target_size(960, 960, 1.5), (1440, 1440))

    def test_decode_json_line_skips_non_json(self):
        self.assertIsNone(decode_json_line("not json\n"))
        self.assertEqual(decode_json_line('{"ok": true, "ready": true}\n'), {"ok": True, "ready": True})


if __name__ == "__main__":
    unittest.main()
