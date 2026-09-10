from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
os.environ.setdefault("WAN22_UPSCALE_ENABLE", "0")
sys.path.insert(0, str(ROOT))

from wan22.infer.upscale import (  # noqa: E402
    decode_json_line,
    snap_4n1,
    target_max_edge,
    target_short_side,
)


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

    def test_decode_json_line_skips_seedvr2_tips(self):
        self.assertIsNone(decode_json_line("💡 Optional: pip install sageattention flash-attn\n"))
        self.assertEqual(decode_json_line('{"ok": true, "ready": true}\n'), {"ok": True, "ready": True})

    def test_imageio_ffmpeg_binary_is_exposed_as_ffmpeg(self):
        from wan22.infer.upscale_sidecar import _link_named_ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ffmpeg-linux-x86_64-v7.0.2"
            src.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            src.chmod(0o755)
            bindir = Path(tmp) / "shim"
            link = _link_named_ffmpeg(src, bindir)
            self.assertEqual(link.name, "ffmpeg")
            self.assertTrue(link.exists())
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = str(bindir) + os.pathsep + old
            try:
                self.assertEqual(Path(shutil.which("ffmpeg") or "").resolve(), src.resolve())
            finally:
                os.environ["PATH"] = old


if __name__ == "__main__":
    unittest.main()
