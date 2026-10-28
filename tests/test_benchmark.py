from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark
from wan22.infer.upscale_sidecar import _write_rgb_mp4


class BenchmarkTests(unittest.TestCase):
    def test_zero_warmup_requires_diagnostic_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            prompt = Path(directory) / "prompt.txt"
            image.touch()
            prompt.write_text("A slow camera pan")
            args = ["--image", str(image), "--prompt-file", str(prompt), "--output", str(Path(directory) / "out"), "--warmup", "0", "--runs", "1"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.arguments(args)
            parsed = benchmark.arguments(args + ["--profile", "on"])
            self.assertEqual(parsed.warmup, 0)
            self.assertEqual(parsed.runs, 1)
            parsed = benchmark.arguments(args + ["--block-profile", "--residency", "on"])
            self.assertTrue(parsed.block_profile)
            for incompatible in (["--profile", "on"], ["--compile", "default"], ["--resolution", "1080p"], ["--residency", "off"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    benchmark.arguments(args + ["--block-profile", "--residency", "on"] + incompatible)

    def test_rejects_compile_with_instrumentation_or_offload(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            prompt = Path(directory) / "prompt.txt"
            image.touch()
            prompt.write_text("A slow camera pan")
            args = ["--image", str(image), "--prompt-file", str(prompt), "--output", str(Path(directory) / "out"), "--compile", "default"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.arguments(args)
            parsed = benchmark.arguments(args + ["--attention", "sdpa", "--residency", "on"])
            self.assertEqual(parsed.compile, "default")

    def test_real_ffmpeg_export_decode_frame_count_and_contact_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            frames = [np.full((16, 32, 3), fill_value=value, dtype=np.uint8) for value in (30, 80, 130, 180)]
            _write_rgb_mp4(path, frames, fps=16)
            result = benchmark.inspect_video(path, 4)
            self.assertEqual(result, {"frames": 4, "size": (32, 16), "fps": 16})
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 3)
            with self.assertRaisesRegex(ValueError, "frame count"):
                benchmark.inspect_video(path, 5)

    def test_report_is_json_and_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            benchmark.save_report(path, {"status": "running"})
            benchmark.save_report(path, {"status": "failed", "error": "OOM"})
            self.assertEqual(json.loads(path.read_text())["error"], "OOM")
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
