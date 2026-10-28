from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wan22.infer import upscale, upscale_sidecar


class FakeSR(torch.nn.Module):
    """A real tensor operation exercises precision, layout, clipping and resize on CPU."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def model(self):
        return self

    def forward(self, tensor):
        return functional.interpolate(tensor, scale_factor=4, mode="nearest") * 1.2 - 0.1


class UpscaleFramesTests(unittest.TestCase):
    def test_default_output_matches_original_uint8_frames_exactly(self):
        rgb = np.random.default_rng(42).integers(0, 256, (12, 20, 3), dtype=np.uint8)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            desc = FakeSR().to(dtype=dtype)
            with self.subTest(dtype=dtype), torch.inference_mode():
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255).unsqueeze(0).to(dtype=dtype)
                original = desc(tensor).float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                expected = (original * 255).round().astype(np.uint8)
                actual = upscale_sidecar._sr_frame(desc, rgb)
                np.testing.assert_array_equal(expected, actual)

    def test_opt_in_resize_returns_final_size_and_valid_rgb(self):
        rgb = np.random.default_rng(1).integers(0, 256, (12, 20, 3), dtype=np.uint8)
        with torch.inference_mode():
            out = upscale_sidecar._sr_frame(FakeSR(), rgb, size=(30, 18))
        self.assertEqual(out.shape, (18, 30, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertGreater(int(out.max()), int(out.min()))

    def test_release_acknowledges_cpu_residency_and_reuses_process(self):
        proc = Mock()
        with patch.object(upscale, "_proc", proc), patch.object(upscale, "_write") as write, patch.object(upscale, "_readline", return_value={"ok": True, "device": "cpu"}), patch.object(upscale, "_kill") as kill:
            upscale.release_gpu()
            write.assert_called_once_with({"cmd": "offload"})
            kill.assert_not_called()

    def test_failed_offload_terminates_process_before_wan_can_resume(self):
        for response in ({"ok": False}, {"ok": True, "device": "cuda"}):
            with self.subTest(response=response), patch.object(upscale, "_proc", Mock()), patch.object(upscale, "_write"), patch.object(upscale, "_readline", return_value=response), patch.object(upscale, "_kill") as kill:
                upscale.release_gpu()
                kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
