from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")

from wan22 import config
from wan22.infer import upscale
from wan22.tasks import runner


class PostprocessGpuTests(unittest.TestCase):
    def run_task(self, *, shared=True, audio=False, resolution="1080p", errors=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(config, "DRY_RUN", False))
            stack.enter_context(patch.object(config, "FOLEY_ENABLE", audio))
            stack.enter_context(patch.object(runner, "_ensure_image", return_value=None))
            stack.enter_context(patch.object(runner, "_cleanup_frames"))
            stack.enter_context(patch.object(runner.status, "write"))
            stack.enter_context(patch.object(runner.generate, "generate_video", return_value=42))
            share = stack.enter_context(patch.object(runner.generate, "can_share_gpu_for_upscale", return_value=shared))
            events = Mock()
            for owner, attr, label in (
                (runner.generate, "pause_gpu", "pause"), (runner.generate, "resume_gpu", "resume"),
                (runner, "upscale_video", "upscale"), (runner, "stop_upscale", "stop"),
                (runner, "add_audio", "foley"), (runner, "upload_video", "upload"),
            ):
                mock = stack.enter_context(patch.object(owner, attr))
                events.attach_mock(mock, label)
            events.upscale.side_effect = errors
            events.upload.return_value = "https://example.com/video.mp4"
            events.foley.return_value = True
            result = runner._run({"id": "gpu-policy-test", "prompt": "test", "duration": 15, "resolution": resolution, "audio": audio, "steps": 4})
            return result, events, share

    def test_shared_upscale_skips_transfers_and_releases_sidecar(self):
        result, events, _ = self.run_task()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual([call[0] for call in events.mock_calls], ["upscale", "stop", "upload"])

    def test_shared_oom_retries_once_after_pausing_wan(self):
        result, events, _ = self.run_task(errors=[upscale.UpscaleOutOfMemory("OOM"), None])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual([call[0] for call in events.mock_calls], ["upscale", "stop", "pause", "upscale", "stop", "resume", "upload"])

    def test_second_oom_fails_and_restores_wan_without_upload(self):
        result, events, _ = self.run_task(errors=[upscale.UpscaleOutOfMemory("OOM"), upscale.UpscaleOutOfMemory("OOM")])
        self.assertEqual(result["error"], "upscale_failed")
        self.assertEqual(events.upscale.call_count, 2)
        events.resume.assert_called_once()
        events.upload.assert_not_called()

    def test_other_errors_are_not_retried(self):
        result, events, _ = self.run_task(errors=upscale.UpscaleError("bad video"))
        self.assertEqual(result["error"], "upscale_failed")
        events.upscale.assert_called_once()
        events.pause.assert_not_called()
        events.stop.assert_called_once()

    def test_low_memory_uses_original_transfer_path(self):
        result, events, _ = self.run_task(shared=False)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual([call[0] for call in events.mock_calls], ["pause", "upscale", "resume", "upload"])

    def test_foley_keeps_original_transfer_policy(self):
        result, events, share = self.run_task(audio=True)
        self.assertEqual(result["status"], "succeeded")
        share.assert_not_called()
        self.assertEqual([call[0] for call in events.mock_calls], ["pause", "upscale", "foley", "resume", "upload"])

    def test_720p_does_not_use_upscale_or_transfers(self):
        result, events, share = self.run_task(resolution="720p")
        self.assertEqual(result["status"], "succeeded")
        share.assert_not_called()
        self.assertEqual([call[0] for call in events.mock_calls], ["upload"])

    def test_sidecar_oom_reply_is_typed_and_other_errors_stay_generic(self):
        with patch.object(upscale, "_write"), patch.object(upscale, "_readline", return_value={"ok": False, "error_type": "cuda_oom", "error": "OOM"}):
            with self.assertRaises(upscale.UpscaleOutOfMemory):
                upscale._request({})
        with patch.object(upscale, "_write"), patch.object(upscale, "_readline", return_value={"ok": False, "error": "bad video"}):
            with self.assertRaises(upscale.UpscaleError) as ctx:
                upscale._request({})
            self.assertNotIsInstance(ctx.exception, upscale.UpscaleOutOfMemory)


if __name__ == "__main__":
    unittest.main()
