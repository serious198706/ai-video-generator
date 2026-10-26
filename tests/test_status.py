from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
os.environ["WAN22_DATA_DIR"] = str(ROOT / "data-test")
os.environ["WAN22_LOG_DIR"] = str(ROOT / "logs-test")
sys.path.insert(0, str(ROOT))

from wan22.media import status  # noqa: E402


class StatusWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = MagicMock()
        patcher = patch.object(status, "s3_client", return_value=self.s3)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher2 = patch.object(status.config, "S3_BUCKET", "bucket")
        patcher2.start()
        self.addCleanup(patcher2.stop)
        patcher3 = patch.object(status.config, "STATUS_PREFIX", "video/wan22/")
        patcher3.start()
        self.addCleanup(patcher3.stop)

    def test_key_uses_status_dir(self) -> None:
        self.assertEqual(status.key("abc"), "video/wan22/status/abc.json")

    def test_write_puts_single_key_with_timestamp(self) -> None:
        status.write("abc", "generating", resolution="720p", seed=None)
        kwargs = self.s3.put_object.call_args.kwargs
        self.assertEqual(kwargs["Bucket"], "bucket")
        self.assertEqual(kwargs["Key"], "video/wan22/status/abc.json")
        body = json.loads(kwargs["Body"])
        self.assertEqual(body["status"], "generating")
        self.assertEqual(body["resolution"], "720p")
        self.assertNotIn("seed", body)
        self.assertTrue(body["updated_at"])
        self.assertTrue(body["worker_id"])

    def test_write_swallows_s3_errors(self) -> None:
        self.s3.put_object.side_effect = RuntimeError("boom")
        status.write("abc", "pending")

    def test_write_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            status.write("abc", "weird")

    def test_write_is_noop_without_bucket(self) -> None:
        with patch.object(status.config, "S3_BUCKET", ""):
            status.write("abc", "pending")
        self.s3.put_object.assert_not_called()

    def test_terminal_write_carries_video_url(self) -> None:
        status.write("abc", "succeeded", video_url="https://cdn/x.mp4", seed=7)
        body = json.loads(self.s3.put_object.call_args.kwargs["Body"])
        self.assertEqual(body["video_url"], "https://cdn/x.mp4")
        self.assertEqual(body["seed"], 7)


if __name__ == "__main__":
    unittest.main()
