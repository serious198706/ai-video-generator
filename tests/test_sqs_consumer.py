from __future__ import annotations

import json
import os
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
os.environ.setdefault("WAN22_FOLEY_ENABLE", "0")
os.environ.setdefault("WAN22_UPSCALE_ENABLE", "0")
os.environ.setdefault("WAN22_IMAGE_HOSTS", "")
os.environ.setdefault("WAN22_WEBHOOK_HOSTS", "")
os.environ["WAN22_S3_BUCKET"] = ""  # 单测不写真实 S3；status 走 mock
os.environ["WAN22_DATA_DIR"] = str(ROOT / "data-test")
os.environ["WAN22_LOG_DIR"] = str(ROOT / "logs-test")
sys.path.insert(0, str(ROOT))

from wan22.net.urlguard import UrlError  # noqa: E402
from wan22.tasks import runner, sqs_consumer  # noqa: E402


def _fake_https(url, _allowlist, *, kind, allow_private=False):
    if not str(url).startswith("https://"):
        raise UrlError(f"{kind} only https allowed")
    return url


def _fake_download(url, dest: Path) -> str:
    path = Path(dest).with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")
    return str(path)


def _message(body: dict, receipt: str = "r1", receive: int = 1) -> dict:
    return {
        "MessageId": uuid.uuid4().hex,
        "ReceiptHandle": receipt,
        "Body": json.dumps(body),
        "Attributes": {"ApproximateReceiveCount": str(receive)},
    }


def _wait_current_gone(task_id: str, tries: int = 300) -> None:
    for _ in range(tries):
        if runner.current() != task_id:
            return
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} still holding the slot")


def _wait_called(mock, tries: int = 200) -> None:
    for _ in range(tries):
        if mock.called:
            return
        time.sleep(0.02)
    raise AssertionError("mock never called")


class SqsConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        sqs_consumer._stop.clear()
        # patch 共享模块函数：consumer 和 runner 各自 import 的 status 都指向这里。
        self.mock_is_terminal = patch("wan22.media.status.is_terminal", return_value=False).start()
        self.mock_write = patch("wan22.media.status.write").start()
        self.addCleanup(patch.stopall)

    def tearDown(self) -> None:
        sqs_consumer.stop()

    def test_parse_accepts_lambda_body(self) -> None:
        body = {
            "task_id": "t1",
            "image_url": "https://cdn.example.com/a.jpg",
            "prompt": "turn",
            "duration": 5,
            "resolution": "540p",
            "webhook_url": "https://api.example.com/cb",
            "audio": True,
        }
        with patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https):
            task_id, request, error = sqs_consumer._parse(json.dumps(body))
        self.assertIsNone(error)
        self.assertEqual(task_id, "t1")
        self.assertEqual(request.image, "https://cdn.example.com/a.jpg")
        self.assertTrue(request.audio)

    def test_parse_rejects_bad_body(self) -> None:
        _, _, error = sqs_consumer._parse("not json")
        self.assertEqual(error, "body is not json")
        with patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https):
            _, _, error = sqs_consumer._parse(json.dumps({"task_id": "t", "image_url": "x"}))
        self.assertIsNotNone(error)

    def test_handle_runs_task_and_deletes_message(self) -> None:
        task_id = uuid.uuid4().hex
        message = _message(
            {
                "task_id": task_id,
                "image_url": "https://cdn.example.com/a.jpg",
                "prompt": "from sqs",
                "duration": 5,
            }
        )
        sqs = MagicMock()
        with (
            patch.object(sqs_consumer, "_sqs", return_value=sqs),
            patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https),
            patch.object(sqs_consumer.config, "SQS_HEARTBEAT_SECONDS", 0.05),
            patch("wan22.media.download.download_image", side_effect=_fake_download),
        ):
            sqs_consumer._handle(message)
            _wait_current_gone(task_id)
            _wait_called(sqs.delete_message)

        # pending -> generating -> succeeded 全写进了状态文件
        written = [call.args[1] for call in self.mock_write.call_args_list]
        self.assertEqual(written, ["pending", "generating", "succeeded"])
        self.assertEqual(sqs.delete_message.call_args.kwargs["ReceiptHandle"], "r1")

        # 终态带计时与基本参数
        final = self.mock_write.call_args
        self.assertEqual(final.args[1], "succeeded")
        fields = final.kwargs
        for key in ("seed", "video_url", "started_at", "finished_at", "duration_s",
                    "generate_s", "upscale_s", "foley_s", "upload_s"):
            self.assertIn(key, fields, f"missing {key} in succeeded status")
        self.assertIsInstance(fields["duration_s"], float)

    def test_handle_duplicate_terminal_deletes(self) -> None:
        task_id = uuid.uuid4().hex
        self.mock_is_terminal.return_value = True
        message = _message({"task_id": task_id, "image_url": "https://cdn.example.com/a.jpg"})
        sqs = MagicMock()
        with (
            patch.object(sqs_consumer, "_sqs", return_value=sqs),
            patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https),
        ):
            sqs_consumer._handle(message)
        sqs.delete_message.assert_called_once()
        runner.release(task_id)  # 不应占过位，保险起见

    def test_handle_busy_requeues_without_spawning(self) -> None:
        task_id = uuid.uuid4().hex
        message = _message({"task_id": task_id, "image_url": "https://cdn.example.com/a.jpg"})
        sqs = MagicMock()
        self.assertTrue(runner.reserve("http-slot"))
        try:
            with (
                patch.object(sqs_consumer, "_sqs", return_value=sqs),
                patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https),
            ):
                sqs_consumer._handle(message)
        finally:
            runner.release("http-slot")
        self.assertIsNone(runner.current())
        self.mock_write.assert_not_called()
        sqs.delete_message.assert_not_called()
        sqs.change_message_visibility.assert_called_once()

    def test_poison_message_failed_and_deleted(self) -> None:
        task_id = uuid.uuid4().hex
        message = _message(
            {"task_id": task_id, "image_url": "https://cdn.example.com/a.jpg"},
            receive=99,
        )
        sqs = MagicMock()
        with (
            patch.object(sqs_consumer, "_sqs", return_value=sqs),
            patch("wan22.tasks.sqs_consumer.assert_image_source", side_effect=_fake_https),
        ):
            sqs_consumer._handle(message)
        self.mock_write.assert_called_once_with(task_id, "failed", error="poisoned")
        sqs.delete_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
