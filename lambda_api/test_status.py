from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DATA_DIR", "/tmp/wan22-lambda-status-test")
os.environ["S3_BUCKET"] = ""  # 单测不写真实 S3
sys.path.insert(0, str(ROOT))

from botocore.exceptions import ClientError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lambda_api import app as lambda_app  # noqa: E402
from lambda_api import status as lambda_status  # noqa: E402


def _s3_body(payload: dict) -> dict:
    return {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}


class LambdaStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = MagicMock()
        for patcher in (
            patch.object(lambda_status, "_s3", self.s3),
            patch.object(lambda_status, "S3_BUCKET", "bucket"),
            patch.object(lambda_status, "STATUS_PREFIX", "video/wan22"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_status_endpoint_returns_queued(self) -> None:
        self.s3.get_object.return_value = _s3_body(
            {"task_id": "t1", "status": "queued", "updated_at": "2026-10-26T00:00:00+00:00"}
        )
        with TestClient(lambda_app.app) as client:
            response = client.get("/v1/tasks/t1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["id"], "t1")
        self.assertEqual(self.s3.get_object.call_args.kwargs["Key"], "video/wan22/status/t1.json")

    def test_missing_object_is_unknown(self) -> None:
        self.s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        with TestClient(lambda_app.app) as client:
            response = client.get("/v1/tasks/nope")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unknown")

    def test_s3_error_is_503(self) -> None:
        self.s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject"
        )
        with TestClient(lambda_app.app) as client:
            response = client.get("/v1/tasks/t1")
        self.assertEqual(response.status_code, 503)

    def test_stale_generating_reports_stalled(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        self.s3.get_object.return_value = _s3_body(
            {"task_id": "t1", "status": "generating", "updated_at": old}
        )
        with patch.object(lambda_status, "STALL_SECONDS", 60):
            with TestClient(lambda_app.app) as client:
                response = client.get("/v1/tasks/t1")
        body = response.json()
        self.assertEqual(body["status"], "stalled")
        self.assertEqual(body["error"], "worker_unresponsive")

    def test_fresh_generating_not_stalled(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.s3.get_object.return_value = _s3_body(
            {"task_id": "t1", "status": "generating", "updated_at": now}
        )
        with patch.object(lambda_status, "STALL_SECONDS", 60):
            with TestClient(lambda_app.app) as client:
                response = client.get("/v1/tasks/t1")
        self.assertEqual(response.json()["status"], "generating")

    def test_succeeded_not_affected_by_stall(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self.s3.get_object.return_value = _s3_body(
            {"task_id": "t1", "status": "succeeded", "updated_at": old, "video_url": "https://x/y.mp4"}
        )
        with patch.object(lambda_status, "STALL_SECONDS", 60):
            with TestClient(lambda_app.app) as client:
                response = client.get("/v1/tasks/t1")
        self.assertEqual(response.json()["status"], "succeeded")
        self.assertEqual(response.json()["video_url"], "https://x/y.mp4")

    def test_enqueue_writes_queued(self) -> None:
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda v, *a, **k: v),
        ):
            sqs.send_message.return_value = {"MessageId": "m1"}
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/a.jpg", "duration": 5},
                )
        task_id = response.json()["task_id"]
        self.s3.put_object.assert_called_once()
        kwargs = self.s3.put_object.call_args.kwargs
        self.assertEqual(kwargs["Key"], f"video/wan22/status/{task_id}.json")
        self.assertEqual(json.loads(kwargs["Body"])["status"], "queued")

    def test_enqueue_writes_queued_before_send(self) -> None:
        order: list[str] = []
        self.s3.put_object.side_effect = lambda *a, **k: order.append("queued")
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda v, *a, **k: v),
        ):
            def _send(*a, **k):
                order.append("send")
                return {"MessageId": "m1"}

            sqs.send_message.side_effect = _send
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/a.jpg", "duration": 5},
                )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(order, ["queued", "send"])

    def test_enqueue_failure_writes_failed(self) -> None:
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda v, *a, **k: v),
        ):
            sqs.send_message.side_effect = RuntimeError("boom")
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/a.jpg", "duration": 5},
                )
        self.assertEqual(response.status_code, 503)
        writes = [json.loads(call.kwargs["Body"]) for call in self.s3.put_object.call_args_list]
        self.assertEqual([w["status"] for w in writes], ["queued", "failed"])
        self.assertEqual(writes[-1]["error"], "enqueue_failed")


if __name__ == "__main__":
    unittest.main()
