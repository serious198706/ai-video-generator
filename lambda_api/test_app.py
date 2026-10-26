from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DATA_DIR", "/tmp/wan22-lambda-test-data")
os.environ["S3_BUCKET"] = ""  # 单测不写真实 S3
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from lambda_api import app as lambda_app  # noqa: E402


class LambdaQueueApiTests(unittest.TestCase):
    def test_generate_enqueues_task(self) -> None:
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda value, *_args, **_kwargs: value),
        ):
            sqs.send_message.return_value = {"MessageId": "message-1"}
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/input.jpg", "duration": 5},
                )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(response.json()["id"], response.json()["task_id"])
        kwargs = sqs.send_message.call_args.kwargs
        self.assertEqual(kwargs["QueueUrl"], "https://sqs.test/queue")
        task = json.loads(kwargs["MessageBody"])
        self.assertEqual(task["task_id"], response.json()["task_id"])
        self.assertEqual(task["image_url"], "https://images.example.com/input.jpg")
        self.assertEqual(task["duration"], 5)

    def test_queue_limit_returns_429_without_sending(self) -> None:
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "SQS_MAX_QUEUE", 2),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda value, *_args, **_kwargs: value),
        ):
            sqs.get_queue_attributes.return_value = {
                "Attributes": {"ApproximateNumberOfMessages": "2"}
            }
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/input.jpg"},
                )

        self.assertEqual(response.status_code, 429)
        sqs.send_message.assert_not_called()

    def test_queue_below_limit_enqueues(self) -> None:
        with (
            patch.object(lambda_app, "QUEUE_URL", "https://sqs.test/queue"),
            patch.object(lambda_app, "SQS_MAX_QUEUE", 2),
            patch.object(lambda_app, "_sqs") as sqs,
            patch.object(lambda_app, "assert_image_source", side_effect=lambda value, *_args, **_kwargs: value),
        ):
            sqs.get_queue_attributes.return_value = {
                "Attributes": {"ApproximateNumberOfMessages": "1"}
            }
            sqs.send_message.return_value = {"MessageId": "message-1"}
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/input.jpg"},
                )

        self.assertEqual(response.status_code, 202)
        sqs.send_message.assert_called_once()

    def test_missing_queue_returns_503(self) -> None:
        with patch.object(lambda_app, "QUEUE_URL", ""):
            with TestClient(lambda_app.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={"image": "https://images.example.com/input.jpg"},
                )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
