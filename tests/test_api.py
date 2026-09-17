from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")
os.environ.setdefault("WAN22_FOLEY_ENABLE", "0")
os.environ.setdefault("WAN22_UPSCALE_ENABLE", "0")
os.environ.setdefault("WAN22_DOCS", "0")
os.environ.setdefault("WAN22_IMAGE_HOSTS", "")
os.environ.setdefault("WAN22_WEBHOOK_HOSTS", "")
os.environ["WAN22_DATA_DIR"] = str(ROOT / "data-test")
os.environ["WAN22_LOG_DIR"] = str(ROOT / "logs-test")
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from wan22 import config  # noqa: E402
from wan22.api.app import app  # noqa: E402
from wan22.net.urlguard import UrlError  # noqa: E402
from wan22.tasks import runner, store  # noqa: E402


def _fake_https(url, _allowlist, *, kind, allow_private=False):
    if not str(url).startswith("https://"):
        raise UrlError(f"{kind} only https allowed")
    return url


def _fake_download(url, dest: Path) -> str:
    path = Path(dest).with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")
    return str(path)


def _poll(client, task_id: str, tries: int = 100) -> dict:
    task = None
    for _ in range(tries):
        polled = client.get(f"/v1/tasks/{task_id}")
        assert polled.status_code == 200, polled.text
        task = polled.json()
        if task["status"] in {"succeeded", "failed"}:
            return task
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never finished: {task}")


class GenerateApiTests(unittest.TestCase):
    def test_post_generate_returns_202_and_can_be_polled(self):
        with (
            patch("wan22.api.app.assert_image_source", side_effect=_fake_https),
            patch("wan22.media.download.download_image", side_effect=_fake_download),
            TestClient(app) as client,
        ):
            created = client.post(
                "/v1/generate",
                json={
                    "image": "https://cdn.example.com/a.jpg",
                    "prompt": "turn her head",
                    "duration": 5,
                    "audio": False,
                },
            )
            self.assertEqual(created.status_code, 202, created.text)
            body = created.json()
            self.assertEqual(body["status"], "running")
            self.assertEqual(body["id"], body["task_id"])
            self.assertTrue(body["id"])

            task = _poll(client, body["id"])
            self.assertEqual(task["status"], "succeeded")
            self.assertTrue(task["video_url"])
            self.assertEqual(task["prompt"], "turn her head")

            # SQLite 留档：进程里查不到内存态也能从库里读回来
            stored = store.get_task(body["id"])
            self.assertIsNotNone(stored)
            self.assertEqual(stored["status"], "succeeded")
            self.assertEqual(stored["worker_id"], config.WORKER_ID)

    def test_post_generate_returns_429_while_busy(self):
        with (
            patch("wan22.api.app.assert_image_source", side_effect=_fake_https),
            patch("wan22.media.download.download_image", side_effect=_fake_download),
            TestClient(app) as client,
        ):
            payload = {
                "image": "https://cdn.example.com/a.jpg",
                "prompt": "busy check",
                "duration": 5,
                "audio": False,
            }
            # 执行位被占住（相当于卡上正在跑），此时不排队，直接 429
            self.assertTrue(runner.reserve("busy-slot"))
            try:
                busy = client.post("/v1/generate", json=payload)
                self.assertEqual(busy.status_code, 429, busy.text)
            finally:
                runner.release("busy-slot")
            self.assertIsNone(runner.current())

            created = client.post("/v1/generate", json=payload)
            self.assertEqual(created.status_code, 202, created.text)
            task = _poll(client, created.json()["id"])
            self.assertEqual(task["status"], "succeeded")

    def test_second_post_during_generate_returns_429(self):
        def _slow_generate(**kwargs):
            time.sleep(0.5)
            return 123

        with (
            patch("wan22.api.app.assert_image_source", side_effect=_fake_https),
            patch("wan22.media.download.download_image", side_effect=_fake_download),
            patch("wan22.tasks.runner.generate.generate_video", side_effect=_slow_generate),
            TestClient(app) as client,
        ):
            payload = {
                "image": "https://cdn.example.com/a.jpg",
                "prompt": "slow generate",
                "duration": 5,
                "audio": False,
            }
            first = client.post("/v1/generate", json=payload)
            self.assertEqual(first.status_code, 202, first.text)
            second = client.post("/v1/generate", json=payload)
            self.assertEqual(second.status_code, 429, second.text)

            task = _poll(client, first.json()["id"])
            self.assertEqual(task["status"], "succeeded")
            self.assertEqual(task["seed"], 123)

    def test_interrupted_running_task_fails_on_restart(self):
        task_id = "interruptedtask0000000000000000ff"
        store.create_task(
            task_id,
            {
                "prompt": "left running",
                "image_url": "https://cdn.example.com/a.jpg",
                "duration": 5,
                "audio": False,
            },
        )
        with TestClient(app) as client:
            polled = client.get(f"/v1/tasks/{task_id}")
            self.assertEqual(polled.status_code, 200, polled.text)
            body = polled.json()
            self.assertEqual(body["status"], "failed")
            self.assertEqual(body["error"], "interrupted")

    def test_post_generate_accepts_local_image_and_480p(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "test.jpg"
            Image.new("RGB", (32, 48), "red").save(image_path)
            with TestClient(app) as client:
                created = client.post(
                    "/v1/generate",
                    json={
                        "image": str(image_path),
                        "prompt": "local bench",
                        "duration": 5,
                        "resolution": "480p",
                        "steps": 4,
                        "audio": False,
                    },
                )
                self.assertEqual(created.status_code, 202, created.text)
                task = _poll(client, created.json()["id"])
                self.assertEqual(task["status"], "succeeded")
                self.assertEqual(task["resolution"], "480p")

    def test_post_generate_accepts_1080p_in_dry_run(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "test.jpg"
            Image.new("RGB", (32, 48), "red").save(image_path)
            with TestClient(app) as client:
                created = client.post(
                    "/v1/generate",
                    json={
                        "image": str(image_path),
                        "prompt": "1080p dry-run",
                        "duration": 5,
                        "resolution": "1080p",
                        "steps": 4,
                        "audio": False,
                    },
                )
                self.assertEqual(created.status_code, 202, created.text)
                task = _poll(client, created.json()["id"])
                self.assertEqual(task["status"], "succeeded")
                self.assertEqual(task["resolution"], "1080p")

    def test_1080p_without_upscale_returns_503(self):
        with (
            patch("wan22.api.app.assert_image_source", side_effect=_fake_https),
            patch("wan22.api.app.assert_s3"),
            patch("wan22.api.app.config.DRY_RUN", False),
            patch("wan22.api.app.config.UPSCALE_ENABLE", False),
            TestClient(app) as client,
        ):
            created = client.post(
                "/v1/generate",
                json={
                    "image": "https://cdn.example.com/a.jpg",
                    "prompt": "need 1080p",
                    "duration": 5,
                    "resolution": "1080p",
                    "audio": False,
                },
            )
            self.assertEqual(created.status_code, 503, created.text)

    def test_health_stays_dependency_free(self):
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200, health.text)
            body = health.json()
            self.assertTrue(body["ok"])
            self.assertIn("model_ready", body)


if __name__ == "__main__":
    unittest.main()
