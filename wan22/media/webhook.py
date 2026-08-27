from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import requests

from wan22 import config
from wan22.log import get_logger
from wan22.net.urlguard import UrlError, assert_https_url

logger = get_logger(__name__)


def webhook_payload(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["id"]
    return {
        "id": task_id,
        "task_id": task_id,
        "status": task.get("status"),
        "video_url": task.get("video_url"),
        "error": task.get("error"),
        "seed": task.get("seed"),
        "duration": task.get("duration"),
        "resolution": task.get("resolution"),
    }


def notify(task: dict[str, Any]) -> None:
    url = task.get("webhook_url")
    if not url:
        return
    try:
        assert_https_url(
            url,
            config.WEBHOOK_HOSTS,
            kind="webhook",
            allow_private=True,
        )
    except UrlError:
        logger.warning("webhook skipped host not allowed task=%s", task.get("id"))
        return

    body = json.dumps(webhook_payload(task), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.WEBHOOK_SECRET:
        digest = hmac.new(
            config.WEBHOOK_SECRET.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-Wan-Signature"] = f"sha256={digest}"

    delays = [1, 2, 4][: max(0, config.WEBHOOK_RETRIES)]
    attempts = max(1, config.WEBHOOK_RETRIES)
    for index in range(attempts):
        try:
            response = requests.post(
                url,
                data=body,
                headers=headers,
                timeout=config.WEBHOOK_TIMEOUT,
                allow_redirects=False,
            )
            if 200 <= response.status_code < 300:
                logger.info(
                    "webhook %s task=%s status=%s",
                    task.get("status"),
                    task.get("id"),
                    response.status_code,
                )
                return
            if response.status_code < 500 and response.status_code != 429:
                logger.warning(
                    "webhook rejected task=%s status=%s",
                    task.get("id"),
                    response.status_code,
                )
                return
        except requests.RequestException:
            logger.exception(
                "webhook attempt %s/%s failed task=%s",
                index + 1,
                attempts,
                task.get("id"),
            )
        if index + 1 < attempts:
            time.sleep(delays[min(index, len(delays) - 1)])
    logger.error("webhook failed task=%s after %s attempts", task.get("id"), attempts)
