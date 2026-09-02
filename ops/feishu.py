#!/usr/bin/env python3
"""POST 文本到飞书群自定义机器人。无 webhook 时只打印。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request

_MAX = 18000


def _sign(secret: str, timestamp: str) -> str:
    raw = f"{timestamp}\n{secret}".encode()
    digest = hmac.new(secret.encode(), raw, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def send(text: str, webhook: str | None = None, secret: str | None = None) -> None:
    text = (text or "").strip()
    if not text:
        return
    if len(text) > _MAX:
        text = text[: _MAX - 20] + "\n…(截断)"
    webhook = (webhook or os.environ.get("WAN22_FEISHU_WEBHOOK") or "").strip()
    secret = (secret or os.environ.get("WAN22_FEISHU_SECRET") or "").strip()
    if not webhook:
        print("[feishu] WAN22_FEISHU_WEBHOOK empty, skip POST", flush=True)
        print(text, flush=True)
        return
    print(f"[feishu] POST {webhook[:48]}…", flush=True)
    body: dict = {"msg_type": "text", "content": {"text": text}}
    if secret:
        ts = str(int(time.time()))
        body["timestamp"] = ts
        body["sign"] = _sign(secret, ts)
    req = urllib.request.Request(
        webhook,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = resp.read().decode()
            print(f"[feishu] HTTP {resp.status} {payload}", flush=True)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode() if exc.fp else ""
        print(f"[feishu] HTTP {exc.code} {err_body}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except urllib.error.URLError as exc:
        print(f"[feishu] send failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {}
    code = parsed.get("code", parsed.get("Code", parsed.get("StatusCode", 0)))
    if code not in (0, "0", None):
        print(f"[feishu] API error {payload}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    print("[feishu] sent", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    args = parser.parse_args()
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    send(text)


if __name__ == "__main__":
    main()
