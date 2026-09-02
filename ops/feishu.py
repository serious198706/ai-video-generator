#!/usr/bin/env python3
"""POST 飞书消息卡片（interactive）。无 webhook 时只打印。"""

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

_MD_MAX = 5000
_TITLE_MAX = 50
_NOTE_MAX = 200


def _sign(secret: str, timestamp: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 8] + "\n…(截断)"


def build_card(
    title: str,
    template: str,
    sections: list[str],
    note: str | None = None,
) -> dict:
    elements: list[dict] = []
    chunks = [s for s in sections if (s or "").strip()]
    if not chunks:
        chunks = ["（无内容）"]
    for i, section in enumerate(chunks):
        if i:
            elements.append({"tag": "hr"})
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": _clip(section, _MD_MAX)},
            }
        )
    if note and note.strip():
        elements.append(
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": _clip(note.strip(), _NOTE_MAX)}],
            }
        )
    color = (template or "green").strip() or "green"
    return {
        "header": {
            "title": {"tag": "plain_text", "content": _clip(title, _TITLE_MAX).replace("\n", " ")},
            "template": color,
        },
        "elements": elements,
    }


def post(payload: dict, webhook: str | None = None, secret: str | None = None) -> None:
    webhook = (webhook or os.environ.get("WAN22_FEISHU_WEBHOOK") or "").strip()
    secret = (secret or os.environ.get("WAN22_FEISHU_SECRET") or "").strip()
    if not webhook:
        print("[feishu] WAN22_FEISHU_WEBHOOK empty, skip POST", flush=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    print(f"[feishu] POST {webhook[:48]}…", flush=True)
    body: dict = dict(payload)
    if secret:
        ts = str(int(time.time()))
        body["timestamp"] = ts
        body["sign"] = _sign(secret, ts)
        print(f"[feishu] signing timestamp={ts}", flush=True)
    req = urllib.request.Request(
        webhook,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode()
            print(f"[feishu] HTTP {resp.status} {raw}", flush=True)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode() if exc.fp else ""
        print(f"[feishu] HTTP {exc.code} {err_body}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except urllib.error.URLError as exc:
        print(f"[feishu] send failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    code = parsed.get("code", parsed.get("Code", parsed.get("StatusCode", 0)))
    if code not in (0, "0", None):
        print(f"[feishu] API error {raw}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    print("[feishu] sent", flush=True)


def send_card(
    title: str,
    sections: list[str],
    *,
    template: str = "green",
    note: str | None = None,
    webhook: str | None = None,
    secret: str | None = None,
) -> None:
    post(
        {"msg_type": "interactive", "card": build_card(title, template, sections, note)},
        webhook=webhook,
        secret=secret,
    )


def send(text: str, webhook: str | None = None, secret: str | None = None) -> None:
    text = (text or "").strip()
    if not text:
        return
    send_card("Wan22", [text], template="grey", webhook=webhook, secret=secret)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Wan22")
    parser.add_argument("--template", default="green")
    parser.add_argument("--note", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--json", action="store_true", help="stdin 为 digest 的 card JSON")
    args = parser.parse_args()
    if args.json:
        spec = json.loads(sys.stdin.read() or "{}")
        send_card(
            spec.get("title") or args.title,
            list(spec.get("sections") or []),
            template=spec.get("template") or args.template,
            note=spec.get("note") or args.note,
        )
        return
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    send_card(args.title, [text], template=args.template, note=args.note or None)


if __name__ == "__main__":
    main()
