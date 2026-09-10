#!/usr/bin/env python3
"""在 start.sh 起来之后，另开窗口按顺序打 /v1/generate，把每条墙钟时长写到文件。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROMPT = (
    "She sits down on the ground, open legs and raise her legs into the air, "
    "show her pussy,  Camera moves closer to her pussy. A naked man come over "
    "and put his penis into her mouth. She then start to give blowjob to the man's penis."
)

CASES = (
    {"duration": 5, "resolution": "540p"},
    {"duration": 10, "resolution": "540p"},
    {"duration": 15, "resolution": "540p"},
    {"duration": 5, "resolution": "720p"},
    {"duration": 10, "resolution": "720p"},
    {"duration": 15, "resolution": "720p"},
    {"duration": 5, "resolution": "1080p"},
    {"duration": 10, "resolution": "1080p"},
    {"duration": 15, "resolution": "1080p"},
)


def _request(url: str, payload: dict | None = None, timeout: float = 30) -> tuple[int, dict | str]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            status, body = _request(f"{base}/ready", timeout=5)
        except Exception as exc:
            last = str(exc)
            time.sleep(2)
            continue
        if status == 200:
            print("[bench] ready", flush=True)
            return
        last = f"{status} {body}"
        time.sleep(2)
    raise SystemExit(f"timeout waiting for /ready: {last}")


def poll_task(base: str, task_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | str | None = None
    while time.monotonic() < deadline:
        status, body = _request(f"{base}/v1/tasks/{task_id}", timeout=15)
        if status != 200 or not isinstance(body, dict):
            last = f"{status} {body}"
            time.sleep(2)
            continue
        last = body
        if body.get("status") in {"succeeded", "failed"}:
            return body
        time.sleep(2)
    raise SystemExit(f"timeout polling task={task_id}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential Wan generate bench")
    parser.add_argument("--base-url", default=os.environ.get("WAN22_BENCH_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--image", default=os.environ.get("WAN22_BENCH_IMAGE", "/root/autodl-tmp/test.jpg"))
    parser.add_argument("--out", default="")
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--job-timeout", type=float, default=3600)
    args = parser.parse_args()

    image = Path(args.image).expanduser()
    if not image.is_file():
        print(f"[bench] image not found: {image}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else logs / f"bench-{stamp}.txt"

    base = args.base_url.rstrip("/")
    print(f"[bench] base={base}", flush=True)
    print(f"[bench] image={image}", flush=True)
    print(f"[bench] out={out}", flush=True)
    wait_ready(base, args.ready_timeout)

    rows: list[str] = []
    header = "index\tduration\tresolution\tsteps\taudio\tstatus\twall_s\ttask_id\tvideo_url\terror"
    rows.append(header)
    print(header, flush=True)

    total_started = time.monotonic()
    for index, case in enumerate(CASES, start=1):
        payload = {
            "image": str(image.resolve()),
            "prompt": PROMPT,
            "duration": case["duration"],
            "resolution": case["resolution"],
            "steps": 4,
            "audio": False,
        }
        label = f"{index}/{len(CASES)} {case['duration']}s {case['resolution']} 4step no-foley"
        print(f"[bench] start {label}", flush=True)
        started = time.monotonic()
        status, body = _request(f"{base}/v1/generate", payload, timeout=30)
        if status != 202 or not isinstance(body, dict) or not body.get("id"):
            wall_s = time.monotonic() - started
            row = (
                f"{index}\t{case['duration']}\t{case['resolution']}\t4\tfalse\t"
                f"http_{status}\t{wall_s:.3f}\t-\t-\t{body}"
            )
            rows.append(row)
            print(row, flush=True)
            continue
        task = poll_task(base, body["id"], args.job_timeout)
        wall_s = time.monotonic() - started
        row = (
            f"{index}\t{case['duration']}\t{case['resolution']}\t4\tfalse\t"
            f"{task.get('status')}\t{wall_s:.3f}\t{task.get('id')}\t"
            f"{task.get('video_url') or '-'}\t{task.get('error') or '-'}"
        )
        rows.append(row)
        print(f"[bench] done {label} status={task.get('status')} wall_s={wall_s:.1f}", flush=True)
        print(row, flush=True)

    total_s = time.monotonic() - total_started
    rows.append("")
    rows.append(f"total_s\t{total_s:.3f}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[bench] wrote {out} total_s={total_s:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
