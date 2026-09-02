#!/usr/bin/env python3
"""时间窗：小时用 UTC 绝对时间；日报是上海昨天 08:00 到今天 08:00。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

SHANGHAI = timezone(timedelta(hours=8))


def hourly() -> tuple[str, str]:
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(hours=1)
    return _utc(start), _utc(end)


def daily(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(SHANGHAI)
    end = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now < end:
        end -= timedelta(days=1)
    start = end - timedelta(days=1)
    return _utc(start.astimezone(timezone.utc)), _utc(end.astimezone(timezone.utc))


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("hourly", "daily"))
    args = parser.parse_args()
    start, end = hourly() if args.kind == "hourly" else daily()
    sys.stdout.write(f"{start}\n{end}\n")


if __name__ == "__main__":
    main()
