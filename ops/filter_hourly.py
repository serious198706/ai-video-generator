#!/usr/bin/env python3
"""从 journal/文件里抽出小时告警行；空输出表示没事。"""

from __future__ import annotations

import re
import sys

KEEP = re.compile(
    r" (ERROR|WARNING) |failed|exception|CUDA|out of memory|OOM|"
    r"preload failed|pipeline load failed|Main process exited|"
    r"timing .*status=failed",
    re.I,
)
DROP = re.compile(
    r"uvicorn\.access|webhook skipped|missing task=",
    re.I,
)


def interesting(line: str) -> bool:
    if DROP.search(line):
        return False
    return bool(KEEP.search(line))


def main() -> None:
    kept = [line.rstrip("\n") for line in sys.stdin if interesting(line)]
    if not kept:
        return
    sys.stdout.write("\n".join(kept[-120:]) + "\n")


if __name__ == "__main__":
    main()
