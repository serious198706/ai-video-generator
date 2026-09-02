#!/usr/bin/env python3
"""从 timing 行统计成功率、按时长/Foley 的平均耗时。"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

_KV = re.compile(r"(\w+)=(\S+)")


def parse_timing(line: str) -> dict[str, str] | None:
    if "timing " not in line:
        return None
    fields = dict(_KV.findall(line.split("timing ", 1)[-1]))
    if "status" not in fields:
        return None
    return fields


def bucket_duration(raw: str) -> str:
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return "unknown"
    if seconds <= 0:
        return "unknown"
    snapped = int(round(seconds / 5.0) * 5)
    snapped = min(max(snapped, 5), 15)
    return f"{snapped}s"


def _mean(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{sum(values) / len(values):.1f}s"


def summarize(lines: list[str], *, start: str, end: str, host: str) -> str:
    rows: list[dict[str, str]] = []
    for line in lines:
        parsed = parse_timing(line)
        if parsed:
            rows.append(parsed)

    total = len(rows)
    ok = [r for r in rows if r.get("status") == "succeeded"]
    failed = [r for r in rows if r.get("status") == "failed"]
    errors: dict[str, int] = defaultdict(int)
    for row in failed:
        errors[row.get("error") or "unknown"] += 1

    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"total": [], "generate": [], "n_ok": [], "n_fail": []}
    )
    for row in rows:
        dur = bucket_duration(row.get("duration", ""))
        foley = "有Foley" if row.get("foley") == "1" else "无声"
        key = (dur, foley)
        if row.get("status") == "succeeded":
            try:
                groups[key]["total"].append(float(row.get("total_s") or 0))
                groups[key]["generate"].append(float(row.get("generate_s") or 0))
            except ValueError:
                pass
            groups[key]["n_ok"].append(1)
        else:
            groups[key]["n_fail"].append(1)

    rate = f"{len(ok) / total * 100:.1f}%" if total else "-"
    out: list[str] = [
        f"【Wan22 日报】{host}",
        f"窗口 {start} → {end}（上海 08:00–08:00）",
        f"完成 {total} 条：成功 {len(ok)}，失败 {len(failed)}，成功率 {rate}",
    ]
    if ok:
        totals = []
        for row in ok:
            try:
                totals.append(float(row["total_s"]))
            except (KeyError, ValueError):
                pass
        out.append(f"成功任务墙钟平均 {_mean(totals)}（I2V+Foley+上传）")

    if groups:
        out.append("")
        out.append("分类型（成功平均时长）：")
        for dur, foley in sorted(groups, key=lambda x: (x[0], x[1])):
            g = groups[(dur, foley)]
            n_ok = len(g["n_ok"])
            n_fail = len(g["n_fail"])
            out.append(
                f"- {dur} {foley}：成功 {n_ok} / 失败 {n_fail}，"
                f"平均 total {_mean(g['total'])}，I2V {_mean(g['generate'])}"
            )

    if errors:
        out.append("")
        out.append("失败短码：")
        for name, count in sorted(errors.items(), key=lambda x: -x[1]):
            out.append(f"- {name}: {count}")

    if total == 0:
        out.append("")
        out.append("这一窗没有 timing 行。worker 没跑、日志权限不够，或时区窗口不对。")

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--host", default="")
    args = parser.parse_args()
    text = sys.stdin.read().splitlines()
    sys.stdout.write(
        summarize(text, start=args.start, end=args.end, host=args.host or "gpu")
    )


if __name__ == "__main__":
    main()
