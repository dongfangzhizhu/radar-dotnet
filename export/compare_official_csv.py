# -*- coding: utf-8 -*-
"""比对 radar-cli 产出的宽表 CSV 与官方 CSV 的格式与数值。

格式：BOM / 换行 / 列数 / 列名逐列
数值：逐列绝对差（max / mean），并给出超差条目
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OURS = REPO / ".tmp_out" / "RADAR_infer_results_demo.csv"
OFFICIAL = REPO.parent / "damo-radar" / "results" / "RADAR_infer_results_demo.csv"

lines = []


def raw(p):
    return p.read_bytes()


a, b = raw(OURS), raw(OFFICIAL)
lines.append("=== 字节级格式 ===")
lines.append("我们的: %d 字节 | 官方: %d 字节" % (len(a), len(b)))
lines.append("BOM      : ours=%s official=%s" % (a[:3] == b"\xef\xbb\xbf", b[:3] == b"\xef\xbb\xbf"))
lines.append("CRLF 数  : ours=%d official=%d" % (a.count(b"\r\n"), b.count(b"\r\n")))
lines.append("LF 数    : ours=%d official=%d" % (a.count(b"\n"), b.count(b"\n")))
lines.append("结尾 CRLF: ours=%s official=%s" % (a.endswith(b"\r\n"), b.endswith(b"\r\n")))

with open(OURS, encoding="utf-8-sig", newline="") as f:
    ho = next(csv.reader(f))
    ro = next(csv.reader(f))
with open(OFFICIAL, encoding="utf-8-sig", newline="") as f:
    hf = next(csv.reader(f))
    rf = next(csv.reader(f))

lines.append("")
lines.append("=== 表头 ===")
lines.append("列数: ours=%d official=%d" % (len(ho), len(hf)))
bad = [(i, x, y) for i, (x, y) in enumerate(zip(ho, hf)) if x != y]
lines.append("列名不一致处: %d" % len(bad))
for i, x, y in bad[:5]:
    lines.append("  col %d: ours=%r official=%r" % (i, x, y))

lines.append("")
lines.append("=== first row ===")
lines.append("file_name: ours=%r official=%r" % (ro[0], rf[0]))

lines.append("")
lines.append("=== 数值比对（逐列）===")
diffs = []
for i in range(1, min(len(ho), len(ro), len(rf))):
    x, y = ro[i], rf[i]
    if x == "" or y == "":
        diffs.append((ho[i], None, x, y))
        continue
    diffs.append((ho[i], abs(float(x) - float(y)), x, y))

numeric = [d for d in diffs if d[1] is not None]
if numeric:
    mx = max(numeric, key=lambda d: d[1])
    mean = sum(d[1] for d in numeric) / len(numeric)
    lines.append("可比列数: %d / %d" % (len(numeric), len(diffs)))
    lines.append("max_diff  = %.3E  (%s)" % (mx[1], mx[0]))
    lines.append("mean_diff = %.3E" % mean)
    over = [d for d in numeric if d[1] >= 1e-3]
    lines.append("超 1e-3 门禁的列数: %d" % len(over))
    for d in over[:5]:
        lines.append("  %s: ours=%s official=%s" % (d[0], d[2], d[3]))
empties = [d for d in diffs if d[1] is None]
lines.append("任一侧为空的列数: %d" % len(empties))

out = REPO / ".tmp_cmp.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
