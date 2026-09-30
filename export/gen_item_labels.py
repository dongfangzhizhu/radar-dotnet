# -*- coding: utf-8 -*-
"""从上游源码生成 C# 的 146 项中英列名映射表。

权威来源：../damo-radar/RADAR_inference/inference_demo.py 中 DataFolder.__init__
的 self.test_items（顺序）与 self.english_mapping（列名）。
用 ast 提取而非解析 CSV 表头 —— 中文名含全角括号、英文含撇号，字符切分易错。

交叉校验：与 testdata/golden/RADAR_infer_results_demo.csv 的表头逐列比对，
确认真实产出 CSV 的列名与列序完全一致。

产物：src/Radar.Core/Support/TestItemLabels.cs（入库，编译进程序集）
运行：C:/Users/Administrator/miniconda3/python.exe export/gen_item_labels.py
"""
import ast
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO.parent / "damo-radar" / "RADAR_inference" / "inference_demo.py"
OFFICIAL_CSV = REPO / "testdata" / "golden" / "RADAR_infer_results_demo.csv"
OUT_CS = REPO / "src" / "Radar.Core" / "Support" / "TestItemLabels.cs"


def extract():
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    test_items = english = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                if t.attr == "test_items":
                    test_items = ast.literal_eval(node.value)
                elif t.attr == "english_mapping":
                    english = ast.literal_eval(node.value)
    if not test_items or not english:
        sys.exit("未能在上游源码中定位 test_items / english_mapping")
    return test_items, english


def main():
    test_items, english = extract()
    print("test_items = %d, english_mapping = %d" % (len(test_items), len(english)))
    assert len(test_items) == 146, "test_items 数量不是 146"

    missing = [i for i in test_items if i not in english]
    if missing:
        sys.exit("english_mapping 缺少条目: %s" % missing[:5])

    ordered = [(i, english[i]) for i in test_items]

    # ---- 交叉校验官方 CSV 表头 ----
    with open(OFFICIAL_CSV, "r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header[0] == "file_name", "首列不是 file_name: %r" % header[0]
    cols = header[1:]
    print("官方 CSV 列数（不含 file_name）= %d" % len(cols))
    if len(cols) != len(ordered):
        sys.exit("列数不一致: csv=%d vs mapping=%d" % (len(cols), len(ordered)))
    bad = []
    for (item, eng), col in zip(ordered, cols):
        expect = "%s (%s)" % (item, eng)
        if col != expect:
            bad.append((col, expect))
    if bad:
        for got, exp in bad[:5]:
            print("  MISMATCH got=%r exp=%r" % (got, exp))
        sys.exit("列名/列序与官方 CSV 不一致（%d 处）" % len(bad))
    print("交叉校验通过：146 列列名与列序与官方 CSV 完全一致")

    # ---- 列名唯一性：反查表必须是一一映射 ----
    # 注意「胃_间质瘤（gist） (Stomach_Gastrointestinal stromal tumor (GIST))」这类
    # 英文名自带半角括号的列，字符切分会切错，因此反查必须走精确字典而非切括号。
    colnames = ["%s (%s)" % (i, e) for i, e in ordered]
    if len(set(colnames)) != len(colnames):
        dupes = sorted({c for c in colnames if colnames.count(c) > 1})
        sys.exit("列名存在重复，无法一一反查: %s" % dupes[:5])
    print("列名唯一性通过：%d 列可精确一一反查" % len(colnames))

    # ---- 生成 C# ----
    lines = []
    lines.append("namespace Radar.Core.Support;")
    lines.append("")
    lines.append("/// <summary>")
    lines.append("/// 146 个打分条目的官方英文列名映射（对应上游 inference_demo.py 的 english_mapping）。")
    lines.append("/// 用于导出与官方对齐的宽表 CSV，列名格式 \"{中文条目} ({英文})\"。")
    lines.append("/// 列序即本表顺序，与 text_feats.json 的 items 及官方 CSV 列序一致。")
    lines.append("/// </summary>")
    lines.append("/// <remarks>")
    lines.append("/// 本文件由 export/gen_item_labels.py 从上游源码生成，请勿手工编辑。")
    lines.append("/// 重新生成：python export/gen_item_labels.py")
    lines.append("/// </remarks>")
    lines.append("public static class TestItemLabels")
    lines.append("{")
    lines.append("    /// <summary>中文条目 -> 官方英文列名（不含括号）。</summary>")
    lines.append("    public static readonly IReadOnlyDictionary<string, string> English =")
    lines.append("        new Dictionary<string, string>(StringComparer.Ordinal)")
    lines.append("    {")
    for item, eng in ordered:
        e = eng.replace("\\", "\\\\").replace('"', '\\"')
        lines.append('        ["%s"] = "%s",' % (item, e))
    lines.append("    };")
    lines.append("")
    lines.append("    /// <summary>")
    lines.append("    /// 官方列名 -> 中文条目（精确反查）。")
    lines.append("    /// 由 <see cref=\"English\"/> 构造：不能靠切分括号反查 —— 英文名本身可能含半角括号，")
    lines.append("    /// 例如 \"胃_间质瘤（gist） (Stomach_Gastrointestinal stromal tumor (GIST))\"。")
    lines.append("    /// </summary>")
    lines.append("    private static readonly IReadOnlyDictionary<string, string> ColumnToItem =")
    lines.append("        English.ToDictionary(kv => $\"{kv.Key} ({kv.Value})\", kv => kv.Key, StringComparer.Ordinal);")
    lines.append("")
    lines.append("    /// <summary>由官方列名精确反查中文条目；未命中时 <paramref name=\"item\"/> 为空串。</summary>")
    lines.append("    public static bool TryItemOfColumn(string column, out string item)")
    lines.append("    {")
    lines.append("        if (ColumnToItem.TryGetValue(column, out var v)) { item = v; return true; }")
    lines.append("        item = string.Empty;")
    lines.append("        return false;")
    lines.append("    }")
    lines.append("")
    lines.append("    /// <summary>官方 CSV 表头列名：\"{中文} ({英文})\"；无映射时回退为中文条目本身。</summary>")
    lines.append("    public static string CsvColumn(string item) =>")
    lines.append('        English.TryGetValue(item, out var en) ? $"{item} ({en})" : item;')
    lines.append("")
    lines.append("    /// <summary>官方宽表 CSV 的表头（含首列 file_name）。</summary>")
    lines.append("    public static string[] CsvHeader(IReadOnlyList<string> items)")
    lines.append("    {")
    lines.append("        var cols = new string[items.Count + 1];")
    lines.append('        cols[0] = "file_name";')
    lines.append("        for (int i = 0; i < items.Count; i++) cols[i + 1] = CsvColumn(items[i]);")
    lines.append("        return cols;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    OUT_CS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CS.write_text("\n".join(lines), encoding="utf-8-sig" if False else "utf-8")
    print("已写入 %s（%d 行）" % (OUT_CS, len(lines)))


if __name__ == "__main__":
    main()
