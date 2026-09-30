using System.Globalization;
using System.Text;
using Radar.Core.Support;

namespace Radar.Core.Postprocessing;

/// <summary>
/// 与官方结果表（上游 <c>inference_demo.py</c> 的 <c>pandas.to_csv</c>）逐字节对齐的宽表 CSV。
/// <para>
/// 格式约定（由 testdata/golden/RADAR_infer_results_demo.csv 实测确定）：
/// UTF-8 BOM + CRLF 行尾 + 末尾 CRLF；147 列 = <c>file_name</c> + 146 个条目；
/// 列名 <c>"{中文} ({英文})"</c>；数值为 Python <c>repr(float)</c> 的最短往返表示；
/// 缺失值（对应上游 NaN，多为器官不完整未打分）写空字段。
/// </para>
/// </summary>
public static class ResultCsv
{
    /// <summary>首列列名。</summary>
    public const string FileNameColumn = "file_name";

    /// <summary>一条病例记录：文件名 + 条目名 -> 阳性概率。</summary>
    public sealed record Row(string FileName, IReadOnlyDictionary<string, double> Values);

    /// <summary>
    /// 复刻 Python <c>repr(float)</c> 的最短往返表示。
    /// 与 C# 默认 <c>ToString()</c>（.NET Core 3.0+ 亦为最短往返）的差异只有两处：
    /// 指数标记大小写（C# <c>E</c> / Python <c>e</c>）与整数形态（Python 补 <c>.0</c>）。
    /// </summary>
    public static string PyRepr(double v)
    {
        if (double.IsNaN(v) || double.IsInfinity(v)) return string.Empty;
        string s = v.ToString("R", CultureInfo.InvariantCulture);
        int e = s.IndexOf('E');
        if (e >= 0) return s[..e] + "e" + s[(e + 1)..];
        return s.Contains('.') ? s : s + ".0";
    }

    /// <summary>组装官方格式宽表字节（含 BOM）。</summary>
    public static byte[] Build(IReadOnlyList<string> items, IEnumerable<Row> rows)
    {
        var sb = new StringBuilder();
        sb.Append(string.Join(",", TestItemLabels.CsvHeader(items).Select(CsvField))).Append("\r\n");

        foreach (var r in rows)
        {
            sb.Append(CsvField(r.FileName));
            foreach (var item in items)
            {
                sb.Append(',');
                if (r.Values.TryGetValue(item, out var v) && !double.IsNaN(v))
                    sb.Append(PyRepr(v));
            }
            sb.Append("\r\n");
        }

        byte[] bom = Encoding.UTF8.GetPreamble();
        byte[] body = Encoding.UTF8.GetBytes(sb.ToString());
        var buf = new byte[bom.Length + body.Length];
        bom.CopyTo(buf, 0);
        body.CopyTo(buf, bom.Length);
        return buf;
    }

    /// <summary>写出官方格式宽表。</summary>
    public static void Write(string path, IReadOnlyList<string> items, IEnumerable<Row> rows)
        => File.WriteAllBytes(path, Build(items, rows));

    /// <summary>
    /// 读回宽表：返回按列序还原的条目名（由 <c>"{中文} ({英文})"</c> 反查）与数据行。
    /// 表头无法反查的列会被保留为空串占位，以维持列对齐。
    /// </summary>
    public static (List<string> Items, List<Row> Rows) Read(string path)
    {
        string text = File.ReadAllText(path, Encoding.UTF8);
        var records = ParseCsv(text);
        if (records.Count == 0) return (new List<string>(), new List<Row>());

        var header = records[0];
        var items = new List<string>();
        for (int c = 1; c < header.Count; c++)
            items.Add(ReverseColumn(header[c]));

        var rows = new List<Row>();
        for (int r = 1; r < records.Count; r++)
        {
            var rec = records[r];
            if (rec.Count == 1 && rec[0].Length == 0) continue;   // 跳过空行
            var values = new Dictionary<string, double>(StringComparer.Ordinal);
            for (int c = 1; c < rec.Count && c - 1 < items.Count; c++)
            {
                if (rec[c].Length == 0) continue;                  // 空字段 = NaN
                if (double.TryParse(rec[c], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    values[items[c - 1]] = v;
            }
            rows.Add(new Row(rec.Count > 0 ? rec[0] : "", values));
        }
        return (items, rows);
    }

    /// <summary>
    /// 由官方列名反查中文条目。优先走 <see cref="TestItemLabels.TryItemOfColumn"/> 的精确字典 ——
    /// 不能靠切分括号：英文名自身可能含半角括号（如 <c>Stomach_Gastrointestinal stromal tumor (GIST)</c>），
    /// 从右往左找第一个 <c>" ("</c> 会切错。
    /// 未命中（外部/未知列）时才退化为按最左侧 <c>" ("</c> 取中文前缀。
    /// </summary>
    private static string ReverseColumn(string column)
    {
        if (TestItemLabels.TryItemOfColumn(column, out var exact)) return exact;

        int p = column.IndexOf(" (", StringComparison.Ordinal);
        return p > 0 && column.EndsWith(')') ? column[..p] : column;
    }

    /// <summary>RFC4180 字段转义。</summary>
    private static string CsvField(string s)
        => s.IndexOfAny([',', '"', '\r', '\n']) < 0 ? s : "\"" + s.Replace("\"", "\"\"") + "\"";

    /// <summary>RFC4180 解析（支持引号包裹与 "" 转义）。</summary>
    internal static List<List<string>> ParseCsv(string text)
    {
        var records = new List<List<string>>();
        var record = new List<string>();
        var field = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (inQuotes)
            {
                if (ch == '"')
                {
                    if (i + 1 < text.Length && text[i + 1] == '"') { field.Append('"'); i++; }
                    else inQuotes = false;
                }
                else field.Append(ch);
                continue;
            }

            switch (ch)
            {
                case '"':
                    inQuotes = true;
                    break;
                case ',':
                    record.Add(field.ToString());
                    field.Clear();
                    break;
                case '\r':
                    break;                                  // 交给 \n 收行（兼容 CRLF 与 LF）
                case '\n':
                    record.Add(field.ToString());
                    field.Clear();
                    records.Add(record);
                    record = new List<string>();
                    break;
                default:
                    field.Append(ch);
                    break;
            }
        }
        if (field.Length > 0 || record.Count > 0)
        {
            record.Add(field.ToString());
            records.Add(record);
        }
        return records;
    }
}
