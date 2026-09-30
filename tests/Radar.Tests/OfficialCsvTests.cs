using System.Text;
using System.Text.Json;
using Radar.Core.Postprocessing;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// 官方宽表 CSV 对齐（实测差距 #1 与 #6）：
/// 上游 <c>inference_demo.py</c> 只产出一张「1 行/病例 × 147 列」的宽表
/// （UTF-8 BOM + CRLF + 列名 <c>"{中文} ({英文})"</c> + Python <c>repr(float)</c>）。
/// 这里用真实官方产物 <c>testdata/golden/RADAR_infer_results_demo.csv</c> 做往返对拍。
/// </summary>
public class OfficialCsvTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();
    private static string OfficialCsv => Path.Combine(Root, "testdata", "golden", "RADAR_infer_results_demo.csv");

    // ---------- 1. 与官方文件逐字节往返 ----------

    [Fact]
    public void ReadThenBuild_ReproducesOfficialFileByteExact()
    {
        if (!File.Exists(OfficialCsv)) return;   // golden 未就位

        byte[] original = File.ReadAllBytes(OfficialCsv);
        var (items, rows) = ResultCsv.Read(OfficialCsv);
        byte[] rebuilt = ResultCsv.Build(items, rows);

        output.WriteLine($"原始 {original.Length} 字节 / 重建 {rebuilt.Length} 字节，列数 {items.Count + 1}，行数 {rows.Count}");

        Assert.Equal(147, items.Count + 1);
        Assert.Single(rows);

        if (!original.AsSpan().SequenceEqual(rebuilt))
        {
            int n = Math.Min(original.Length, rebuilt.Length);
            int at = -1;
            for (int i = 0; i < n; i++) if (original[i] != rebuilt[i]) { at = i; break; }
            if (at >= 0)
            {
                int s = Math.Max(0, at - 40);
                output.WriteLine($"首个差异 @{at}:\n 原: {Encoding.UTF8.GetString(original, s, Math.Min(120, original.Length - s))}"
                    + $"\n 新: {Encoding.UTF8.GetString(rebuilt, s, Math.Min(120, rebuilt.Length - s))}");
            }
            else output.WriteLine($"长度不同: {original.Length} vs {rebuilt.Length}");
        }
        Assert.True(original.AsSpan().SequenceEqual(rebuilt), "重建结果与官方文件不是逐字节一致");
    }

    [Fact]
    public void File_HasBomCrlfAndTrailingNewline()
    {
        if (!File.Exists(OfficialCsv)) return;
        byte[] b = File.ReadAllBytes(OfficialCsv);

        Assert.True(b.Length > 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF, "缺少 UTF-8 BOM");
        Assert.True(b[^2] == '\r' && b[^1] == '\n', "缺少末尾 CRLF");
        // 全文不含裸 LF（每个 \n 前面必须是 \r）
        for (int i = 0; i < b.Length; i++)
            if (b[i] == '\n') Assert.True(i > 0 && b[i - 1] == '\r', $"第 {i} 字节处出现裸 LF");
    }

    // ---------- 2. 表头列名列序 ----------

    [Fact]
    public void Header_MatchesOfficialColumnsInOrder()
    {
        if (!File.Exists(OfficialCsv)) return;

        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "testdata", "golden", "meta.json")));
        var testItems = meta.RootElement.GetProperty("test_items").EnumerateArray()
            .Select(v => v.GetString()!).ToList();
        Assert.Equal(146, testItems.Count);

        string headerLine = File.ReadAllLines(OfficialCsv, Encoding.UTF8)[0].TrimStart('\uFEFF');
        string[] official = headerLine.Split(',');

        string[] mine = TestItemLabels.CsvHeader(testItems);
        Assert.Equal(official.Length, mine.Length);
        for (int i = 0; i < official.Length; i++)
            Assert.Equal(official[i], mine[i]);
    }

    [Fact]
    public void ColumnWithHalfWidthParenInEnglish_ReverseLookupIsExact()
    {
        // 英文名自带半角括号，是切括号反查法的天然陷阱
        const string item = "胃_间质瘤（gist）";
        string column = TestItemLabels.CsvColumn(item);
        Assert.Equal("胃_间质瘤（gist） (Stomach_Gastrointestinal stromal tumor (GIST))", column);

        Assert.True(TestItemLabels.TryItemOfColumn(column, out string back));
        Assert.Equal(item, back);

        // 全部 146 列都必须能精确反查回自身
        int ok = 0;
        foreach (var (cn, en) in TestItemLabels.English)
            if (TestItemLabels.TryItemOfColumn($"{cn} ({en})", out string r) && r == cn) ok++;
        Assert.Equal(146, ok);
    }

    [Fact]
    public void Read_RecoversAll146ItemNamesFromOfficialHeader()
    {
        if (!File.Exists(OfficialCsv)) return;
        var (items, _) = ResultCsv.Read(OfficialCsv);

        Assert.Equal(146, items.Count);
        Assert.DoesNotContain(items, s => s.Length == 0);          // 没有反查失败的占位
        Assert.All(items, s => Assert.True(TestItemLabels.English.ContainsKey(s), $"反查结果不是合法条目: {s}"));
        // 列序须与官方一致（Read 是按列序还原的）
        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "testdata", "golden", "meta.json")));
        var testItems = meta.RootElement.GetProperty("test_items").EnumerateArray()
            .Select(v => v.GetString()!).ToList();
        Assert.Equal(testItems, items);
    }

    // ---------- 3. 数值格式：Python repr(float) ----------

    [Theory]
    [InlineData(0.5, "0.5")]
    [InlineData(0.1, "0.1")]
    [InlineData(1.0, "1.0")]                       // Python 给整数补 .0
    [InlineData(0.0, "0.0")]
    [InlineData(3.0, "3.0")]
    [InlineData(123456789.0, "123456789.0")]
    [InlineData(1e-7, "1e-07")]                    // 指数标记小写 + 两位指数
    [InlineData(1.5e-8, "1.5e-08")]
    [InlineData(2.5e20, "2.5e+20")]
    [InlineData(0.3333333333333333, "0.3333333333333333")]
    [InlineData(0.37641215324401855, "0.37641215324401855")]
    public void PyRepr_MatchesPythonRepr(double value, string expected)
        => Assert.Equal(expected, ResultCsv.PyRepr(value));

    [Fact]
    public void PyRepr_NaNIsEmptyField_NotLiteralNaN()
    {
        Assert.Equal(string.Empty, ResultCsv.PyRepr(double.NaN));
        Assert.Equal(string.Empty, ResultCsv.PyRepr(double.PositiveInfinity));
    }

    [Fact]
    public void Build_NaNBecomesEmptyField_AndMissingKeyAlsoEmpty()
    {
        var items = new List<string> { "主动脉_主动脉夹层", "主动脉_主动脉瘤" };
        var row = new ResultCsv.Row("case.nii.gz", new Dictionary<string, double>
        {
            ["主动脉_主动脉夹层"] = double.NaN,       // 显式 NaN
            // 主动脉_主动脉瘤 缺失           -> 同样应写空字段
        });

        string text = Encoding.UTF8.GetString(ResultCsv.Build(items, [row])).TrimStart('\uFEFF');
        string[] lines = text.Split("\r\n");
        Assert.Equal(3, lines.Length);                              // 表头 + 1 行 + 末尾空串
        Assert.Equal("", lines[2]);
        Assert.Equal("case.nii.gz,,", lines[1]);                    // 空字段保留逗号占位
    }

    [Fact]
    public void BuildAndRead_RoundTripMultiRowKeepsOrderAndValues()
    {
        var items = TestItemLabels.English.Keys.Take(146).ToList();
        var rnd = new Random(20260101);
        var rows = new List<ResultCsv.Row>();
        for (int r = 0; r < 3; r++)
        {
            var v = new Dictionary<string, double>();
            foreach (var it in items) v[it] = rnd.NextDouble();
            rows.Add(new ResultCsv.Row($"case_{r}.nii.gz", v));
        }

        string tmp = Path.Combine(Path.GetTempPath(), $"radar_csv_rt_{Guid.NewGuid():N}.csv");
        try
        {
            ResultCsv.Write(tmp, items, rows);
            var (backItems, backRows) = ResultCsv.Read(tmp);

            Assert.Equal(items, backItems);
            Assert.Equal(3, backRows.Count);
            for (int r = 0; r < 3; r++)
            {
                Assert.Equal(rows[r].FileName, backRows[r].FileName);
                foreach (var it in items)
                    Assert.Equal(rows[r].Values[it], backRows[r].Values[it]);   // 最短往返 -> 位级可逆
            }
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [Fact]
    public void CsvField_QuotesWhenNeeded()
    {
        // 文件名含逗号/引号时必须按 RFC4180 转义，否则列错位
        var items = new List<string> { "主动脉_主动脉夹层" };
        var row = new ResultCsv.Row("a,\"b\".nii.gz", new Dictionary<string, double> { ["主动脉_主动脉夹层"] = 0.5 });

        string body = Encoding.UTF8.GetString(ResultCsv.Build(items, [row])).TrimStart('\uFEFF');
        Assert.Contains("\"a,\"\"b\"\".nii.gz\"", body);

        string tmp = Path.Combine(Path.GetTempPath(), $"radar_csv_q_{Guid.NewGuid():N}.csv");
        try
        {
            ResultCsv.Write(tmp, items, [row]);
            var (_, rows) = ResultCsv.Read(tmp);
            Assert.Equal("a,\"b\".nii.gz", rows[0].FileName);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }
}
