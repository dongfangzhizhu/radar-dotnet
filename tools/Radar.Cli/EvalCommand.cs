using System.Text.Json;
using Radar.Core.Postprocessing;
using Radar.Core.Support;

namespace Radar.Cli;

/// <summary>
/// <c>radar eval</c>：复刻上游 <c>calc_metrics_merlin_testset.py</c> 的逐项 AUC 与 AvgAUC。
/// 输入为 <c>radar infer</c> 产出的宽表 CSV + MERLIN 标签 JSON。
/// </summary>
internal static class EvalCommand
{
    public static int Run(Options o)
    {
        string csvPath = o.Get("results-csv") ?? throw new ArgumentException("必须指定 --results-csv");
        string labelsPath = o.Get("labels") ?? throw new ArgumentException("必须指定 --labels");
        string? sagPath = o.Get("sag-counts");

        var (items, rows) = ResultCsv.Read(csvPath);
        if (rows.Count == 0)
        {
            Console.Error.WriteLine($"结果表没有数据行: {csvPath}");
            return 3;
        }

        var itemSet = items.ToHashSet(StringComparer.Ordinal);
        var labels = LoadNested(labelsPath);
        var sagCounts = sagPath is not null ? LoadFlat(sagPath) : null;

        Console.WriteLine($"结果表  : {csvPath}（{rows.Count} 例，{items.Count} 项）");
        Console.WriteLine($"标签表  : {labelsPath}（{labels.Count} 个疾病键）");
        if (sagCounts is not null)
            Console.WriteLine($"胆囊体素: {sagPath}（{sagCounts.Count} 例）");
        Console.WriteLine();

        var aucs = new List<double>();
        int notScored = 0, noLabel = 0, singleClass = 0;

        // 按结果表列序遍历，保证输出顺序与上游一致
        foreach (var item in items)
        {
            if (!MerlinDiseases.Map.TryGetValue(item, out var diseaseKey)) continue;

            if (!labels.TryGetValue(diseaseKey, out var perPatient))
            {
                Console.WriteLine($"{diseaseKey,-34} 跳过：标签中无该疾病键");
                notScored++;
                continue;
            }

            bool useSagCount = item == MerlinDiseases.GallbladderAbsentItem;
            if (useSagCount && sagCounts is null)
            {
                Console.WriteLine($"{diseaseKey,-34} 跳过：该口径需要 --sag-counts（体素数），上游用 <"
                    + $"{MerlinDiseases.GallbladderAbsentThreshold} 判定");
                notScored++;
                continue;
            }

            var scores = new List<double>();
            var gt = new List<int>();
            foreach (var row in rows)
            {
                string pid = PatientId(row.FileName);
                if (!perPatient.TryGetValue(pid, out var label)) continue;
                if (label == -1) continue;                      // 上游口径：-1 表示不参与评估
                if (useSagCount)
                {
                    if (!sagCounts!.TryGetValue(pid, out var n)) continue;
                    scores.Add(n < MerlinDiseases.GallbladderAbsentThreshold ? 1.0 : 0.0);
                }
                else
                {
                    // 空字段 = 该器官不完整未打分，上游按 0 处理
                    scores.Add(row.Values.TryGetValue(item, out var v) && !double.IsNaN(v) ? v : 0.0);
                }
                gt.Add(label != 0 ? 1 : 0);
            }

            if (scores.Count == 0)
            {
                Console.WriteLine($"{diseaseKey,-34} 跳过：无任何可配对样本（{rows.Count} 例均无标签）");
                noLabel++;
                continue;
            }

            double auc = RocAuc.Compute(scores, gt);
            if (double.IsNaN(auc))
            {
                Console.WriteLine($"{diseaseKey,-34} 跳过：样本只有单一类别（n={scores.Count}）");
                singleClass++;
                continue;
            }

            aucs.Add(auc);
            Console.WriteLine($"{diseaseKey,-34} {auc:F4}   (n={scores.Count})");
        }

        Console.WriteLine();
        if (aucs.Count == 0)
        {
            Console.Error.WriteLine("没有任何条目可计算 AUC。");
            return 1;
        }
        Console.WriteLine($"AvgAUC: {aucs.Average():F4}   （{aucs.Count} 项）");

        // 结果表里存在但不在 MERLIN 映射中的项，说明该权重与 MERLIN 评估口径不同
        int unmapped = items.Count(i => !MerlinDiseases.Map.ContainsKey(i));
        if (unmapped > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"说明：结果表有 {unmapped} 项不在 MERLIN 映射内（该集合是 RAD-CT 的 146 项，");
            Console.WriteLine("      而 MERLIN 评估用的是另一套 finding 表），这些项不参与评估。");
        }
        if (notScored + noLabel + singleClass > 0)
            Console.WriteLine($"未评估：标签缺失/口径不符 {notScored + noLabel} 项，单类别 {singleClass} 项。");

        return 0;
    }

    /// <summary>上游口径：patient_id = file_name[:-7]（去掉 ".nii.gz"）。</summary>
    private static string PatientId(string fileName)
        => fileName.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase) && fileName.Length > 7
            ? fileName[..^7]
            : Path.GetFileNameWithoutExtension(fileName);

    /// <summary>解析 {疾病键: {病例ID: 标签}}。</summary>
    private static Dictionary<string, Dictionary<string, double>> LoadNested(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var result = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
        foreach (var disease in doc.RootElement.EnumerateObject())
        {
            var inner = new Dictionary<string, double>(StringComparer.Ordinal);
            if (disease.Value.ValueKind == JsonValueKind.Object)
                foreach (var p in disease.Value.EnumerateObject())
                    inner[p.Name] = ToDouble(p.Value);
            result[disease.Name] = inner;
        }
        return result;
    }

    /// <summary>解析 {病例ID: 数值}。</summary>
    private static Dictionary<string, double> LoadFlat(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var p in doc.RootElement.EnumerateObject())
            result[p.Name] = ToDouble(p.Value);
        return result;
    }

    private static double ToDouble(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Number => e.GetDouble(),
        JsonValueKind.String when double.TryParse(e.GetString(), out var d) => d,
        _ => double.NaN,
    };
}
