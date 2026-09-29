using System.Text;
using Radar.Core.Inference;
using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Core.Support;

/// <summary>
/// M5 性能基准 + 扩展数据集冒烟：
///   ① demo case × 三设备（cpu/directml/cuda）端到端耗时 + 阶段分解 + 对拍 golden；
///   ② merlin 扩展数据 × CPU 冒烟（不崩溃、无 NaN、条目数合理）。
/// 结果写 benchmarks.md（仓库根）+ 控制台输出。
/// </summary>
internal static class Program
{
    private static void Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        string root = FindRepoRoot();
        string modelsDir = Path.Combine(root, "models");
        string benchPath = Path.Combine(root, "benchmarks.md");

        var head = new OrganHead(
            OrganHeadWeights.Load(Path.Combine(modelsDir, "organ_head.npz")),
            TextFeatures.Load(
                Path.Combine(modelsDir, "text_feats.bin"),
                Path.Combine(modelsDir, "text_feats.json")));
        var items = head.Text.Items.ToList();
        var testOrgans = items.Select(i => i.Split('_')[0]).Distinct().ToList();
        string modelPath = Path.Combine(modelsDir, "radar_vision.onnx");

        // ---- ① 设备矩阵（demo case） ----
        string demoNii = Path.Combine(root, "testdata", "AC423ccbe.nii.gz");
        var golden = LoadGolden(root);
        var rows = new List<string>();
        rows.Add("## 设备矩阵（demo case AC423ccbe，单次端到端）");
        rows.Add("");
        rows.Add("| 设备 | 预处理(ms) | 会话创建(ms) | 端到端(ms) | 窗数 | ORT合计(ms) | 轻头合计(ms) | vs golden max |");
        rows.Add("|---|---|---|---|---|---|---|---|");

        foreach (var device in new[] { ExecutionDevice.Cpu, ExecutionDevice.DirectMl, ExecutionDevice.Cuda })
        {
            string dev = device.ToString().ToLowerInvariant();
            string used;
            try
            {
                var swSession = System.Diagnostics.Stopwatch.StartNew();
                var probe = new VisionEngine(modelPath, device);
                used = probe.Device.ToString();
                probe.Dispose();
            }
            catch (Exception ex)
            {
                rows.Add($"| {dev} | - | - | - | - | - | - | 不可用: {Truncate(ex.Message)} |");
                continue;
            }

            var swPre = System.Diagnostics.Stopwatch.StartNew();
            var img = NiftiReader.Read(demoNii);
            var prep = Preprocessor.Run(img, Path.GetFileName(demoNii));
            swPre.Stop();

            var swSess = System.Diagnostics.Stopwatch.StartNew();
            var engine = new VisionEngine(modelPath, device);
            swSess.Stop();
            try
            {
                using var radar = new RadarInferenceEngine(engine, head, RadarLabels.Organs, items);
                var result = radar.Run(prep.Data, prep.D, prep.H, prep.W, testOrgans);
                double ort = result.PhaseMs!.GetValueOrDefault("ort_ms");
                double headMs = result.PhaseMs.GetValueOrDefault("head_ms");
                string diff = "-";
                if (golden is not null)
                {
                    double max = result.Probabilities.Max(p =>
                        golden.TryGetValue(p.Key, out var g) ? Math.Abs(p.Value - g) : 0);
                    diff = max.ToString("E2");
                }
                rows.Add($"| {dev}（实际 {used}）| {swPre.ElapsedMilliseconds} | {swSess.ElapsedMilliseconds} | " +
                         $"{result.PhaseMs["total_ms"]:F0} | {result.WindowCount} | {ort:F0} | {headMs:F0} | {diff} |");
            }
            finally
            {
                engine.Dispose();
            }
        }

        // ---- ② 扩展数据冒烟（CPU） ----
        rows.Add("");
        rows.Add("## 扩展数据集冒烟（merlin_data_train_demo/resized_images × CPU）");
        rows.Add("");
        rows.Add("| 文件 | 结果 | 打分条目 | 阳性(p≥0.5) | prob 范围 | 耗时(ms) |");
        rows.Add("|---|---|---|---|---|---|");

        string merlinDir = Path.Combine(root, "..", "damo-radar", "data", "merlin_data_train_demo", "resized_images");
        merlinDir = Path.GetFullPath(merlinDir);
        var files = Directory.Exists(merlinDir)
            ? Directory.GetFiles(merlinDir, "*.nii.gz").Concat(Directory.GetFiles(merlinDir, "*.nii")).ToArray()
            : Array.Empty<string>();

        if (files.Length == 0)
        {
            rows.Add("| （未找到 merlin 数据目录） | - | - | - | - | - |");
        }
        else foreach (var f in files)
        {
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var img = NiftiReader.Read(f);
                var prep = Preprocessor.Run(img, Path.GetFileName(f));
                var engine = new VisionEngine(modelPath, ExecutionDevice.Cpu);
                Dictionary<string, double> probs;
                int windows;
                try
                {
                    using var radar = new RadarInferenceEngine(engine, head, RadarLabels.Organs, items);
                    var result = radar.Run(prep.Data, prep.D, prep.H, prep.W, testOrgans);
                    probs = result.Probabilities.ToDictionary(p => p.Key, p => p.Value);
                    windows = result.WindowCount;
                }
                finally { engine.Dispose(); }

                bool nan = probs.Values.Any(double.IsNaN);
                string ok = nan ? "FAIL(含NaN)" : "PASS";
                rows.Add($"| {Path.GetFileName(f)}（{img.DimX}×{img.DimY}×{img.DimZ}，{windows} 窗）| {ok} | " +
                         $"{probs.Count} | {probs.Values.Count(v => v >= 0.5)} | " +
                         $"{probs.Values.Min():F4} ~ {probs.Values.Max():F4} | {sw.ElapsedMilliseconds} |");
            }
            catch (Exception ex)
            {
                rows.Add($"| {Path.GetFileName(f)} | FAIL({Truncate(ex.Message)}) | - | - | - | - |");
            }
        }

#if DEBUG
        const string buildCfg = "Debug";
#else
        const string buildCfg = "Release";
#endif
        string md = "# RADAR .NET 性能基准（M5）\n\n" +
                    $"生成时间：{DateTime.Now:yyyy-MM-dd HH:mm:ss} · 机内实测（{buildCfg} 构建，绝对值为同环境相对比较用）\n\n" +
                    string.Join("\n", rows) + "\n";
        File.WriteAllText(benchPath, md);
        Console.WriteLine(md);
    }

    private static string Truncate(string s) =>
        s.Length <= 60 ? s : s[..60].Replace("|", "/").Replace("\n", " ");

    private static Dictionary<string, double>? LoadGolden(string root)
    {
        string p = Path.Combine(root, "testdata", "golden", "organ_probs.json");
        if (!File.Exists(p)) return null;
        var json = System.Text.Json.JsonDocument.Parse(File.ReadAllText(p));
        return json.RootElement.EnumerateObject().ToDictionary(x => x.Name, x => x.Value.GetDouble());
    }

    private static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "models", "radar_vision.onnx")))
                return dir.FullName;
        throw new InvalidOperationException("未找到仓库根（含 models/radar_vision.onnx）");
    }
}
