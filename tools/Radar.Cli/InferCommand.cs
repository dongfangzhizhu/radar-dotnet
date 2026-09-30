using System.Diagnostics;
using Radar.Core.Inference;
using Radar.Core.Nifti;
using Radar.Core.Postprocessing;
using Radar.Core.Preprocessing;
using Radar.Core.Support;
using Radar.Core.Visualization;

namespace Radar.Cli;

/// <summary>
/// <c>radar infer</c>：目录级批量推理，输出与官方对齐的宽表 CSV。
/// 对应上游 <c>inference_demo.py --img_dir/--save_dir/--save_tag</c>。
/// 多 worker × 多设备按输入顺序切分与聚合，等价上游多卡 <c>all_gather</c> 的有序汇总。
/// </summary>
internal static class InferCommand
{
    /// <summary>可识别的病例扩展名（按长度降序，保证 .nii.gz 先于 .nii 匹配）。</summary>
    private static readonly string[] Extensions = [".nii.gz", ".nii", ".dcm", ".zip"];

    private sealed record CaseResult(
        string FileName, Dictionary<string, double>? Probs, string Status,
        string? Note, long ElapsedMs, int Windows);

    public static int Run(Options o)
    {
        string repoRoot = CliEnv.FindRepoRoot();
        string imgDir = o.Get("img-dir") ?? throw new ArgumentException("必须指定 --img-dir");
        if (!Directory.Exists(imgDir))
            throw new DirectoryNotFoundException($"病例目录不存在: {imgDir}");

        string saveDir = o.Get("save-dir") ?? Path.Combine(repoRoot, "results");
        string saveTag = o.GetOr("save-tag", "demo");
        string model = o.GetOr("model", "radar").ToLowerInvariant();
        string variant = o.GetOr("variant", "merlin_ft").ToLowerInvariant();
        bool saveMask = o.Flag("save-mask");
        int limit = o.GetInt("limit", 0);

        var devices = o.GetList("devices");
        if (devices.Length == 0) devices = [o.GetOr("device", "cpu")];
        int workers = Math.Max(1, o.GetInt("workers", 1));

        if (model is not ("radar" or "radar_plus"))
            throw new ArgumentException($"--model 只支持 radar | radar_plus，收到 {model}");
        bool isPlus = model == "radar_plus";

        string modelsDir = Path.Combine(repoRoot, "models");
        string modelPath = Path.Combine(modelsDir, "radar_vision.onnx");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"缺少模型产物 {modelPath}，请先按 README §4 重建（models/ 不入库）");

        var cases = ScanCases(imgDir);
        if (limit > 0) cases = cases.Take(limit).ToList();
        if (cases.Count == 0)
        {
            Console.Error.WriteLine($"目录下没有可识别的病例（支持 {string.Join(" / ", Extensions)}）: {imgDir}");
            return 3;
        }

        string csvPath = Path.Combine(saveDir, $"RADAR_infer_results_{saveTag}.csv");
        string masksDir = Path.Combine(saveDir, "masks");

        // ---- 共享只读依赖（纯数学，线程安全）----
        OrganHead? organHead = null;
        PlusHead? plusHead = null;
        List<string> items;
        List<string> testOrgans = [];

        if (isPlus)
        {
            string suffix = variant == "pretrain" ? "" : "_merlin_ft";
            plusHead = new PlusHead(
                PlusHeadWeights.Load(Path.Combine(modelsDir, $"plus_head{suffix}.npz")),
                PlusTextFeatures.Load(
                    Path.Combine(modelsDir, $"plus_text_feats{suffix}.bin"),
                    Path.Combine(modelsDir, $"plus_text_feats{suffix}.json")));
            items = plusHead.Text.Diseases.Select(d => d.Name).ToList();
        }
        else
        {
            organHead = new OrganHead(
                OrganHeadWeights.Load(Path.Combine(modelsDir, "organ_head.npz")),
                TextFeatures.Load(
                    Path.Combine(modelsDir, "text_feats.bin"),
                    Path.Combine(modelsDir, "text_feats.json")));
            items = organHead.Text.Items.ToList();
            testOrgans = items.Select(i => i.Split('_')[0]).Distinct().ToList();
        }

        Console.WriteLine($"仓库根    : {repoRoot}");
        Console.WriteLine($"病例目录  : {imgDir}（{cases.Count} 例）");
        Console.WriteLine($"模型      : {model}{(isPlus ? $" / {variant}" : "")}"
            + $"（{items.Count} 个打分条目）");
        Console.WriteLine($"设备      : {string.Join(", ", devices)}"
            + (workers > 1 ? $"，{workers} 个 worker" : ""));
        Console.WriteLine($"输出      : {csvPath}");
        if (saveMask)
        {
            if (isPlus)
                Console.WriteLine("掩码      : radar_plus 路径不产出分割掩码，--save-mask 已忽略");
            else
                Console.WriteLine($"掩码      : {masksDir}");
        }
        Console.WriteLine();

        // ---- 每个 worker 独占一个推理会话 ----
        var engines = new VisionEngine[workers];
        for (int i = 0; i < workers; i++)
            engines[i] = new VisionEngine(modelPath, CliEnv.ParseDevice(devices[i % devices.Length]));

        var results = new CaseResult[cases.Count];
        var done = 0;
        var printLock = new object();
        var totalSw = Stopwatch.StartNew();

        try
        {
            Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, k =>
            {
                for (int i = k; i < cases.Count; i += workers)
                {
                    var r = ProcessCase(cases[i], engines[k], isPlus, saveMask, masksDir,
                        organHead, plusHead, testOrgans, items);
                    results[i] = r;

                    lock (printLock)
                    {
                        int n = Interlocked.Increment(ref done);
                        string mark = r.Status switch { "ok" => "OK  ", "skipped" => "SKIP", _ => "FAIL" };
                        Console.WriteLine($"[{n,3}/{cases.Count}] {mark} {r.FileName}"
                            + (r.Status == "ok"
                                ? $"  {r.ElapsedMs / 1000.0:F1}s {r.Windows}窗"
                                : $"  {r.Note}"));
                    }
                }
            });
        }
        finally
        {
            foreach (var e in engines) e.Dispose();
        }
        totalSw.Stop();

        // ---- 汇总输出（保序）----
        var ok = results.Where(r => r.Status == "ok").ToList();
        var skipped = results.Where(r => r.Status == "skipped").ToList();
        var failed = results.Where(r => r.Status == "failed").ToList();

        if (ok.Count > 0)
        {
            Directory.CreateDirectory(saveDir);
            ResultCsv.Write(csvPath, items,
                ok.Select(r => new ResultCsv.Row(r.FileName, r.Probs!)));
        }

        Console.WriteLine();
        Console.WriteLine("──────── 汇总 ────────");
        Console.WriteLine($"成功 : {ok.Count}");
        Console.WriteLine($"跳过 : {skipped.Count}" + (skipped.Count > 0 ? "（空间维超 1000，与上游一致）" : ""));
        Console.WriteLine($"失败 : {failed.Count}");
        Console.WriteLine($"总耗时: {totalSw.Elapsed.TotalSeconds:F1}s");
        if (ok.Count > 0) Console.WriteLine($"结果表: {csvPath}");

        foreach (var f in failed)
            Console.Error.WriteLine($"  失败 {f.FileName}: {f.Note}");

        return failed.Count > 0 ? 1 : 0;
    }

    private static CaseResult ProcessCase(
        string path, VisionEngine engine, bool isPlus, bool saveMask, string masksDir,
        OrganHead? organHead, PlusHead? plusHead, List<string> testOrgans, List<string> items)
    {
        string fileName = Path.GetFileName(path);
        var sw = Stopwatch.StartNew();
        try
        {
            var img = ReadImage(path);

            if (isPlus)
            {
                var prep = PlusPreprocessor.Run(img, fileName);
                var plus = new RadarPlusEngine(engine, plusHead!);
                var res = plus.Run(prep.Data);
                sw.Stop();
                var probs = res.Probabilities.ToDictionary(p => p.Key, p => (double)p.Value);
                return new CaseResult(fileName, probs, "ok", null, sw.ElapsedMilliseconds, 1);
            }

            var pre = Preprocessor.Run(img, fileName);
            using var radar = new RadarInferenceEngine(engine, organHead!, RadarLabels.Organs, items);
            var r = radar.Run(pre.Data, pre.D, pre.H, pre.W, testOrgans);

            if (saveMask && r.StitchedMask is not null)
            {
                Directory.CreateDirectory(masksDir);
                var png = MaskOverlayPng.Render(pre.Data, r.D, r.H, r.W, r.StitchedMask);
                File.WriteAllBytes(Path.Combine(masksDir, StemOf(fileName) + ".png"), png);
            }

            sw.Stop();
            return new CaseResult(fileName, new Dictionary<string, double>(r.Probabilities),
                "ok", null, sw.ElapsedMilliseconds, r.WindowCount);
        }
        catch (CaseSkippedException ex)
        {
            return new CaseResult(fileName, null, "skipped", ex.Message, 0, 0);
        }
        catch (Exception ex)
        {
            return new CaseResult(fileName, null, "failed", ex.Message, 0, 0);
        }
    }

    private static NiftiImage ReadImage(string path)
        => path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ? DicomReader.ReadZip(path)
         : path.EndsWith(".dcm", StringComparison.OrdinalIgnoreCase) ? DicomReader.ReadFile(path)
         : NiftiReader.Read(path);

    /// <summary>目录扫描，按文件名排序保证多 worker 分片与结果聚合的确定性。</summary>
    private static List<string> ScanCases(string dir)
    {
        var found = new List<string>();
        foreach (var ext in Extensions)
            found.AddRange(Directory.GetFiles(dir, "*" + ext, SearchOption.TopDirectoryOnly));
        return found
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(p => Path.GetFileName(p), StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>去掉已知的病例扩展名（处理 .nii.gz 双扩展名）。</summary>
    private static string StemOf(string fileName)
    {
        foreach (var ext in Extensions)
            if (fileName.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                return fileName[..^ext.Length];
        return Path.GetFileNameWithoutExtension(fileName);
    }
}
