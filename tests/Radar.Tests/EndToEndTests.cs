using System.Text.Json;
using Radar.Core.Inference;
using Radar.Core.Preprocessing;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// M3-2 端到端对拍：.NET 全链路（预处理 → 滑窗 → 器官头 → 补推 → 聚合）
/// vs Python 黄金基线 organ_probs.json（146 项）。
/// 硬门禁：全部 diff &lt; 1e-3（PLAN.md §6）。
/// </summary>
public class EndToEndTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    [Fact]
    public void FullPipeline_MatchesGoldenOrganProbs()
    {
        string onnx = Path.Combine(Root, "models", "radar_vision.onnx");
        string npz = Path.Combine(Root, "models", "organ_head.npz");
        string bin = Path.Combine(Root, "models", "text_feats.bin");
        string json = Path.Combine(Root, "models", "text_feats.json");
        string input = Path.Combine(Root, "testdata", "golden", "input_window.npy");
        string goldenPath = Path.Combine(Root, "testdata", "golden", "organ_probs.json");
        if (new[] { onnx, npz, bin, json, input, goldenPath }.Any(p => !File.Exists(p)))
            return; // 数据未就位（golden 可由 export/golden_dump.py 重建）

        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "testdata", "golden", "meta.json")));
        var organs = meta.RootElement.GetProperty("organs").EnumerateArray().Select(v => v.GetString()!).ToList();
        var testItems = meta.RootElement.GetProperty("test_items").EnumerateArray().Select(v => v.GetString()!).ToList();
        var testOrgans = meta.RootElement.GetProperty("test_organs").EnumerateArray().Select(v => v.GetString()!).ToList();

        var volume = NpyReader.LoadFloat32(input, out var inShape);
        int d = inShape[2], h = inShape[3], w = inShape[4];

        var golden = JsonDocument.Parse(File.ReadAllText(goldenPath)).RootElement.EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetDouble());

        using var engine = new VisionEngine(onnx, ExecutionDevice.Cpu);
        var head = new OrganHead(OrganHeadWeights.Load(npz), TextFeatures.Load(bin, json));
        using var radar = new RadarInferenceEngine(engine, head, organs, testItems);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = radar.Run(volume, d, h, w, testOrgans);
        sw.Stop();
        output.WriteLine($"端到端耗时 {sw.Elapsed.TotalSeconds:F1}s（{result.WindowCount} 窗）");

        // 逐项对拍
        double maxDiff = 0;
        int matched = 0, missing = 0;
        foreach (var (item, goldenVal) in golden)
        {
            if (!result.Probabilities.TryGetValue(item, out var ours))
            {
                missing++;
                output.WriteLine($"缺失: {item}");
                continue;
            }
            double diff = Math.Abs(ours - goldenVal);
            if (diff > maxDiff) maxDiff = diff;
            if (diff < 1e-3) matched++;
            else output.WriteLine($"超差({diff:E3}): {item} ours={ours:E6} golden={goldenVal:E6}");
        }
        output.WriteLine($"匹配 {matched}/{golden.Count}，缺失 {missing}，max_diff={maxDiff:E3}");
        Assert.Equal(0, missing);
        Assert.Equal(golden.Count, matched); // 硬门禁：全部 < 1e-3
    }
}
