using System.Text.Json;
using Radar.Core.Inference;
using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// M7 对拍测试：RADAR+ whole-image 路径 vs Python 黄金基线（export/plus_golden.py）。
/// 黄金基线：ORT CPU 视觉主干 + monai 预处理（merlin 变体）+ numpy 手写 whole head。
/// </summary>
public class PlusTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    private static string NiiPath => Path.Combine(Root, "testdata", "AC423ccbe.nii.gz");
    private static string GoldenVolPath => Path.Combine(Root, "testdata", "golden", "plus_preproc_merlin_ft.npy");

    [Fact]
    public void PlusPreprocessor_MatchesGoldenVolume()
    {
        if (!File.Exists(NiiPath) || !File.Exists(GoldenVolPath)) return; // 数据未就位时跳过

        var golden = NpyReader.LoadFloat32(GoldenVolPath, out var gShape);
        Assert.Equal(new[] { 96, 256, 384 }, gShape);

        var img = NiftiReader.Read(NiiPath);
        var result = PlusPreprocessor.Run(img, "AC423ccbe.nii.gz");

        Assert.Equal(96, result.D);
        Assert.Equal(256, result.H);
        Assert.Equal(384, result.W);

        double maxDiff = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            double diff = Math.Abs(golden[i] - result.Data[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        output.WriteLine($"预处理 max_diff = {maxDiff:0.###e+00}");
        Assert.True(maxDiff < 1e-5, $"预处理 max_diff={maxDiff} 超门限 1e-5");
    }

    [Fact]
    public void PlusHead_BothVariants_MatchGolden()
    {
        string merlinJson = Path.Combine(Root, "testdata", "golden", "plus_golden_merlin_ft.json");
        string pretrainJson = Path.Combine(Root, "testdata", "golden", "plus_golden_pretrain.json");
        if (!File.Exists(NiiPath) || !File.Exists(GoldenVolPath)
            || !File.Exists(merlinJson) || !File.Exists(pretrainJson)) return;

        // 预处理（有 golden 体积则直接用，保证与 golden 同一输入）
        float[] volume;
        var goldenVol = NpyReader.LoadFloat32(GoldenVolPath, out var gShape);
        Assert.Equal(new[] { 96, 256, 384 }, gShape);
        volume = goldenVol;

        using var engine = new VisionEngine(Path.Combine(Root, "models", "radar_vision.onnx"),
            ExecutionDevice.Cpu);
        var outs = engine.Run(volume);

        VerifyVariant("merlin_ft", merlinJson,
            Path.Combine(Root, "models", "plus_head_merlin_ft.npz"),
            Path.Combine(Root, "models", "plus_text_feats_merlin_ft.bin"),
            Path.Combine(Root, "models", "plus_text_feats_merlin_ft.json"),
            outs);
        VerifyVariant("pretrain", pretrainJson,
            Path.Combine(Root, "models", "plus_head.npz"),
            Path.Combine(Root, "models", "plus_text_feats.bin"),
            Path.Combine(Root, "models", "plus_text_feats.json"),
            outs);
    }

    private void VerifyVariant(string name, string goldenJsonPath,
        string headNpz, string textBin, string textJson, WindowOutputs outs)
    {
        var golden = JsonDocument.Parse(File.ReadAllText(goldenJsonPath)).RootElement;
        var goldenFeat = golden.GetProperty("feature").EnumerateArray()
            .Select(v => v.GetSingle()).ToArray();
        var goldenProbs = golden.GetProperty("probs").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetDouble());

        var head = new PlusHead(
            PlusHeadWeights.Load(headNpz),
            PlusTextFeatures.Load(textBin, textJson));
        var (feat, probs) = head.Score(outs.ResX1, outs.ResX2, outs.ResX3);

        double featMax = 0;
        for (int i = 0; i < goldenFeat.Length; i++)
            featMax = Math.Max(featMax, Math.Abs(goldenFeat[i] - feat[i]));

        double probMax = 0;
        foreach (var (k, v) in goldenProbs)
        {
            Assert.True(probs.ContainsKey(k), $"缺疾病: {k}");
            probMax = Math.Max(probMax, Math.Abs(v - probs[k]));
        }

        output.WriteLine($"[{name}] feature max_diff = {featMax:0.###e+00}, prob max_diff = {probMax:0.###e+00}");
        foreach (var (k, v) in goldenProbs)
            output.WriteLine($"    {k,-24} golden={v:0.000000} dotnet={probs[k]:0.000000}");

        Assert.True(featMax < 1e-4, $"[{name}] feature max_diff={featMax} 超门限 1e-4");
        Assert.True(probMax < 1e-3, $"[{name}] prob max_diff={probMax} 超门限 1e-3");
    }

    [Fact]
    public void PlusHeadWeights_Load_Contract()
    {
        string npz = Path.Combine(Root, "models", "plus_head_merlin_ft.npz");
        if (!File.Exists(npz)) return;
        var w = PlusHeadWeights.Load(npz);
        Assert.Equal(768 * 256, w.InProjW.Length);
        Assert.Equal(768, w.InProjB.Length);
        Assert.Equal(256 * 256, w.OutProjW.Length);
        Assert.Equal(256 * 256, w.ProjW.Length);
        Assert.Equal(256, w.Query.Length);
        Assert.True(w.Temp is > 0.01f and < 0.2f, $"temp 异常: {w.Temp}");
    }

    [Fact]
    public void PlusTextFeatures_Load_Contract()
    {
        string bin = Path.Combine(Root, "models", "plus_text_feats_merlin_ft.bin");
        string json = Path.Combine(Root, "models", "plus_text_feats_merlin_ft.json");
        if (!File.Exists(bin)) return;
        var t = PlusTextFeatures.Load(bin, json);
        Assert.Equal(9, t.Diseases.Count);
        Assert.Equal(134, t.TotalPrompts);
        Assert.All(t.Diseases, d =>
        {
            Assert.True(d.NumPositive > 0 && d.Count >= d.NumPositive, $"{d.Name} 计数异常");
            Assert.True(d.Start + d.Count <= t.TotalPrompts, $"{d.Name} 越界");
        });
    }
}
