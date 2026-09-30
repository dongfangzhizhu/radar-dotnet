using System.Text.Json;
using Radar.Core.Inference;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// 超大病例跳过守卫（实测差距 #5）。
/// 上游 <c>inference_demo.py</c> 在预处理后检查 <c>image.shape[1:]</c>，任一维 &gt; 1000 就
/// <c>continue</c>（跳过该病例，不中断整批）。批处理入口必须能区分「跳过」与「失败」。
/// </summary>
public class SkipGuardTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    [Fact]
    public void MaxSpatialDim_Is1000()
        => Assert.Equal(1000, RadarInferenceEngine.MaxSpatialDim);

    [Theory]
    [InlineData(1001, 100, 100, true)]
    [InlineData(100, 1001, 100, true)]
    [InlineData(100, 100, 1001, true)]
    [InlineData(2000, 2000, 2000, true)]
    [InlineData(1000, 1000, 1000, false)]      // 边界：等于 1000 不算超限（上游是 > 1000）
    [InlineData(96, 339, 384, false)]
    [InlineData(1, 1, 1, false)]
    public void ExceedsSpatialLimit_BoundaryBehaviour(int d, int h, int w, bool expected)
        => Assert.Equal(expected, RadarInferenceEngine.ExceedsSpatialLimit(d, h, w));

    [Fact]
    public void CaseSkippedException_IsNotAnInferenceFailure()
    {
        var ex = new CaseSkippedException("x");
        Assert.IsAssignableFrom<Exception>(ex);
        Assert.IsNotType<InvalidOperationException>(ex);
        // 关键约定：批处理靠这个类型区分「跳过」，不能与普通异常混淆
        Assert.Equal("Radar.Core.Inference.CaseSkippedException", ex.GetType().FullName);
    }

    /// <summary>
    /// 真实引擎上的守卫：构造完整推理链后喂一个 d=1001 的病例，
    /// 必须在<b>进入滑窗之前</b>抛出 <see cref="CaseSkippedException"/>（否则会白跑几分钟）。
    /// </summary>
    [Fact]
    public void Run_ThrowsCaseSkipped_BeforeAnyInference()
    {
        string onnx = Path.Combine(Root, "models", "radar_vision.onnx");
        string npz = Path.Combine(Root, "models", "organ_head.npz");
        string bin = Path.Combine(Root, "models", "text_feats.bin");
        string json = Path.Combine(Root, "models", "text_feats.json");
        if (new[] { onnx, npz, bin, json }.Any(p => !File.Exists(p))) return;

        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "testdata", "golden", "meta.json")));
        var organs = meta.RootElement.GetProperty("organs").EnumerateArray().Select(v => v.GetString()!).ToList();
        var testItems = meta.RootElement.GetProperty("test_items").EnumerateArray().Select(v => v.GetString()!).ToList();

        using var engine = new VisionEngine(onnx, ExecutionDevice.Cpu);
        var head = new OrganHead(OrganHeadWeights.Load(npz), TextFeatures.Load(bin, json));
        using var radar = new RadarInferenceEngine(engine, head, organs, testItems);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var ex = Assert.Throws<CaseSkippedException>(() =>
        {
            radar.Run(new float[1001], 1001, 1, 1, ["主动脉"]);
        });
        sw.Stop();

        output.WriteLine($"抛出耗时 {sw.ElapsedMilliseconds} ms：{ex.Message}");
        Assert.Contains("1001", ex.Message);
        Assert.Contains("1000", ex.Message);
        // 守卫在滑窗之前，必须远快于一次真实推理（秒级以内）
        Assert.True(sw.Elapsed.TotalSeconds < 30, $"守卫耗时 {sw.Elapsed.TotalSeconds:F1}s，疑似未提前返回");
    }
}
