using Radar.Core.Inference;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// M3-1 对拍：滑窗调度 + ORT 视觉引擎 + 拼接 argmax vs Python 黄金基线。
/// 需要 models/radar_vision.onnx 与 testdata/golden/（golden_dump.py 可重建）。
/// </summary>
public class SlidingWindowTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    private static bool DataReady(out string onnx, out string input, out string stitched)
    {
        onnx = Path.Combine(Root, "models", "radar_vision.onnx");
        input = Path.Combine(Root, "testdata", "golden", "input_window.npy");
        stitched = Path.Combine(Root, "testdata", "golden", "stitched_mask.npy");
        return File.Exists(onnx) && File.Exists(input) && File.Exists(stitched);
    }

    [Fact]
    public void DensePatchSlices_MatchesGoldenMeta()
    {
        var metaPath = Path.Combine(Root, "testdata", "golden", "meta.json");
        if (!File.Exists(metaPath)) return;

        var json = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath));
        var size = json.RootElement.GetProperty("input_shape");
        int d = size[2].GetInt32(), h = size[3].GetInt32(), w = size[4].GetInt32();
        var interval = json.RootElement.GetProperty("scan_interval");
        var expected = json.RootElement.GetProperty("slices")
            .EnumerateArray()
            .Select(s => (s[0].GetInt32(), s[1].GetInt32(), s[2].GetInt32(),
                          s[3].GetInt32(), s[4].GetInt32(), s[5].GetInt32()))
            .ToList();

        var actual = SlidingWindowInference.DensePatchSlices(
            (d, h, w), (96, 256, 384), (interval[0].GetInt32(), interval[1].GetInt32(), interval[2].GetInt32()));

        Assert.Equal(expected.Count, actual.Count);
        Assert.Equal(expected, actual);
        output.WriteLine($"窗调度一致: {actual.Count} 窗");
    }

    [Fact]
    public void SlidingWindowArgmax_MatchesGoldenStitchedMask()
    {
        if (!DataReady(out var onnx, out var input, out var stitched)) return;

        var volume = NpyReader.LoadFloat32(input, out var inShape);
        int d = inShape[2], h = inShape[3], w = inShape[4];
        var goldenRaw = NpyReader.Load(stitched); // int16 存储（dump 时为 (1,96,339,384)，取后三维）
        var gShape = goldenRaw.Shape;
        Assert.Equal((d, h, w), (gShape[^3], gShape[^2], gShape[^1]));
        var flatGolden = ((short[])goldenRaw.Data).Select(v => (double)v).ToArray();

        using var engine = new VisionEngine(onnx, ExecutionDevice.Cpu);
        output.WriteLine($"EP = {engine.Device}");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = new SlidingWindowInference(engine, (96, 256, 384), 0.25).Run(volume, d, h, w);
        sw.Stop();
        output.WriteLine($"滑窗 {result.WindowCount} 窗耗时 {sw.Elapsed.TotalSeconds:F1}s");

        long mismatch = 0;
        double maxDiff = 0;
        for (int i = 0; i < flatGolden.Length; i++)
        {
            double diff = Math.Abs(flatGolden[i] - result.ArgmaxMask[i]);
            if (diff > maxDiff) maxDiff = diff;
            if (diff > 0) mismatch++;
        }
        output.WriteLine($"argmax 不匹配 voxel: {mismatch}/{flatGolden.Length} ({(double)mismatch / flatGolden.Length:P5}) max_diff={maxDiff}");
        // 与 ONNX 对拍标准一致：不匹配率 < 1e-4
        Assert.True((double)mismatch / flatGolden.Length < 1e-4, $"argmax 不匹配率超阈值: {mismatch}");
    }
}
