using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// M2 对拍测试：NiftiReader + Preprocessor 全管线 vs Python 黄金基线。
/// 黄金基线由 export/golden_dump.py 生成（testdata/golden/，不入库，可重建）。
/// </summary>
public class PreprocessorTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    [Fact]
    public void NiftiReader_DemoCase_MatchesNibabelFacts()
    {
        var path = Path.Combine(Root, "testdata", "AC423ccbe.nii.gz");
        if (!File.Exists(path)) return; // 数据未就位时跳过

        var img = NiftiReader.Read(path);
        Assert.Equal(512, img.DimX);
        Assert.Equal(512, img.DimY);
        Assert.Equal(368, img.DimZ);
        // nibabel 参考值：zooms (0.703125, 0.703125, 1.25)
        Assert.Equal(0.703125, img.Spacing.Sx, 6);
        Assert.Equal(0.703125, img.Spacing.Sy, 6);
        Assert.Equal(1.25, img.Spacing.Sz, 6);
        // nibabel 参考值：min=-3024.0 max=1758.0
        float mn = float.MaxValue, mx = float.MinValue;
        foreach (float v in img.Data)
        {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        Assert.Equal(-3024.0f, mn);
        Assert.Equal(1758.0f, mx);
    }

    [Fact]
    public void Preprocessor_FullPipeline_MatchesGoldenVolume()
    {
        var niiPath = Path.Combine(Root, "testdata", "AC423ccbe.nii.gz");
        var goldenPath = Path.Combine(Root, "testdata", "golden", "input_window.npy");
        if (!File.Exists(niiPath) || !File.Exists(goldenPath)) return;

        var golden = NpyReader.LoadFloat32(goldenPath, out var gShape);
        Assert.Equal(new[] { 1, 1, 96, 339, 384 }, gShape);

        var img = NiftiReader.Read(niiPath);
        var result = Preprocessor.Run(img, "AC423ccbe.nii.gz");

        Assert.Equal(96, result.D);
        Assert.Equal(339, result.H);
        Assert.Equal(384, result.W);
        Assert.Equal(golden.Length, result.Data.Length);

        double maxDiff = 0;
        long mismatch = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            double d = Math.Abs((double)golden[i] - result.Data[i]);
            if (d > maxDiff) maxDiff = d;
            if (d > 1e-5) mismatch++;
        }
        output.WriteLine($"max_diff={maxDiff:E3} mismatch(>1e-5)={mismatch}/{golden.Length}");
        Assert.Equal(0, mismatch);
    }
}
