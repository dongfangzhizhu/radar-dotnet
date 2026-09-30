using System.Text.Json;
using Radar.Core.Inference;
using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// M8 DICOM 有损压缩精度评估（PLAN.md M6 遗留待办的闭环）：
/// JPEG Baseline 8bit（HU 重标定 + 有损压缩，q90/q75 两档）解码后
/// 与无损 NIfTI 链路逐级对比——HU 误差 → 预处理误差 → 端到端 146 项概率偏移。
/// 结论（实测数据）写入 PLAN.md；本测试门禁为「记录性断言」：HU 误差超上界即失败。
/// </summary>
public class DicomLossyTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    private static string NiiPath => Path.Combine(Root, "testdata", "AC423ccbe.nii.gz");

    [Fact]
    public void LossyJpeg_QuantifiesError()
    {
        if (!File.Exists(NiiPath)) return;
        bool anySample = false;
        foreach (var q in new[] { 90, 75 })
        {
            string zip = Path.Combine(Root, "testdata", $"dicom_lossy_q{q}.zip");
            if (!File.Exists(zip)) continue;
            anySample = true;
            Evaluate($"q{q}", zip, endToEnd: q == 90);
        }
        if (!anySample)
            output.WriteLine("有损样例未生成（testdata/make_dicom_lossy.py 可重建），跳过");
    }

    private void Evaluate(string tag, string zipPath, bool endToEnd)
    {
        // ---- 参考链路：NIfTI 直读 ----
        var nifti = NiftiReader.Read(NiiPath);
        var prepRef = Preprocessor.Run(nifti, "ref");

        // ---- 有损链路：JPEG zip 解码 ----
        var lossy = DicomReader.ReadZip(zipPath);
        Assert.Equal(nifti.DimX, lossy.DimX);
        Assert.Equal(nifti.DimZ, lossy.DimZ);   // 368 切片数一致（轴序一致）

        // ---- HU 级误差 ----
        long n = Math.Min(nifti.Data.Length, lossy.Data.Length);
        double huMax = 0, huSum = 0;
        double huClipMax = 0, huClipSum = 0; long clipCount = 0;
        for (long i = 0; i < n; i++)
        {
            float a = nifti.Data[i], b = lossy.Data[i];
            double diff = Math.Abs(a - b);
            if (diff > huMax) huMax = diff;
            huSum += diff;
            if (a is >= Preprocessor.HuMin and <= Preprocessor.HuMax)
            {
                if (diff > huClipMax) huClipMax = diff;
                huClipSum += diff;
                clipCount++;
            }
        }
        output.WriteLine($"[{tag}] HU 全域: max={huMax:F1} mean={huSum / n:F2} | " +
            $"clip[-300,400]域内({clipCount * 100 / n}%体素): max={huClipMax:F1} mean={huClipSum / Math.Max(clipCount, 1):F2}");

        // 门禁口径（2026-09-30 实测确立，pydicom 独立解码交叉验证一致）：
        //   8bit 重标定量化步长 = (1758-(-3024))/255 ≈ 18.75 HU，量化误差 ≤ 半步长 ≈ 9.4 HU；
        //   实测均值 13.3(q90)/20.0(q75) HU 含 JPEG 振铃；max 出现在骨/气高对比边缘
        //   （q90 417 HU @ ref=-112 → rec=-530，块效应振铃），属编码固有特性非链路缺陷。
        // 门禁取保守上界：均值 < 30 HU、max < 1000 HU（后者只防解码链路崩坏）。
        Assert.True(huSum / n < 30.0, $"[{tag}] HU 均值误差 {huSum / n:F2} 超上界 30 HU");
        Assert.True(huMax < 1000.0, $"[{tag}] HU max 误差 {huMax:F1} 超上界 1000 HU（疑似解码链路异常）");

        // ---- 预处理级误差 ----
        var prepLossy = Preprocessor.Run(lossy, "lossy");
        Assert.Equal((prepRef.D, prepRef.H, prepRef.W), (prepLossy.D, prepLossy.H, prepLossy.W));
        double prepMax = 0;
        for (long i = 0; i < prepRef.Data.Length; i++)
            prepMax = Math.Max(prepMax, Math.Abs(prepRef.Data[i] - prepLossy.Data[i]));
        output.WriteLine($"[{tag}] 预处理 max_diff = {prepMax:E3}（归一化后 [0,1] 域）");

        // ---- 端到端（q90 一档，CPU 全链路 ~60s）----
        if (!endToEnd) return;
        string onnx = Path.Combine(Root, "models", "radar_vision.onnx");
        string npz = Path.Combine(Root, "models", "organ_head.npz");
        string bin = Path.Combine(Root, "models", "text_feats.bin");
        string jsonPath = Path.Combine(Root, "models", "text_feats.json");
        if (!File.Exists(onnx)) return;

        using var engine = new VisionEngine(onnx, ExecutionDevice.Cpu);
        var head = new OrganHead(OrganHeadWeights.Load(npz), TextFeatures.Load(bin, jsonPath));
        using var radar = new RadarInferenceEngine(engine, head,
            RadarLabels.Organs,
            TextFeatures.Load(bin, jsonPath).Items.ToList());
        var result = radar.Run(prepLossy.Data, prepLossy.D, prepLossy.H, prepLossy.W,
            RadarLabels.Organs);

        // 对比无损端到端 golden（organ_probs.json，146 项）
        string goldenPath = Path.Combine(Root, "testdata", "golden", "organ_probs.json");
        if (!File.Exists(goldenPath)) return;
        var golden = JsonDocument.Parse(File.ReadAllText(goldenPath)).RootElement.EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetDouble());
        double probMax = 0; int over50 = 0, flip = 0;
        foreach (var (item, g) in golden)
        {
            if (!result.Probabilities.TryGetValue(item, out var ours)) continue;
            double diff = Math.Abs(ours - g);
            if (diff > probMax) probMax = diff;
            if ((g >= 0.5) != (ours >= 0.5)) flip++;
            if (ours >= 0.5) over50++;
        }
        output.WriteLine($"[{tag}] 端到端: prob max_diff={probMax:E3}（无损 golden，{golden.Count} 项）| " +
            $"阳性判定翻转 {flip} 项 | 有损链路阳性 {over50} 项");
    }
}
