namespace Radar.Core.Inference;

/// <summary>
/// RADAR+ 端到端推理（复刻 infer_merlin_whole.evaluate 主循环）：
/// 预处理（PlusPreprocessor）产物 (96,256,384) 单窗整推（无滑窗、无器官头、无补推），
/// 三级 embeds 全量拼接 → whole head → 9 疾病阳性概率。
/// </summary>
public sealed class RadarPlusEngine
{
    private readonly VisionEngine _engine;
    private readonly PlusHead _head;

    public RadarPlusEngine(VisionEngine engine, PlusHead head)
    {
        _engine = engine;
        _head = head;
    }

    public sealed record Result(IReadOnlyDictionary<string, float> Probabilities,
        float[] Feature, IReadOnlyDictionary<string, double>? PhaseMs = null);

    public Result Run(float[] volume)
    {
        if (volume.Length != VisionEngine.InputD * VisionEngine.InputH * VisionEngine.InputW)
            throw new InvalidDataException(
                $"RADAR+ 输入须为 ({VisionEngine.InputD},{VisionEngine.InputH},{VisionEngine.InputW})，" +
                $"实际长度 {volume.Length}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var outs = _engine.Run(volume);
        long ortMs = sw.ElapsedMilliseconds;

        var (feat, probs) = _head.Score(outs.ResX1, outs.ResX2, outs.ResX3);
        long headMs = sw.ElapsedMilliseconds - ortMs;

        var phaseMs = new Dictionary<string, double>
        {
            ["ort_ms"] = ortMs,
            ["head_ms"] = headMs,
            ["total_ms"] = sw.ElapsedMilliseconds,
        };
        return new Result(probs, feat, phaseMs);
    }
}
