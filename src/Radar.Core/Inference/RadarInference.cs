using Radar.Core.Preprocessing;

namespace Radar.Core.Inference;

/// <summary>
/// RADAR 端到端推理（复刻 inference_demo.evaluate + RADAR.forward_test_win）：
/// 滑窗 → 每窗 intact 器官打分（窗口级边界排除 margin=2）→ 缺失条目对应器官
/// center_crop 补推（DivisiblePad k=32）→ 多窗概率平均 → 146 项阳性概率。
/// </summary>
public sealed class RadarInferenceEngine : IDisposable
{
    private readonly VisionEngine _engine;
    private readonly OrganHead _head;
    private readonly IReadOnlyList<string> _organs;
    private readonly IReadOnlyList<string> _testItems;
    private readonly int _margin = 2;

    public RadarInferenceEngine(VisionEngine engine, OrganHead head,
        IReadOnlyList<string> organs, IReadOnlyList<string> testItems)
    {
        _engine = engine;
        _head = head;
        _organs = organs;
        _testItems = testItems;
    }

    public sealed record Result(IReadOnlyDictionary<string, double> Probabilities, int WindowCount,
        IReadOnlyDictionary<string, double>? PhaseMs = null);

    /// <summary>test_items 按 organ 名分组索引（item 名格式「器官_病灶」）。</summary>
    private readonly Dictionary<string, List<int>> _itemsByOrgan = new();

    public Result Run(float[] volume, int d, int h, int w, IReadOnlyList<string> testOrgans)
    {
        foreach (var group in _testItems.Select((it, i) => (it, i)).GroupBy(t => t.it.Split('_')[0]))
            _itemsByOrgan[group.Key] = group.Select(t => t.i).ToList();

        var organFeats = new Dictionary<string, float[]>();          // organ -> 已打分特征缓存
        var organProbs = new Dictionary<int, List<float[]>>();        // itemIndex -> 概率列表
        foreach (int i in Enumerable.Range(0, _testItems.Count))
            organProbs[i] = new List<float[]>();

        var testOrganSet = testOrgans.ToHashSet();
        int windowCount = 0;
        var phaseMs = new Dictionary<string, double>();   // 阶段计时（M5 基准用）
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // ---- 主滑窗（同时累加 seg_probs，供 argmax 拼接 → 补推用）----
        int di = Math.Max((int)(96 * 0.75), 1), hi = Math.Max((int)(256 * 0.75), 1), wi = Math.Max((int)(384 * 0.75), 1);
        var windows = SlidingWindowInference.DensePatchSlices((d, h, w), (96, 256, 384), (di, hi, wi));
        var fullMask = new float[37L * d * h * w];
        var countMap = new float[37L * d * h * w];

        foreach (var win in windows)
        {
            var window = ExtractWindow(volume, d, h, w, win);
            ForwardWindow(window, win.D1 - win.D0, win.H1 - win.H0, win.W1 - win.W0,
                organFeats, organProbs, testOrganSet, skipOrgan: null,
                fullMask, countMap, d, h, w, (win.D0, win.H0, win.W0), phaseMs);
            windowCount++;
        }

        // argmax 拼接 → stitched mask（补推的 bbox 搜索基于它）
        long vox = (long)d * h * w;
        phaseMs["interp_accum_ms"] = sw.ElapsedMilliseconds; // 主滑窗（含 interp 累加）累计到此
        var stitched = ArgmaxStitched(fullMask, countMap, d, h, w);
        phaseMs["argmax_ms"] = sw.ElapsedMilliseconds - phaseMs["interp_accum_ms"];

        // ---- 缺失条目补推（center_crop + DivisiblePad k=32）----
        foreach (var (item, itemIdx) in _testItems.Select((it, i) => (it, i)))
        {
            if (organProbs[itemIdx].Count > 0) continue;
            string organName = item.Split('_')[0];
            int organId = -1;
            for (int oi = 0; oi < _organs.Count; oi++)
                if (_organs[oi] == organName) { organId = oi; break; }
            if (organId < 0)
                throw new InvalidOperationException($"未知器官: {organName}");

            var (window, wd, wh, ww) = CenterCropWindow(volume, stitched, d, h, w, organId);
            if (wd == 0)
                continue; // 该器官在全图 mask 中无体素（与 Python 端 mask 边界翻转差异所致），无概率
            (window, wd, wh, ww) = DivisiblePad(window, wd, wh, ww, 32);

            if (wd == 96 && wh == 256 && ww == 384)
            {
                // 常规路径：crop 恰为固定 ONNX 输入（Python 原版行为：单窗整推）
                ForwardWindow(window, wd, wh, ww, organFeats, organProbs, testOrganSet,
                    skipOrgan: organId, phaseMs: phaseMs);
                windowCount++;
            }
            else
            {
                // R4 路径：bbox 超出 ROI（如器官 z 向跨 96+ 层），crop+pad 后 > 固定输入。
                // 对该区域做二次滑窗（子窗恒为 (96,256,384)），skip_organ 保证目标器官
                // 在每个子窗都不算边界，organFeats 缓存保证其特征只在首个非边界子窗计算一次。
                // 数值说明：与 Python 原版"单窗整推"有差（多窗各自视角），但主链路 146 项
                // 硬门禁（demo case 不触发此路径）不受影响。
                int di2 = Math.Max((int)(96 * 0.75), 1), hi2 = Math.Max((int)(256 * 0.75), 1),
                    wi2 = Math.Max((int)(384 * 0.75), 1);
                var subWindows = SlidingWindowInference.DensePatchSlices(
                    (wd, wh, ww), (96, 256, 384), (di2, hi2, wi2));
                if (subWindows.Count == 0)
                    throw new InvalidOperationException(
                        $"补推区域 ({wd},{wh},{ww}) 未能切出任何 (96,256,384) 子窗");
                foreach (var sub in subWindows)
                {
                    var subWin = ExtractWindow(window, wd, wh, ww, sub);
                    ForwardWindow(subWin, sub.D1 - sub.D0, sub.H1 - sub.H0, sub.W1 - sub.W0,
                        organFeats, organProbs, testOrganSet, skipOrgan: organId,
                        phaseMs: phaseMs);
                    windowCount++;
                }
            }
        }
        phaseMs["crop_pass_ms"] = sw.ElapsedMilliseconds - phaseMs["interp_accum_ms"] - phaseMs["argmax_ms"];

        // ---- 聚合：多窗概率平均，取阳性（index 1）----
        var result = new Dictionary<string, double>();
        for (int i = 0; i < _testItems.Count; i++)
        {
            var list = organProbs[i];
            if (list.Count == 0) continue;
            double pos = 0;
            foreach (var p in list) pos += p[1];
            result[_testItems[i]] = pos / list.Count;
        }
        phaseMs["total_ms"] = sw.ElapsedMilliseconds;
        return new Result(result, windowCount, phaseMs);
    }

    /// <summary>单窗前向 + intact 器官打分（复刻 forward_test_win）。</summary>
    private void ForwardWindow(float[] window, int wd, int wh, int ww,
        Dictionary<string, float[]> organFeats,
        Dictionary<int, List<float[]>> organProbs,
        HashSet<string> testOrganSet,
        int? skipOrgan,
        float[]? fullMask = null, float[]? countMap = null, int D = 0, int H = 0, int W = 0,
        (int D0, int H0, int W0)? origin = null,
        Dictionary<string, double>? phaseMs = null)
    {
        var swPhase = System.Diagnostics.Stopwatch.StartNew();
        var outs = _engine.Run(window);
        if (phaseMs != null)
            phaseMs["ort_ms"] = phaseMs.GetValueOrDefault("ort_ms") + swPhase.ElapsedMilliseconds;

        // seg_probs (37,96,128,192) → 上采样到窗尺寸 → 累加（与 evaluate 一致）
        if (fullMask != null && countMap != null && origin.HasValue)
        {
            int ld = 96, lh = 128, lw = 192;
            for (int c = 0; c < 37; c++)
            {
                var ch = new float[ld * lh * lw];
                Array.Copy(outs.SegProbs, (long)c * ld * lh * lw, ch, 0, ch.Length);
                var up = Resampler.TrilinearResize(ch, ld, lh, lw, wd, wh, ww);
                for (int zi = 0; zi < wd; zi++)
                {
                    long fmBase = ((long)c * D + origin.Value.D0 + zi) * H * W + (long)origin.Value.H0 * W + origin.Value.W0;
                    int cbBase = zi * wh * ww;
                    for (int yi = 0; yi < wh; yi++)
                    {
                        long fm = fmBase + (long)yi * W;
                        int cb = cbBase + yi * ww;
                        for (int xi = 0; xi < ww; xi++)
                        {
                            fullMask[fm + xi] += up[cb + xi];
                            countMap[fm + xi] += 1f;
                        }
                    }
                }
            }
        }

        // 窗口级 boundary organs：mask 边缘 margin=2 内出现的非零标签
        var boundary = CollectBoundaryOrgans(outs.PredMask, wd, wh, ww, _margin);
        if (skipOrgan.HasValue) boundary.Remove(skipOrgan.Value + 1);

        // organ id -> 体积计数（unique with counts 的等价：字典统计）
        var counts = new Dictionary<int, long>();
        for (long v = 0; v < (long)wd * wh * ww; v++)
        {
            int label = outs.PredMask[v];
            if (label != 0) counts[label] = counts.GetValueOrDefault(label) + 1;
        }

        foreach (var (label, _) in counts)
        {
            if (boundary.Contains(label)) continue;
            int organId = label - 1;
            string organName = _organs[organId];
            if (!testOrganSet.Contains(organName)) continue;
            if (organFeats.ContainsKey(organName)) continue;

            // 三尺度选中 token -> key 序列（顺序 x1, x2, x3）
            var flags1 = outs.Flags1.AsSpan(organId * 1152, 1152);
            var flags2 = outs.Flags2.AsSpan(organId * 9216, 9216);
            var flags3 = outs.Flags3.AsSpan(organId * 73728, 73728);

            swPhase.Restart();
            var (feat, scores) = _head.Score(organId, outs.ResX1, outs.ResX2, outs.ResX3, flags1, flags2, flags3);
            if (phaseMs != null)
                phaseMs["head_ms"] = phaseMs.GetValueOrDefault("head_ms") + swPhase.ElapsedMilliseconds;
            organFeats[organName] = feat; // 缓存特征向量（与原版 organ_feat_dict 对齐）

            if (_itemsByOrgan.TryGetValue(organName, out var items))
                foreach (int itemIdx in items)
                    organProbs[itemIdx].Add(new[] { 1f - scores[itemIdx], scores[itemIdx] });
        }
    }

    /// <summary>(full/count) 沿 37 类 argmax → (D,H,W) 标签图。</summary>
    internal static byte[] ArgmaxStitched(float[] fullMask, float[] countMap, int d, int h, int w)
    {
        long vox = (long)d * h * w;
        var argmax = new byte[vox];
        for (long v = 0; v < vox; v++)
        {
            float cnt = Math.Max(countMap[v], 1f);
            byte best = 0;
            float bestP = -1f;
            for (int c = 0; c < 37; c++)
            {
                float p = fullMask[c * vox + v] / cnt;
                if (p > bestP) { bestP = p; best = (byte)c; }
            }
            argmax[v] = best;
        }
        return argmax;
    }

    /// <summary>mask 边缘 margin 内出现的非零标签集合。</summary>
    internal static HashSet<int> CollectBoundaryOrgans(byte[] mask, int d, int h, int w, int margin)
    {
        var result = new HashSet<int>();
        // 直接三重循环最外层两片（axis=D）
        for (int zi = 0; zi < d; zi++)
        {
            if (zi >= margin && zi < d - margin) continue;
            for (int yi = 0; yi < h; yi++)
                for (int xi = 0; xi < w; xi++)
                    AddIfNonzero(result, mask, d, h, w, zi, yi, xi);
        }
        // axis=H
        for (int yi = 0; yi < h; yi++)
        {
            if (yi >= margin && yi < h - margin) continue;
            for (int zi = 0; zi < d; zi++)
                for (int xi = 0; xi < w; xi++)
                    AddIfNonzero(result, mask, d, h, w, zi, yi, xi);
        }
        // axis=W
        for (int xi = 0; xi < w; xi++)
        {
            if (xi >= margin && xi < w - margin) continue;
            for (int zi = 0; zi < d; zi++)
                for (int yi = 0; yi < h; yi++)
                    AddIfNonzero(result, mask, d, h, w, zi, yi, xi);
        }
        return result;
    }

    private static void AddIfNonzero(HashSet<int> into, byte[] mask, int d, int h, int w, int zi, int yi, int xi)
    {
        byte v = mask[(long)zi * h * w + (long)yi * w + xi];
        if (v != 0) into.Add(v);
    }

    /// <summary>
    /// center_crop：bbox 中心对齐、尺寸 max(roi, bbox)、clamp 到图像内（复刻 center_crop）。
    /// bbox 从 mask 找，返回窗口是 CT 影像体素（volume），与 Python 端 center_crop(image, mask) 一致。
    /// </summary>
    internal (float[] Window, int D, int H, int W) CenterCropWindow(
        float[] volume, byte[] mask, int d, int h, int w, int organId)
    {
        int label = organId + 1;
        long n = (long)d * h * w;
        int xMin = int.MaxValue, yMin = int.MaxValue, zMin = int.MaxValue;
        int xMax = int.MinValue, yMax = int.MinValue, zMax = int.MinValue;
        for (int zi = 0; zi < d; zi++)
            for (int yi = 0; yi < h; yi++)
            {
                int baseIdx = zi * h * w + yi * w;
                for (int xi = 0; xi < w; xi++)
                {
                    if (mask[baseIdx + xi] == label)
                    {
                        if (xi < xMin) xMin = xi;
                        if (xi > xMax) xMax = xi;
                        if (yi < yMin) yMin = yi;
                        if (yi > yMax) yMax = yi;
                        if (zi < zMin) zMin = zi;
                        if (zi > zMax) zMax = zi;
                    }
                }
            }
        if (xMax < xMin)
            return (Array.Empty<float>(), 0, 0, 0); // 该器官不在全图 mask 中

        int cropD = Math.Max(96, zMax - zMin);
        int cropH = Math.Max(256, yMax - yMin);
        int cropW = Math.Max(384, xMax - xMin);
        int cz = (zMin + zMax) / 2, cy = (yMin + yMax) / 2, cx = (xMin + xMax) / 2;

        int zStart = Math.Max(0, cz - cropD / 2);
        int zEnd = Math.Min(d, zStart + cropD);
        if (zEnd - zStart < cropD) zStart = Math.Max(0, zEnd - cropD);
        int yStart = Math.Max(0, cy - cropH / 2);
        int yEnd = Math.Min(h, yStart + cropH);
        if (yEnd - yStart < cropH) yStart = Math.Max(0, yEnd - cropH);
        int wStart = Math.Max(0, cx - cropW / 2);
        int wEnd = Math.Min(w, wStart + cropW);
        if (wEnd - wStart < cropW) wStart = Math.Max(0, wEnd - cropW);

        int wd = zEnd - zStart, wh = yEnd - yStart, ww = wEnd - wStart;
        var window = new float[wd * wh * ww];
        for (int zi = 0; zi < wd; zi++)
        {
            int srcBase = (zStart + zi) * h * w + yStart * w + wStart;
            int dstBase = zi * wh * ww;
            for (int yi = 0; yi < wh; yi++)
                Array.Copy(volume, srcBase + yi * w, window, dstBase + yi * ww, ww);
        }
        return (window, wd, wh, ww);
    }

    /// <summary>尾部 pad 到 k 的倍数（原版 DivisiblePadd method="end"）。</summary>
    internal static (float[] Data, int D, int H, int W) DivisiblePad(float[] data, int d, int h, int w, int k)
    {
        int pd = (int)Math.Ceiling((double)d / k) * k;
        int ph = (int)Math.Ceiling((double)h / k) * k;
        int pw = (int)Math.Ceiling((double)w / k) * k;
        if (pd == d && ph == h && pw == w) return (data, d, h, w);
        var padded = new float[pd * ph * pw];
        for (int zi = 0; zi < d; zi++)
        {
            int srcBase = zi * h * w;
            int dstBase = zi * ph * pw;
            for (int yi = 0; yi < h; yi++)
                Array.Copy(data, srcBase + yi * w, padded, dstBase + yi * pw, w);
        }
        return (padded, pd, ph, pw);
    }

    private static float[] ExtractWindow(float[] volume, int d, int h, int w,
        (int D0, int H0, int W0, int D1, int H1, int W1) win)
    {
        int wd = win.D1 - win.D0, wh = win.H1 - win.H0, ww = win.W1 - win.W0;
        var window = new float[wd * wh * ww];
        for (int zi = 0; zi < wd; zi++)
        {
            int srcBase = (win.D0 + zi) * h * w + win.H0 * w + win.W0;
            int dstBase = zi * wh * ww;
            for (int yi = 0; yi < wh; yi++)
                Array.Copy(volume, srcBase + yi * w, window, dstBase + yi * ww, ww);
        }
        return window;
    }

    /// <summary>引擎生命周期归调用方管理（支持会话跨任务复用），此处不级联 Dispose。</summary>
    public void Dispose() { }
}
