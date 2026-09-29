namespace Radar.Core.Inference;

/// <summary>
/// 滑窗推理：窗调度（等价 monai dense_patch_slices）→ 逐窗 VisionEngine →
/// seg_probs 上采样累加 / 计数平均 → argmax → 全图器官标签图。
/// 逐行对应原版 evaluate 主循环（inference_demo.py 436-481 行）。
/// </summary>
public sealed class SlidingWindowInference
{
    public sealed record Result(byte[] ArgmaxMask, int D, int H, int W, int WindowCount);

    private readonly VisionEngine _engine;
    private readonly int _roiD, _roiH, _roiW;
    private readonly double _overlap;

    public SlidingWindowInference(VisionEngine engine, (int D, int H, int W) roi, double overlap = 0.25)
    {
        _engine = engine;
        _roiD = roi.D; _roiH = roi.H; _roiW = roi.W;
        _overlap = overlap;
    }

    /// <summary>等价 monai dense_patch_slices：返回每窗 (d0,h0,w0,d1,h1,w1)。</summary>
    public static List<(int D0, int H0, int W0, int D1, int H1, int W1)> DensePatchSlices(
        (int D, int H, int W) imageSize, (int D, int H, int W) patchSize, (int Di, int Hi, int Wi) scanInterval)
    {
        static int ScanCount(int imSize, int patchSize, int interval) =>
            patchSize >= imSize ? 1 : (int)Math.Ceiling((double)(imSize - patchSize) / interval) + 1;

        int nd = ScanCount(imageSize.D, patchSize.D, scanInterval.Di);
        int nh = ScanCount(imageSize.H, patchSize.H, scanInterval.Hi);
        int nw = ScanCount(imageSize.W, patchSize.W, scanInterval.Wi);

        static List<int> Starts(int imSize, int patchSize, int interval, int count)
        {
            var starts = new List<int>(count);
            for (int i = 0; i < count; i++)
                starts.Add(Math.Min(interval * i, imSize - patchSize));
            return starts;
        }

        var ds = Starts(imageSize.D, patchSize.D, scanInterval.Di, nd);
        var hs = Starts(imageSize.H, patchSize.H, scanInterval.Hi, nh);
        var ws = Starts(imageSize.W, patchSize.W, scanInterval.Wi, nw);

        var slices = new List<(int, int, int, int, int, int)>(nd * nh * nw);
        foreach (int d in ds)
            foreach (int h in hs)
                foreach (int w in ws)
                    slices.Add((d, h, w, d + patchSize.D, h + patchSize.H, w + patchSize.W));
        return slices;
    }

    /// <summary>
    /// 对整卷 (D,H,W) 行主序执行滑窗推理，返回 argmax 标签图。
    /// volume 尺寸任意（≥roi 时滑窗，否则整体 pad 由调用方预处理保证）。
    /// </summary>
    public Result Run(float[] volume, int D, int H, int W)
    {
        // scan_interval = int(roi * (1 - overlap))，0 时取 1（原版 _get_scan_interval）
        int di = Math.Max((int)(_roiD * (1 - _overlap)), 1);
        int hi = Math.Max((int)(_roiH * (1 - _overlap)), 1);
        int wi = Math.Max((int)(_roiW * (1 - _overlap)), 1);
        var windows = DensePatchSlices((D, H, W), (_roiD, _roiH, _roiW), (di, hi, wi));

        var fullMask = new float[37 * D * H * W];
        var countMap = new float[37 * D * H * W];

        foreach (var (d0, h0, w0, d1, h1, w1) in windows)
        {
            int wd = d1 - d0, wh = h1 - h0, ww = w1 - w0;

            var window = new float[_roiD * _roiH * _roiW];
            for (int zi = 0; zi < wd; zi++)
            {
                int srcBase = (d0 + zi) * H * W + h0 * W + w0;
                int dstBase = zi * _roiH * _roiW;
                for (int yi = 0; yi < wh; yi++)
                    Array.Copy(volume, srcBase + yi * W, window, dstBase + yi * _roiW, ww);
            }

            var outs = _engine.Run(window);

            // seg_probs (37,96,128,192) → interpolate 到窗尺寸 → 累加
            // （原版 evaluate: F.interpolate(mode="trilinear")，与 Resampler 同公式）
            int logitsD = 96, logitsH = 128, logitsW = 192;
            var channels = new float[37][];
            for (int c = 0; c < 37; c++)
            {
                var ch = new float[logitsD * logitsH * logitsW];
                Array.Copy(outs.SegProbs, c * logitsD * logitsH * logitsW, ch, 0, ch.Length);
                channels[c] = Resampler(ch, wd, wh, ww);
            }

            for (int c = 0; c < 37; c++)
            {
                var chv = channels[c];
                for (int zi = 0; zi < wd; zi++)
                {
                    int fmBase = ((c * D) + d0 + zi) * H * W + h0 * W + w0;
                    int chBase = zi * wh * ww;
                    for (int yi = 0; yi < wh; yi++)
                    {
                        int fm = fmBase + yi * W;
                        int cb = chBase + yi * ww;
                        for (int xi = 0; xi < ww; xi++)
                        {
                            fullMask[fm + xi] += chv[cb + xi];
                            countMap[fm + xi] += 1f;
                        }
                    }
                }
            }
        }

        // stitched = (full/count).argmax(1)
        var argmax = new byte[D * H * W];
        long vox = (long)D * H * W;
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

        return new Result(argmax, D, H, W, windows.Count);
    }

    private float[] Resampler(float[] channel, int wd, int wh, int ww) =>
        Preprocessing.Resampler.TrilinearResize(channel, 96, 128, 192, wd, wh, ww);
}
