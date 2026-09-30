namespace Radar.Core.Visualization;

/// <summary>
/// 分割掩码透叠图：把预处理后的 CT 体数据按三个正交方向做最大密度投影（MIP），
/// 再把 stitched 掩码半透明着色叠上去，横向拼成一张 PNG。
/// 对应 PLAN §3.3 承诺的「分割掩码可视化」。
/// </summary>
public static class MaskOverlayPng
{
    /// <summary>视图间留白（像素）。</summary>
    private const int Gap = 8;

    /// <summary>掩码着色不透明度。</summary>
    private const double Alpha = 0.55;

    /// <summary>
    /// 渲染掩码透叠图。<paramref name="volume"/> 为预处理后的归一化体数据 [0,1]，
    /// <paramref name="mask"/> 为同尺寸 argmax 标签（0=背景，1..N=器官 id+1）。
    /// </summary>
    public static byte[] Render(float[] volume, int d, int h, int w, byte[] mask)
    {
        if (volume.Length < (long)d * h * w)
            throw new ArgumentException($"volume 长度不足: {volume.Length} < {d * h * w}");
        if (mask.Length < (long)d * h * w)
            throw new ArgumentException($"mask 长度不足: {mask.Length} < {d * h * w}");

        // 三个正交视图（沿 axis 做 MIP）
        var views = new List<(int W, int H, float[] Gray, byte[] Label)>
        {
            Project(volume, mask, d, h, w, axis: 0),   // 沿 D（轴状位）：出图 H×W
            Project(volume, mask, d, h, w, axis: 1),   // 沿 H（冠状位）：出图 D×W
            Project(volume, mask, d, h, w, axis: 2),   // 沿 W（矢状位）：出图 D×H
        };

        int canvasH = views.Max(v => v.H);
        int canvasW = views.Sum(v => v.W) + Gap * (views.Count - 1);
        var rgb = new byte[(long)canvasW * canvasH * 3];

        int x0 = 0;
        foreach (var v in views)
        {
            for (int y = 0; y < v.H; y++)
            {
                for (int x = 0; x < v.W; x++)
                {
                    int src = y * v.W + x;
                    double g = Math.Clamp(v.Gray[src], 0f, 1f) * 255.0;
                    byte label = v.Label[src];
                    byte r, gg, b;
                    if (label == 0)
                    {
                        r = gg = b = (byte)g;
                    }
                    else
                    {
                        var (cr, cg, cb) = OrganColor(label);
                        r = (byte)(g * (1 - Alpha) + cr * Alpha);
                        gg = (byte)(g * (1 - Alpha) + cg * Alpha);
                        b = (byte)(g * (1 - Alpha) + cb * Alpha);
                    }
                    long dst = ((long)y * canvasW + (x0 + x)) * 3;
                    rgb[dst] = r; rgb[dst + 1] = gg; rgb[dst + 2] = b;
                }
            }
            x0 += v.W + Gap;   // 视图之间的留白保持全黑
        }

        return PngEncoder.Encode(canvasW, canvasH, rgb);
    }

    /// <summary>沿指定轴做最大密度投影：灰度取最大值，标签取最大非零值。</summary>
    private static (int W, int H, float[] Gray, byte[] Label) Project(
        float[] volume, byte[] mask, int d, int h, int w, int axis)
    {
        int ow = axis switch { 0 => w, 1 => w, _ => h };
        int oh = axis switch { 0 => h, 1 => d, _ => d };
        var gray = new float[ow * oh];
        var label = new byte[ow * oh];

        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                long rowBase = ((long)z * h + y) * w;
                for (int x = 0; x < w; x++)
                {
                    int oy, ox;
                    switch (axis)
                    {
                        case 0: oy = y; ox = x; break;   // 投影掉 z
                        case 1: oy = z; ox = x; break;   // 投影掉 y
                        default: oy = z; ox = y; break;  // 投影掉 x
                    }
                    int o = oy * ow + ox;
                    float v = volume[rowBase + x];
                    if (v > gray[o]) gray[o] = v;
                    byte lb = mask[rowBase + x];
                    if (lb > label[o]) label[o] = lb;
                }
            }
        }
        return (ow, oh, gray, label);
    }

    /// <summary>器官 id（1..36）-> 可区分颜色，用 HSV 色环等分生成。</summary>
    internal static (byte R, byte G, byte B) OrganColor(int label)
    {
        double hue = (label - 1) * 360.0 / 36.0;
        double c = 0.95, vv = 1.0;
        double hp = hue / 60.0;
        double x = c * (1 - Math.Abs(hp % 2 - 1));
        double r1, g1, b1;
        switch ((int)hp)
        {
            case 0: r1 = c; g1 = x; b1 = 0; break;
            case 1: r1 = x; g1 = c; b1 = 0; break;
            case 2: r1 = 0; g1 = c; b1 = x; break;
            case 3: r1 = 0; g1 = x; b1 = c; break;
            case 4: r1 = x; g1 = 0; b1 = c; break;
            default: r1 = c; g1 = 0; b1 = x; break;
        }
        double m = vv - c;
        return ((byte)Math.Round((r1 + m) * 255),
                (byte)Math.Round((g1 + m) * 255),
                (byte)Math.Round((b1 + m) * 255));
    }
}
