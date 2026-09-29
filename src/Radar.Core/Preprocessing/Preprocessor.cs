using Radar.Core.Nifti;

namespace Radar.Core.Preprocessing;

/// <summary>
/// RADAR 预处理管线，逐行对应原版 DataFolder.__getitem__（inference_demo.py 163-239 行）：
///   spacing 重采样目标 → 三线性 resize → HU clip [-300,400] → min-max 归一化
///   → 非零域裁剪 + 外扩 (5,20,20) → 尾部 pad 到 (96,256,384)。
/// 内存布局 (D,H,W) 行主序；D 对应 z 轴（重采样到 5mm）。
/// </summary>
public static class Preprocessor
{
    public const int TargetD = 96;
    public const int TargetH = 256;
    public const int TargetW = 384;
    public const float HuMin = -300f;
    public const float HuMax = 400f;
    public const int ExtendD = 5;
    public const int ExtendHw = 20;
    /// <summary>参考 spacing (x, y, z) mm，与原版 REF_SPACING 一致。</summary>
    public static readonly (double X, double Y, double Z) RefSpacing = (1.0, 1.0, 5.0);

    public sealed record Result(float[] Data, int D, int H, int W, string FileName);

    public static Result Run(NiftiImage img, string fileName)
    {
        // 原版：_, h, w, d = image.shape（image 为 (1, x, y, z)），spacing 取 |affine 对角|
        int x = img.DimX, y = img.DimY, z = img.DimZ;
        double sx = img.Spacing.Sx, sy = img.Spacing.Sy, sz = img.Spacing.Sz;

        double scaleX = sx / RefSpacing.X;
        double scaleY = sy / RefSpacing.Y;
        double scaleZ = sz / RefSpacing.Z;
        int targetX = (int)(x * scaleY);  // target[0] = int(h * scale[1])
        int targetY = (int)(y * scaleX);  // target[1] = int(w * scale[0])
        int targetZ = (int)(z * scaleZ);  // target[2] = int(d * scale[2])

        // NIfTI (X,Y,Z) fortran 序 → (Z,Y,X) 行主序（即原版 Transpose(0,3,2,1) 后的轴序）
        var zyx = ToZyxRowMajor(img.Data, x, y, z);

        // resize（各轴独立三线性，等价于原版对 (1,1,X,Y,Z) 的 interpolate 后转置）
        var resized = Resampler.TrilinearResize(zyx, z, y, x, targetZ, targetY, targetX);

        // HU clip
        for (int i = 0; i < resized.Length; i++)
        {
            if (resized[i] > HuMax) resized[i] = HuMax;
            else if (resized[i] < HuMin) resized[i] = HuMin;
        }

        // min-max 归一化（全卷，fp32 + 1e-8 与原版一致）
        float mn = float.MaxValue, mx = float.MinValue;
        foreach (float v in resized)
        {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float denom = mx - mn + 1e-8f;
        for (int i = 0; i < resized.Length; i++)
            resized[i] = (resized[i] - mn) / denom;

        // 非零域裁剪 + 外扩（clip 后背景恰为 0）
        int d = targetZ, h = targetY, w = targetX;
        int dMin = int.MaxValue, dMax = int.MinValue;
        int hMin = int.MaxValue, hMax = int.MinValue;
        int wMin = int.MaxValue, wMax = int.MinValue;
        for (int zi = 0; zi < d; zi++)
        {
            int baseZ = zi * h * w;
            for (int yi = 0; yi < h; yi++)
            {
                int baseY = baseZ + yi * w;
                for (int xi = 0; xi < w; xi++)
                {
                    if (resized[baseY + xi] != 0f)
                    {
                        if (zi < dMin) dMin = zi;
                        if (zi > dMax) dMax = zi;
                        if (yi < hMin) hMin = yi;
                        if (yi > hMax) hMax = yi;
                        if (xi < wMin) wMin = xi;
                        if (xi > wMax) wMax = xi;
                    }
                }
            }
        }

        dMin = Math.Max(dMin - ExtendD, 0);
        hMin = Math.Max(hMin - ExtendHw, 0);
        wMin = Math.Max(wMin - ExtendHw, 0);
        dMax = Math.Min(dMax + ExtendD, d);
        hMax = Math.Min(hMax + ExtendHw, h);
        wMax = Math.Min(wMax + ExtendHw, w);

        int cd = dMax - dMin, ch = hMax - hMin, cw = wMax - wMin;

        // 裁剪
        var cropped = new float[cd * ch * cw];
        for (int zi = 0; zi < cd; zi++)
        {
            int srcBase = (dMin + zi) * h * w + hMin * w + wMin;
            int dstBase = zi * ch * cw;
            for (int yi = 0; yi < ch; yi++)
                Array.Copy(resized, srcBase + yi * w, cropped, dstBase + yi * cw, cw);
        }

        // 尾部 pad 到 (96,256,384)（原版 SpatialPadd method="end"）。
        // 注意：SpatialPadd 只 pad 不裁剪——crop 后大于目标的维度保持原尺寸
        // （demo case H=339 > 256 即此情况，golden 体积为 (96,339,384)）。
        int outD = Math.Max(cd, TargetD);
        int outH = Math.Max(ch, TargetH);
        int outW = Math.Max(cw, TargetW);
        var padded = new float[outD * outH * outW];
        for (int zi = 0; zi < cd; zi++)
        {
            int srcBase = zi * ch * cw;
            int dstBase = zi * outH * outW;
            for (int yi = 0; yi < ch; yi++)
                Array.Copy(cropped, srcBase + yi * cw, padded, dstBase + yi * outW, cw);
        }

        return new Result(padded, outD, outH, outW, fileName);
    }

    /// <summary>NIfTI fortran 序 (X 最快) → (Z,Y,X) 行主序 float32。</summary>
    internal static float[] ToZyxRowMajor(float[] data, int x, int y, int z)
    {
        var outp = new float[x * y * z];
        long i = 0;
        for (int zi = 0; zi < z; zi++)
        {
            int baseZ = zi * y;
            for (int yi = 0; yi < y; yi++)
            {
                long srcBase = (long)zi * x * y + yi * x;
                for (int xi = 0; xi < x; xi++)
                    outp[i++] = data[srcBase + xi];
            }
        }
        return outp;
    }
}
