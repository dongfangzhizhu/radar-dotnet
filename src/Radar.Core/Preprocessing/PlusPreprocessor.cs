using Radar.Core.Nifti;

namespace Radar.Core.Preprocessing;

/// <summary>
/// RADAR+（whole image）预处理，复刻 infer_merlin_whole.py DataFolder.__getitem__（L493-531）：
///   spacing 重采样目标 → 三线性 resize → Transpose(0,3,2,1)（与前半段同 Preprocessor）
///   → SpatialPad(96,256,384) 尾部补 → CenterSpatialCrop(96,256,384)
///   → clip ±1000 → min-max 归一化（无 1e-8）。
/// 与器官路径差异：无非零域裁剪；clip/归一化发生在 pad/crop 之后；clip 域 ±1000。
/// 输出恒为 (96,256,384)（单窗整推，无滑窗）。
/// </summary>
public static class PlusPreprocessor
{
    public const float HuMin = -1000f;
    public const float HuMax = 1000f;

    public static Preprocessor.Result Run(NiftiImage img, string fileName)
    {
        // ---- 前半段与 Preprocessor 完全一致（resize 目标 + ZYX 重排 + 三线性）----
        int x = img.DimX, y = img.DimY, z = img.DimZ;
        double sx = img.Spacing.Sx, sy = img.Spacing.Sy, sz = img.Spacing.Sz;

        double scaleX = sx / Preprocessor.RefSpacing.X;
        double scaleY = sy / Preprocessor.RefSpacing.Y;
        double scaleZ = sz / Preprocessor.RefSpacing.Z;
        int targetX = (int)(x * scaleY);
        int targetY = (int)(y * scaleX);
        int targetZ = (int)(z * scaleZ);

        var zyx = Preprocessor.ToZyxRowMajor(img.Data, x, y, z);
        var resized = Resampler.TrilinearResize(zyx, z, y, x, targetZ, targetY, targetX);

        // ---- merlin 变体：pad → center crop ----
        var (data, d, h, w) = PadCenterCrop(resized, targetZ, targetY, targetX);

        // ---- clip ±1000 → min-max 归一化（无 1e-8，逐式对应 L528-530）----
        float mn = float.MaxValue, mx = float.MinValue;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] > HuMax) data[i] = HuMax;
            else if (data[i] < HuMin) data[i] = HuMin;
            if (data[i] < mn) mn = data[i];
            if (data[i] > mx) mx = data[i];
        }
        float denom = mx - mn;
        for (int i = 0; i < data.Length; i++)
            data[i] = (data[i] - mn) / denom;

        return new Preprocessor.Result(data, d, h, w, fileName);
    }

    /// <summary>
    /// SpatialPad(method=end) + CenterSpatialCrop 合成：每轴
    ///   padded = max(size, target)；start = padded/2 - target/2（monai
    ///   CenterSpatialCrop.compute_slices：roi_center=size//2，starts=center - roi//2）。
    /// 输出恒为 (tD,tH,tW)。
    /// </summary>
    internal static (float[] Data, int D, int H, int W) PadCenterCrop(
        float[] src, int d, int h, int w)
    {
        int tD = Preprocessor.TargetD, tH = Preprocessor.TargetH, tW = Preprocessor.TargetW;
        int pd = Math.Max(d, tD), ph = Math.Max(h, tH), pw = Math.Max(w, tW);
        int sd = pd / 2 - tD / 2, sh = ph / 2 - tH / 2, sw = pw / 2 - tW / 2;

        var dst = new float[tD * tH * tW];
        // dst 恒为 0 初始化：pad 区域保持 0，仅拷贝 crop 窗口与源的交集
        int srcZ0 = Math.Max(0, sd), srcZ1 = Math.Min(d, sd + tD);
        int srcY0 = Math.Max(0, sh), srcY1 = Math.Min(h, sh + tH);
        int srcW0 = Math.Max(0, sw), srcW1 = Math.Min(w, sw + tW);
        int cw = srcW1 - srcW0;
        if (cw <= 0) return (dst, tD, tH, tW);
        for (int zi = srcZ0; zi < srcZ1; zi++)
        {
            int dz = zi - sd;
            for (int yi = srcY0; yi < srcY1; yi++)
            {
                long srcBase = (long)zi * h * w + (long)yi * w + srcW0;
                int dstBase = (zi - sd) * tH * tW + (yi - sh) * tW + (srcW0 - sw);
                Array.Copy(src, srcBase, dst, dstBase, cw);
            }
        }
        return (dst, tD, tH, tW);
    }
}
