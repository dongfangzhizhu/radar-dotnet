using System.Numerics;

namespace Radar.Core.Preprocessing;

/// <summary>
/// 三线性重采样（half_pixel + 边界 clamp）。
/// 公式与 torch upsample_trilinear3d CPU / export/verify_trilinear.py 的 numpy
/// 参考实现逐式对应，全部 fp32 计算（含坐标）——实测与 torch 最大差 &lt; 7.2e-7。
/// 内存布局统一为 (D,H,W) 行主序：i = (d * H + h) * W + w。
/// </summary>
public static class Resampler
{
    public static float[] TrilinearResize(float[] src, int dIn, int hIn, int wIn, int dOut, int hOut, int wOut)
    {
        var dst = new float[dOut * hOut * wOut];

        // 每输出轴的取样表（fp32 全程，与 torch CPU kernel 一致）
        var (z0, z1, wz) = Grid(dIn, dOut);
        var (y0, y1, wy) = Grid(hIn, hOut);
        var (x0, x1, wx) = Grid(wIn, wOut);

        for (int zo = 0; zo < dOut; zo++)
        {
            float tz = wz[zo];
            float w0z = 1f - tz, w1z = tz;
            int rzi0 = z0[zo] * hIn * wIn, rzi1 = z1[zo] * hIn * wIn;
            int rozi = zo * hOut * wOut;

            for (int yo = 0; yo < hOut; yo++)
            {
                float ty = wy[yo];
                float w0y = 1f - ty, w1y = ty;
                int ryi0 = y0[yo] * wIn, ryi1 = y1[yo] * wIn;
                int royi = rozi + yo * wOut;

                for (int xo = 0; xo < wOut; xo++)
                {
                    float tx = wx[xo];
                    float w0x = 1f - tx, w1x = tx;

                    int b00 = ryi0 + x0[xo], b01 = ryi0 + x1[xo];
                    int b10 = ryi1 + x0[xo], b11 = ryi1 + x1[xo];

                    // 加权顺序与 numpy 参考实现一致：
                    // c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111
                    float v =
                        src[rzi0 + b00] * w0z * w0y * w0x +
                        src[rzi1 + b00] * w1z * w0y * w0x +
                        src[rzi0 + b10] * w0z * w1y * w0x +
                        src[rzi1 + b10] * w1z * w1y * w0x +
                        src[rzi0 + b01] * w0z * w0y * w1x +
                        src[rzi1 + b01] * w1z * w0y * w1x +
                        src[rzi0 + b11] * w0z * w1y * w1x +
                        src[rzi1 + b11] * w1z * w1y * w1x;

                    dst[royi + xo] = v;
                }
            }
        }
        return dst;
    }

    /// <summary>half_pixel 网格：返回 (i0, i1, t)，均 fp32。</summary>
    private static (int[] I0, int[] I1, float[] T) Grid(int nIn, int nOut)
    {
        var i0 = new int[nOut];
        var i1 = new int[nOut];
        var t = new float[nOut];

        // scale/src/t 全部 fp32（fp64 计算会引入 1.4e-4 偏差，实测）
        float scale = (float)nIn / (float)nOut;
        for (int o = 0; o < nOut; o++)
        {
            float src = (o + 0.5f) * scale - 0.5f;
            int f = (int)MathF.Floor(src);
            t[o] = src - f;
            i0[o] = Math.Clamp(f, 0, nIn - 1);
            i1[o] = Math.Clamp(f + 1, 0, nIn - 1);
        }
        return (i0, i1, t);
    }
}
