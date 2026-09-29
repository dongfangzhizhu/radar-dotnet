# -*- coding: utf-8 -*-
"""numpy 三线性重采样参考实现 + 与 torch F.interpolate 对拍。

为什么不用 ONNX Resize 子图：实测 ORT Resize 与 torch 在强下采样
（512->100）时 max diff 2e-4（实现路径差异），会污染端到端对拍。
改为手写实现（half_pixel + 边界 clamp，与 torch CPU kernel 同公式），
C# 侧 Radar.Core 逐公式对应。

torch upsample_trilinear3d CPU 语义（align_corners=False, half_pixel）：
  src = (dst + 0.5) * (in_size / out_size) - 0.5
  i0 = floor(src), t = src - i0
  i0c = clamp(i0, 0, in-1), i1c = clamp(i0+1, 0, in-1)
  v = Σ (w * x[i0c/i1c, ...])，8 邻域加权
"""
import numpy as np
import torch
import torch.nn.functional as F


def _grid(n_in: int, n_out: int):
    # 全程 fp32（与 torch CPU kernel 一致：scale/src/t 均为 float32 计算）
    scale = np.float32(n_in) / np.float32(n_out)
    src = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    i0 = np.floor(src).astype(np.int64)
    t = (src - i0.astype(np.float32)).astype(np.float32)
    i0c = np.clip(i0, 0, n_in - 1)
    i1c = np.clip(i0 + 1, 0, n_in - 1)
    return i0c, i1c, t


def trilinear_resize(x: np.ndarray, out_shape) -> np.ndarray:
    """x (N,C,D,H,W) fp32 -> (N,C,d,h,w) fp32，half_pixel 三线性。"""
    n, c, din, hin, win = x.shape
    dout, hout, wout = out_shape
    zd, zu, wz = _grid(din, dout)
    yd, yu, wy = _grid(hin, hout)
    xd, xu, wx = _grid(win, wout)

    def take(zi, yi, xi):
        return x[np.ix_(range(n), range(c), zi, yi, xi)]

    c000 = take(zd, yd, xd); c100 = take(zu, yd, xd)
    c010 = take(zd, yu, xd); c110 = take(zu, yu, xd)
    c001 = take(zd, yd, xu); c101 = take(zu, yd, xu)
    c011 = take(zd, yu, xu); c111 = take(zu, yu, xu)

    wz = wz[None, None, :, None, None]
    wy = wy[None, None, None, :, None]
    wx = wx[None, None, None, None, :]

    out = (
        c000 * (1 - wz) * (1 - wy) * (1 - wx) + c100 * wz * (1 - wy) * (1 - wx) +
        c010 * (1 - wz) * wy * (1 - wx) + c110 * wz * wy * (1 - wx) +
        c001 * (1 - wz) * (1 - wy) * wx + c101 * wz * (1 - wy) * wx +
        c011 * (1 - wz) * wy * wx + c111 * wz * wy * wx
    )
    return out.astype(np.float32)


def main():
    cases = [
        ((1, 1, 100, 512, 512), (48, 100, 100)),   # 强下采样（此前 ORT 超差点）
        ((1, 1, 64, 256, 256), (24, 64, 64)),
        ((1, 1, 32, 64, 64), (96, 128, 160)),
        ((1, 1, 7, 33, 45), (10, 41, 57)),          # 非整除边界
        ((1, 1, 96, 384, 384), (96, 339, 384)),     # demo case 实际目标
        ((1, 1, 1, 1, 1), (5, 5, 5)),               # 退化输入
    ]
    max_d = 0.0
    for shape, sz in cases:
        x = torch.randn(*shape)
        ref = F.interpolate(x, size=list(sz), mode="trilinear", align_corners=False).numpy()
        out = trilinear_resize(x.numpy(), sz)
        d = float(np.max(np.abs(ref - out)))
        max_d = max(max_d, d)
        print(f"  {shape}->{sz}: max_diff={d:.3e}")
    assert max_d < 1e-6, f"对拍超差: {max_d}"
    print(f"PASS — numpy 参考实现与 torch 最大差 {max_d:.3e} < 1e-6")
    print("C# Radar.Core.Resampler 按同一公式实现（见 src/Radar.Core/Preprocessing）")


if __name__ == "__main__":
    main()
