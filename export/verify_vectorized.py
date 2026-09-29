# -*- coding: utf-8 -*-
"""对拍 1：原版 VisionBranch vs 向量化 VecVisionBranch（torch 内部，导出前硬门槛）。

判定标准：8 个输出张量 max abs diff < 1e-6（pred_mask/flags 为离散值，要求完全一致）。
输入使用固定种子的随机张量（BN 已 eval，统计量固定，随机输入即可暴露语义差异）。
"""
import torch

from common import max_abs_diff
from vec_vision_branch import VecVisionBranch, build_vectorized


def build_original():
    """原版 VisionBranch（直接 import 上游模块，不改一行）。"""
    from dynamic_network_architectures.vision_branch import VisionBranch
    m = VisionBranch()
    from common import load_visual_state_dict
    load_visual_state_dict(m)
    m.eval()
    return m


def main():
    torch.manual_seed(42)
    x = torch.randn(1, 1, 96, 256, 384)

    print("构建原版 VisionBranch ...")
    orig = build_original()
    print("构建向量化 VecVisionBranch ...")
    vec = build_vectorized()

    # 两模型参数应逐键一致（结构等价的静态证明）
    sd_o = orig.state_dict()
    sd_v = vec.state_dict()
    assert set(sd_o.keys()) == set(sd_v.keys()), "state_dict 键集合不一致"
    for k in sd_o:
        if sd_o[k].dtype.is_floating_point:
            d = max_abs_diff(sd_o[k], sd_v[k])
            assert d == 0.0, f"权重不一致: {k} diff={d}"
    print(f"权重逐键一致（{len(sd_o)} 键）")

    print("原版前向 ...")
    with torch.no_grad():
        o = orig(x, None)
    print("向量化前向 ...")
    with torch.no_grad():
        v = vec(x)

    names = ["seg_probs", "pred_mask", "res_x1", "res_x2", "res_x3", "flags1", "flags2", "flags3"]
    ok = True
    for n, to, tv in zip(names, o, v):
        d = max_abs_diff(to, tv)
        exact = torch.equal(to.long(), tv.long()) if n in ("pred_mask", "flags1", "flags2", "flags3") else None
        status = "OK"
        if n in ("pred_mask", "flags1", "flags2", "flags3"):
            if not exact:
                status = "FAIL(离散值不一致)"
                ok = False
        elif d >= 1e-6:
            status = "FAIL(超差)"
            ok = False
        print(f"{n}: max_abs_diff={d:.3e} {status}")

    print("\n结果:", "PASS — 向量化与原版语义等价" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
