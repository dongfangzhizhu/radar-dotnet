# -*- coding: utf-8 -*-
"""导出器官轻头权重 -> models/organ_head.npz。

从官方 checkpoint 提取推理必需的轻量头（共约 2.64M 参数）：
  attention.in_proj_weight  (768,256)   MHA 合并 QKV，行序 Q[0:256] K[256:512] V[512:768]
  attention.in_proj_bias    (768,)
  attention.out_proj.weight (256,256)
  attention.out_proj.bias   (256,)
  vision_projs.{0..35}.weight (256,256) 每器官一个投影层
  vision_projs.{0..35}.bias   (256,)
  query_tokens              (36,256)
  temp                      标量

npz 为无损容器，键名与 checkpoint 一致，C# 侧用自定义解析或转储二进制。
"""
import numpy as np
import torch

from common import MODELS_DIR, OFFICIAL_CKPT

OUT_PATH = MODELS_DIR / "organ_head.npz"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(str(OFFICIAL_CKPT), map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    out = {}
    keep_prefixes = ("attention.", "vision_projs.", "query_tokens", "temp")
    for k, v in sd.items():
        if k.startswith(keep_prefixes):
            out[k] = v.detach().cpu().numpy()

    n_params = sum(v.size for v in out.values())
    np.savez(str(OUT_PATH), **out)
    print(f"保存 {OUT_PATH}  键数={len(out)}  参数量={n_params} ({n_params*4/1048576:.2f} MB fp32)")

    # 回读校验
    z = np.load(str(OUT_PATH))
    assert set(z.files) == set(out.keys())
    for k in out:
        assert np.array_equal(z[k], out[k]), f"回读不一致: {k}"
    print("npz 回读校验通过")

    # 关键形状断言（C# 端契约）
    assert out["attention.in_proj_weight"].shape == (768, 256)
    assert out["query_tokens"].shape == (36, 256)
    assert out["vision_projs.0.weight"].shape == (256, 256)
    assert out["temp"].shape == ()
    print("形状契约校验通过（Q=行0:256, K=256:512, V=512:768）")


if __name__ == "__main__":
    main()
