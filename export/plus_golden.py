# -*- coding: utf-8 -*-
"""RADAR+ 黄金基线转储：ORT CPU 视觉主干 + numpy 手写 whole head + 预计算文本特征。

复刻 infer_merlin_whole.py 的 whole-image 路径：
  merlin 预处理变体（pad→center crop→clip ±1000→min-max 无 1e-8）
  → radar_vision.onnx 单窗整推（CPU EP，与 .NET 同一模型文件同一 EP）
  → cat(x1,x2,x3) → attention_whole → vision_projs_whole → L2 norm
  → 与 plus_text_feats*.bin 打分（pos/neg 分组均值 → softmax）

产物（testdata/golden/）:
  plus_preproc_merlin_ft.npy   预处理体积 (96,256,384) fp32（.NET 预处理比对）
  plus_golden_merlin_ft.json   {feature(256), probs{9 疾病}}
  plus_golden_pretrain.json    同上（pretrain 权重；共享同一预处理与 embeds）

运行解释器: C:/Users/Administrator/miniconda3/python.exe
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from common import MODELS_DIR, TESTDATA_DIR

GOLDEN = TESTDATA_DIR / "golden"
NII_PATH = TESTDATA_DIR / "AC423ccbe.nii.gz"

VARIANTS = {
    "merlin_ft": {
        "head": "plus_head_merlin_ft.npz",
        "text_bin": "plus_text_feats_merlin_ft.bin",
        "text_json": "plus_text_feats_merlin_ft.json",
        "prefix": "merlin_ft",
    },
    "pretrain": {
        "head": "plus_head.npz",
        "text_bin": "plus_text_feats.bin",
        "text_json": "plus_text_feats.json",
        "prefix": "pretrain",
    },
}


def preprocess_merlin(image_path: str) -> np.ndarray:
    """复刻 infer_merlin_whole DataFolder.__getitem__（L493-531），monai 同参。"""
    from monai import transforms
    res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(
        {"image": image_path})
    image = res["image"]

    affine = res["image_meta_dict"]["affine"]
    spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
    _, h, w, d = image.shape

    ref_spacing = (1.0, 1.0, 5.0)
    scale = [spacing[i] / ref_spacing[i] for i in range(3)]
    target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]

    trans = transforms.Compose([
        transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
        transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
    ])
    image = trans(res)["image"]

    # pad(end) → center crop（monai 同参）
    pad_func = transforms.SpatialPadd(keys=["image"], spatial_size=(96, 256, 384),
                                      mode="constant", constant_values=0, method="end")
    image = pad_func({"image": image})["image"]
    image = transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 256, 384))(
        {"image": image})["image"]

    # clip ±1000 → min-max 归一化（无 1e-8）
    image[image > 1000] = 1000
    image[image < -1000] = -1000
    image = (image - image.min()) / (image.max() - image.min())
    return image[0].numpy().astype(np.float32)   # (D,H,W) = (96,256,384)


def whole_head_forward(x1, x2, x3, head_npz: Path):
    """朴素 numpy whole head（与 torch nn.MultiheadAttention 等价，独立于 C# 优化式）。"""
    z = np.load(str(head_npz))
    ipw = z["attention_whole.in_proj_weight"].astype(np.float32)   # (768,256)
    ipb = z["attention_whole.in_proj_bias"].astype(np.float32)
    ow = z["attention_whole.out_proj.weight"].astype(np.float32)
    ob = z["attention_whole.out_proj.bias"].astype(np.float32)
    pw = z["vision_projs_whole.weight"].astype(np.float32)
    pb = z["vision_projs_whole.bias"].astype(np.float32)
    query = z["query_tokens_whole"].astype(np.float32).ravel()      # (256)
    temp = float(z["temp"])

    key = np.concatenate([x1, x2, x3], axis=0).astype(np.float32)   # (84096,256)
    n_heads, head_dim = 4, 64

    q = query @ ipw[:256].T + ipb[:256]                              # (256)
    kv = key @ ipw[256:].T + ipb[256:]                               # (84096,512)
    k_mat, v_mat = kv[:, :256], kv[:, 256:]

    out = np.zeros(256, dtype=np.float32)
    for h in range(n_heads):
        off = h * head_dim
        q_h = q[off:off + head_dim]
        attn = (k_mat[:, off:off + head_dim] @ q_h) / np.sqrt(head_dim)
        attn = np.exp(attn - attn.max())
        attn /= attn.sum()
        out[off:off + head_dim] = attn @ v_mat[:, off:off + head_dim]
    out = out @ ow.T + ob
    feat = out @ pw.T + pb
    feat = feat / np.linalg.norm(feat)
    return feat.astype(np.float32), temp


def load_text_feats(bin_path: Path, json_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    feats = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 256)
    return meta["diseases"], feats


def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)

    # 1) merlin 预处理 → 落盘供 .NET 比对
    vol = preprocess_merlin(str(NII_PATH))
    assert vol.shape == (96, 256, 384), f"预处理形状异常: {vol.shape}"
    np.save(str(GOLDEN / "plus_preproc_merlin_ft.npy"), vol)
    print(f"预处理体积: shape={vol.shape} min={vol.min():.6f} max={vol.max():.6f} "
          f"sum={vol.sum():.3f}")

    # 2) ORT CPU 单窗整推
    sess = ort.InferenceSession(str(MODELS_DIR / "radar_vision.onnx"),
                                providers=["CPUExecutionProvider"])
    outs = sess.run(["res_x1", "res_x2", "res_x3"],
                    {sess.get_inputs()[0].name: vol[None, None]})
    x1, x2, x3 = (o.reshape(-1, 256) for o in outs)
    print(f"embeds: x1{x1.shape} x2{x2.shape} x3{x3.shape}")

    # 3) 两个变体 whole head + 打分
    for name, spec in VARIANTS.items():
        feat, temp = whole_head_forward(x1, x2, x3, MODELS_DIR / spec["head"])
        diseases, tfeats = load_text_feats(
            MODELS_DIR / spec["text_bin"], MODELS_DIR / spec["text_json"])
        probs = {}
        for dis in diseases:
            sims = (tfeats @ feat) / temp
            pos = sims[dis["start"]:dis["start"] + dis["num_positive"]].mean()
            neg_start = dis["start"] + dis["num_positive"]
            neg = sims[neg_start:neg_start + dis["count"] - dis["num_positive"]].mean()
            e = np.exp([neg - max(neg, pos), pos - max(neg, pos)])
            probs[dis["name"]] = float(e[1] / e.sum())
        out = {"feature": feat.tolist(), "probs": probs, "temp": temp}
        out_path = GOLDEN / f"plus_golden_{spec['prefix']}.json"
        out_path.write_text(json.dumps(out), encoding="utf-8")
        print(f"[{name}] temp={temp:.6f} probs:")
        for k, v in probs.items():
            print(f"    {k:24s} {v:.6f}")
        print(f"  保存 {out_path}")

    print("golden 转储完成")


if __name__ == "__main__":
    main()
