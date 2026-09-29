# -*- coding: utf-8 -*-
"""黄金基线 dump：对官方 demo case (AC423ccbe.nii.gz) 复刻 inference_demo.evaluate
的完整推理，落盘逐级中间结果，供 .NET (Radar.Core) 逐阶段对拍。

不 import inference_demo（其 import 链依赖旧版 transformers）：
- 预处理/_get_scan_interval/center_crop 逐行复刻自 inference_demo.py（163-122 行段）；
- test_items 大列表用 ast.literal_eval 从原源码字面量提取，零手抄误差；
- 模型用向量化 VecVisionBranch（已证与原版逐位一致，见 verify_vectorized.py）。

最终正确性门禁：organ_probs.json 与官方 CSV (testdata/golden/RADAR_infer_results_demo.csv)
逐值 diff < 1e-3（M3 验收）。
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data.utils import dense_patch_slices

from common import (
    DAMO_RADAR_INFERENCE, MODELS_DIR, N_ORGANS, NUM_CLASSES, ORGANS,
    ROI_SIZE, SLIDING_OVERLAP, TESTDATA_DIR, TEXT_EMBEDDING_PT,
)

GOLDEN = TESTDATA_DIR / "golden"
DUMP_FULL_WINDOWS = 3  # 前 N 窗 + 末窗完整 dump（含 8 输出）


def extract_test_items() -> list:
    """从原源码提取 test_items 列表字面量（零手抄误差）。"""
    src = (DAMO_RADAR_INFERENCE / "inference_demo.py").read_text(encoding="utf-8")
    m = re.search(r"self\.test_items = (\[[^\]]*\])", src, re.S)
    if not m:
        raise RuntimeError("未能从 inference_demo.py 提取 test_items")
    return ast.literal_eval(m.group(1))


# ---- 以下三个函数逐行复刻自 inference_demo.py ----

def get_scan_interval(image_size, roi_size, num_spatial_dims, overlap):
    """复刻 inference_demo._get_scan_interval。"""
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def masks_to_boxes_3d(masks):
    """复刻 inference_demo.masks_to_boxes_3d。"""
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)
    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing="ij")
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def center_crop(image, mask, crop_size):
    """复刻 inference_demo.center_crop。"""
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    crop_d, crop_h, crop_w = max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min), max(crop_size[2], x_max - x_min)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    return (
        image[..., z_start:z_end, y_start:y_end, x_start:x_end],
        mask[..., z_start:z_end, y_start:y_end, x_start:x_end],
    )


def preprocess(image_path: str):
    """复刻 DataFolder.__getitem__ 的预处理段（inference_demo.py 163-239 行）。

    返回: (image (1,D,W,H) tensor, file_name)
    """
    data = {"image": image_path}
    res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
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
    image = trans(res)["image"]  # [C, D', W', H']（注意：传 LoadImaged 的输出 res，非原始 data）

    image[image > 400] = 400
    image[image < -300] = -300
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img = image

    roi_coords = np.nonzero(img[0].cpu().numpy())
    min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
    max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

    extend_d, extend_hw = 5, 20
    min_dhw = torch.max(min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]), torch.tensor([0, 0, 0]))
    max_dhw = torch.min(max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
                        torch.tensor([img.shape[1], img.shape[2], img.shape[3]]))

    cropped = img[:, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]]

    pad_func = transforms.SpatialPadd(keys=["image"], spatial_size=(96, 256, 384), mode="constant", constant_values=0, method="end")
    padded = pad_func({"image": cropped})["image"]

    file_name = Path(image_path).name
    return padded, file_name


def main():
    with torch.no_grad():
        return _main_impl()


def _main_impl():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "windows").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    import onnxruntime as ort
    from verify_trilinear import trilinear_resize

    # 黄金基线 v2：视觉主干用 ORT CPU（与 .NET 同一模型文件同一 EP），
    # 轻头用 numpy 手写（与 C# 逐式对应）。torch/monai 仅用于预处理。
    sess = ort.InferenceSession(str(MODELS_DIR / "radar_vision.onnx"), providers=["CPUExecutionProvider"])
    OUT_NAMES = ["seg_probs", "pred_mask", "res_x1", "res_x2", "res_x3", "flags1", "flags2", "flags3"]

    test_items = extract_test_items()
    print(f"test_items 共 {len(test_items)} 项")

    nii_path = TESTDATA_DIR / "AC423ccbe.nii.gz"
    image, file_name = preprocess(str(nii_path))
    print("预处理后 shape:", tuple(image.shape), "file:", file_name)

    image = image[None].to(device)  # (1,1,D,W,H)
    np.save(GOLDEN / "input_window.npy", image.cpu().numpy().astype(np.float32))

    image_size = list(image.shape[2:])
    scan_interval = get_scan_interval(image_size, ROI_SIZE, 3, SLIDING_OVERLAP)
    slices = dense_patch_slices(image_size, ROI_SIZE, scan_interval)
    num_win = len(slices)
    print(f"滑窗数 = {num_win}, scan_interval = {scan_interval}")

    test_organs = sorted({k.split("_")[0] for k in test_items})
    meta = {
        "file_name": file_name,
        "input_shape": list(image.shape),
        "roi_size": list(ROI_SIZE),
        "overlap": SLIDING_OVERLAP,
        "scan_interval": list(scan_interval),
        "num_windows": num_win,
        "slices": [[s[0].start, s[1].start, s[2].start, s[0].stop, s[1].stop, s[2].stop] for s in slices],
        "test_items": test_items,
        "test_organs": test_organs,
        "organs": ORGANS,
    }
    (GOLDEN / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=int), encoding="utf-8")

    text_feat_dict = torch.load(str(TEXT_EMBEDDING_PT), map_location="cpu")

    # 器官轻头权重（来自 organ_head.npz，M1-3 产物）；MHA 手写实现与 C# 端同一算法形态
    import math
    z = np.load(str(MODELS_DIR / "organ_head.npz"))
    qt = torch.from_numpy(z["query_tokens"]).to(device)
    query_tokens = z["query_tokens"].astype(np.float32)
    in_proj_w = z["attention.in_proj_weight"].astype(np.float32)
    in_proj_b = z["attention.in_proj_bias"].astype(np.float32)
    out_w = z["attention.out_proj.weight"].astype(np.float32)
    out_b = z["attention.out_proj.bias"].astype(np.float32)
    proj_w = np.stack([z[f"vision_projs.{i}.weight"] for i in range(N_ORGANS)]).astype(np.float32)
    proj_b = np.stack([z[f"vision_projs.{i}.bias"] for i in range(N_ORGANS)]).astype(np.float32)                      # (36,256)
    ipw = torch.from_numpy(z["attention.in_proj_weight"]).to(device)         # (768,256) Q|K|V
    ipb = torch.from_numpy(z["attention.in_proj_bias"]).to(device)
    ow = torch.from_numpy(z["attention.out_proj.weight"]).to(device)         # (256,256)
    ob = torch.from_numpy(z["attention.out_proj.bias"]).to(device)
    temp = float(z["temp"])
    pw = [torch.from_numpy(z[f"vision_projs.{i}.weight"]).to(device) for i in range(N_ORGANS)]
    pb = [torch.from_numpy(z[f"vision_projs.{i}.bias"]).to(device) for i in range(N_ORGANS)]
    N_HEADS, HEAD_DIM = 4, 64

    def mha(query, key):
        """等价 nn.MultiheadAttention(256, 4, batch_first=True) 单 query 前向。"""
        q = F.linear(query, ipw[:256], ipb[:256]).view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = F.linear(key, ipw[256:512], ipb[256:512]).view(1, -1, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = F.linear(key, ipw[512:], ipb[512:]).view(1, -1, N_HEADS, HEAD_DIM).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)).softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(1, 1, 256)
        return F.linear(out, ow, ob)

    organ_logits = dict(zip(test_items, [[] for _ in test_items]))
    organ_feat_dict = {}
    fid = file_name

    pad_func = transforms.DivisiblePadd(keys=["image", "label"], k=32, mode="constant", constant_values=0, method="end")

    full_mask = torch.zeros((1, NUM_CLASSES) + tuple(image_size), device=device)
    count_map = torch.zeros_like(full_mask)
    seg_probs_all = {}

    import math as _math

    def linear(x, W, B):
        """fp32 线性层：y[r,o] = b[o] + sum_d x[r,d]*W[o,d]（与 C# Linear 累加顺序一致）。"""
        rows = x.shape[0]
        inD, outD = W.shape[1], W.shape[0]
        y = np.empty((rows, outD), dtype=np.float32)
        for r in range(rows):
            for o in range(outD):
                acc = np.float32(B[o])
                xb = x[r]
                wb = W[o]
                for d in range(inD):
                    acc += np.float32(xb[d] * wb[d])
                y[r, o] = acc
        return y

    def organ_head_forward(oid, key):
        """手写 MHA + vision_proj + L2（与 C# OrganHead 逐式对应）。key (K,256)。"""
        K = key.shape[0]
        q = linear(query_tokens[oid:oid + 1], in_proj_w[:256], in_proj_b[:256])[0]     # (256,)
        kv = linear(key, in_proj_w[256:768], in_proj_b[256:768])                        # (K,512) 列拼接 K|V
        kmat = kv[:, :256]
        vmat = kv[:, 256:]
        outp = np.empty(256, dtype=np.float32)
        for h in range(4):
            off = h * 64
            attn = np.empty(K, dtype=np.float32)
            mx = -np.inf
            for j in range(K):
                sacc = np.float32(0)
                for d in range(64):
                    sacc += np.float32(q[off + d] * kmat[j, off + d])
                sacc = np.float32(sacc / np.float32(8.0))
                attn[j] = sacc
                if sacc > mx:
                    mx = sacc
            sm = np.float32(0)
            for j in range(K):
                attn[j] = np.float32(np.exp(attn[j] - mx))
                sm += attn[j]
            for j in range(K):
                attn[j] /= sm
            for d in range(64):
                acc = np.float32(0)
                for j in range(K):
                    acc += np.float32(attn[j] * vmat[j, off + d])
                outp[off + d] = acc
        o = linear(outp[None, :], out_w, out_b)[0]
        f = linear(o[None, :], proj_w[oid], proj_b[oid])[0]
        nrm = np.float32(np.sqrt(np.sum(f * f)))
        return (f / nrm).astype(np.float32)

    def forward_test_win(images, organ_logits, skip_organ=None, win_tag=None):
        """numpy 版 forward_test_win：ORT 视觉 + 手写轻头，与 C# RadarInferenceEngine 同构。"""
        feed = {"x": images.cpu().numpy()}
        o_seg, o_mask, o_x1, o_x2, o_x3, o_f1, o_f2, o_f3 = sess.run(OUT_NAMES, feed)
        if win_tag is not None:
            wd = GOLDEN / "windows" / win_tag
            wd.mkdir(exist_ok=True)
            np.save(wd / "win_in.npy", images.cpu().numpy().astype(np.float32))
            for name, arr in zip(OUT_NAMES, [o_seg, o_mask, o_x1, o_x2, o_x3, o_f1, o_f2, o_f3]):
                np.save(wd / f"{name}.npy", arr)

        seg_prob_t = torch.from_numpy(o_seg)  # 供 torch interpolate 拼接（与 C# Resampler 同 half_pixel 公式）
        seg_probs = o_seg
        masks = o_mask  # (96,256,384) uint8

        margin = 2
        for i in range(1):
            mk = masks[i].astype(np.int64)  # (1,96,256,384) -> 剥 batch 维，必须与 V1 的 masks[i] 一致
            boundary_organs = set()
            for ax in range(3):
                sl0 = [slice(None)] * 3
                sl1 = [slice(None)] * 3
                sl0[ax] = slice(None, margin)
                sl1[ax] = slice(-margin, None)
                for seg_ in (mk[tuple(sl0)], mk[tuple(sl1)]):
                    boundary_organs.update(int(v) for v in np.unique(seg_[seg_ > 0]))
            if skip_organ is not None:
                boundary_organs.discard(skip_organ + 1)

            uniq, cnts = np.unique(mk, return_counts=True)
            intact = [int(u) for u, c in zip(uniq, cnts) if u != 0 and int(u) not in boundary_organs]

            for label in intact:
                oid = label - 1
                organ_name = ORGANS[oid]
                if organ_name not in test_organs:
                    continue
                if organ_name in organ_feat_dict:
                    continue
                f1 = o_f1[0, oid] > 0  # flags (1,36,N) -> 第 0 维是 batch，(1152,) bool
                f2 = o_f2[0, oid] > 0
                f3 = o_f3[0, oid] > 0
                key = np.concatenate([o_x1[0][f1], o_x2[0][f2], o_x3[0][f3]], axis=0)  # (K,256)
                feat = organ_head_forward(oid, key)  # (256,) L2 归一化
                organ_feat_dict[organ_name] = feat
                for item in organ_logits.keys():
                    if item.split("_")[0] != organ_name:
                        continue
                    tf = text_feat_dict[item].numpy()  # (2,256)
                    neg = float(np.dot(feat, tf[0]))
                    pos = float(np.dot(feat, tf[1]))
                    l0, l1 = neg / temp, pos / temp
                    m = max(l0, l1)
                    e0, e1 = np.exp(l0 - m), np.exp(l1 - m)
                    organ_logits[item].append(np.array([[e0 / (e0 + e1), e1 / (e0 + e1)]], dtype=np.float32))
        return organ_logits, seg_prob_t

    # ---- 滑窗主循环 ----
    for w in range(num_win):
        win = image[:, :, slices[w][0], slices[w][1], slices[w][2]]
        dump_tag = f"win_{w:04d}" if (w < DUMP_FULL_WINDOWS or w == num_win - 1) else None
        organ_logits, seg_prob = forward_test_win(win, organ_logits, win_tag=dump_tag)
        # 逐通道 numpy 三线性重采样（与 C# Resampler 同 half_pixel 公式），
        # 攒满 (C, D, H, W) 后一次性转 torch —— 不能用 empty_like(win)（仅 1 通道）
        n_cls = seg_prob.shape[1]
        interp_np = np.empty((n_cls,) + tuple(win.shape[2:]), dtype=np.float32)
        for c in range(n_cls):
            ch = seg_prob[0, c].reshape(1, 1, *seg_prob.shape[2:]).numpy()  # (1,1,96,128,192)
            up = trilinear_resize(ch, win.shape[2:])
            interp_np[c] = up[0, 0]
        interp = torch.from_numpy(interp_np)[None].to(device)  # (1, C, D, H, W)，与 full_mask 同设备
        full_mask[:, :, slices[w][0], slices[w][1], slices[w][2]] += interp
        count_map[:, :, slices[w][0], slices[w][1], slices[w][2]] += 1
        seg_probs_all[str(w)] = interp.cpu().numpy().astype(np.float16)
        if w % 10 == 0:
            print(f"窗 {w + 1}/{num_win}")

    np.savez_compressed(GOLDEN / "seg_probs_all.npz", **seg_probs_all)

    count_map = torch.clamp(count_map, min=1)
    stitched = (full_mask / count_map).argmax(1).unsqueeze(0)
    np.save(GOLDEN / "stitched_mask.npy", stitched.squeeze(0).cpu().numpy().astype(np.int16))

    # ---- 缺失器官补推 ----
    squeeze_mask = stitched.squeeze(0).squeeze(0)
    boundary_organs = set()
    margin = 2
    for d in range(3):
        start = [slice(None)] * 3
        end = [slice(None)] * 3
        start[d] = slice(None, margin)
        end[d] = slice(-margin, None)
        for seg_ in (squeeze_mask[tuple(start)], squeeze_mask[tuple(end)]):
            boundary_organs.update(int(v) for v in torch.unique(seg_[seg_ > 0]))
    organ_ids, organ_counts = torch.unique(squeeze_mask, return_counts=True)
    organ_ids = organ_ids.long()
    keep = organ_ids != 0
    organ_counts = organ_counts[keep]
    organ_ids = organ_ids[keep]
    intact = [int(oid) for oid, c in zip(organ_ids.tolist(), organ_counts.tolist()) if oid not in boundary_organs]

    for k in list(organ_logits.keys()):
        if organ_logits[k]:
            continue
        organ_name = k.split("_")[0]
        organ_id = ORGANS.index(organ_name)
        win_patch, win_mask = center_crop(image, torch.eq(stitched, organ_id + 1), crop_size=ROI_SIZE)
        win_mask = win_mask.float()
        win_mask[win_mask == 1] = organ_id + 1
        pad_data = pad_func({"image": win_patch[0], "label": win_mask[0]})
        organ_logits, _ = forward_test_win(
            pad_data["image"][None].to(device), organ_logits, skip_organ=organ_id,
            win_tag=f"win_crop_{organ_id}",
        )

    # ---- 最终概率（复刻 evaluate 聚合）----
    result = {}
    for item, probs in organ_logits.items():
        if len(probs) > 0:
            result[item] = float(np.concatenate(probs).reshape(-1, 2).mean(0)[1])
    (GOLDEN / "organ_probs.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(GOLDEN / "organ_feats.npz", **organ_feat_dict)

    print(f"\n完成：打分条目 {len(result)}/{len(test_items)}，完整 dump 窗 {DUMP_FULL_WINDOWS + 1} 个")
    print(f"输出目录 {GOLDEN}")


if __name__ == "__main__":
    main()
