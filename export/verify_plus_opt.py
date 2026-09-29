# -*- coding: utf-8 -*-
"""验证 whole head 优化式（u=W_kᵀq_h 先行）与朴素式在 fp32 下的数值差异。"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import MODELS_DIR, TESTDATA_DIR

GOLDEN = TESTDATA_DIR / "golden"

vol = np.load(str(GOLDEN / "plus_preproc_merlin_ft.npy"))
sess = ort.InferenceSession(str(MODELS_DIR / "radar_vision.onnx"),
                            providers=["CPUExecutionProvider"])
outs = sess.run(["res_x1", "res_x2", "res_x3"],
                {sess.get_inputs()[0].name: vol[None, None]})
x1, x2, x3 = (o.reshape(-1, 256).astype(np.float32) for o in outs)

z = np.load(str(MODELS_DIR / "plus_head_merlin_ft.npz"))
ipw = z["attention_whole.in_proj_weight"].astype(np.float32)
ipb = z["attention_whole.in_proj_bias"].astype(np.float32)
ow = z["attention_whole.out_proj.weight"].astype(np.float32)
ob = z["attention_whole.out_proj.bias"].astype(np.float32)
pw = z["vision_projs_whole.weight"].astype(np.float32)
pb = z["vision_projs_whole.bias"].astype(np.float32)
query = z["query_tokens_whole"].astype(np.float32).ravel()
temp = float(z["temp"])

key = np.concatenate([x1, x2, x3], axis=0)
nh, hd = 4, 64


def finish(out):
    out = out @ ow.T + ob
    feat = out @ pw.T + pb
    return feat / np.linalg.norm(feat)


# 朴素式（golden 同款）
q = query @ ipw[:256].T + ipb[:256]
kv = key @ ipw[256:].T + ipb[256:]
km, vm = kv[:, :256], kv[:, 256:]
out_naive = np.zeros(256, dtype=np.float32)
for h in range(nh):
    off = h * hd
    attn = (km[:, off:off + hd] @ q[off:off + hd]) / np.sqrt(hd)
    attn = np.exp(attn - attn.max()); attn /= attn.sum()
    out_naive[off:off + hd] = attn @ vm[:, off:off + hd]
feat_naive = finish(out_naive)

# 优化式（C# 同款：fp32 逐步）
out_opt = np.zeros(256, dtype=np.float32)
for h in range(nh):
    off = h * hd
    q_h = q[off:off + hd]
    u = (ipw[256 + off:256 + off + hd, :].T.astype(np.float32) @ q_h.astype(np.float32)).astype(np.float32)
    bK = np.float32(q_h @ ipb[256 + off:256 + off + hd])
    scores = (key @ u + bK) / np.sqrt(hd)
    attn = np.exp(scores - scores.max()); attn = (attn / attn.sum()).astype(np.float32)
    s = (attn @ key).astype(np.float32)
    out_opt[off:off + hd] = s @ ipw[512 + off:512 + off + hd, :].T + ipb[512 + off:512 + off + hd]
feat_opt = finish(out_opt)

print("feature naive vs opt max_diff =", np.abs(feat_naive - feat_opt).max())
print("scores head0 前5:", scores[:5])
print("scores 范围:", scores.min(), scores.max())

# 打分
meta = json.loads((MODELS_DIR / "plus_text_feats_merlin_ft.json").read_text(encoding="utf-8"))
tfeats = np.fromfile(str(MODELS_DIR / "plus_text_feats_merlin_ft.bin"), dtype=np.float32).reshape(-1, 256)
for tag, f in [("naive", feat_naive), ("opt", feat_opt)]:
    probs = {}
    for d in meta["diseases"]:
        sims = (tfeats @ f) / temp
        s0, s1 = d["start"], d["num_positive"]
        pos = sims[s0:s0 + s1].mean()
        neg = sims[s0 + s1:s0 + d["count"]].mean()
        e = np.exp([neg - max(neg, pos), pos - max(neg, pos)])
        probs[d["name"]] = float(e[1] / e.sum())
    print(tag, {k: round(v, 4) for k, v in probs.items()})
