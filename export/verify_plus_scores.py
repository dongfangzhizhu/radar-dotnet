# -*- coding: utf-8 -*-
"""定位 whole head 优化式与朴素式差异根因：score 级 / attn 级 / fp64 对照。"""
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
query = z["query_tokens_whole"].astype(np.float32).ravel()

key = np.concatenate([x1, x2, x3], axis=0)
nh, hd = 4, 64
q = query @ ipw[:256].T + ipb[:256]
ipw64, ipb64, q64, key64 = [a.astype(np.float64) for a in (ipw, ipb, query, key)]
q64v = q64 @ ipw64[:256].T + ipb64[:256]

print("x1/x2/x3 范围:", x1.min(), x1.max(), "|", x3.min(), x3.max())
print("key 元素绝对值 max:", np.abs(key).max())

h = 0
off = 0
q_h = q[off:off + hd]

# naive scores: q_h · (W_k x_j + b_k)[off:off+hd]
kv32 = key @ ipw[256:].T + ipb[256:]
km = kv32[:, :256]
s_naive = (km[:, off:off + hd] @ q_h) / np.sqrt(hd)

# opt scores: (W_kᵀ q_h)·x_j + q_h·b_k[off:off+hd]
u = ipw[256 + off:256 + off + hd, :].T @ q_h
bK = q_h @ ipb[256 + off:256 + off + hd]
s_opt = (key @ u + bK) / np.sqrt(hd)

# fp64 reference
km64 = key64 @ ipw64[256:256 + hd, :].T + ipb64[256:256 + hd]
s64 = (km64 @ q64v[off:off + hd]) / np.sqrt(np.float64(hd))

print("head0 scores: max|naive-opt| =", np.abs(s_naive - s_opt).max())
print("head0 scores: max|naive-fp64| =", np.abs(s_naive - s64.astype(np.float32)).max())
print("head0 scores: max|opt-fp64|   =", np.abs(s_opt - s64.astype(np.float32)).max())
print("scores[0:8] naive:", s_naive[:8])
print("scores[0:8] opt  :", s_opt[:8])
print("scores[0:8] fp64 :", s64[:8].astype(np.float32))

# attn 差异
a_naive = np.exp(s_naive - s_naive.max()); a_naive /= a_naive.sum()
a_opt = np.exp(s_opt - s_opt.max()); a_opt /= a_opt.sum()
print("attn: max|naive-opt| =", np.abs(a_naive - a_opt).max(),
      " top10 mass naive =", np.sort(a_naive)[-10:].sum(),
      " opt =", np.sort(a_opt)[-10:].sum())
