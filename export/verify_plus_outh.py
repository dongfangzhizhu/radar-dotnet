# -*- coding: utf-8 -*-
"""隔离 out_h（V 加权求和）各实现差异：naive32 / naive64 / opt32 / opt64。"""
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
hd = 64
q = query @ ipw[:256].T + ipb[:256]

ipw64 = ipw.astype(np.float64)
ipb64 = ipb.astype(np.float64)
key64 = key.astype(np.float64)
q64 = query.astype(np.float64) @ ipw64[:256].T + ipb64[:256].astype(np.float64)

h = 2
off = h * hd
q_h = q[off:off + hd]

# 同一 attn（fp64 算好共享，排除 attn 差异）
u64 = ipw64[256 + off:256 + off + hd, :].T @ q64[off:off + hd]
bK64 = q64[off:off + hd] @ ipb64[256 + off:256 + off + hd]
s64_scores = (key64 @ u64 + bK64) / np.sqrt(64.0)
attn64 = np.exp(s64_scores - s64_scores.max())
attn64 /= attn64.sum()
attn32 = attn64.astype(np.float32)

# 1) naive32: v32 = key@W_v.T 然后 attn32 @ v32
v32 = key @ ipw[512 + off:512 + off + hd, :].T + ipb[512 + off:512 + off + hd]
out_naive32 = attn32 @ v32

# 2) naive64
v64 = key64 @ ipw64[512 + off:512 + off + hd, :].T + ipb64[512 + off:512 + off + hd]
out_naive64 = attn64 @ v64

# 3) opt32: s32 = attn32@key32，再 W_v@s
s32 = attn32 @ key
out_opt32 = s32 @ ipw[512 + off:512 + off + hd, :].T + ipb[512 + off:512 + off + hd]

# 4) opt64
s64 = attn64 @ key64
out_opt64 = s64 @ ipw64[512 + off:512 + off + hd, :].T + ipb64[512 + off:512 + off + hd]

print("out_h 量级: naive32 范围", out_naive32.min(), out_naive32.max())
print("max|naive32-naive64| =", np.abs(out_naive32 - out_naive64.astype(np.float32)).max())
print("max|opt32  -naive64| =", np.abs(out_opt32 - out_naive64.astype(np.float32)).max())
print("max|opt64  -naive64| =", np.abs(out_opt64 - out_naive64).max())
print("s32 范围:", s32.min(), s32.max(), " s64 范围:", s64.min(), s64.max())
print("max|s32-s64| =", np.abs(s32 - s64.astype(np.float32)).max())
