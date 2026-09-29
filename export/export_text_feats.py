# -*- coding: utf-8 -*-
"""导出预计算文本特征 -> models/text_feats.bin + models/text_feats.json。

源: ../damo-radar/ckpt/infer_text_embedding_radar.pt
    dict[str, Tensor(2,256)]，146 项「器官_病灶」条目；
    每项 2 行 = [负向 prompt 特征, 正向 prompt 特征]（打分时 softmax 取 index 1）。

二进制格式（C# 零依赖解析）：
  text_feats.bin : 全部特征按 json 索引顺序连续排列，float32 小端，
                   每项 2*256=512 个 float。
  text_feats.json: {"dim":256, "rows_per_item":2,
                    "items":[{"name":"主动脉_主动脉夹层"}, ...],
                    "count":146}
  第 i 项特征位于 bin 的 [i*512, (i+1)*512) 个 float。
"""
import json

import numpy as np
import torch

from common import MODELS_DIR, TEXT_EMBEDDING_PT

BIN_PATH = MODELS_DIR / "text_feats.bin"
JSON_PATH = MODELS_DIR / "text_feats.json"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    d = torch.load(str(TEXT_EMBEDDING_PT), map_location="cpu", weights_only=False)

    names = list(d.keys())
    dim = d[names[0]].shape[-1]
    rows = d[names[0]].shape[0]
    for k in names:
        assert d[k].shape == (rows, dim), f"形状不一致: {k} {d[k].shape}"

    feats = np.stack([d[k].detach().cpu().numpy().astype(np.float32).ravel() for k in names])
    feats.tofile(str(BIN_PATH))

    meta = {
        "dim": int(dim),
        "rows_per_item": int(rows),
        "count": len(names),
        "items": [{"name": n} for n in names],
    }
    JSON_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"保存 {BIN_PATH} ({BIN_PATH.stat().st_size/1024:.1f} KB)")
    print(f"保存 {JSON_PATH}  count={len(names)} rows={rows} dim={dim}")

    # 回读校验
    back = np.fromfile(str(BIN_PATH), dtype=np.float32).reshape(len(names), rows * dim)
    for i, k in enumerate(names):
        assert np.array_equal(back[i], d[k].numpy().ravel()), f"回读不一致: {k}"
    print("bin 回读校验通过")


if __name__ == "__main__":
    main()
