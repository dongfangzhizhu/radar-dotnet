# models/ — 模型产物（不入库）

本目录内容由 `export/` 一次性工具链生成，体积大，git 忽略。丢失后按下列命令重建
（解释器：`C:/Users/Administrator/miniconda3/python.exe`，torch 2.9.1+cu128）。

| 文件 | 生成命令 | 说明 | 实测 |
|---|---|---|---|
| `radar_vision.onnx` | `python ../export/export_onnx.py` | 视觉主干，opset 17，固定输入 (1,1,96,256,384)，8 输出 | 98.9 MB |
| `organ_head.npz` | `python ../export/export_weights.py` | 轻头权重 78 键（MHA in_proj Q\|K\|V 布局 + 36×vision_projs + query_tokens + temp） | 10.07 MB |
| `text_feats.bin` + `text_feats.json` | `python ../export/export_text_feats.py` | 146 项 × (2,256) 预计算文本特征，连续 float32 + json 索引（C# 零依赖解析） | 292 KB |

上游依赖（不在本仓库，缺失则无法重建）：
- `../RADAR/checkpoint_radar_pretrain.pth`（1.49 GB，HuggingFace）
- `../damo-radar/ckpt/infer_text_embedding_radar.pt`（338 KB）
- `../damo-radar/RADAR_inference/`（网络结构源码，export 脚本 sys.path 引用）

对拍结论（2026-09-29，全部 PASS，详见各 export 脚本与 PLAN.md）：
- 向量化 vs 原版：8 输出 diff = 0
- ONNX vs torch：res_x < 1.5e-5，pred_mask 不匹配 3/9,437,184，flags 全等
- golden vs 官方 CSV：146 项全匹配，mean 5.8e-5
