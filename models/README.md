# models/ — 模型产物（不入库）

本目录内容由 `export/` 一次性工具链生成，体积大，git 忽略。丢失后按下列命令重建
（解释器：`C:/Users/Administrator/miniconda3/python.exe`，torch 2.9.1+cu128）。

| 文件 | 生成命令 | 说明 | 实测 |
|---|---|---|---|
| `radar_vision.onnx` | `python ../export/export_onnx.py` | 视觉主干，opset 17，固定输入 (1,1,96,256,384)，8 输出 | 98.9 MB |
| `organ_head.npz` | `python ../export/export_weights.py` | 轻头权重 78 键（MHA in_proj Q\|K\|V 布局 + 36×vision_projs + query_tokens + temp） | 10.07 MB |
| `text_feats.bin` + `text_feats.json` | `python ../export/export_text_feats.py` | 146 项 × (2,256) 预计算文本特征，连续 float32 + json 索引（C# 零依赖解析） | 292 KB |
| `plus_head.npz` | `python ../export/export_plus.py` | RADAR+ whole-head 权重 8 键（pretrain 变体，temp=0.053180） | 1.26 MB |
| `plus_head_merlin_ft.npz` | 同上 | RADAR+ whole-head 权重 8 键（merlin 微调变体，temp=0.041191） | 1.26 MB |
| `plus_text_feats.bin` + `.json` | 同上 | 9 疾病 134 条 prompt 文本特征（pretrain 变体，含 num_positive/start/count） | 134 KB |
| `plus_text_feats_merlin_ft.bin` + `.json` | 同上 | 同上（merlin 微调变体） | 134 KB |
| `radar_vision_fp16.onnx` | `python ../export/export_onnx_fp16.py` | fp16 变体（keep_io_types=True，51.9 MB）——**评估用，未启用**：res_x max 2.4e-3 超门禁 240 倍且 pred_mask/flags 有翻转，见 PLAN.md M8-2 | 51.9 MB |

DICOM 解码依赖（NuGet）：`fo-dicom` 5.2.5 + `fo-dicom.Codecs` 5.16.7（原生编解码器，
DicomReader 静态构造注册 NativeTranscoderManager）。覆盖 RLE / JPEG Baseline / JPEG Extended /
JPEG Lossless / JPEG-LS(Lossy) / JPEG2000 / HT-JPEG2000。Windows 目标机需 VC++ 2015-2022 运行库。

上游依赖（不在本仓库，缺失则无法重建）：
- `../RADAR/checkpoint_radar_pretrain.pth`（1.49 GB，HuggingFace）
- `../RADAR/checkpoint_radar_plus.pth`（1.57 GB，M7 RADAR+ pretrain）
- `../RADAR/checkpoint_radar_plus_finetuned_on_merlin.pth`（1.57 GB，M7 RADAR+ merlin 微调）
- `../RADAR/bert-base-uncased/`（文本特征计算用 tokenizer + BertConfig）
- `../damo-radar/ckpt/infer_text_embedding_radar.pt`（338 KB）
- `../damo-radar/RADAR_inference/`（网络结构源码，export 脚本 sys.path 引用）
- `../damo-radar/RADAR_train/`（M7 文本特征复刻 infer_merlin_whole.py 的 prompt 清单与编码参数）

对拍结论（2026-09-29，全部 PASS，详见各 export 脚本与 PLAN.md）：
- 向量化 vs 原版：8 输出 diff = 0
- ONNX vs torch：res_x < 1.5e-5，pred_mask 不匹配 3/9,437,184，flags 全等
- golden vs 官方 CSV：146 项全匹配，mean 5.8e-5
