# RADAR .NET 推理平台 — 完整实施计划

> 项目：将 damo-radar（全身 CT 多器官分割 + 器官级病灶概率诊断）的推理功能完整迁移到 .NET 10。
> 仓库：`radar-dotnet/`（本仓库，git 管理）
> 上游：`../damo-radar/`（Python 原版，只读参照，不修改）；权重：`../RADAR/`
> 制定日期：2026-09-29

---

## 1. 目标与范围

### 1.1 必须交付

| # | 交付物 | 说明 |
|---|--------|------|
| D1 | **Radar.Core 推理库** | 预处理、滑窗推理、器官轻头、后处理，纯 C#，CPU/GPU 双路径 |
| D2 | **Radar.Backend Web 服务** | ASP.NET Core minimal API，任务提交/查询/结果下载，托管测试 Web 页 |
| D3 | **Web 测试端** | 上传/选择 NIfTI 输入 → 触发推理 → 展示 146 项病灶概率 + 分割掩码可视化 |
| D4 | **Python 导出工具链** | .pth → ONNX（视觉主干）+ safetensors/npz（轻头权重 + 文本特征），一次性 |
| D5 | **对齐测试套件** | 与官方参考 CSV（`../damo-radar/results/RADAR_infer_results_demo.csv`）逐值比对 |

### 1.2 明确不做

- 不迁移训练链路（RADAR_train/ 整体排除）。
- 不迁移 BERT 文本编码器——推理时文本特征为离线预计算（`infer_text_embedding_radar.pt`），BERT 的 117.9M 参数是死重。
- 不支持 DICOM 输入（首版仅 `.nii/.nii.gz`，DICOM 列为后续扩展）。
- 不迁移 `radar_plus` 系列权重（首版只做 `checkpoint_radar_pretrain.pth`，管线设计上预留换权重能力）。

> **范围变更（后续里程碑推翻的条目）**：
> - 「不支持 DICOM 输入」→ **M6 已推翻**，现支持单文件/多帧/目录序列/zip，含 7 种压缩传输语法。
> - 「不迁移 `radar_plus`」→ **M7 已推翻**，两个变体（pretrain / merlin_ft）均已迁移并对拍通过。
> - 「AUC 评估属评估而非推理由此不做」→ **M9 已补做**（算子 + CLI `eval`），因差距审计要求与上游 `calc_metrics_merlin_testset.py` 能力对齐。
> - 始终不做的只有：训练链路、BERT 文本编码器。

---

## 2. 迁移可行性依据（已取证）

对 `checkpoint_radar_pretrain.pth`（1.49 GB，693 键）实测分析：

| 部件 | 参数量 | 迁移决策 |
|---|---|---|
| visual_encoder（PlainConvUNetLightD 6-stage Conv3d + 3×1×1×1 投影） | 77.69M | **导出 ONNX**，推理核心 |
| text_encoder（XBertEncoder/bert-base-chinese） | 117.92M | 不迁移（推理零调用） |
| text_proj | 0.20M | 不迁移 |
| attention（单 query 4 头 MHA 256d） | 0.26M | C# 手写（~80 行） |
| vision_projs（36 × Linear 256→256） | 2.37M | safetensors 加载，C# matmul |
| query_tokens (36,256) + temp 标量 | 0.01M | 同上 |

**需迁移总量 ≈ 80.3M 参数（fp32 约 321 MB，fp16 约 161 MB）**，全部为规则结构，无自定义算子。
模型输入 shape 固定 `(1, 1, 96, 256, 384)`，滑窗 overlap 0.25，DivisiblePad k=32。

### 2.1 图拆分（关键设计）

`forward_test_win`（`inference_demo.py:299`）含 `torch.unique`、布尔索引、Python dict、per-organ 循环，**不可整体导出**。拆为：

- **子图 A → ONNX**（`radar_vision.onnx`）：`VisionBranch` 张量部分。输入 `(1,1,96,256,384)`，输出 8 个张量：`seg_probs`、`pred_mask`、`res_x1/2/3`（三尺度 token 序列）、`organ_token_flags1/2/3`。flags 计算需先**向量化重写**（对 36 器官通道直接 `eq → max_pool3d((8,32,32)/(4,16,16)/(2,8,8))`，语义等价、纯张量）。
- **子图 B → C# 手写**：单 query 对 K token 的 4 头 attention → `vision_projs[i]` → L2 归一化 → 与预载 text_feat 点积 ÷temp → softmax。动态 token 数天然支持。
- **重采样子图 → ONNX**（`radar_resize.onnx`）：MONAI `Resized(mode="trilinear")` 语义对齐是最大数值风险，用 ONNX Resize op 导出小模型规避手写插值。
- **滑窗/拼接/argmax/边界器官排除/center_crop → C#**：纯张量索引逻辑。

---

## 3. 系统架构

```
radar-dotnet/
├── PLAN.md                     ← 本文档
├── Radar.sln
├── src/
│   ├── Radar.Core/             ← D1 核心推理库
│   │   ├── Preprocessing/      │  NIfTI 读取、(1,1,5)mm 重采样、HU clip[-300,400]、
│   │   │                       │  非零域裁剪+外扩(5,20,20)、pad 到 (96,256,384)
│   │   ├── Inference/          │  ORT Session 管理、滑窗调度、概率累加、argmax 拼接
│   │   ├── OrganHead/          │  单 query MHA + vision_projs + 相似度打分（C# 手写）
│   │   ├── Postprocessing/     │  边界器官排除、多窗概率平均、CSV 输出（147 列 = file_name + 146 项）
│   │   └── Configuration/      │  ExecutionProvider 选择（CPU/DirectML/CUDA）
│   ├── Radar.Backend/          ← D2 Web 服务（ASP.NET Core minimal API, Kestrel）
│   │   ├── Endpoints/          │  POST /api/jobs（上传 NIfTI 或选内置样例）
│   │   │                       │  GET  /api/jobs/{id}（状态轮询）
│   │   │                       │  GET  /api/jobs/{id}/result（JSON/CSV 下载）
│   │   │                       │  GET  /api/devices（可用 EP 列表）
│   │   ├── Services/           │  后台任务队列（Channel + N 个 worker，互斥 GPU）
│   │   └── wwwroot/            ← D3 测试页（原生 HTML/JS，无框架依赖）
│   └── Radar.Web（并入 Backend wwwroot，不单独建项目）
├── export/                     ← D4 Python 工具链（一次性，torch 2.9.1 / miniconda）
│   ├── export_onnx.py          │  向量化 VisionBranch → ONNX opset 17
│   ├── export_resize.py        │  重采样子图
│   ├── export_weights.py       │  轻头权重 + temp → safetensors/npz
│   └── export_text_feats.py    │  infer_text_embedding_radar.pt → npz + json 索引
├── models/                     ← 导出产物（.gitignore，不入库，见 §8）
├── testdata/                   ← 测试数据（.gitignore，软引用 ../damo-radar/data）
└── tests/
    └── Radar.Tests/            ← D5 xUnit 对齐测试
```

### 3.1 推理数据流

```
NIfTI 文件 → [C# 预处理] → 重采样(ORT 小模型) → 裁剪/pad → (1,1,96,256,384)
  → [滑窗调度] 每窗 → radar_vision.onnx (ORT, IOBinding)
      → seg_probs / pred_mask / res_x1-3 / flags1-3
  → 概率累加 + count map → argmax → 全图分割掩码
  → 边界器官排除 → 完整器官：per-organ token 聚合 → [C# 器官头] → 165 项概率
  → 缺失器官：center_crop 二次滑窗补推
  → 多窗平均 → CSV/JSON 结果
```

### 3.2 CPU / GPU 支持方案

| 执行提供程序 | 包 | 状态 | 说明 |
|---|---|---|---|
| CPU | Microsoft.ML.OnnxRuntime（已引入） | 首版必选 | AVX2/AVX512 自动加速，零配置 |
| DirectML | Microsoft.ML.OnnxRuntime.DirectML（已引入） | 首版必选 | Windows GPU 默认路径，免 CUDA 版本匹配，AMD/NVIDIA/Intel 通吃 |
| CUDA | Microsoft.ML.OnnxRuntime.Gpu（按需引入） | 开发机已确认 CUDA 可用（torch cuda OK），M5 启用 |

EP 选择策略（`Radar.Core/Configuration`）：
1. 启动参数 `--ep auto|cpu|dml|cuda`，默认 `auto`；
2. `auto`：依次尝试创建 CUDA → DirectML → CPU session，探活成功者写入 `/api/devices` 暴露；
3. GPU session 全局单例 + 请求级排队（Channel worker 数=1），避免显存竞争；
4. Conv3d fp16 优化：DirectML/CUDA 下可选 fp16 ONNX（精度对齐测试通过后启用，预期提速 1.5–2×）。

### 3.3 Web 测试端（wwwroot 单页）

- 输入方式：① 上传 `.nii/.nii.gz`（multipart，≤200 MB）；② 选择内置 demo（AC423ccbe.nii.gz）；③ 选择 testdata/ 目录下任意已放置样例。
- 设备选择下拉（cpu/dml/cuda，来自 /api/devices）。
- 提交后轮询任务状态，展示：耗时分解（预处理/推理/后处理）、146 项「器官_病灶」阳性概率表（中文列名 + 英文映射，支持排序/筛选/阈值高亮）、与参考 CSV 的偏差列（对齐模式）。
- 结果导出：CSV（列序与官方完全一致）/ JSON。
- 分割掩码可视化：按器官切片数最多的轴出 PNG 透叠图（首版静态，3D 交互列为扩展）。

---

## 4. 里程碑与任务分解

### M0 — 仓库与工具链就绪（本次提交，已完成）
- [x] .NET 10 解决方案骨架（Radar.Core / Radar.Backend / Radar.Tests）
- [x] NuGet：Microsoft.ML.OnnxRuntime + .DirectML + .Managed
- [x] git init + .gitignore（权重/数据不入库）
- [x] PLAN.md

### M1 — Python 导出与黄金基线（预计 2–3 天）
- [x] `export_onnx.py`：向量化 flags 重写（先在 Python 内对拍：改造前后 `forward_test_win` 输出 max abs diff < 1e-6）→ `model.eval()` → `torch.onnx.export` opset 17 → `onnxruntime` 回读对拍（CPU EP，diff < 1e-5）
- [ ] `export_resize.py`、`export_weights.py`（safetensors 或 npz：attention in/out_proj、36×vision_projs、query_tokens、temp）、`export_text_feats.py`（165 项 text_feat + 索引 json）
- [x] 产出 Python 黄金基线：对 demo case 逐窗 dump（预处理后体数据、每窗 8 输出张量、最终 CSV），存 `testdata/golden/`（gitignore，本地约 2–4 GB，供 .NET 逐级比对）
- **验收（已达成 2026-09-29）**：向量化 vs 原版 8 输出逐位一致；ONNX vs torch 按分层标准 PASS；golden vs 官方 CSV 146 项全匹配 mean 5.8e-5

### M2 — Radar.Core 预处理 + NIfTI（预计 2–3 天）✅ 2026-09-29
- [x] NIfTI/.nii.gz 手写读取器（gzip 流 + sform/qform affine + 8 种数据类型）
- [x] 预处理管线：spacing→(1,1,5)mm 目标尺寸 → C# 手写 fp32 三线性重采样（与 torch CPU kernel 逐位一致，`export/verify_trilinear.py` 钉死公式）→ HU clip → min-max 归一化 → 非零域裁剪 + (5,20,20) 外扩 → 尾部 pad（SpatialPadd 只 pad 不 crop，`Math.Max(cd, TargetD)`）
- **验收（已达成）**：预处理输出与黄金基线 mismatch=0（逐元素 diff < 1e-5）

### M3 — 滑窗推理与器官头（预计 3–4 天）✅ 2026-09-29
- [x] ORT Session 封装（EP 回退 CUDA→DirectML→CPU，NamedOnnxValue API）
- [x] 滑窗调度（dense_patch_slices 等价，overlap 0.25）+ 概率累加 + argmax 拼接
- [x] C# 器官头：手写 MHA（关键坑：torch F.linear 合并 K/V 输出是**行内列拼接** [K_j|V_j]，非行分块；修正后特征 diff 0.25→5e-5）
- [x] 边界器官排除（margin=2）+ 缺失器官 center_crop 补推（坑：裁剪体素必须来自 CT volume，mask 只用于定位 bbox）
- [x] 黄金基线 V2：视觉主干 ORT CPU + numpy 手写 MHA + numpy 三线性（与 .NET 同数值路径；坑：pred_mask (1,96,256,384) 漏剥 batch 维会使边界器官全集、intact 恒空，v1↔v2 mean 偏差 1.1e-2 → 修复后 2.3e-5）
- **验收（已达成）**：① 每窗 8 张量对齐 golden；② 端到端 146/146 匹配 max_diff=5.96e-7（门禁 1e-3），CPU 端到端 119s；③ v2 基线 vs 官方 CSV mean=7.4e-5 PASS、vs v1 torch 路径 mean=2.3e-5；全量测试 6/6 通过

### M4 — Radar.Backend + Web 测试端（预计 2–3 天）✅ 2026-09-29
- [x] 任务队列（Channel 单 worker 串行，GPU 会话独占）、上传/暂存、设备探活缓存
- [x] /api/jobs 系列 + /api/devices + /api/samples + CSV 下载
- [x] wwwroot 单页（设备选择/样例/上传/队列轮询/146 项结果表/CSV 下载）
- **验收（已达成）**：并发 2 任务串行排队正确；Web 端 demo case 146 项 vs golden max=1.1e-6 mean=5.3e-8，与 CLI 结果逐位一致，单任务 120s

### M5 — 性能测试与多 EP 矩阵（预计 2 天）✅ 2026-09-29
- [x] 基准程序 `tools/Radar.Benchmark`（阶段分解：预处理/会话创建/ORT/轻头/端到端，Debug/Release 配置自适应标注）
- [x] 扩展数据集冒烟：merlin 4 例全 PASS（146 条目、无 NaN）
- [x] fp16 ONNX 可选路径（2026-09-30 评估完成：不达标，结论为不启用，见 M8-2）
- **R4 缺口已修复（补推区二次滑窗）**：bbox 超出 ROI 时对 crop+pad 区域做二次滑窗（子窗恒为 96×256×384），skip_organ 保证目标器官在每个子窗均视为非边界、organFeats 缓存保证特征只算一次；数值与 Python 原版"单窗整推"有差（多窗视角），demo case 不触发此路径、146 项硬门禁不受影响。AC4242a2f 由 FAIL 转 PASS。
- **实测结论（benchmarks.md，Release 构建）**：① DirectML/CUDA 探活当时失败系 Gpu 包缺失（M6 已修复）；② Release demo CPU 端到端 **60.2s**（Debug 120.7s 的一半），ORT 占 86%（51.6s）、手写轻头 0.9s（Debug 5.8s → JIT 优化 6.5 倍）；③ Release 优化未破坏 fp 语义（vs golden max 仍 1.10e-6）；④ 探活必须校验返回的实际 EP（Create 失败会静默回退）。

### M6 — CUDA EP 修复 + DICOM 支持（2026-09-29 追加）✅
- [x] **CUDA EP 修通**（前次"不可用"结论系包缺失所致）：根因 = 项目只引用了 CPU/DirectML 包，无 `Microsoft.ML.OnnxRuntime.Gpu`（CUDA EP native 库）；包策略改为 Managed + Gpu（Gpu 包自带 CPU EP，原生包二选一勿混用）；`OnnxSessionFactory.EnsureCudaRuntime` 自动探测本机 CUDA 12.x toolkit bin 与 `NVIDIA\CUDNN\v9.*\bin\12.*`（cudart/cublas/cudnn64_9.dll）前插进程 PATH。本机环境：RTX 4090 24GB + CUDA 12.4/12.8/13.0/13.3 + cuDNN 9.17/9.8
- [x] **CUDA 实测**（Release，demo case 3 窗）：数值 vs golden max=4e-4 mean=2.6e-5（GPU fp32 几乎无 argmax 翻转）；端到端 99.7s（首跑）/104.1s（稳态）——**慢于 CPU 60.2s**，瓶颈是大输出张量（seg_probs 349MB/窗）PCIe 拷贝 + 每 job 会话重建，GPU 加速收益需 IOBinding/会话复用才能兑现（后续项）
- [x] **DICOM 支持**（`DicomReader.cs`，fo-dicom 5.1）：三种输入——单文件 .dcm（多帧 Enhanced CT）、目录序列（按 ImagePositionPatient 沿法向投影排序）、.zip 序列；RescaleSlope/Intercept → HU；Backend 上传与样例接口同步放宽；前端提示更新
- [x] **DICOM 对拍**（DicomTests）：testdata/dicom_demo.zip（make_dicom.py 从 AC423ccbe.nii.gz 无损转换，368 切片）读回 HU 与 NIfTI 直读**逐位一致（max_diff=0）**、spacing 一致、预处理输出一致（<1e-5）——DICOM 链路与 NIfTI 链路数值等价
- [x] **CUDA 性能兑现（会话复用 + 输出缓冲复用）**：JobRunner 按 EP 缓存 VisionEngine 跨任务复用（RadarInferenceEngine 不再级联 Dispose）；VisionEngine 输出缓冲跨窗复用（~450MB/窗免重复分配）。实测（Release，RTX 4090）：demo case CUDA 首跑 120.9s（含会话/上下文初始化）→ **稳态 24.7s**（CPU 60.2s 的 2.4 倍快），数值 vs golden 不变（max=4e-4 mean=2.6e-5）
- [x] **DICOM 压缩传输语法**：DicomReader 自动把压缩语法（JPEG/JPEG-LS/JPEG2000/RLE 等）经 `Imaging.Codec.DicomTranscoder` 转码为 Explicit VR LE 后解码；RLE 无损压缩回环对拍逐位一致（CompressedTransferSyntax 测试）
- [x] **部署双变体（DirectML + CUDA）**：ORT 同进程只能加载一个 onnxruntime.dll（DML build 与 CUDA build 互斥），csproj 按 `-p:OnnxEp=Cuda|Dml` 条件选择原生包变体（Cuda=Gpu 1.30 / Dml=DirectML 1.24.4，托管层版本跟随），部署时按目标机器 GPU 类型选产物；本机验证 DML 变体探活 `DirectMl: OK（实际 EP = DirectMl）`
- [x] **DICOM 有损压缩（JPEG 有损族）精度影响评估**（2026-09-30 完成，见 M8）
- [x] **fp16 ONNX 可选路径评估**（2026-09-30 完成，见 M8：不达标，结论为不启用）

### M7 — radar_plus 系列权重迁移（2026-09-29 追加）✅
- [x] **权重结构探明**（probe_plus_ckpt.py）：两个 checkpoint（`checkpoint_radar_plus.pth` pretrain / `checkpoint_radar_plus_finetuned_on_merlin.pth` merlin_ft）各含 8 个 whole-head 键（attention_whole.in_proj (768,256) Q|K|V 行序同器官头、vision_projs_whole (256,256)、query_tokens_whole (1,256)、temp 标量）+ text_proj (256,768) + text_encoder 199 键（BLIP med.py 裁剪版 BertModel：无 token_type_embeddings、无 pooler）。temp 两套不同（pretrain=0.053180 / merlin_ft=0.041191），故权重+文本特征各导一套
- [x] **导出脚本**（export/export_plus.py）：`models/plus_head{,_merlin_ft}.npz`（329217 参数/1.26MB）+ `models/plus_text_feats{,_merlin_ft}.bin/.json`（9 疾病 134 条 prompt 特征，HF BertTokenizer + BertModel(ckpt text_encoder 权重, token_type_embeddings 清零) + text_proj + L2 norm；缺失/多余键集合精确断言）
- [x] **merlin 预处理变体**（`PlusPreprocessor.cs`）：与器官路径差异 = ① 无非零域裁剪，SpatialPad(end)→CenterSpatialCrop（monai 取中 start=size//2−target//2）；② clip ±1000（原 [-300,400]）；③ min-max 归一无 1e-8；④ clip/归一在 pad/crop 之后。对拍 vs monai 黄金体积 max_diff<1e-5
- [x] **whole-image 引擎**（`PlusHead.cs` + `RadarPlusInference.cs`）：x1|x2|x3 全量拼接（84096 token）→ 4 头 attention → vision_projs_whole → L2 norm → 与 9 疾病多 prompt 特征点积/temp → pos/neg 分组均值 → softmax。KV 打分用恒等式 u=W_kᵀq_h 先行（对 84k token 的 512 维展开 ~11G 乘加降为 ~0.4G）
- [x] **踩坑记录**：优化式首版漏掉 attention 缩放 `/√head_dim`（OrganHead.Mha 有此步），feature diff 高达 0.156；逐层隔离定位（score/attn/out_h 各自对拍）后补上，恢复 1e-5 级。教训：跨实现移植 MHA 必须逐式清单化（scale/softmax/mask 三件套）
- [x] **黄金对拍**（export/plus_golden.py → testdata/golden/plus_golden_*.json，ORT CPU + numpy 朴素 whole head）：PlusTests 4 项全过——merlin_ft feature max_diff=2.4e-5 / prob max_diff=1.3e-4；pretrain feature 6.3e-7 / prob 9.7e-7（门限 1e-4 / 1e-3）
- [x] **后端 + Web 接入**：JobRunner 支持 `model=radar|radar_plus` + `variant=pretrain|merlin_ft`（whole head 按变体缓存）；API `/api/jobs/*` 透传；前端加「推理模型 / 权重变体」选择器与模型列
- [x] **测试基建**：测试并行禁用（重测试各自 ONNX 会话并行跑触发 BFCArena 1.2GB OOM）；全量 12/12 PASS（5m23s）

### M8 — 遗留待办闭环：DICOM 有损压缩评估 + fp16 路径评估（2026-09-30）✅

#### M8-1 DICOM 有损压缩（JPEG 有损族）精度评估
- **前置修复（实测发现的真 bug）**：`DicomReader.DecodeFrameHu` 的 8bit 分支无视 `PixelRepresentation`，把 unsigned 8bit（JPEG Baseline 解码后的常规形态）按 `sbyte` 解释，HU 偏差均值 2392 HU / max 5218 HU；已按符号分支修复。
- **codec 接入**：fo-dicom 5.1.3 核心包不含 JPEG 解码器，报 `DicomCodecException: ... is not supported`；引入官方扩展包 `fo-dicom.Codecs`（原生 C/C++ 编解码，覆盖 JPEG Baseline/Extended/Lossless、JPEG-LS(Lossy)、JPEG2000、HT-JPEG2000、RLE），升级 `fo-dicom` 5.1.3 → **5.2.5**（Codecs 5.16.7 要求 ≥5.2.5），并在 `DicomReader` 静态构造注册 `NativeTranscoderManager`。Windows 目标机需 VC++ 2015-2022 运行库（未装报 `Unable to load DLL 'Dicom.Native'`）。
- **评估样例**：`testdata/make_dicom_lossy.py` 生成 JPEG Baseline 8bit 有损序列（HU 线性重标到 0-255，整卷统一 slope=18.7529/intercept=-3024；q90/q75 两档），368 切片。
- **实测结论（`DicomLossyTests`，pydicom 独立解码交叉验证一致）**：

| 指标 | q90 | q75 |
|---|---|---|
| HU 误差 max / mean | 417.9 / 13.33 | 910.0 / 20.02 |
| HU 误差 p99 / 中位 | 82.4 / 8.5 | 168.8 / 11.3 |
| clip[-300,400] 域内 mean | 15.51 | 21.67 |
| 预处理输出 max_diff（[0,1] 域） | 0.325 | 0.614 |
| 端到端 146 项 prob max_diff | 0.166 | 未跑 |
| 阳性判定（p≥0.5）翻转 | 0 项 | — |

- **判定**：8bit 量化步长 ≈ 18.75 HU（量化固有误差 ≤ 半步长 9.4 HU），实测均值 13~20 HU 还叠加了 JPEG 振铃；max 误差出现在骨/气高对比边缘（q90: ref=-112 → rec=-530），属编码固有特性非链路缺陷（与 pydicom 解码完全一致）。对最终概率的影响达 0.166（超 1e-3 门禁两个量级），虽然本 demo 病例阳性判定未翻转，但器官分割对边界 HU 跳变敏感 → **诊断链路不接受 8bit 有损 JPEG**。若必须压缩，建议 16bit JPEG2000 有损（近无损）或仅用于预览/存档，不进推理链路。

#### M8-2 fp16 ONNX 路径评估
- **导出**：`export/export_onnx_fp16.py`（onnxconverter-common 1.16.0，隔离安装于 `export/.deps`，`keep_io_types=True`），103.7 MB → 51.9 MB。
- **踩坑**：转换器对原图自带 `Cast` 节点处理不一致——value_info 已改 fp16 但 `to` 属性仍为 FLOAT，ORT 载入报 `Type (tensor(float16)) of output arg (/Cast_output_0) ... does not match expected type (tensor(float))`；脚本内按 value_info 回写 `to` 属性后 `onnx.checker` 通过、会话可建。
- **实测对拍（同输入 (96,256,384)，fp16 vs fp32 输出，CPU/CUDA 两 EP 结论一致）**：

| 输出 | CPU max | CPU mean | 门禁 | 判定 |
|---|---|---|---|---|
| seg_probs | 3.38e-1 | 2.6e-5 | 1e-5 | ✗（p99.9 6.2e-3） |
| res_x1/x2/x3 | 2.4e-3 ~ 3.6e-3 | ~2e-4 | 1e-5 | ✗（超 240 倍） |
| pred_mask | 4267/9.4M 像素不同 | — | 0 | ✗（argmax 翻转） |
| flags1/2/3 | 8 / 21 / 76 位翻转 | — | 0 | ✗（token 选择掩码变化） |

- **判定**：fp16 全模型转换**不满足现有对齐门禁**（res_x 超门禁 240 倍；pred_mask/flags 存在翻转，flags 翻转会改变 key 序列结构）。**结论：不启用 fp16 路径**；保持 fp32（CUDA fp32 稳态 24.7s 已兑现 2.4× CPU）。若将来确需 fp16，须做混合精度（Conv/BatchNorm 等敏感算子保 fp32，仅 GEMM 类降精）并重跑全链路复验，而非整体 `convert_float_to_float16`。

### M9 — 实测差距 8 项收口（2026-09-30）✅

**背景**：迁移审计（逐行通读上游 `inference_demo.py` 628 行 + `infer_merlin_whole.py`）发现推理主链路已完整迁移，但有 8 项「上游有 / 本项目无」或「PLAN 自承诺未落地」的差距，全部在本轮补齐。

**关键澄清**：上游 `inference_demo.py` **本身不产出**分割掩码、可视化与 JSON，唯一输出是一张概率 CSV（L552-555）。因此差距表中 #2（掩码可视化）与 #6（JSON 导出）属 PLAN 自承诺或增补，#1/#3/#4/#5/#7 为真实功能差距。

#### M9-1 官方宽表 CSV（差距 #1、#6）
- **权威来源**：列名表由 `export/gen_item_labels.py` 从上游 `inference_demo.py` 用 `ast` 提取 `self.test_items`（顺序）+ `self.english_mapping`（列名），并与官方 CSV 表头逐列交叉校验（146 列列名与列序完全一致）。
- **踩坑（实测发现的真 bug）**：`english_mapping` 中存在英文名自带半角括号的条目 —— `胃_间质瘤（gist） (Stomach_Gastrointestinal stromal tumor (GIST))`。原先按「从右往左找第一个 `" ("`」切分反查会切错（切出 `...stromal tumor`），导致宽表**往返不是逐字节一致**。改为在 `TestItemLabels` 里构建 `列名 -> 条目` 的**精确字典**反查（由 `English` 派生，生成器内加「列名唯一性」断言保证可一一反查），未知列才退化为最左侧 `" ("` 前缀。
- **格式约定（实测确定）**：UTF-8 BOM + CRLF + 末尾 CRLF；147 列；数值为 Python `repr(float)` 最短往返（与 C# `"R"` 仅差指数大小写与整数补 `.0`）；NaN 写空字段。
- **验收**：`testdata/golden/RADAR_infer_results_demo.csv` → `Read` → `Build` **逐字节一致（9993 = 9993 字节）**；11 组 `repr` 取值全覆盖；多行走查回位级可逆。
- **落地**：`Radar.Core/Postprocessing/ResultCsv.cs`；API `/api/jobs/{id}/csv?format=official` 与 `/json`；CLI `--output`。

#### M9-2 分割掩码可视化（差距 #2）
- **引擎改造**：`RadarInferenceEngine.Result` 暴露 `StitchedMask`（滑窗概率累加 → argmax 拼接的 (D,H,W) 标签图）+ `D/H/W`；补推阶段不修改它。
- **实现**：`PngEncoder`（真彩色 RGB8、ZLibStream 生成 IDAT、CRC32 表；无图像库依赖）+ `MaskOverlayPng`（沿三轴做 MIP，掩码 0.55 不透明度叠色，36 器官按 HSV 色环分配，三视图横向拼接 + 8px 黑色留白）。
- **独立预言机（避免「自己测自己」）**：`export/gen_png_ref.py` 用 Python 标准库 `zlib` 按 PNG 规范手工拼一份参考 PNG（5×3，非 16 整数倍宽度），落盘 IHDR/IEND 的**完整块字节**与 IDAT **解压后的原始扫描线**；测试端逐字节比对（IHDR/IEND 与压缩实现无关，IDAT 比语义内容以绕过压缩级别差异）。
- **验收**：编码器对 Python 参考**结构与语义全等**；与黄金 `stitched_mask.npy` **逐体素全等（0 / 12,496,896 不一致）**，前景体素 1,645,197 完全相同、标签上界同为 36。
- **落地**：`Radar.Core/Visualization/`；API `/api/jobs/{id}/mask.png`；CLI `--save-mask`。

#### M9-3 目录级批处理与独立 CLI（差距 #3、#4）
- **新增 `tools/Radar.Cli`**（程序集名 `radar`，加入 `Radar.slnx`）：`infer` / `eval` / `help`；`Options.cs` 做 `--key value` 与 `--flag` 解析；`CliEnv.FindRepoRoot()` 上溯定位 `models/`（`tests` 里的 `RepoRoot` 不可跨程序集复用）。
- **批处理语义**：递归扫描 `.nii/.nii.gz/.dcm/.zip` → 按路径排序（保证确定性）→ 按 `--workers` × 设备做索引分片 → `results[i]` 按序聚合 → **一次**写出宽表（不会多 worker 并发写同一文件）。
- **实测**：4 worker × 4 个 merlin 病例，单个 ~100.5s，总计 105.0s（串行约 400s）；结果互不相同且行序 == 输入排序。

#### M9-4 超大病例跳过守卫（差距 #5）
- 上游在预处理后检查 `image.shape[1:]`，任一维 > 1000 就 `continue`（跳过而非失败）。
- **实现**：`RadarInferenceEngine.MaxSpatialDim = 1000` + 可独立调用的静态谓词 `ExceedsSpatialLimit(d,h,w)`（批处理预筛用）+ `CaseSkippedException`（独立类型，便于与「推理失败」区分）。
- **验收**：真实引擎上喂 `(1001,1,1)` 病例，**进入滑窗之前**抛出异常，实测 **3 ms** 返回；边界 `== 1000` 不抛。批处理汇总单列跳过数，Web 端新增 `Skipped` 状态与「超限跳过」文案。

#### M9-5 AUC 评估（差距 #7）
- `Radar.Core/Support/RocAuc.cs`：秩形式 Mann-Whitney U，平局取平均秩，单类（全正/全负）返回 NaN —— 与 `sklearn.metrics.roc_auc_score` 语义一致。
- `Radar.Core/Support/MerlinDiseases.cs`：`map_radar_merlin` 21 项 key 逐字保留；胆囊缺失项用分割体素数（< 1000）判定而非概率。
- **交叉验证**：测试端用**朴素两两配对**（O(n²)，平局计 0.5）作为独立算法，与秩形式在随机/平局/单类等场景下逐值一致（< 1e-12）。
- **口径提醒**：`MerlinDiseases.Map` 中 `大肠_粘膜下水肿`、`心脏_主动脉瓣钙化`、`肝_肝大`、`肾_肾积水`、`胆囊_术后胆囊缺失` **不在** RAD-CT 的 146 项内 —— 上游 MERLIN 评估用另一套 finding 名表。`eval` 对缺失项显式报告。

#### M9-6 差距 #8（多卡 `all_gather`）
- **明确不做**：属训练/大集评估场景，需引入 NCCL。改以 CLI `--workers` 做**多进程级**并行（同机按设备分片），无分布式依赖。

#### M9-7 前端与端点补齐
- **新端点**：`/api/jobs/{id}/json`（`UnsafeRelaxedJsonEscaping` 保持中文可读）、`/api/jobs/{id}/mask.png`、`POST /api/jobs/batch`；`/csv?format=official`；`JobView` 增 `hasMask`。
- **前端**：样例下拉改 `multiple`（多选即批量）、任务行末列改为「结果 / 下载」链接组（查看 / 官方CSV / 长表 / JSON / 掩码）、结果面板加 KPI 下载区与掩码图、`Skipped` 中文文案。
- **前端回归方法（沿用本项目的实证手法）**：**不做静态检查** —— 编译器不校验裸 JS，模板字符串里字段名拼错只在运行时暴露（上一轮 `undefined` bug 即由此漏过）。改为：启动后端 → 探活 → 从 `GET /` 抓**实际被服务的那份 `<script>`** → 在 Node `vm` 里配 DOM shim + 真实 `/api/devices`、`/api/samples`、`/api/jobs` payload 执行 → 断言 38 项（无 `undefined`/`NaN`、状态文案、链接路径、掩码显隐切换、多选走 batch / 单选走 sample / 空选不请求、`init()` 确实填充了下拉）。
  - 脚本：`export/verify_frontend.py`。**必须校验服务端返回的脚本**，否则静态资源没更新会「假绿」。

#### M9-8 测试与文档
- 新增 4 个测试类共 51 个用例（官方宽表 CSV 20 / ROC AUC 11 / PNG 与掩码 10 / 跳过守卫 10），并在既有端到端用例上追加**掩码逐体素对拍**（复用同一次推理，零额外耗时）。
- 全量 **64/64 PASS（8m30s，0 跳过）**；`dotnet build -c Release` **0 warning / 0 error**。
- README §8.2 由「未实现清单」改写为「已收口清单」，附落地位置与验收数据。

> **已知遗留**：`eval` 子命令的端到端验证仅覆盖算子与格式（与朴素配对交叉验证），未在真实 MERLIN 测试集上跑出全量 AUC 数值 —— 缺少 MERLIN 标签 JSON（Stanford AIMI 需注册下载）。属数据可及性问题，非实现缺陷。

---

> 总工期估算：**11–15 个工作日**（单人）。M1 硬门槛已于 2026-09-29 通过（4 个 commit）。
>
> **M1 实测结论**：① 向量化改造与原版逐位一致（8 输出 diff=0）；② ONNX 导出 PASS（res_x < 1.5e-5、pred_mask 不匹配 3/9.4M、flags 全等）；③ 官方 CSV 软参考 PASS（146 项全匹配 mean 5.8e-5）；④ 本机 CUDA 可用（golden dump GPU 秒级）。

---

## 5. 测试数据管理

| 数据 | 来源 | 用途 | 入库 |
|---|---|---|---|
| AC423ccbe.nii.gz | `../damo-radar/data/demo_cases/` | 主对齐样例（有官方参考 CSV） | 否（软引用） |
| RADAR_infer_results_demo.csv | `../damo-radar/results/` | 对齐金标准（147 列 = `file_name` + 146 项） | **是**（14 KB，拷入 `testdata/golden/`） |
| png_ref_5x3.{png,json} | `export/gen_png_ref.py` 自产 | PNG 编码器的独立预言机 | **是**（< 1 KB） |
| merlin_data_train_demo | `../damo-radar/data/` | M5 冒烟扩展 | 否（软引用） |
| golden 逐窗 dump | M1 自产 | 逐级对拍 | 否（gitignore） |
| 更多样例 | MERLIN（Stanford AIMI，需注册下载） | 可选扩展 | 否 |

> 测试数据获取：demo case 与官方参考 CSV 从 `../damo-radar/` 对应目录拷入 `testdata/`；
> golden 逐窗基线与 `testdata/dicom_*.zip` 分别由 `export/golden_dump.py` 与 `testdata/make_dicom*.py` 生成。
> （原计划中的 `export/fetch_testdata.py` 未实现，数据获取按上述手动步骤执行。）

---

## 6. 对齐验收标准（汇总）

| 层级 | 指标 | 阈值 |
|---|---|---|
| Python 内部（向量化改造） | forward 输出 max abs diff | < 1e-6 |
| ONNX 导出 | ORT vs torch（逐窗 8 张量） | < 1e-5 |
| .NET 预处理 | vs 黄金基线逐元素 | < 1e-5 |
| .NET 滑窗/器官头 | 逐窗张量 / 单器官概率 | < 1e-4 / < 1e-3 |
| 端到端（**核心门禁**） | demo case 146 值 vs 本地 golden | 全部 < 1e-3 |
| 分割掩码（**核心门禁**，M9 追加） | argmax 拼接标签图 vs 黄金 `stitched_mask.npy` | 逐体素全等 |
| 官方 CSV 格式（M9 追加） | 本地 golden → 宽表 → 对比官方文件 | 逐字节全等 |
| ROC AUC 算子（M9 追加） | vs 朴素两两配对（独立算法） | < 1e-12 |

浮点注意事项：ORT 与 torch 的 Conv3d/interpolate 在不同 EP/线程数下存在 1e-6 级浮点差异，逐窗累加（数百窗 × 概率平均）会放大到 1e-4 级——阈值已按此留量。若个别列超差，优先排查重采样语义，其次 max_pool3d 的 ceil/floor 模式。

## 7. 风险清单

| # | 风险 | 概率 | 对策 |
|---|---|---|---|
| R1 | MONAI trilinear 重采样语义与手写实现对不齐 | 高 | 用 ONNX Resize 子图规避手写（§2.1）；对拍不过再回退查 align_corners |
| R2 | 向量化 flags 改造与 torch.unique 语义不完全等价 | 中 | M1 首个任务即 Python 内对拍，改到过为止 |
| R3 | DirectML 对 Conv3d/opset17 覆盖不全 | 中 | `auto` 探活回退 CPU；CUDA EP 兜底；M5 矩阵实测 |
| R4 | 大体积 NIfTI（>512³）内存压力 | 中 | float32 单卷 + 流式重采样；跳过 >1000 维样例（与原版一致，**M9-4 已落地**为 `CaseSkippedException`） |
| R5 | NIfTI 读取器兼容性（数据类型/压缩变体） | 中 | 首版支持 float32/int16 + gzip；用 demo + merlin demo 实测 |
| R6 | fp16 精度损失 | 低 | 默认 fp32，fp16 仅在对齐通过后开 |

## 8. 模型产物管理（models/，gitignore）

导出产物不入库（>100 MB）。`models/README.md`（入库）记录：
- 每个文件的 md5、参数量、导出日期、对应上游 ckpt 的 md5；
- 重建命令（按序执行，无 `export_all.py` 汇总脚本；依赖 `C:/Users/Administrator/miniconda3/python.exe`，torch 2.9.1+cu128）：
  `export/export_onnx.py` → `export/export_weights.py` → `export/export_text_feats.py` → `export/export_plus.py`。
  各脚本自带对拍验收；完整清单见 README.md §4 与 models/README.md。

## 9. Git 工作流

- 分支：`main`（稳定，里程碑合并）+ `feature/m1-export`、`feature/m2-preprocess` 等短命分支。
- 提交粒度：每个可验证单元一次提交（如「M2: NIfTI 读取器 + 黄金对拍用例」），提交信息带里程碑前缀。
- 门禁：M3 起任何合并到 main 的提交必须通过 `dotnet test`（对齐用例 green）。
- 已配置身份：dongfangzhizhu <axu0606@gmail.com>（本机全局配置）。

## 10. 待确认清单

- [x] ~~部署目标机 GPU 型号~~（2026-09-29 已确认：同时支持 DirectML 与 CUDA，双变体构建已落地，见 M6）
- [x] ~~merlin_data_train_demo 是否够用~~（2026-09-29 已确认：够用，M5 冒烟 4 例 + M7 golden 均基于现有数据）
- [x] ~~Web 端是否需要鉴权~~（2026-09-29 已确认：仅功能演示，无需鉴权，维持现状）
- [x] ~~radar_plus 系列权重的迁移排期~~（2026-09-29 已完成，见 M7）
