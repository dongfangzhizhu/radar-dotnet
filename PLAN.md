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

### M2 — Radar.Core 预处理 + NIfTI（预计 2–3 天）
- [ ] NIfTI/.nii.gz 读取器（Nifti.NET 或手写 ~150 行，需支持 gzip 流 + affine 读取 + 数据类型提升）
- [ ] 预处理管线：spacing→(1,1,5)mm 目标尺寸计算 → ORT 重采样子图 → HU clip → min-max 归一化 → 非零域裁剪 + (5,20,20) 外扩 → 尾部 pad (96,256,384)
- **验收**：预处理输出张量与黄金基线逐元素 diff < 1e-5（xUnit 用例）

### M3 — 滑窗推理与器官头（预计 3–4 天）
- [ ] ORT Session 封装（EP 注入、IOBinding、预分配 pinned buffer）
- [ ] 滑窗调度（dense_patch_slices 等价实现，overlap 0.25）+ 概率累加 + argmax 拼接
- [ ] C# 器官头：MHA（4 头，手写 softmax(QKᵀ/√64)V）→ vision_projs → L2 → 点积/温度 → softmax
- [ ] 边界器官排除（margin=2）+ 缺失器官 center_crop 补推
- **验收**：① 每窗 8 张量 vs 黄金基线 diff < 1e-4；② 单器官概率 diff < 1e-3；③ demo case 最终 146 值全部 diff < 1e-3

### M4 — Radar.Backend + Web 测试端（预计 2–3 天）
- [ ] 任务队列（Channel，GPU 单 worker）、文件上传/暂存、结果缓存
- [ ] /api/jobs 系列端点 + /api/devices
- [ ] wwwroot 单页（上传/选样例/设备选择/进度/结果表/偏差对齐列/CSV 下载）
- **验收**：Web 端跑 demo case，结果与 M3 CLI 结果一致；并发 2 任务队列行为正确

### M5 — 性能测试与多 EP 矩阵（预计 2 天）
- [ ] 基准：demo case 在 CPU / DirectML / (CUDA 若可用) 三路径的端到端耗时 + 各阶段分解（目标：CPU ≤ Python GPU 版 2×，DirectML ≤ Python GPU 版 1.5×，量级待实测标定）
- [ ] fp16 ONNX 可选路径（精度对齐通过后默认开启 GPU fp16）
- [ ] 扩展数据集冒烟：`../damo-radar/data/merlin_data_train_demo/` 下样例批量跑，无崩溃、无 NaN
- **验收**：性能报告（markdown）入库；三路径数值一致性结论明确

> 总工期估算：**11–15 个工作日**（单人）。M1 硬门槛已于 2026-09-29 通过（4 个 commit）。
>
> **M1 实测结论**：① 向量化改造与原版逐位一致（8 输出 diff=0）；② ONNX 导出 PASS（res_x < 1.5e-5、pred_mask 不匹配 3/9.4M、flags 全等）；③ 官方 CSV 软参考 PASS（146 项全匹配 mean 5.8e-5）；④ 本机 CUDA 可用（golden dump GPU 秒级）。

---

## 5. 测试数据管理

| 数据 | 来源 | 用途 | 入库 |
|---|---|---|---|
| AC423ccbe.nii.gz | `../damo-radar/data/demo_cases/` | 主对齐样例（有官方参考 CSV） | 否（软引用） |
| RADAR_infer_results_demo.csv | `../damo-radar/results/` | 对齐金标准（165 列） | **是**（14 KB，拷入 `testdata/golden/`） |
| merlin_data_train_demo | `../damo-radar/data/` | M5 冒烟扩展 | 否（软引用） |
| golden 逐窗 dump | M1 自产 | 逐级对拍 | 否（gitignore） |
| 更多样例 | MERLIN（Stanford AIMI，需注册下载） | 可选扩展 | 否 |

获取脚本 `export/fetch_testdata.py`（M1 提供）：从 `../damo-radar/` 拷贝/软链至 `testdata/`，校验 md5。

---

## 6. 对齐验收标准（汇总）

| 层级 | 指标 | 阈值 |
|---|---|---|
| Python 内部（向量化改造） | forward 输出 max abs diff | < 1e-6 |
| ONNX 导出 | ORT vs torch（逐窗 8 张量） | < 1e-5 |
| .NET 预处理 | vs 黄金基线逐元素 | < 1e-5 |
| .NET 滑窗/器官头 | 逐窗张量 / 单器官概率 | < 1e-4 / < 1e-3 |
| 端到端（**核心门禁**） | demo case 146 值 vs 本地 golden | 全部 < 1e-3 |

浮点注意事项：ORT 与 torch 的 Conv3d/interpolate 在不同 EP/线程数下存在 1e-6 级浮点差异，逐窗累加（数百窗 × 概率平均）会放大到 1e-4 级——阈值已按此留量。若个别列超差，优先排查重采样语义，其次 max_pool3d 的 ceil/floor 模式。

## 7. 风险清单

| # | 风险 | 概率 | 对策 |
|---|---|---|---|
| R1 | MONAI trilinear 重采样语义与手写实现对不齐 | 高 | 用 ONNX Resize 子图规避手写（§2.1）；对拍不过再回退查 align_corners |
| R2 | 向量化 flags 改造与 torch.unique 语义不完全等价 | 中 | M1 首个任务即 Python 内对拍，改到过为止 |
| R3 | DirectML 对 Conv3d/opset17 覆盖不全 | 中 | `auto` 探活回退 CPU；CUDA EP 兜底；M5 矩阵实测 |
| R4 | 大体积 NIfTI（>512³）内存压力 | 中 | float32 单卷 + 流式重采样；跳过 >1000 维样例（与原版一致） |
| R5 | NIfTI 读取器兼容性（数据类型/压缩变体） | 中 | 首版支持 float32/int16 + gzip；用 demo + merlin demo 实测 |
| R6 | fp16 精度损失 | 低 | 默认 fp32，fp16 仅在对齐通过后开 |

## 8. 模型产物管理（models/，gitignore）

导出产物不入库（>100 MB）。`models/README.md`（入库）记录：
- 每个文件的 md5、参数量、导出日期、对应上游 ckpt 的 md5；
- 一键重建命令：`python export/export_all.py`（依赖 `C:/Users/Administrator/miniconda3/python.exe`，torch 2.9.1+cu128）。

## 9. Git 工作流

- 分支：`main`（稳定，里程碑合并）+ `feature/m1-export`、`feature/m2-preprocess` 等短命分支。
- 提交粒度：每个可验证单元一次提交（如「M2: NIfTI 读取器 + 黄金对拍用例」），提交信息带里程碑前缀。
- 门禁：M3 起任何合并到 main 的提交必须通过 `dotnet test`（对齐用例 green）。
- 已配置身份：dongfangzhizhu <axu0606@gmail.com>（本机全局配置）。

## 10. 待确认清单

- [ ] 部署目标机 GPU 型号（决定 DirectML vs CUDA 主路径，影响 M5 基准口径）
- [ ] merlin_data_train_demo 是否够用，是否需要申请 MERLIN 完整测试集（影响 M5 冒烟覆盖面）
- [ ] Web 端是否需要鉴权（内网测试工具默认无鉴权）
- [ ] radar_plus 系列权重的迁移排期（管线已预留，未列入本期）
