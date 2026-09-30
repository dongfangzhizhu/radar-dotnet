# radar-dotnet

将 [damo-radar](https://github.com/alibaba-damo-academy/damo-radar)（RADAR / RADAR+ —— 全腹增强 CT 的多器官分割与器官级病灶概率诊断模型，*Science* 2026）的**推理链路**完整迁移到 **.NET 10** 的实现。

纯 C# 推理，无 Python 运行时依赖；支持 **CPU / DirectML / CUDA** 三种执行后端，提供 Web 服务与单页测试端。
数值上与官方 Python 实现逐值对齐（端到端 146 项概率误差 < 1e-3，实测 5.96e-7；分割掩码逐体素全等）。

> 上游 Python 项目（`../damo-radar`）与权重（`../RADAR`）**只读参照，不在本仓库内**。
> 完整迁移计划、里程碑与踩坑记录见 [PLAN.md](PLAN.md)。

---

## 1. 能力概览

### 1.1 输入

| 格式 | 说明 |
|---|---|
| `.nii` / `.nii.gz` | NIfTI，含 gzip 流、sform/qform affine、8 种像素类型 |
| `.dcm` | 单文件 DICOM（含多帧 Enhanced CT） |
| DICOM 目录序列 | 按 `ImagePositionPatient` 沿法向投影排序 |
| `.zip` | DICOM 序列打包 |

DICOM 侧自动处理压缩传输语法（RLE / JPEG Baseline / JPEG Extended / JPEG Lossless / JPEG-LS / JPEG2000 / HT-JPEG2000），
非压缩路径经 `RescaleSlope/Intercept` 还原 HU。**DICOM 支持为本项目新增能力**，上游仅有 NIfTI。

### 1.2 两条推理链路

| 链路 | 模型 | 打分粒度 | 输出 |
|---|---|---|---|
| `radar` | `checkpoint_radar_pretrain.pth` | 器官级（36 器官 → **146 项**「器官_病灶」） | 滑窗 + 边界/缺失器官补推 |
| `radar_plus` | `checkpoint_radar_plus{,_finetuned_on_merlin}.pth` | 全图级（**9 种疾病**，多 prompt 正负分组） | 单窗整图 |

两条链路共用同一视觉主干（ONNX），差异全部在预处理与打分头上。

### 1.3 已实现的关键环节

- **预处理**：各向异性重采样至 `(1,1,5) mm`（C# 手写 fp32 三线性，与 torch CPU kernel 逐位一致）→ HU clip → min-max 归一化 → 非零域裁剪 + 外扩 → 填充至 `(96,256,384)`
- **滑窗推理**：`roi=(96,256,384)`、`overlap=0.25`，等价 monai `dense_patch_slices`；概率累加 + count map → argmax 拼接
- **器官头**：手写 4 头 MHA（256 维）+ per-organ `vision_projs` + L2 归一化 + 与预计算文本特征点积 / temp → softmax
- **边界器官排除**（margin=2）与**缺失器官 center_crop 二次补推**
- **RADAR+ whole head**：84096 token 全局 attention → L2 → 9 疾病多 prompt 分组均值 → softmax（KV 打分用恒等式 `u = W_kᵀq` 先行，将约 11G 乘加降至约 0.4G）
- **多 EP 与部署变体**：按 `-p:OnnxEp=Cuda|Dml` 条件选择原生包；CUDA 路径下会话与输出缓冲跨任务复用
- **官方格式导出**：宽表 CSV（BOM + CRLF + `{中文} ({英文})` 列名 + Python `repr(float)` 数值），与官方文件**逐字节往返一致**；另有 JSON 导出端点
- **分割掩码可视化**：滑窗 argmax 拼接的标签图 → 三正交 MIP 透叠 PNG（零依赖编码器，无图像库）
- **目录级批处理**：`radar infer --input <dir>` 多 worker × 设备分片，输出顺序与输入排序对齐
- **AUC 评估**：`radar eval` 按 MERLIN 21 项 finding 出 AUC + AvgAUC，算子与 `sklearn.roc_auc_score` 逐值一致
- **超大病例守卫**：预处理后任一空间维 > 1000 直接 `CaseSkippedException`（跳过 ≠ 失败，批处理继续）

---

## 2. 目录结构

```
radar-dotnet/
├── PLAN.md                  完整实施计划（M0–M9）+ 验收记录 + 踩坑
├── benchmarks.md            性能基准实测记录
├── Radar.slnx               .NET 10 解决方案
├── src/
│   ├── Radar.Core/          D1 推理库（纯 C#，无 Web 依赖）
│   │   ├── Preprocessing/   两条链路的预处理（Preprocessor / PlusPreprocessor / Resampler）
│   │   ├── Inference/       ORT 会话、视觉引擎、滑窗、器官头、RADAR+ whole head
│   │   ├── Postprocessing/  官方格式宽表 CSV（ResultCsv）
│   │   ├── Visualization/   零依赖 PNG 编码器 + 掩码透叠图（PngEncoder / MaskOverlayPng）
│   │   ├── Nifti/           NIfTI 与 DICOM 读取器
│   │   └── Support/         NPY/NPZ 解析、器官/标签/疾病映射表、ROC AUC
│   └── Radar.Backend/       D2 Web 服务（ASP.NET Core Minimal API）
│       ├── InferenceJobRunner.cs   任务队列（Channel 单 worker 串行，GPU 独占）
│       ├── wwwroot/index.html      D3 单页测试端（原生 HTML/JS，无框架）
│       └── appsettings*.json
├── tests/Radar.Tests/       D5 xUnit 对齐测试（48 个用例）
├── tools/
│   ├── Radar.Benchmark/     性能基准 + 设备探活 + 扩展集冒烟
│   └── Radar.Cli/           独立 CLI（`radar infer` 目录级批推理 / `radar eval` AUC 评估）
├── export/                  D4 Python 导出工具链与对拍/验证脚本（一次性，见 §4）
├── models/                  导出产物（gitignore，见 §4 与 models/README.md）
└── testdata/                测试数据（gitignore，见 §5）
```

---

## 3. 快速开始

### 3.1 前置条件

- **.NET 10 SDK**
- **模型产物**：`models/` 内容不入库（体积大），首次使用必须自行生成或获取 —— 见 §4
- *（仅 DICOM 路径需要）* Windows 上安装 **VC++ 2015-2022 可再发行组件**，否则原生编解码器加载失败（`Unable to load DLL 'Dicom.Native'`）

### 3.2 构建

```bash
# CUDA 变体（NVIDIA GPU）
dotnet build src/Radar.Backend/Radar.Backend.csproj -c Release -p:OnnxEp=Cuda

# DirectML 变体（AMD / Intel / NVIDIA 通用，免 CUDA 版本匹配）
dotnet build src/Radar.Backend/Radar.Backend.csproj -c Release -p:OnnxEp=Dml
```

> **同一进程只能加载一个 `onnxruntime.dll`**，DirectML 与 CUDA 原生包互斥，因此二选一构建。默认（不传 `OnnxEp`）为 CPU + DirectML。

### 3.3 运行 Web 测试端

```bash
dotnet run --project src/Radar.Backend -c Release --no-build
```

打开 **http://localhost:5202** （端口由 `src/Radar.Backend/Properties/launchSettings.json` 决定，命令行 `--urls` 会被它覆盖）。

页面上选择【设备 / 样例（**可多选 = 批量**）/ 推理模型 / 权重变体】→「运行选中样例」→ 队列中查看状态与耗时 → 「查看」展示概率明细；每行可直接下载官方格式 CSV / 长表 CSV / JSON / 掩码 PNG。

**HTTP 接口：**

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/devices` | 可用执行后端（真实 EP 校验，非仅尝试创建） |
| `GET` | `/api/samples` | `testdata/` 下的可用样例 |
| `POST` | `/api/jobs/sample` | 用样例建任务：`{sample, device, model, variant}` |
| `POST` | `/api/jobs/batch` | 批量建任务：`{samples[], device, model, variant}` |
| `POST` | `/api/jobs/upload` | 上传建任务：`multipart/form-data`，query 带 `device/model/variant` |
| `GET` | `/api/jobs` | 任务列表（JSON，**camelCase**；含 `hasMask`） |
| `GET` | `/api/jobs/{id}` | 单任务详情 |
| `GET` | `/api/jobs/{id}/csv` | 结果 CSV（**长表** `item,probability`） |
| `GET` | `/api/jobs/{id}/csv?format=official` | 官方格式**宽表** CSV（1 行 × 147 列，BOM + CRLF） |
| `GET` | `/api/jobs/{id}/json` | 结果 JSON（`UnsafeRelaxedJsonEscaping`，中文不转义） |
| `GET` | `/api/jobs/{id}/mask.png` | 分割掩码透叠图（仅 `radar` 路径，`204/404` 表示本任务无掩码） |

> 任务状态多一个 `Skipped`：空间维超限的病例被**跳过**而非标记失败，避免污染批处理的失败率。

### 3.4 命令行（`radar` CLI）

```bash
# 单文件或整目录批量推理，直接产出官方格式宽表 CSV
dotnet run --project tools/Radar.Cli -c Release -- infer \
    --input testdata --output out/results.csv --device cpu --workers 4 --save-mask

# MERLIN 测试集 AUC 评估（21 项 finding + AvgAUC）
dotnet run --project tools/Radar.Cli -c Release -- eval \
    --pred out/results.csv --labels merlin_labels.json --device cpu
```

| 参数 | 适用命令 | 说明 |
|---|---|---|
| `--input` | `infer` | 文件或目录（递归扫描 `.nii/.nii.gz/.dcm/.zip`） |
| `--output` | `infer` | 输出宽表 CSV 路径 |
| `--model` / `--variant` | 两条 | `radar`(默认) / `radar_plus`；变体 `pretrain` / `merlin_ft` |
| `--device` / `--workers` | 两条 | 执行后端；并行 worker 数（按设备分片，输出顺序仍与输入排序一致） |
| `--save-mask` | `infer` | 为每个 `radar` 病例额外落盘掩码 PNG |
| `--pred` / `--labels` | `eval` | 预测宽表 CSV；MERLIN 标签 JSON |
| `--sag-counts` | `eval` | 胆囊缺失项改用分割体素计数的数据源（上游口径：体素 < 1000 判阳性） |

> 批处理中超出 1000 维的病例会被跳过并在汇总里单列，不计入失败。

### 3.5 测试

```bash
dotnet test tests/Radar.Tests/Radar.Tests.csproj -c Release -p:OnnxEp=Cuda
```

64 个用例，覆盖：NIfTI 读取、预处理、滑窗调度与拼接、器官头逐器官特征、端到端 146 项与**掩码逐体素**门禁、DICOM 等价性、DICOM 压缩转码、有损压缩误差量化、RADAR+ 预处理与双变体对拍，以及官方宽表 CSV 往返、Python `repr(float)` 数值格式、ROC AUC 与朴素配对交叉验证、PNG 编码器对 Python zlib 参考实现、超大病例跳过守卫。

> 测试**禁用了并行执行**（`AssemblyInfo.cs`）：多个重测试各自持有 ONNX 会话并行跑会触发 BFCArena 显存 OOM。
> 用例在依赖的 golden / 模型缺失时会**静默跳过**（`return`），因此「全绿」前请确认数据已就位。

### 3.6 设备探活与基准

```bash
dotnet run --project tools/Radar.Benchmark -c Release -- probe   # 秒级，只探活三后端
dotnet run --project tools/Radar.Benchmark -c Release            # 完整基准，覆写 benchmarks.md
```

---

## 4. 模型产物（`models/`，不入库）

`models/` 下的文件由 `export/` 工具链生成，**必须自备**，否则程序无法启动。
逐文件说明、md5、依赖条件见 [`models/README.md`](models/README.md)。

**上游依赖（均不在本仓库，缺失则无法重建）：**

- `../RADAR/checkpoint_radar_pretrain.pth`、`checkpoint_radar_plus.pth`、`checkpoint_radar_plus_finetuned_on_merlin.pth`
- `../RADAR/bert-base-uncased/`、`../RADAR/bert-base-chinese/`
- `../damo-radar/RADAR_inference/`（网络结构源码）、`../damo-radar/RADAR_train/`（RADAR+ 提示词清单）、`../damo-radar/ckpt/infer_text_embedding_radar.pt`

**重建顺序**（解释器：`C:/Users/Administrator/miniconda3/python.exe`，torch 2.9.1+cu128；在仓库根执行）：

```bash
python export/export_onnx.py         # → models/radar_vision.onnx        （本链路核心，98.9 MB）
python export/export_weights.py      # → models/organ_head.npz           （器官轻头权重）
python export/export_text_feats.py   # → models/text_feats.{bin,json}    （146 项文本特征）
python export/export_plus.py         # → models/plus_head{,_merlin_ft}.npz
                                     #   models/plus_text_feats{,_merlin_ft}.{bin,json}
```

各导出脚本自带**对拍验收**（向量化 vs 原版、ONNX vs torch、golden vs 官方 CSV），失败会显式报错。
`export/verify_*.py`、`probe_plus_ckpt.py` 为过程性对拍/探查脚本，保留用于复现结论。

---

## 5. 测试数据（`testdata/`，不入库）

| 数据 | 来源 | 用途 |
|---|---|---|
| `AC423ccbe.nii.gz` | 上游 `data/demo_cases/` | 主对齐样例（有官方参考 CSV） |
| `golden/RADAR_infer_results_demo.csv` | 上游 `results/`（**入库**，14 KB） | 对齐金标准（宽表往返 + 列名/列序 + 数值格式） |
| `golden/png_ref_5x3.{png,json}` | `export/gen_png_ref.py`（**入库**） | PNG 编码器的独立预言机（Python zlib 手工拼块） |
| `golden/*`（其余） | `export/golden_dump.py` 重建 | 逐级对拍基线（本地约 1.1 GB） |
| `dicom_demo.zip` | `testdata/make_dicom.py` 由 NIfTI 无损转换 | DICOM 链路等价性 |
| `dicom_lossy_q{75,90}.zip` | `testdata/make_dicom_lossy.py` | 有损压缩误差量化 |

---

## 6. 对齐验收门禁

| 层级 | 指标 | 阈值 | 实测 |
|---|---|---|---|
| 预处理 | vs 黄金基线逐元素 | < 1e-5 | mismatch = 0 |
| 滑窗/器官头 | 逐窗张量 / 单器官概率 | < 1e-4 / < 1e-3 | PASS |
| **端到端（核心门禁）** | demo case 146 项 vs 本地 golden | 全部 < 1e-3 | max **5.96e-7**（另一轮 1.10e-6） |
| **分割掩码（核心门禁）** | argmax 拼接标签图 vs 黄金 `stitched_mask.npy` | 逐体素全等 | **0 / 12,496,896 不一致** |
| 官方 CSV 格式 | 本地 golden → 宽表 → 逐字节 vs 官方文件 | 全等 | 9993 = 9993 字节 |
| 数值格式 | Python `repr(float)` | 全等 | 11 组取值全覆盖 |
| ROC AUC | vs 朴素两两配对（独立算法） | < 1e-12 | 一致 |
| 软参考 | 本地 golden vs 官方 CSV | mean < 1e-4 | mean 5.8e-5，146/146 匹配 |
| RADAR+ merlin_ft | 特征 / 概率 | < 1e-4 / < 1e-3 | 2.4e-5 / 1.3e-4 |
| RADAR+ pretrain | 特征 / 概率 | < 1e-4 / < 1e-3 | 6.3e-7 / 9.7e-7 |

浮点说明：ORT 与 torch 的 Conv3d / interpolate 存在 1e-6 级差异，滑窗逐窗累加会放大到 1e-4 级，阈值已按此留量。
掩码门禁之所以能取「逐体素全等」：概率差 5.96e-7 远小于任何 argmax 翻转所需的间隔，实测 12,496,896 个体素零差异。

---

## 7. 性能（Release，单机实测）

| 场景 | 耗时 |
|---|---|
| CPU（demo case，3 窗，端到端） | **60.2 s** |
| CUDA（RTX 4090，首次含会话初始化） | 120.9 s |
| CUDA（稳态，会话 + 缓冲复用） | **24.7 s**（CPU 的 2.4 倍快） |

ORT 推理占端到端的约 86%，手写轻头 < 1 s。GPU 收益主要来自**跨任务会话复用 + 输出缓冲复用**，
而非算子本身（单窗 `seg_probs` 达 349 MB，PCIe 拷贝是瓶颈）。明细见 [benchmarks.md](benchmarks.md)。

---

## 8. 迁移完整度

### 8.1 已完整迁移

推理主链路（上游 `inference_demo.py` / `infer_merlin_whole.py`）的全部功能：
NIfTI 读取 → 预处理 → 滑窗调度 → 视觉主干 → 器官头 / whole head 打分 → 边界与缺失器官回退 → 多窗平均 → 146 项 / 9 疾病概率输出；
外加 DICOM 支持、多 EP 部署、Web 服务与单页测试端（上游为 CLI）。

### 8.2 与上游的差异（8 项实测差距，已全部收口）

审计方法：逐行通读上游 `inference_demo.py`（628 行）与 `infer_merlin_whole.py`，与本地实现做能力对照。
**关键澄清**：上游 `inference_demo.py` 本身**不产出**分割掩码、可视化与 JSON，唯一输出是一张概率 CSV；
因此下面 2 / 6 两项属本项目 PLAN 的自承诺或增补，其余为真实功能差距。

| # | 项 | 状态 | 落地位置与验收 |
|---|---|---|---|
| 1 | **CSV 导出格式** | ✅ 已对齐 | `Radar.Core/Postprocessing/ResultCsv.cs` + `Support/TestItemLabels.cs`（由 `export/gen_item_labels.py` 从上游源码 `ast` 提取，并与官方 CSV 表头 146 列逐列交叉校验）。BOM + CRLF + `{中文} ({英文})` + Python `repr(float)`；对官方文件**逐字节往返一致**（9993 = 9993 字节）。API：`/csv?format=official`；CLI：`radar infer --output` |
| 2 | **分割掩码可视化** | ✅ 已实现 | `Radar.Core/Visualization/{PngEncoder,MaskOverlayPng}.cs`：零依赖 PNG 编码器（真彩色 RGB8 + ZLibStream + CRC32）+ 三正交 MIP 透叠（36 器官 HSV 色环）。数据源为引擎暴露的 `Result.StitchedMask`，与黄金 `stitched_mask.npy` **逐体素全等**（0/12,496,896）。API：`/mask.png`；CLI：`--save-mask` |
| 3 | **目录级批量推理** | ✅ 已实现 | `tools/Radar.Cli` `infer` 子命令：递归扫描 `.nii/.nii.gz/.dcm/.zip`，按输入排序确定性并行（`--workers` × 设备分片），结果按序聚合后一次写出宽表；Web 端 `/api/jobs/batch` + 样例多选 |
| 4 | **独立 CLI 推理入口** | ✅ 已实现 | `tools/Radar.Cli`（程序集名 `radar`）：`infer` / `eval` / `help`，`--key value` 风格参数，自动上溯仓库根定位 `models/` |
| 5 | **超大样本跳过保护** | ✅ 已实现 | `RadarInferenceEngine.ExceedsSpatialLimit` + `CaseSkippedException`：任一空间维 > 1000 在**进入滑窗之前**抛出（实测 3 ms 返回）。批处理据此区分「跳过」与「失败」，Web 端新增 `Skipped` 状态 |
| 6 | **结果 JSON 文件导出** | ✅ 已实现 | `GET /api/jobs/{id}/json`，`UnsafeRelaxedJsonEscaping` 保持中文可读，带 `Content-Disposition` 下载 |
| 7 | **AUC 评估脚本** | ✅ 已实现 | `Radar.Core/Support/RocAuc.cs`（秩形式 Mann-Whitney U，平局取平均秩，单类返回 NaN）+ `Support/MerlinDiseases.cs`（`map_radar_merlin` 21 项）+ CLI `eval` 子命令：逐项 AUC + `AvgAUC`，`label == -1` 过滤、缺失置 0、胆囊缺失项按体素计数（< 1000）口径。算子与朴素两两配对交叉验证一致 |
| 8 | **多卡 `all_gather`** | ➖ 明确不做 | 上游多 GPU 结果聚合为训练/大集评估场景。本项目面向单机单卡/CPU，改以 CLI `--workers` 做**多进程级**并行（同一台机器上按设备分片），不引入 NCCL 依赖 |

> PLAN §1.2 明确**不做**的：训练链路、BERT 文本编码器（推理时文本特征为离线预计算，117.9M 参数是死重）。
>
> **已知口径差异**：`MerlinDiseases.Map` 中若干键（`大肠_粘膜下水肿`、`心脏_主动脉瓣钙化`、`肝_肝大`、`肾_肾积水`、`胆囊_术后胆囊缺失`）**不在** RAD-CT 的 146 项里 —— 上游 MERLIN 评估用的是另一套 finding 名表（对应 MERLIN 微调权重）。`eval` 对缺失项**显式报告**而非静默跳过。

### 8.3 已评估后否决的路径

- **fp16 ONNX**：精度不达标（`res_x` 最大偏差 2.4e-3，超门禁 240 倍；`pred_mask` 翻转 4267 像素、`flags` 掩码翻转）。结论：**保持 fp32**。详见 PLAN M8-2。
- **8bit 有损 JPEG DICOM**：HU 误差均值 13.3（q90），端到端概率偏差达 0.166，超门禁两个量级。结论：**诊断链路不接受**，仅可用于预览/存档。详见 PLAN M8-1。

---

## 9. 上游与许可

- 上游：`alibaba-damo-academy/damo-radar`，代码 **Apache-2.0**；模型权重 **CC BY-NC-SA 4.0**（**仅限非商业用途**）。
- 本仓库为**推理重构**，不含上游源码副本；网络结构、预处理语义、后端依赖（ONNX Runtime）与上游保持一致。
- 引用上游论文请见 [damo-radar README](https://github.com/alibaba-damo-academy/damo-radar)。
