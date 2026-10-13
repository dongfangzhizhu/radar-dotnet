# radar-dotnet

[![.NET 10](https://img.shields.io/badge/.NET-10.0-512BD4?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20%2F%20DirectML%20%2F%20CUDA-005CED)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-7C3F58?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

将 [RADAR](https://www.science.org/doi/10.1126/science.aec6129)（*Science* 2026，阿里达摩院 [damo-radar](https://github.com/alibaba-damo-academy/damo-radar)）——全腹增强 CT 的多器官分割与器官级病灶概率诊断模型——的**推理链路完整迁移到 .NET 10** 的实现。

纯 C# 推理，**无 Python 运行时依赖**；支持 CPU / DirectML / CUDA 三种执行后端；提供开箱即用的 Web 服务、单页测试端和命令行批处理工具。数值上与官方 Python 实现逐值对齐：端到端 146 项概率误差实测 **5.96e-7**（门禁 < 1e-3），分割掩码 **12,496,896 个体素逐体素全等**。

## 1. 项目背景

RADAR 是基于 40 万例全腹增强 CT、1500 万解剖感知图文对训练的通用视觉-语言模型，可对 36 个腹部器官给出 146 项「器官_病灶」概率，并能以 9 种疾病粒度做全图诊断（RADAR+）。上游官方实现为 Python + PyTorch + MONAI 技术栈，推理依赖 conda 环境、GPU 版 torch 及一系列科学计算库，在 Windows 临床环境、.NET 企业系统（HIS / PACS / 影像后处理工作站）中集成成本高。

本项目的目标只有一个：**让 .NET 技术栈的团队无需安装 Python，即可在自己的系统里直接跑 RADAR 推理。**

实现方式为推理重构而非源码移植：将视觉主干导出为 ONNX，器官打分头、RADAR+ whole head、预处理与滑窗调度全部用 C# 手写重实现，逐级与 Python 黄金基线对拍验收。上游 Python 项目与权重只读参照，不在本仓库内。

## 2. 功能特性

### 2.1 输入格式

| 格式 | 说明 |
|---|---|
| `.nii` / `.nii.gz` | NIfTI，含 gzip 流、sform/qform affine、8 种像素类型 |
| `.dcm` | 单文件 DICOM（含多帧 Enhanced CT） |
| DICOM 目录序列 | 按 `ImagePositionPatient` 沿法向投影自动排序 |
| `.zip` | DICOM 序列打包 |

DICOM 侧自动处理压缩传输语法（RLE / JPEG Baseline / JPEG Extended / JPEG Lossless / JPEG-LS / JPEG2000 / HT-JPEG2000），非压缩路径经 `RescaleSlope/Intercept` 还原 HU。**DICOM 支持为本项目新增能力**，上游仅有 NIfTI。

### 2.2 两条推理链路

| 链路 | 模型 | 打分粒度 | 输出 |
|---|---|---|---|
| `radar` | `checkpoint_radar_pretrain.pth` | 器官级：36 器官 → **146 项**「器官_病灶」概率 | 滑窗推理 + 分割掩码 |
| `radar_plus` | `checkpoint_radar_plus{,_finetuned_on_merlin}.pth` | 全图级：**9 种疾病**概率（多 prompt 正负分组） | 单窗整图 |

两条链路共用同一视觉主干（ONNX），差异在预处理与打分头。自动处理边界器官排除与缺失器官的 center_crop 二次补推。

### 2.3 交付形态

- **Web 服务**（ASP.NET Core Minimal API）：上传或选择样例建推理任务，异步队列执行，结果可下载为官方格式 CSV / 长表 CSV / JSON / 分割掩码透叠 PNG。附带原生 HTML/JS 单页测试端，无需前端构建。
- **CLI 工具**（`radar`）：`infer` 子命令整目录批量推理（多 worker × 多设备分片，结果按输入顺序聚合），`eval` 子命令按 MERLIN 21 项 finding 计算 AUC / AvgAUC。
- **推理库**（`Radar.Core`）：纯 C# 类库，无 Web 依赖，可直接嵌入任意 .NET 应用。

### 2.4 工程细节

- **预处理**：各向异性重采样至 `(1,1,5) mm`（C# 手写 fp32 三线性插值）→ HU clip → 归一化 → 非零域裁剪 → 填充至 `(96,256,384)`
- **滑窗推理**：`roi=(96,256,384)`、`overlap=0.25`，概率累加 + count map → argmax 拼接
- **输出兼容**：官方格式宽表 CSV（BOM + CRLF + `{中文} ({英文})` 列名 + Python `repr(float)` 数值格式），与官方文件**逐字节往返一致**
- **掩码可视化**：分割标签图 → 三正交 MIP 透叠 PNG（内置零依赖 PNG 编码器，不依赖任何图像库）
- **健壮性**：预处理后任一空间维 > 1000 的超大病例自动跳过（区分「跳过」与「失败」，批处理不中断）
- **GPU 复用**：CUDA 路径下推理会话与输出缓冲跨任务复用

## 3. 适用场景与用户

| 用户 | 用途 |
|---|---|
| **.NET 企业医疗系统开发者** | 将腹部 CT 病灶概率分析直接嵌入 HIS / PACS / 报告系统 / 影像后处理工作站，无需维护 Python 环境 |
| **医院信息科 / 内网部署** | Windows 服务器上一条 `dotnet` 命令部署；DirectML 变体免 CUDA 版本匹配，AMD / Intel / NVIDIA 显卡通用 |
| **医学影像 AI 研究者** | 用 CLI 批量推理 + 内置 AUC 评估复现论文指标，或在 RADAR 之上做二次实验 |
| **RADAR 上游关注者** | 需要 Web 服务化、DICOM 输入、批量处理等上游不具备的工程能力时，直接使用本项目 |

## 4. 快速开始

### 4.1 前置条件

- **.NET 10 SDK**
- **模型产物**：`models/` 目录体积大、不入库，首次使用必须生成或获取，见 [§6](#6-模型产物models不入库)
- *（仅 DICOM 路径需要）* Windows 上安装 **VC++ 2015-2022 可再发行组件**，否则原生编解码器加载失败（`Unable to load DLL 'Dicom.Native'`）

### 4.2 构建

```bash
# CUDA 变体（NVIDIA GPU）
dotnet build src/Radar.Backend/Radar.Backend.csproj -c Release -p:OnnxEp=Cuda

# DirectML 变体（AMD / Intel / NVIDIA 通用，免 CUDA 版本匹配）
dotnet build src/Radar.Backend/Radar.Backend.csproj -c Release -p:OnnxEp=Dml
```

> **同一进程只能加载一个 `onnxruntime.dll`**，DirectML 与 CUDA 原生包互斥，二选一构建。默认（不传 `OnnxEp`）为 CPU + DirectML。

### 4.3 运行 Web 服务

```bash
dotnet run --project src/Radar.Backend -c Release --no-build
```

打开 **http://localhost:5202**（端口由 `src/Radar.Backend/Properties/launchSettings.json` 决定，命令行 `--urls` 会被它覆盖）。

页面上选择【设备 / 样例（可多选 = 批量）/ 推理模型 / 权重变体】→「运行选中样例」→ 队列中查看状态与耗时 →「查看」展示概率明细；每行可下载官方格式 CSV / 长表 CSV / JSON / 掩码 PNG。

**HTTP 接口：**

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/devices` | 可用执行后端（真实 EP 校验） |
| `GET` | `/api/samples` | `testdata/` 下的可用样例 |
| `POST` | `/api/jobs/sample` | 用样例建任务：`{sample, device, model, variant}` |
| `POST` | `/api/jobs/batch` | 批量建任务：`{samples[], device, model, variant}` |
| `POST` | `/api/jobs/upload` | 上传建任务：`multipart/form-data`，query 带 `device/model/variant` |
| `GET` | `/api/jobs` | 任务列表（JSON，camelCase，含 `hasMask`） |
| `GET` | `/api/jobs/{id}` | 单任务详情 |
| `GET` | `/api/jobs/{id}/csv` | 结果 CSV（长表 `item,probability`） |
| `GET` | `/api/jobs/{id}/csv?format=official` | 官方格式宽表 CSV（1 行 × 147 列，BOM + CRLF） |
| `GET` | `/api/jobs/{id}/json` | 结果 JSON（中文不转义） |
| `GET` | `/api/jobs/{id}/mask.png` | 分割掩码透叠图（仅 `radar` 路径；`204/404` 表示本任务无掩码） |

> 任务状态含 `Skipped`：空间维超限的病例被跳过而非标记失败，不污染批处理失败率。

### 4.4 命令行

```bash
# 整目录批量推理，产出官方格式宽表 CSV + 分割掩码图
dotnet run --project tools/Radar.Cli -c Release -- infer \
    --img-dir testdata --save-dir results --workers 4 --device cpu --save-mask

# 多设备分片：cpu + cuda 各扛一半，结果按输入顺序聚合
dotnet run --project tools/Radar.Cli -c Release -- infer \
    --img-dir testdata --devices cpu,cuda --workers 4 --save-tag demo_cuda

# MERLIN 测试集 AUC 评估（21 项 finding + AvgAUC）
dotnet run --project tools/Radar.Cli -c Release -- eval \
    --results-csv results/RADAR_infer_results_demo.csv --labels merlin_labels.json

# 完整参数说明
dotnet run --project tools/Radar.Cli -c Release -- help
```

**`infer` 选项：**

| 参数 | 说明 |
|---|---|
| `--img-dir <目录>` | **必填**。待推理病例目录，递归识别 `.nii` / `.nii.gz` / `.dcm` / `.zip` |
| `--save-dir <目录>` | 结果目录，默认 `<仓库根>/results` |
| `--save-tag <标签>` | 输出文件名后缀，默认 `demo` → `<save-dir>/RADAR_infer_results_<标签>.csv` |
| `--device <设备>` | `cpu` / `directml` / `cuda`，默认 `cpu` |
| `--devices <列表>` | 多设备分片，逗号分隔（如 `cpu,cuda`），与 `--workers` 配合 |
| `--workers <N>` | 并行度，默认 1；每个 worker 独占一个推理会话 |
| `--model <模型>` | `radar`（默认）/ `radar_plus` |
| `--variant <变体>` | `radar_plus` 专用：`pretrain` / `merlin_ft`（默认） |
| `--save-mask` | 额外输出掩码透叠图 `<save-dir>/masks/<病例>.png`（仅 `radar` 路径产出） |
| `--limit <N>` | 只处理前 N 个病例（调试用） |

**`eval` 选项：**

| 参数 | 说明 |
|---|---|
| `--results-csv <路径>` | **必填**。`infer` 产出的宽表 CSV |
| `--labels <路径>` | **必填**。MERLIN 标签 JSON `{疾病键: {病例ID: 标签}}` |
| `--sag-counts <路径>` | 可选。胆囊分割体素数 JSON，复刻上游对「胆囊_术后胆囊缺失」的体素数判定口径 |

> - 空间任一维 > 1000 的病例按上游约定跳过（不计入结果表，也不视为失败）。
> - 多设备 / 多 worker 的结果按输入文件顺序聚合。

### 4.5 测试

```bash
dotnet test tests/Radar.Tests/Radar.Tests.csproj -c Release -p:OnnxEp=Cuda
```

64 个 xUnit 用例，覆盖：NIfTI / DICOM 读取、预处理、滑窗调度与拼接、器官头、端到端概率与掩码逐体素门禁、RADAR+ 双变体对拍、官方 CSV 往返、ROC AUC、PNG 编码器、超大病例守卫。

> 测试禁用了并行执行（多会话并行会触发显存 OOM）；依赖的 golden / 模型缺失时用例会**静默跳过**，「全绿」前请确认数据已就位。

### 4.6 设备探活与基准

```bash
dotnet run --project tools/Radar.Benchmark -c Release -- probe   # 秒级，只探活三后端
dotnet run --project tools/Radar.Benchmark -c Release            # 完整基准，覆写 benchmarks.md
```

## 5. 目录结构

```
radar-dotnet/
├── PLAN.md                  设计与实施记录（里程碑、验收数据、技术决策）
├── benchmarks.md            性能基准实测记录
├── Radar.slnx               .NET 10 解决方案
├── src/
│   ├── Radar.Core/          推理库（纯 C#，无 Web 依赖，二次开发的主要接入点）
│   │   ├── Preprocessing/   两条链路的预处理（Preprocessor / PlusPreprocessor / Resampler）
│   │   ├── Inference/       ORT 会话、视觉引擎、滑窗、器官头、RADAR+ whole head
│   │   ├── Postprocessing/  官方格式宽表 CSV（ResultCsv）
│   │   ├── Visualization/   零依赖 PNG 编码器 + 掩码透叠图
│   │   ├── Nifti/           NIfTI 与 DICOM 读取器
│   │   └── Support/         NPY/NPZ 解析、器官/标签/疾病映射表、ROC AUC
│   └── Radar.Backend/       Web 服务（ASP.NET Core Minimal API + 任务队列 + 单页测试端）
├── tests/Radar.Tests/       xUnit 对齐测试（64 个用例）
├── tools/
│   ├── Radar.Benchmark/     性能基准 + 设备探活
│   └── Radar.Cli/           独立 CLI（infer / eval / help）
├── export/                  Python 导出工具链（一次性，见 §6）
├── models/                  导出产物（gitignore）
└── testdata/                测试数据（gitignore）
```

**二次开发提示**：`Radar.Core` 是纯推理库，推荐以它为依赖嵌入自己的应用——构造 `RadarInferenceEngine`，传入影像路径或体数据，取回概率表与掩码。Web 层（`Radar.Backend`）可整体替换为自己的宿主。逐级对拍基线与验收数据见 `PLAN.md`。

## 6. 模型产物（`models/`，不入库）

`models/` 下的文件由 `export/` 工具链生成，必须自备，否则程序无法启动。逐文件说明、md5、依赖条件见 [`models/README.md`](models/README.md)。

**上游依赖（不在本仓库，缺失则无法重建）：**

- `../RADAR/checkpoint_radar_pretrain.pth`、`checkpoint_radar_plus.pth`、`checkpoint_radar_plus_finetuned_on_merlin.pth`
- `../RADAR/bert-base-uncased/`、`../RADAR/bert-base-chinese/`
- `../damo-radar/RADAR_inference/`（网络结构源码）、`../damo-radar/RADAR_train/`（RADAR+ 提示词清单）、`../damo-radar/ckpt/infer_text_embedding_radar.pt`

**重建顺序**（解释器：torch 2.9.1+cu128；在仓库根执行）：

```bash
python export/export_onnx.py         # → models/radar_vision.onnx        （视觉主干，98.9 MB）
python export/export_weights.py      # → models/organ_head.npz           （器官打分头权重）
python export/export_text_feats.py   # → models/text_feats.{bin,json}    （146 项文本特征）
python export/export_plus.py         # → models/plus_head{,_merlin_ft}.npz
                                     #   models/plus_text_feats{,_merlin_ft}.{bin,json}
```

各导出脚本自带对拍验收（向量化 vs 原版、ONNX vs torch、golden vs 官方 CSV），失败会显式报错。

> 说明：BERT 文本编码器不在运行时使用——推理所需的文本特征为离线预计算导出，避免 117.9M 参数的死重。

## 7. 精度与性能

### 7.1 数值对齐验收

| 层级 | 指标 | 阈值 | 实测 |
|---|---|---|---|
| **端到端概率（核心门禁）** | demo case 146 项 vs 官方实现 | 全部 < 1e-3 | max **5.96e-7** |
| **分割掩码（核心门禁）** | argmax 拼接标签图 vs 黄金基线 | 逐体素全等 | **0 / 12,496,896 不一致** |
| 预处理 | vs 黄金基线逐元素 | < 1e-5 | mismatch = 0 |
| 官方 CSV 格式 | 逐字节 vs 官方文件 | 全等 | 9993 = 9993 字节 |
| ROC AUC | vs 独立实现交叉验证 | < 1e-12 | 一致 |
| RADAR+ 两个权重变体 | 概率 | < 1e-3 | 1.3e-4 / 9.7e-7 |

浮点说明：ONNX Runtime 与 torch 的 Conv3d / interpolate 存在 1e-6 级固有差异，滑窗累加会放大到 1e-4 级，阈值已按此留量。掩码能取「逐体素全等」是因为概率差 5.96e-7 远小于任何 argmax 翻转所需的间隔。

### 7.2 性能（Release，单机实测）

| 场景 | 耗时 |
|---|---|
| CPU（demo case，3 窗，端到端） | **60.2 s** |
| CUDA（RTX 4090，首次含会话初始化） | 120.9 s |
| CUDA（稳态，会话 + 缓冲复用） | **24.7 s**（CPU 的 2.4 倍快） |

ORT 推理占端到端约 86%，手写打分头 < 1 s。GPU 收益主要来自跨任务会话复用 + 输出缓冲复用（单窗 `seg_probs` 达 349 MB，PCIe 拷贝是瓶颈）。明细见 [benchmarks.md](benchmarks.md)。

## 8. 范围与限制

- **仅推理**：不含训练链路。需要微调请使用上游 Python 实现。
- **多卡聚合不做**：上游多 GPU `all_gather` 结果聚合面向训练/大集评估场景；本项目以 CLI `--workers` 多进程并行（同机按设备分片）替代，不引入 NCCL 依赖。
- **保持 fp32**：fp16 ONNX 已实测评估并否决（`res_x` 最大偏差 2.4e-3，超门禁 240 倍；掩码翻转 4267 像素），精度不达标。
- **诊断链路不接受有损压缩 DICOM**：8bit 有损 JPEG 实测 HU 误差均值 13.3（q90），端到端概率偏差达 0.166，仅可用于预览/存档。
- **MERLIN 评估口径**：`radar eval` 所用 finding 名表（如 `大肠_粘膜下水肿`）部分不在 RAD-CT 的 146 项内——上游 MERLIN 评估使用另一套名表（对应 MERLIN 微调权重）；缺失项会显式报告而非静默跳过。

## 9. 许可与引用

- 上游：[alibaba-damo-academy/damo-radar](https://github.com/alibaba-damo-academy/damo-radar)，代码 **Apache-2.0**；模型权重 **CC BY-NC-SA 4.0**（**仅限非商业用途**）。
- 本仓库为推理重构，不含上游源码副本；网络结构、预处理语义、后端依赖（ONNX Runtime）与上游保持一致。模型权重的许可约束（非商业）同样适用于本项目的推理产物。
- 引用上游论文：

```bibtex
@article{damo-radar-2026,
    title   = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume  = {393},
    number  = {6817},
    pages   = {eaec6129},
    year    = {2026},
    doi     = {10.1126/science.aec6129}
}
```
