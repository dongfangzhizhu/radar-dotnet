# -*- coding: utf-8 -*-
"""公共模块：构建 RADAR 视觉主干网络并加载官方 checkpoint 权重。

仅供一次性导出脚本使用（export_*.py / verify_*.py）。
依赖原项目模块：../damo-radar/RADAR_inference/dynamic_network_architectures
运行解释器：C:/Users/Administrator/miniconda3/python.exe（torch 2.9.1）
"""
import sys
from pathlib import Path

import torch
from torch import nn

# 原项目根目录（仓库布局：radar-dotnet/export/ 本文件，上游在 ../damo-radar）
RADAR_DOTNET_ROOT = Path(__file__).resolve().parent.parent
DAMO_RADAR_INFERENCE = RADAR_DOTNET_ROOT.parent / "damo-radar" / "RADAR_inference"
OFFICIAL_CKPT = RADAR_DOTNET_ROOT.parent / "RADAR" / "checkpoint_radar_pretrain.pth"
TEXT_EMBEDDING_PT = RADAR_DOTNET_ROOT.parent / "damo-radar" / "ckpt" / "infer_text_embedding_radar.pt"
MODELS_DIR = RADAR_DOTNET_ROOT / "models"
TESTDATA_DIR = RADAR_DOTNET_ROOT / "testdata"

sys.path.insert(0, str(DAMO_RADAR_INFERENCE))

# 与原版 VisionBranch（damo-radar/RADAR_inference/dynamic_network_architectures/vision_branch.py）
# 完全一致的网络构造参数，逐字复刻，不做任何改动
UNET_KWARGS = dict(
    input_channels=1,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    conv_op=nn.Conv3d,
    kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
    conv_bias=True,
    norm_op=nn.BatchNorm3d,
    norm_op_kwargs={},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.ReLU,
    nonlin_kwargs={"inplace": True},
    num_classes=37,
    deep_supervision=True,
)

# 36 个器官（原版 organs 列表，索引 = organ_id，标签值 = organ_id + 1）
ORGANS = [
    "肾上腺", "主动脉", "竖脊肌", "脑", "锁骨", "大肠", "十二指肠", "食管", "面部", "股骨",
    "胆囊", "臀肌", "心脏", "髋关节", "肱骨", "髂动脉", "髂静脉", "髂腰肌", "下腔静脉", "肾",
    "肝", "肺", "胰腺", "门静脉", "肺动脉", "肋骨", "骶骨", "肩胛骨", "小肠", "脾",
    "胃", "气管", "膀胱", "颈椎", "腰椎", "胸椎",
]
N_ORGANS = len(ORGANS)  # 36
NUM_CLASSES = 37        # 0 = 背景, 1..36 = 器官

# 滑窗与预处理常量（原版 inference_demo.py）
ROI_SIZE = (96, 256, 384)
SLIDING_OVERLAP = 0.25
HU_CLIP = (-300.0, 400.0)
REF_SPACING = (1.0, 1.0, 5.0)  # (x, y, z) mm
PAD_EXTEND = (5, 20, 20)       # (D, H, W) 非零域外扩


def build_unet() -> nn.Module:
    """构建 PlainConvUNetLightD 视觉主干（结构 = 原版 VisionBranch.UNet）。"""
    from dynamic_network_architectures.architectures.unet_lightdecoder import PlainConvUNetLightD
    return PlainConvUNetLightD(**UNET_KWARGS)


def load_visual_state_dict(module: nn.Module, ckpt_path: Path = OFFICIAL_CKPT, prefix: str = "visual_encoder.") -> None:
    """从官方 checkpoint 提取 visual_encoder.* 键并加载（strict 校验）。"""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    vis_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not vis_sd:
        raise RuntimeError(f"checkpoint 中未找到前缀 {prefix} 的键: {ckpt_path}")
    module.load_state_dict(vis_sd, strict=True)


def max_abs_diff(a, b) -> float:
    """两个 torch 张量 / numpy 数组的最大绝对差。"""
    import numpy as np
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    import numpy as _np
    return float(_np.max(_np.abs(a.astype(_np.float64) - b.astype(_np.float64))))
