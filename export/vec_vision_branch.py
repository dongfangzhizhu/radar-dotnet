# -*- coding: utf-8 -*-
"""向量化 VisionBranch：把原版 forward 中不可导出 ONNX 的动态控制流
（torch.unique / 布尔索引 / per-organ Python 循环）重写为纯张量运算。

语义等价性论证（对照原版 vision_branch.py forward 的 mask 处理段）：
- 原版：对 torch.unique(pred_mask) 中每个出现值 uv 建 eq 掩码，max_pool 后
       按 uv-1 行写入 flags；未出现的器官行保持全 False。
- 向量化：直接对全部 36 个器官 id（1..36）建立 one-hot 掩码后池化。
  未出现器官的掩码全 0，池化后仍全 0 → 与原版 False 等价；
  出现器官的行索引由「值」决定（uv-1），与 unique 的遍历顺序无关 → 等价。
- 原版「unique_values 为空则 continue」：向量化中表现为所有 flags 全 False，等价。

输出尺寸契约（输入 (1,1,96,256,384) 时）：
  seg_probs  (1,37,96,128,192)  fp32   最浅层 deep-supervision 分割概率
  pred_mask  (1,96,256,384)     uint8  interpolate x2 后 argmax 的器官标签图
  res_x1     (1,1152,256)       fp32   proj1(skips[-1]) 展平 token
  res_x2     (1,9216,256)       fp32   proj2(skips[-2])
  res_x3     (1,73728,256)      fp32   proj3(skips[-3])
  flags1     (1,36,1152)        bool   organ->token 池化 (8,32,32)
  flags2     (1,36,9216)        bool   (4,16,16)
  flags3     (1,36,73728)       bool   (2,8,8)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import N_ORGANS, NUM_CLASSES, UNET_KWARGS, build_unet, load_visual_state_dict


class VecVisionBranch(nn.Module):
    """原版 VisionBranch 的可导出等价实现（属性名与原版一致以复用 state_dict）。"""

    def __init__(self):
        super().__init__()
        self.UNet = self._build_network()
        self.proj1 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj2 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj3 = nn.Conv3d(256, 256, kernel_size=1)

    @staticmethod
    def _build_network() -> nn.Module:
        from dynamic_network_architectures.architectures.unet_lightdecoder import PlainConvUNetLightD
        return PlainConvUNetLightD(**UNET_KWARGS)

    def forward(self, x):
        skips, segs = self.UNet(x)

        scale1 = self.proj1(skips[-1])
        scale2 = self.proj2(skips[-2])
        scale3 = self.proj3(skips[-3])

        pred_logit = segs[0]
        seg_probs = torch.softmax(pred_logit, 1)
        target_size = [pred_logit.shape[-3], pred_logit.shape[-2] * 2, pred_logit.shape[-1] * 2]
        pred_logit_up = F.interpolate(pred_logit, size=target_size, mode="trilinear", align_corners=False)
        pred_mask = torch.softmax(pred_logit_up, 1).argmax(1)  # (B, D, H, W) int64

        # ---- 向量化 organ_token_flags（原版动态控制流的等价替换）----
        organ_ids = torch.arange(1, N_ORGANS + 1, device=x.device, dtype=pred_mask.dtype)
        masks = (pred_mask.unsqueeze(1) == organ_ids.view(1, -1, 1, 1, 1)).float()  # (B,36,D,H,W)

        flags1 = F.max_pool3d(masks, kernel_size=(8, 32, 32), stride=(8, 32, 32)).flatten(2) > 0
        flags2 = F.max_pool3d(masks, kernel_size=(4, 16, 16), stride=(4, 16, 16)).flatten(2) > 0
        flags3 = F.max_pool3d(masks, kernel_size=(2, 8, 8), stride=(2, 8, 8)).flatten(2) > 0

        res_x1 = scale1.flatten(2).transpose(1, 2)
        res_x2 = scale2.flatten(2).transpose(1, 2)
        res_x3 = scale3.flatten(2).transpose(1, 2)

        return (
            seg_probs,
            pred_mask.to(torch.uint8),
            res_x1, res_x2, res_x3,
            flags1, flags2, flags3,
        )


def build_vectorized(ckpt_path=None) -> VecVisionBranch:
    model = VecVisionBranch()
    load_visual_state_dict(model, ckpt_path) if ckpt_path else load_visual_state_dict(model)
    model.eval()
    return model


if __name__ == "__main__":
    # 自检：结构 + 权重加载 + 输出尺寸
    m = build_vectorized()
    x = torch.randn(1, 1, 96, 256, 384)
    with torch.no_grad():
        outs = m(x)
    names = ["seg_probs", "pred_mask", "res_x1", "res_x2", "res_x3", "flags1", "flags2", "flags3"]
    for n, t in zip(names, outs):
        print(f"{n}: {tuple(t.shape)} {t.dtype}")
