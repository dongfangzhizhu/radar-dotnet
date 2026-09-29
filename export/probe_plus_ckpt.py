# -*- coding: utf-8 -*-
"""探查 radar_plus 两个 checkpoint 的 whole-head / text 分支键结构。

输出: export/probe_plus_ckpt.log
"""
from pathlib import Path

import torch

CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "RADAR"
CKPTS = {
    "pretrain": CKPT_DIR / "checkpoint_radar_plus.pth",
    "merlin_ft": CKPT_DIR / "checkpoint_radar_plus_finetuned_on_merlin.pth",
}

WHOLE_KEYS = (
    "query_tokens_whole",
    "attention_whole.",
    "vision_projs_whole.",
    "temp",
)
TEXT_KEYS = ("text_proj.", "text_encoder.")


def brief(k: str, v: torch.Tensor) -> str:
    val = ""
    if k == "temp":
        val = f" value={v.item():.6f}"
    return f"  {k}  {tuple(v.shape)}{val}"


def main() -> None:
    lines = []
    for name, path in CKPTS.items():
        lines.append(f"=== {name}: {path.name} ({path.stat().st_size/1048576:.0f} MB) ===")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        lines.append(f"顶层键: {list(ckpt.keys())[:8]}  state_dict 键数: {len(sd)}")

        whole = {k: v for k, v in sd.items() if k.startswith(WHOLE_KEYS) and "momentum" not in k}
        lines.append(f"-- whole head 键 ({len(whole)}):")
        for k in sorted(whole):
            lines.append(brief(k, whole[k]))

        text_proj = {k: v for k, v in sd.items() if k.startswith("text_proj.")}
        lines.append("-- text_proj:")
        for k in sorted(text_proj):
            lines.append(brief(k, text_proj[k]))

        te = [k for k in sd if k.startswith("text_encoder.")]
        tem = [k for k in sd if k.startswith("text_encoder_m.")]
        lines.append(f"-- text_encoder 键数: {len(te)}  (momentum 版: {len(tem)})")
        sample = [k for k in te if "embeddings.word_embeddings.weight" in k
                  or "pooler" in k or "position_embeddings" in k]
        for k in sorted(sample)[:6]:
            lines.append(f"  {k}  {tuple(sd[k].shape)}")

        # 与主 checkpoint 视觉编码器键一致性粗查
        vis = [k for k in sd if k.startswith("visual_encoder.")]
        lines.append(f"-- visual_encoder 键数: {len(vis)}")

        lines.append("")

    out = Path(__file__).with_name("probe_plus_ckpt.log")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
