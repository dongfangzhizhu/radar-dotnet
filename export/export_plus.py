# -*- coding: utf-8 -*-
"""导出 RADAR+ whole-head 权重与 9 疾病文本特征。

来源: RADAR/checkpoint_radar_plus.pth 与 checkpoint_radar_plus_finetuned_on_merlin.pth
（两个 checkpoint 的 whole head / text 分支权重不同，temp 也不同：
 pretrain=0.053180, merlin_ft=0.041191 —— 各导一套，C# 侧按 variant 加载。）

产物（models/ 目录）:
  plus_head.npz                  pretrain checkpoint 的 whole head 权重
  plus_head_merlin_ft.npz        merlin 微调 checkpoint 的 whole head 权重
  plus_text_feats.bin/.json      pretrain 的 9 疾病文本特征
  plus_text_feats_merlin_ft.bin/.json  merlin 微调的 9 疾病文本特征

whole head npz 键（与 checkpoint 一致）:
  query_tokens_whole (1,256)  attention_whole.in_proj_weight (768,256)
  attention_whole.in_proj_bias (768,)  attention_whole.out_proj.weight (256,256)
  attention_whole.out_proj.bias (256,)  vision_projs_whole.weight (256,256)
  vision_projs_whole.bias (256,)  temp 标量
  in_proj 行序: Q=0:256, K=256:512, V=512:768（与器官头 MHA 相同布局）

文本特征二进制格式（C# 零依赖解析，与 text_feats.bin 同风格）:
  .bin : 所有 prompt 特征按 json items 顺序连续 float32 小端，每条 256 float
  .json: {"dim":256, "diseases":[{"name":"coronary_calcification",
          "num_positive":3, "start":0, "count":3}, ...], "count":总prompt数}
  打分: sim = feat·text[i] / temp; pos=前 num_positive 条均值, neg=其余均值,
        softmax([neg,pos]) 取 pos 即阳性概率。
  注意: 特征本身已 L2 归一（F.normalize(text_proj(CLS))），
        image 特征在 C# 侧归一后点积。

文本特征计算复刻 infer_merlin_whole.py L628-649:
  tokenizer = BertTokenizer(bert-base-uncased 本地目录)
  prompts = pos+neg 原文（仅做去反斜杠，不加句号不 refine）
  encoding: padding="max_length", truncation=True, max_length=100
  text_encoder(input_ids, attention_mask) → last_hidden_state[:,0,:]
  → text_proj → F.normalize(dim=-1)

运行解释器: C:/Users/Administrator/miniconda3/python.exe（torch 2.9.1 + transformers 5.4.0）
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer

RADAR_DOTNET_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = RADAR_DOTNET_ROOT.parent / "RADAR"
BERT_DIR = CKPT_DIR / "bert-base-uncased"
MODELS_DIR = RADAR_DOTNET_ROOT / "models"

VARIANTS = {
    "pretrain": {
        "ckpt": CKPT_DIR / "checkpoint_radar_plus.pth",
        "head_npz": "plus_head.npz",
        "text_bin": "plus_text_feats.bin",
        "text_json": "plus_text_feats.json",
    },
    "merlin_ft": {
        "ckpt": CKPT_DIR / "checkpoint_radar_plus_finetuned_on_merlin.pth",
        "head_npz": "plus_head_merlin_ft.npz",
        "text_bin": "plus_text_feats_merlin_ft.bin",
        "text_json": "plus_text_feats_merlin_ft.json",
    },
}

WHOLE_PREFIXES = ("query_tokens_whole", "attention_whole.", "vision_projs_whole.", "temp")

# 9 疾病 prompt（逐字复刻 infer_merlin_whole.py L38-252，顺序 = dict 插入序）
DISEASE_PROMPTS = {
    "coronary_calcification": (
        ["marked coronary calcification", "severe coronary calcifications", "coronary calcification"],
        ["heart are normal", "heart is normal", "heart appears normal"],
    ),
    "thrombosis": (
        ["there is thrombosis", "there has been interval thrombosis", "partial thrombosis",
         "portal vein thrombosis", "complete thrombosis", "stable thrombosis",
         "likely represents thrombosis", "in keeping with thrombosis", "possible thrombosis",
         "with thrombosis", "chronic thrombosis", "occlusive portal venous thrombus",
         "occlusion of the portal vein with thrombus", "nonocclusive thrombus",
         "there is a thrombus", "occlusive thrombus"],
        ["no evidence of deep vein thrombosis", "no portal venous thrombosis", "no thrombosis",
         "no definite evidence of thrombosis", "occlusive thrombosis",
         "no evidence of venous thrombosis", "no evidence of thrombosis", "no venous thrombosis",
         "without evidence of thrombosis", "no evidence of ivc or pelvic vein thrombosis",
         "no splenic , or portal vein thrombosis", "negative for portal vein thrombus",
         "no definite evidence for thrombus", "no evidence of thrombus",
         "no portal venous thrombus", "no evidence of thrombus", "no thrombus",
         "no occlusion or thrombus", "no venous thrombus"],
    ),
    "metastatic_disease": (
        ["consistent with metastatic disease", "concerning for metastatic disease",
         "may represent metastatic disease", "suggestive of pleural metastatic disease",
         "may reflect a metastatic lesion", "consistent with worsening metastatic disease",
         "concerning for nodal metastatic disease", "lung bases concerning for metastatic disease",
         "lesions consistent with metastatic disease", "multiple intrahepatic metastatic lesions",
         "compatible with metastatic pancreatic cancer", "reflecting metastatic disease",
         "concerning for worsening metastatic disease",
         "likely secondary to peritoneal metastatic disease",
         "compatible with metastatic rectal cancer"],
        ["likely benign", "no evidence of metastatic disease",
         "no definite evidence of metastatic disease"],
    ),
    "osteopenia": (
        ["diffuse osteopenia", "marked osteopenia", "mild osteopenia", "patchy osteopenia",
         "significant osteopenia", "severe osteopenia", "markedly osteopenic",
         "bones are osteopenic", "diffusely osteopenic", "osteoporosis", "osteoporotic"],
        ["musculoskeletal : normal "],
    ),
    "anasarca": (
        ["anasarca"],
        ["abdominal wall : normal ."],
    ),
    "lymphadenopathy": (
        ["lymphadenopathy is again seen", "mesenteric lymphadenopathy",
         "upper retroperitoneal lymphadenopathy", "mild retroperitoneal lymphadenopathy",
         "lower retroperitoneal lymphadenopathy", "stable gastrohepatic ligament lymphadenopathy",
         "stable gastrohepatic lymphadenopathy", "stable lymphadenopathy",
         "necrotic mesenteric lymphadenopathy", "bilateral inguinal lymphadenopathy",
         "right hilar lymphadenopathy", "lymph nodes : extensive lymphadenopathy",
         "aortic lymphadenopathy"],
        ["no lymphadenopathy"],
    ),
    "prostatomegaly": (
        ["mild prostatomegaly", "marked prostatomegaly", "moderate prostatomegaly",
         "prostate and seminal vesicles : prostatomegaly", "nonspecific prostatomegaly",
         "moderate non specific prostatomegaly", "there is prostatomegaly", "enlarged prostate ."],
        ["no prostatomegaly", "prostate and seminal vesicles : normal ."],
    ),
    "ascites": (
        ["medium volume ascites", "large volume ascites", "moderate ascites", "large ascites",
         "trace ascites", "small volume ascites", "amount of ascites", "volume of ascites",
         "volume ascites"],
        ["no ascites", "no evidence of ascites", "no amount of ascites",
         "no volume of ascites", "no volume ascites"],
    ),
    "free_air": (
        ["there is free air", "foci of free air", "small amount of free air",
         "small focus of free air", "free air is seen",
         "a few locules of intraperitoneal free air", "with surrounding free air",
         "with intraperitoneal free air", "amount of intraperitoneal free air",
         "volume of free air", "is extensive associated free air",
         "is evidence of small intraperitoneal free air", "free air is also present",
         "with increased adjacent intra - abdominal free air", "volume intraperitoneal free air"],
        ["no free air", "or free air", "no evidence of free air",
         "no intraperitoneal free air", "there is no intra - peritoneal free air",
         "no evidence of extraluminal free air", "no extraluminal free air",
         "no intra - abdominal free air"],
    ),
}


def export_ckpt(name: str, spec: dict) -> None:
    print(f"=== {name}: {spec['ckpt'].name} ===")
    ckpt = torch.load(str(spec["ckpt"]), map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    # ---- whole head 权重 ----
    out = {k: v.detach().cpu().float().numpy() for k, v in sd.items()
           if k.startswith(WHOLE_PREFIXES)}
    n_keys = len(out)
    assert n_keys == 8, f"whole head 键数异常: {n_keys}"
    head_path = MODELS_DIR / spec["head_npz"]
    np.savez(str(head_path), **out)
    n_params = sum(v.size for v in out.values())
    print(f"  保存 {head_path.name}  键数={n_keys}  参数量={n_params} ({n_params*4/1048576:.2f} MB)")
    # 形状契约
    assert out["attention_whole.in_proj_weight"].shape == (768, 256)
    assert out["query_tokens_whole"].shape == (1, 256)
    assert out["vision_projs_whole.weight"].shape == (256, 256)
    assert out["temp"].shape == ()
    temp_val = float(out["temp"])
    print(f"  temp = {temp_val:.6f}")

    # ---- 文本特征 ----
    tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR))
    # BLIP med.py 的 text_encoder = BertModel(config, add_pooling_layer=False)，
    # 且 BertEmbeddings 不含 token_type_embeddings（forward_text 不传 token_type_ids）。
    # HF 标准版前向恒加 token_type_ids=0 查表，故加载后将该表清零（等价于"不加"）。
    cfg = BertConfig.from_json_file(str(BERT_DIR / "config.json"))
    text_encoder = BertModel(cfg, add_pooling_layer=False)
    te_sd = {k[len("text_encoder."):]: v for k, v in sd.items() if k.startswith("text_encoder.")}
    assert len(te_sd) == 199, f"text_encoder 键数异常: {len(te_sd)}"
    report = text_encoder.load_state_dict(te_sd, strict=False)
    # 缺失键必须恰好是 [token_type_embeddings]（清零后不参与数学），
    # 多余键必须是 MLM 头 / position_ids 缓冲（forward 不使用）。
    assert set(report.missing_keys) == {"embeddings.token_type_embeddings.weight"}, \
        f"意外缺失键: {report.missing_keys}"
    assert all(k.startswith(("cls.", "embeddings.position_ids")) for k in report.unexpected_keys), \
        f"意外多余键: {report.unexpected_keys}"
    with torch.no_grad():
        text_encoder.embeddings.token_type_embeddings.weight.zero_()
    text_proj_w = sd["text_proj.weight"].float()
    text_proj_b = sd["text_proj.bias"].float()
    text_encoder.eval()

    items = []
    feats_list = []
    with torch.no_grad():
        for disease, (pos, neg) in DISEASE_PROMPTS.items():
            prompts = [p.replace("\\", "") for p in pos] + [p.replace("\\", "") for p in neg]
            enc = tokenizer(prompts, padding="max_length", truncation=True,
                            max_length=100, return_tensors="pt")
            te_out = text_encoder(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"])
            cls = te_out.last_hidden_state[:, 0, :]            # (P,768)
            feat = F.normalize(cls @ text_proj_w.t() + text_proj_b, dim=-1)  # (P,256)
            feats_list.append(feat.numpy().astype(np.float32))
            items.append({
                "name": disease,
                "num_positive": len(pos),
                "start": sum(len(f) for f in feats_list[:-1]),
                "count": len(prompts),
            })

    feats = np.concatenate(feats_list, axis=0)
    assert feats.dtype == np.float32 and feats.shape[1] == 256
    # L2 归一校验
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"L2 归一偏差: {norms.min()}~{norms.max()}"

    bin_path = MODELS_DIR / spec["text_bin"]
    json_path = MODELS_DIR / spec["text_json"]
    feats.tofile(str(bin_path))
    meta = {"dim": 256, "count": int(feats.shape[0]), "temp": temp_val, "diseases": items}
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  保存 {bin_path.name} ({bin_path.stat().st_size/1024:.1f} KB, {feats.shape[0]} prompts)")
    print(f"  保存 {json_path.name}")

    # 回读校验
    back = np.fromfile(str(bin_path), dtype=np.float32).reshape(feats.shape)
    assert np.array_equal(back, feats), "bin 回读不一致"
    print("  回读校验通过")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in VARIANTS.items():
        export_ckpt(name, spec)
    print("全部导出完成")


if __name__ == "__main__":
    main()
