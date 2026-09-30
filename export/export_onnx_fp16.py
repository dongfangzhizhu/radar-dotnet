# -*- coding: utf-8 -*-
"""RADAR 视觉主干 fp16 变体导出与精度评估（PLAN.md M5/M7 遗留待办）。

流程：
  1) radar_vision.onnx → fp16（keep_io_types=True：输入输出仍 fp32，内部权重/计算 fp16）
  2) 对拍：fp32 模型 vs fp16 模型在同一 (96,256,384) 输入上的 8 个输出
  3) 分别用 CPU EP / CUDA EP 跑，记录各 EP 是否支持 fp16 算子

产物：
  models/radar_vision_fp16.onnx（gitignore 不入库）
  export/fp16_eval.log（结论数据）

运行：C:/Users/Administrator/miniconda3/python.exe export/export_onnx_fp16.py
依赖：onnxconverter-common（隔离安装在 export/.deps，--no-deps）
"""
import sys
from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR / ".deps"))   # 隔离依赖（仅 onnxconverter-common）

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

from common import MODELS_DIR, TESTDATA_DIR

FP32_PATH = MODELS_DIR / "radar_vision.onnx"
FP16_PATH = MODELS_DIR / "radar_vision_fp16.onnx"
INPUT_NPY = TESTDATA_DIR / "golden" / "plus_preproc_merlin_ft.npy"  # (96,256,384) fp32
OUT_NAMES = ["seg_probs", "pred_mask", "res_x1", "res_x2", "res_x3",
             "flags1", "flags2", "flags3"]


def convert():
    print(f"加载 {FP32_PATH.name}（{FP32_PATH.stat().st_size / 1e6:.1f} MB）")
    m = onnx.load(str(FP32_PATH))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
    fixed = fix_cast_types(m16)
    onnx.save(m16, str(FP16_PATH))
    print(f"保存 {FP16_PATH.name}（{FP16_PATH.stat().st_size / 1e6:.1f} MB，"
          f"keep_io_types=True，修正 Cast 类型 {fixed} 处）")
    onnx.checker.check_model(str(FP16_PATH))
    print("onnx.checker 通过")


def fix_cast_types(model):
    """修 onnxconverter-common 的已知不一致：原图自带 Cast 节点（如 bool→float，
    Equal→MaxPool 掩码分支）的 value_info 已被改成 fp16，但 `to` 属性仍是 FLOAT，
    ORT 载入时报 "Type (tensor(float16)) of output arg (...) does not match expected
    type (tensor(float))"。按 value_info 声明回写 `to` 属性。"""
    types = {}
    for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        out = node.output[0]
        want = types.get(out)
        if want is None:
            continue
        attr = next((a for a in node.attribute if a.name == "to"), None)
        if attr is not None and attr.i != want:
            print(f"  修正 {node.name}: to {attr.i} -> {want}")
            attr.i = want
            fixed += 1
    return fixed


def run(providers, tag):
    x = np.load(str(INPUT_NPY)).astype(np.float32)[None, None]  # (1,1,96,256,384)
    out = {}
    for name, model in (("fp32", FP32_PATH), ("fp16", FP16_PATH)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        try:
            sess = ort.InferenceSession(str(model), so, providers=providers)
        except Exception as e:
            print(f"[{tag}] {name} 会话创建失败: {type(e).__name__}: {str(e)[:200]}")
            return
        used = sess.get_providers()
        names = [o.name for o in sess.get_outputs()]
        in_name = sess.get_inputs()[0].name
        print(f"[{tag}] {name} 会话 OK，providers={used}，输入 {sess.get_inputs()[0].type}")
        try:
            out[name] = sess.run(OUT_NAMES if all(n in names for n in OUT_NAMES) else names,
                                {in_name: x})
        except Exception as e:
            print(f"[{tag}] {name} 推理失败: {type(e).__name__}: {str(e)[:300]}")
            return

    names = OUT_NAMES
    print(f"[{tag}] === 输出对拍（fp16 vs fp32）===")
    for i, n in enumerate(names):
        a, b = out["fp32"][i], out["fp16"][i]
        if a.dtype == np.bool_ or b.dtype == np.bool_:
            mism = int((a.astype(np.int8) != b.astype(np.int8)).sum())
            print(f"  {n:10s} 布尔不匹配 {mism}/{a.size}")
        elif a.dtype == np.uint8:
            mism = int((a != b).sum())
            print(f"  {n:10s} uint8 不匹配 {mism}/{a.size}")
        else:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"  {n:10s} max={d.max():.3e} mean={d.mean():.3e} "
                  f"p99.9={np.percentile(d, 99.9):.3e}")


def main():
    convert()
    run(["CPUExecutionProvider"], "CPU")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        run(["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA")
    else:
        print("[CUDA] 不可用，跳过")


if __name__ == "__main__":
    main()
