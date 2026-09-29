# -*- coding: utf-8 -*-
"""导出 radar_resize.onnx：三线性重采样子图（opset 17，手写 onnx 图）。

torch F.interpolate(mode="trilinear", align_corners=False) 等价于 ONNX Resize:
  mode="linear", coordinate_transformation_mode="half_pixel", antialias=0
sizes 作为动态输入（int64 (3,)），避免 torch tracer 的 size 限制。

输入: x     (1, 1, D, H, W) fp32 动态 shape
      sizes (5,)            int64 目标 (1, 1, D', H', W')，N/C 维保持 1
输出: y     (1, 1, D', H', W') fp32

验证：F.interpolate vs ORT 回读，多组真实目标尺寸对拍 < 1e-6。
"""
import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from common import MODELS_DIR

OUT_PATH = MODELS_DIR / "radar_resize.onnx"


def build_model():
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", "C", "D", "H", "W"])
    SIZES = helper.make_tensor_value_info("sizes", TensorProto.INT64, [5])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", "C", "d", "h", "w"])

    roi_empty = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
    scales_empty = numpy_helper.from_array(np.array([], dtype=np.float32), "scales")

    node = helper.make_node(
        "Resize",
        inputs=["x", "roi", "scales", "sizes"],
        outputs=["y"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "radar_resize", [X, SIZES], [Y], [roi_empty, scales_empty])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), str(OUT_PATH))
    print(f"导出 {OUT_PATH} ({OUT_PATH.stat().st_size} 字节)，checker 通过")

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    cases = [
        ((1, 1, 100, 512, 512), (48, 100, 100)),   # 典型 CT (spacing 1,1,5)
        ((1, 1, 64, 256, 256), (24, 64, 64)),
        ((1, 1, 32, 64, 64), (96, 128, 160)),
        ((1, 1, 7, 33, 45), (10, 41, 57)),          # 非整除边界
    ]
    max_d = 0.0
    for shape, sz in cases:
        x = torch.randn(*shape)
        ref = F.interpolate(x, size=sz, mode="trilinear", align_corners=False).numpy()
        out = sess.run(["y"], {"x": x.numpy(), "sizes": np.array([1, 1, *sz], dtype=np.int64)})[0]
        d = float(np.max(np.abs(ref - out)))
        max_d = max(max_d, d)
        print(f"  {shape}->{sz}: max_diff={d:.3e} out={out.shape}")
    assert max_d < 1e-6, f"对拍超差: {max_d}"
    print(f"PASS — F.interpolate 与 ORT 最大差 {max_d:.3e} < 1e-6")


if __name__ == "__main__":
    main()
