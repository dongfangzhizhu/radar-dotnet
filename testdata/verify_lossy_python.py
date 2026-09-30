# -*- coding: utf-8 -*-
"""独立交叉验证：pydicom 解码有损 JPEG DICOM → HU，与 NIfTI 对比误差分布。

用来判定 C# 侧 DicomReader 报出的 max 误差（417 HU）是否来自 JPEG 编码本身
（振铃/块效应），而非解码链路 bug。
"""
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parent.parent
NII = ROOT / "testdata" / "AC423ccbe.nii.gz"


def check(q: int):
    zpath = ROOT / "testdata" / f"dicom_lossy_q{q}.zip"
    if not zpath.exists():
        print(f"q{q}: 样例缺失，跳过")
        return
    img = nib.load(str(NII))
    ref = np.asanyarray(img.dataobj).astype(np.float32)  # (X,Y,Z)

    with zipfile.ZipFile(zpath) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".dcm"))
        rec = np.zeros_like(ref)
        for k, name in enumerate(names):
            ds = pydicom.dcmread(zf.open(name))
            arr = ds.pixel_array.astype(np.float32)          # (Y,X) stored
            hu = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            rec[:, :, k] = hu.T                              # -> (X,Y) 与 NIfTI 同布局

    diff = np.abs(rec - ref)
    print(f"q{q}: 切片 {len(names)}  HU 误差 max={diff.max():.1f} mean={diff.mean():.2f} "
          f"p99={np.percentile(diff, 99):.1f} 中位={np.median(diff):.1f}")
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"     最大误差位置 {idx}  ref={ref[idx]:.0f} rec={rec[idx]:.0f}")
    c = (ref >= -300) & (ref <= 400)
    print(f"     clip 域内({c.mean()*100:.0f}%): max={diff[c].max():.1f} mean={diff[c].mean():.2f}")


for q in (90, 75):
    check(q)
