# -*- coding: utf-8 -*-
"""把 demo NIfTI 转成 JPEG 有损压缩的 DICOM 序列（两个质量档），供有损精度评估。

评估目的（PLAN.md M6 待办）：JPEG 有损族对 HU / 预处理 / 端到端概率的影响量级。

编码方案（最坏情况评估）：
- JPEG Baseline 8bit：HU 线性重标到 [0,255]（整卷统一 slope/intercept），
  HU = stored_value × slope + intercept，slope=(HUmax-HUmin)/255, intercept=HUmin。
  量化步长 ≈ 18.75 HU + JPEG 压缩噪声 —— 8bit 有损是误差上界最大的常见编码。
- 质量档：q90（临床存档常用档）、q75（对照下限）。

产物（gitignore 不入库）：
  testdata/_dicom_tmp_q90/ + testdata/dicom_lossy_q90.zip
  testdata/_dicom_tmp_q75/ + testdata/dicom_lossy_q75.zip

用法：python make_dicom_lossy.py
"""
import io
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGBaseline8Bit, generate_uid

ROOT = Path(__file__).resolve().parent.parent
NII = ROOT / "testdata" / "AC423ccbe.nii.gz"
QUALITIES = [90, 75]


def build_series(quality: int) -> Path:
    img = nib.load(str(NII))
    data = np.asanyarray(img.dataobj).astype(np.float32)  # (X, Y, Z) HU
    sx, sy, sz = [float(z) for z in img.header.get_zooms()[:3]]
    nx, ny, nz = data.shape

    hu_min = float(data.min())
    hu_max = float(data.max())
    slope = (hu_max - hu_min) / 255.0
    intercept = hu_min

    tmp = ROOT / "testdata" / f"_dicom_tmp_q{quality}"
    tmp.mkdir(exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()

    for k in range(nz):
        plane = np.transpose(data[:, :, k])  # (Y, X) HU
        stored = np.round((plane - hu_min) / slope).astype(np.uint8)  # 0..255
        buf = io.BytesIO()
        Image.fromarray(stored, mode="L").save(buf, format="JPEG", quality=quality)
        jpeg_bytes = buf.getvalue()

        ds = pydicom.Dataset()
        sop_class = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientName = f"JPEG^LOSSY_Q{quality}"
        ds.Rows, ds.Columns = ny, nx
        ds.BitsAllocated = ds.BitsStored = 8   # JPEG Baseline 只支持 8bit
        ds.HighBit = 7
        ds.PixelRepresentation = 0             # unsigned
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, k * sz]
        ds.PixelSpacing = [sy, sx]
        ds.SliceThickness = sz
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.InstanceNumber = k + 1
        fm = pydicom.dataset.FileMetaDataset()
        fm.MediaStorageSOPClassUID = sop_class
        fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        fm.TransferSyntaxUID = JPEGBaseline8Bit
        ds.file_meta = fm
        ds.PixelData = encapsulate([jpeg_bytes])
        ds["PixelData"].is_undefined_length = True  # 压缩像素必须封装（undefined length）
        ds.save_as(tmp / f"IM_{k + 1:04d}.dcm", enforce_file_format=True)

    zip_path = ROOT / "testdata" / f"dicom_lossy_q{quality}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(tmp.glob("*.dcm")):
            zf.write(f, f.name)
    print(f"q{quality}: {nz} 切片 -> {zip_path}（{zip_path.stat().st_size / 1e6:.1f} MB）"
          f"  HU 域 [{hu_min:.0f},{hu_max:.0f}] slope={slope:.4f}")
    return zip_path


def main():
    for q in QUALITIES:
        build_series(q)
    print("有损样例生成完成")


if __name__ == "__main__":
    main()
