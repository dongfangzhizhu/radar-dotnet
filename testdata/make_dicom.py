# -*- coding: utf-8 -*-
"""把 demo NIfTI (AC423ccbe.nii.gz) 转成标准轴向 CT DICOM 序列并打 zip 包，
供 .NET DicomReader 对拍测试（testdata/dicom_demo.zip，gitignore 不入库）。

约定：
- 切片 = NIfTI z 层，IPP 沿 z 递增（与 NIfTI z 索引同向）
- 像素 int16 直存 HU（源数据全整数），RescaleSlope=1 / Intercept=0
- PixelSpacing = [row(y), col(x)]，IOP = 标准轴向 (1,0,0,0,1,0)

用法：python make_dicom.py [输出目录，默认 testdata]
"""
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parent.parent  # radar-dotnet/
NII = ROOT / "testdata" / "AC423ccbe.nii.gz"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "testdata"


def main():
    img = nib.load(str(NII))
    data = np.asanyarray(img.dataobj).astype(np.int16)  # (X, Y, Z)
    sx, sy, sz = [float(z) for z in img.header.get_zooms()[:3]]
    nx, ny, nz = data.shape

    tmp = OUT_DIR / "_dicom_tmp"
    tmp.mkdir(exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()

    for k in range(nz):
        # DICOM 切片像素 (row=r, col=c) = NIfTI (x=c, y=r, z=k)
        plane = np.transpose(data[:, :, k])  # (Y, X)
        ds = pydicom.Dataset()
        sop_class = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientName = "DICOM^ROUNDTRIP"
        ds.Rows, ds.Columns = ny, nx
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # int16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, k * sz]
        ds.PixelSpacing = [sy, sx]
        ds.SliceThickness = sz
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.InstanceNumber = k + 1
        fm = pydicom.dataset.FileMetaDataset()
        fm.MediaStorageSOPClassUID = sop_class
        fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = fm
        ds.PixelData = plane.tobytes()
        ds.save_as(tmp / f"IM_{k + 1:04d}.dcm", enforce_file_format=True)

    zip_path = OUT_DIR / "dicom_demo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(tmp.glob("*.dcm")):
            zf.write(f, f.name)
    print(f"完成: {nz} 切片 -> {zip_path}（{zip_path.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
