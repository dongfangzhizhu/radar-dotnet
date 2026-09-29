using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Tests.Support;

namespace Radar.Tests;

/// <summary>
/// DICOM 读取对拍：testdata/dicom_demo.zip（由 testdata/make_dicom.py 从
/// AC423ccbe.nii.gz 无损转换）读回结果必须与 NIfTI 直读逐位一致、
/// 预处理输出一致——证明 DICOM 链路与 NIfTI 链路数值等价。
/// </summary>
public class DicomTests
{
    private static string Root => RepoRoot.Find();

    [Fact]
    public void DicomZip_MatchesNiftiRawAndPreprocessed()
    {
        string zip = Path.Combine(Root, "testdata", "dicom_demo.zip");
        string nii = Path.Combine(Root, "testdata", "AC423ccbe.nii.gz");
        if (!File.Exists(zip) || !File.Exists(nii))
            return; // 测试数据未生成（testdata/make_dicom.py）

        var viaDicom = DicomReader.ReadZip(zip);
        var viaNifti = NiftiReader.Read(nii);

        // 尺寸与 spacing（DICOM: PixelSpacing + 相邻 IPP 投影间距）
        Assert.Equal(viaNifti.DimX, viaDicom.DimX);
        Assert.Equal(viaNifti.DimY, viaDicom.DimY);
        Assert.Equal(viaNifti.DimZ, viaDicom.DimZ);
        double sxN = viaNifti.Spacing.Sx, syN = viaNifti.Spacing.Sy, szN = viaNifti.Spacing.Sz;
        Assert.InRange(Math.Abs(viaDicom.Spacing.Sx - sxN), 0, 1e-6);
        Assert.InRange(Math.Abs(viaDicom.Spacing.Sy - syN), 0, 1e-6);
        Assert.InRange(Math.Abs(viaDicom.Spacing.Sz - szN), 0, 1e-6);

        // HU 逐位一致（无损转换：int16 直存 + slope=1/intercept=0）
        long vox = (long)viaNifti.DimX * viaNifti.DimY * viaNifti.DimZ;
        Assert.Equal(viaNifti.Data.Length, vox);
        double maxDiff = 0;
        for (long i = 0; i < vox; i++)
        {
            double d = Math.Abs(viaDicom.Data[i] - viaNifti.Data[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Assert.Equal(0.0, maxDiff);

        // 预处理输出一致（同一管线吃两种来源）
        var prepDicom = Preprocessor.Run(viaDicom, "dicom");
        var prepNifti = Preprocessor.Run(viaNifti, "nifti");
        Assert.Equal(prepNifti.D, prepDicom.D);
        Assert.Equal(prepNifti.H, prepDicom.H);
        Assert.Equal(prepNifti.W, prepDicom.W);
        double prepMax = 0;
        for (long i = 0; i < prepNifti.Data.Length; i++)
        {
            double d = Math.Abs(prepDicom.Data[i] - prepNifti.Data[i]);
            if (d > prepMax) prepMax = d;
        }
        Assert.True(prepMax < 1e-5, $"预处理输出 max_diff={prepMax}");
    }
}
