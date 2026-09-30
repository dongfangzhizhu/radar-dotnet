using System.IO.Compression;
using FellowOakDicom;

namespace Radar.Core.Nifti;

/// <summary>
/// DICOM CT 读取器：支持三种输入形态，输出与 NiftiImage 同构（Data 为 (x,y,z) 轴序
/// fortran 序，index = x + y·DimX + z·DimX·DimY，HU 值），可直接进 Preprocessor。
///   ① 单文件 .dcm（多帧 Enhanced CT）
///   ② 目录（单序列多文件，按 ImagePositionPatient 沿切片法向排序）
///   ③ .zip（单序列打包，解压到临时目录后走目录逻辑）
/// 压缩传输语法（JPEG/JPEG-LS/JPEG2000/RLE 等）自动转码为 Explicit VR LE 后解码。
/// 注意：切片排序按几何位置（法向投影），与 InstanceNumber 无关。
/// </summary>
public static class DicomReader
{
    /// <summary>
    /// 注册 fo-dicom.Codecs 的原生转码器（JPEG Baseline/Extended/Lossless、JPEG-LS、
    /// JPEG2000、RLE）。fo-dicom 5 默认只用托管解码器，缺失时对压缩语法报
    /// DicomCodecException：「Decoding dataset with transfer syntax ... is not supported」。
    /// 静态构造在首次使用 DicomReader 时执行一次。
    /// </summary>
    static DicomReader()
    {
        new DicomSetupBuilder()
            .RegisterServices(s => s.AddFellowOakDicom()
                .AddTranscoderManager<FellowOakDicom.Imaging.NativeCodec.NativeTranscoderManager>())
            .SkipValidation()
            .Build();
    }

    /// <summary>读取 zip 包内的 DICOM 序列。</summary>
    public static NiftiImage ReadZip(string zipPath)
    {
        string tmp = Path.Combine(Path.GetTempPath(), "radar_dicom_" + Guid.NewGuid().ToString("N"));
        try
        {
            ZipFile.ExtractToDirectory(zipPath, tmp, overwriteFiles: true);
            return ReadDirectory(tmp);
        }
        finally
        {
            try { Directory.Delete(tmp, recursive: true); } catch { /* 清理失败不影响结果 */ }
        }
    }

    /// <summary>读取目录下的 DICOM 序列（多序列时取文件数最多的一组）。</summary>
    public static NiftiImage ReadDirectory(string dir)
    {
        // 大小写不敏感单次枚举（Windows 上 *.dcm/*.DCM 二次枚举会把同批文件算双份）
        var opts = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MatchCasing = MatchCasing.CaseInsensitive,
        };
        var files = Directory.EnumerateFiles(dir, "*.dcm", opts).ToList();
        if (files.Count == 0)
            throw new InvalidDataException($"目录中未找到 .dcm 文件: {dir}");
        return ReadFiles(files);
    }

    /// <summary>读取单文件（多帧 Enhanced CT）；单帧文件直接按 1 帧序列处理。</summary>
    public static NiftiImage ReadFile(string path) => ReadFiles([path]);

    public static NiftiImage Read(string path)
    {
        if (Directory.Exists(path)) return ReadDirectory(path);
        if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase)) return ReadZip(path);
        return ReadFile(path);
    }

    internal static NiftiImage ReadFiles(IReadOnlyList<string> paths)
    {
        // ---- 1) 读头、按 SeriesInstanceUID 分组 ----
        var groups = new Dictionary<string, List<DicomDataset>>();
        foreach (var p in paths)
        {
            DicomFile df;
            try { df = DicomFile.Open(p); }
            catch (Exception ex) { throw new InvalidDataException($"DICOM 文件打开失败: {p} —— {ex.Message}"); }

            var ds = df.Dataset;
            var syntax = ds.InternalTransferSyntax;
            if (syntax != null && syntax.IsEncapsulated)
            {
                // 压缩传输语法（JPEG/JPEG-LS/JPEG2000/RLE 等）→ 转码为未压缩 Explicit VR LE
                ds = new FellowOakDicom.Imaging.Codec.DicomTranscoder(
                    syntax, DicomTransferSyntax.ExplicitVRLittleEndian)
                    .Transcode(ds);
            }
            if (!ds.Contains(DicomTag.ImagePositionPatient)) continue; // 非图像切片（定位片除外延）
            string series = ds.GetSingleValue<string>(DicomTag.SeriesInstanceUID);
            if (!groups.TryGetValue(series, out var list)) groups[series] = list = [];
            list.Add(ds);
        }
        if (groups.Count == 0)
            throw new InvalidDataException("未找到包含 ImagePositionPatient 的 DICOM 切片");
        var sliceFiles = groups.Values.MaxBy(g => g.Count)!;

        // ---- 2) 几何排序：法向 = row方向 × col方向，按 IPP·n 升序 ----
        // 切片条目 = (数据集, 帧号, 法向投影)；单帧文件帧号恒 0，
        // 多帧（Enhanced CT）按帧序展开（假定帧沿法向均匀排布，间距取 SliceThickness）
        var first = sliceFiles[0];
        double[] iop = first.GetValues<double>(DicomTag.ImageOrientationPatient);
        var rowDir = (iop[0], iop[1], iop[2]);   // 行方向余弦（沿列索引增大方向）
        var colDir = (iop[3], iop[4], iop[5]);   // 列方向余弦（沿行索引增大方向）
        var normal = Cross(rowDir, colDir);

        var entries = new List<(DicomDataset ds, int frame, double proj)>();
        foreach (var ds in sliceFiles)
        {
            int frames = ds.Contains(DicomTag.NumberOfFrames)
                ? ds.GetSingleValue<int>(DicomTag.NumberOfFrames) : 1;
            if (frames == 1)
            {
                entries.Add((ds, 0, Dot(GetIPP(ds), normal)));
            }
            else
            {
                var baseIpp = GetIPP(ds);
                float thick = ds.Contains(DicomTag.SliceThickness)
                    ? ds.GetSingleValue<float>(DicomTag.SliceThickness) : 1f;
                for (int f = 0; f < frames; f++)
                    entries.Add((ds, f, Dot(baseIpp, normal) + f * thick));
            }
        }

        var ordered = entries.OrderBy(t => t.proj).ToList();

        int rows = first.GetSingleValue<int>(DicomTag.Rows);
        int cols = first.GetSingleValue<int>(DicomTag.Columns);
        foreach (var (ds, _, _) in ordered)
        {
            if (ds.GetSingleValue<int>(DicomTag.Rows) != rows || ds.GetSingleValue<int>(DicomTag.Columns) != cols)
                throw new InvalidDataException("序列内切片尺寸不一致，暂不支持");
        }
        int x = cols, y = rows, z = ordered.Count;

        // ---- 3) spacing：PixelSpacing=[行间距(y), 列间距(x)]；z = 相邻法向投影间距 ----
        double[] ps = first.GetValues<double>(DicomTag.PixelSpacing);
        double sx = ps[1], sy = ps[0];
        double sz = z > 1
            ? ordered[1].proj - ordered[0].proj
            : first.GetSingleValue<float>(DicomTag.SliceThickness);
        if (sz <= 0) sz = first.GetSingleValue<float>(DicomTag.SliceThickness);

        // ---- 4) 逐切片解码像素 → HU，写入 (x,y,z) fortran 布局 ----
        var data = new float[(long)x * y * z];
        for (int k = 0; k < z; k++)
        {
            var (ds, frame, _) = ordered[k];
            float slope = 1f, inter = 0f;
            if (ds.Contains(DicomTag.RescaleSlope)) slope = ds.GetSingleValue<float>(DicomTag.RescaleSlope);
            if (ds.Contains(DicomTag.RescaleIntercept)) inter = ds.GetSingleValue<float>(DicomTag.RescaleIntercept);
            short[] hu = DecodeFrameHu(ds, frame, x * y);
            long baseIdx = (long)k * x * y;
            for (int r = 0; r < y; r++)          // 行 = y
            {
                int rowOff = r * x;
                for (int c = 0; c < x; c++)      // 列 = x
                    data[baseIdx + c + (long)r * x] = hu[rowOff + c] * slope + inter;
            }
        }

        var affine = new double[3, 4];
        affine[0, 0] = sx; affine[1, 1] = sy; affine[2, 2] = sz;
        return new NiftiImage { Data = data, DimX = x, DimY = y, DimZ = z, Affine = affine };
    }

    /// <summary>解码第 frameIndex 帧为 int16 HU（Rescale 前的存储值）。</summary>
    private static short[] DecodeFrameHu(DicomDataset ds, int frameIndex, int pixelCount)
    {
        var pd = FellowOakDicom.Imaging.DicomPixelData.Create(ds);
        int bits = pd.BitsAllocated;
        bool signed = pd.PixelRepresentation == FellowOakDicom.Imaging.PixelRepresentation.Signed;
        if (pd.SamplesPerPixel != 1)
            throw new NotSupportedException($"仅支持灰度 DICOM（SamplesPerPixel=1），实际 {pd.SamplesPerPixel}");
        string photo = ds.GetSingleValue<string>(DicomTag.PhotometricInterpretation);
        if (photo == "MONOCHROME1")
            throw new NotSupportedException("MONOCHROME1（反白）暂不支持，请先转换为 MONOCHROME2");

        byte[] raw = pd.GetFrame(frameIndex).Data; // IByteBuffer -> byte[]
        if (bits == 16 && raw.Length != pixelCount * 2 || bits == 8 && raw.Length != pixelCount)
            throw new InvalidDataException($"像素数据长度不符: {raw.Length} 字节（期望 {pixelCount} 像素 × {bits / 8} 字节）");

        var result = new short[pixelCount];
        if (bits == 16 && signed)
            Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        else if (bits == 16 && !signed)
        {
            for (int i = 0; i < pixelCount; i++)
                result[i] = unchecked((short)BitConverter.ToUInt16(raw, i * 2));
        }
        else if (bits == 8 && signed)
        {
            for (int i = 0; i < pixelCount; i++)
                result[i] = unchecked((sbyte)raw[i]);
        }
        else if (bits == 8)
        {
            // 8bit 无符号（JPEG Baseline 解码后的常规形态）：0..255 原值
            for (int i = 0; i < pixelCount; i++)
                result[i] = raw[i];
        }
        else
            throw new NotSupportedException($"仅支持 BitsAllocated 8/16，实际 {bits}");
        return result;
    }

    private static (double, double, double) GetIPP(DicomDataset ds)
    {
        double[] v = ds.GetValues<double>(DicomTag.ImagePositionPatient);
        return (v[0], v[1], v[2]);
    }

    private static (double, double, double) Cross((double X, double Y, double Z) a, (double X, double Y, double Z) b) =>
        (a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

    private static double Dot((double X, double Y, double Z) a, (double X, double Y, double Z) b) =>
        a.X * b.X + a.Y * b.Y + a.Z * b.Z;
}
