namespace Radar.Core.Nifti;

/// <summary>
/// NIfTI-1 / NIfTI-1.5 文件读取器（.nii 与 .nii.gz）。
/// 仅支持推理所需子集：常用标量数据类型 + sform/qform 仿射矩阵 + scl_slope/inter 缩放。
/// 输出统一转为 float32 体数据（x,y,z 轴序，与 MONAI LoadImage(ensure_channel_first=True)
/// 产生的 (1,x,y,z) 形状对齐）。
/// </summary>
public sealed class NiftiImage
{
    public required float[] Data { get; init; }     // 长度 = DimX*DimY*DimZ，轴序 x,y,z
    public required int DimX { get; init; }
    public required int DimY { get; init; }
    public required int DimZ { get; init; }
    /// <summary>仿射矩阵 3x4（sform 优先，qform 兜底），行主序 [3,4]。</summary>
    public required double[,] Affine { get; init; }

    /// <summary>各轴 spacing（mm），= |Affine[i,i]|，与原项目取法一致。</summary>
    public (double Sx, double Sy, double Sz) Spacing => (
        Math.Abs(Affine[0, 0]),
        Math.Abs(Affine[1, 1]),
        Math.Abs(Affine[2, 2]));
}

public static class NiftiReader
{
    private const int HeaderSize = 348;

    // NIfTI 数据类型码
    private const short DtUint8 = 2;
    private const short DtInt16 = 4;
    private const short DtInt32 = 8;
    private const short DtFloat32 = 16;
    private const short DtFloat64 = 64;
    private const short DtInt8 = 256;
    private const short DtUint16 = 512;
    private const short DtUint32 = 768;

    public static NiftiImage Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"NIfTI 文件不存在: {path}");

        byte[] raw;
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
        {
            using var fs = File.OpenRead(path);
            using var gz = new System.IO.Compression.GZipStream(fs, System.IO.Compression.CompressionMode.Decompress);
            using var ms = new MemoryStream();
            gz.CopyTo(ms);
            raw = ms.ToArray();
        }
        else
        {
            raw = File.ReadAllBytes(path);
        }

        return Parse(raw);
    }

    internal static NiftiImage Parse(byte[] raw)
    {
        if (raw.Length < HeaderSize)
            throw new InvalidDataException($"文件过小，不是有效 NIfTI: {raw.Length} 字节");

        bool littleEndian = true;
        int sizeofHdr = BitConverter.ToInt32(raw, 0);
        if (sizeofHdr == 348_000_000) // 0x014C0000 大端表示
        {
            littleEndian = false;
            sizeofHdr = 348;
        }
        if (sizeofHdr != 348)
            throw new InvalidDataException($"sizeof_hdr={sizeofHdr}，不是 NIfTI-1 文件");

        string magic = System.Text.Encoding.ASCII.GetString(raw, 344, 4).TrimEnd('\0');
        if (magic != "n+1" && magic != "ni1")
            throw new InvalidDataException($"magic={magic}，不是 NIfTI-1 文件");

        short[] dim = new short[8];
        for (int i = 0; i < 8; i++)
            dim[i] = GetInt16(raw, 40 + i * 2, littleEndian);

        int dimX = dim[1], dimY = dim[2], dimZ = dim[3];
        if (dimX <= 0 || dimY <= 0 || dimZ <= 0)
            throw new InvalidDataException($"非法维度: ({dimX},{dimY},{dimZ})");

        short datatype = GetInt16(raw, 70, littleEndian);
        float voxOffset = GetSingle(raw, 108, littleEndian);
        float sclSlope = GetSingle(raw, 112, littleEndian);
        float sclInter = GetSingle(raw, 116, littleEndian);
        short qformCode = GetInt16(raw, 252, littleEndian);
        short sformCode = GetInt16(raw, 254, littleEndian);

        long dataOffset = voxOffset <= 0 ? 352 : (long)voxOffset;
        int voxCount = dimX * dimY * dimZ;
        if (raw.Length < dataOffset + (long)voxCount * BytesPerVoxel(datatype))
            throw new InvalidDataException($"数据段不完整: 文件 {raw.Length} 字节，需要 {dataOffset + (long)voxCount * BytesPerVoxel(datatype)}");

        float[] data = ReadVoxels(raw, dataOffset, voxCount, datatype, littleEndian);

        // scl_slope/inter 缩放（0 slope = 无缩放）
        if (sclSlope != 0 && (sclSlope != 1 || sclInter != 0))
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = data[i] * sclSlope + sclInter;
        }

        // 仿射：sform 优先（sform_code != 0），否则 qform（此处按单位四元数近似，
        // RADAR 只用 |Affine[i,i]| 做 spacing，若 sform 缺失再退 pixdim）
        var affine = new double[3, 4];
        if (sformCode != 0)
        {
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                    affine[r, c] = GetSingle(raw, 280 + r * 16 + c * 4, littleEndian);
            }
        }
        else if (qformCode != 0)
        {
            // 完整四元数解码，偏移按 NIfTI-1 标准 §2.5：
            // quatern_b=256, quatern_c=260, quatern_d=264, qoffset_x/y/z=268/272/276,
            // pixdim[1](quatern_a 符号)=80, pixdim[2..4]=84/88/92
            float b = GetSingle(raw, 256, littleEndian);
            float c = GetSingle(raw, 260, littleEndian);
            float d = GetSingle(raw, 264, littleEndian);
            float quaternA = GetSingle(raw, 80, littleEndian);
            double a = Math.Sqrt(Math.Max(0, 1.0 - b * b - c * c - d * d)) * (quaternA >= 0 ? 1 : -1);
            affine[0, 0] = a * a + b * b - c * c - d * d;
            affine[0, 1] = 2 * b * c - 2 * a * d;
            affine[0, 2] = 2 * b * d + 2 * a * c;
            affine[1, 0] = 2 * b * c + 2 * a * d;
            affine[1, 1] = a * a + c * c - b * b - d * d;
            affine[1, 2] = 2 * c * d - 2 * a * b;
            affine[2, 0] = 2 * b * d - 2 * a * c;
            affine[2, 1] = 2 * c * d + 2 * a * b;
            affine[2, 2] = a * a + d * d - b * b - c * c;
            affine[0, 3] = GetSingle(raw, 268, littleEndian);
            affine[1, 3] = GetSingle(raw, 272, littleEndian);
            affine[2, 3] = GetSingle(raw, 276, littleEndian);
            double pixdimX = GetSingle(raw, 84, littleEndian);
            double pixdimY = GetSingle(raw, 88, littleEndian);
            double pixdimZ = GetSingle(raw, 92, littleEndian);
            affine[0, 0] *= pixdimX; affine[0, 1] *= pixdimX; affine[0, 2] *= pixdimX;
            affine[1, 0] *= pixdimY; affine[1, 1] *= pixdimY; affine[1, 2] *= pixdimY;
            affine[2, 0] *= pixdimZ; affine[2, 1] *= pixdimZ; affine[2, 2] *= pixdimZ;
        }
        else
        {
            // 退 pixdim 对角（pixdim[2..4] = offset 84/88/92）
            affine[0, 0] = GetSingle(raw, 84, littleEndian);
            affine[1, 1] = GetSingle(raw, 88, littleEndian);
            affine[2, 2] = GetSingle(raw, 92, littleEndian);
        }

        return new NiftiImage
        {
            Data = data,
            DimX = dimX,
            DimY = dimY,
            DimZ = dimZ,
            Affine = affine,
        };
    }

    private static int BytesPerVoxel(short datatype) => datatype switch
    {
        DtUint8 or DtInt8 => 1,
        DtInt16 or DtUint16 => 2,
        DtInt32 or DtUint32 or DtFloat32 => 4,
        DtFloat64 => 8,
        _ => throw new NotSupportedException($"暂不支持的 NIfTI 数据类型: {datatype}（仅支持 uint8/int8/int16/uint16/int32/uint32/float32/float64）"),
    };

    private static float[] ReadVoxels(byte[] raw, long offset, int count, short datatype, bool le)
    {
        var result = new float[count];
        switch (datatype)
        {
            case DtFloat32:
            {
                Buffer.BlockCopy(raw, (int)offset, result, 0, count * 4);
                if (!le) ReverseFloatEndianness(result);
                return result;
            }
            case DtFloat64:
            {
                var tmp = new double[count];
                Buffer.BlockCopy(raw, (int)offset, tmp, 0, count * 8);
                if (!le) ReverseDoubleEndianness(tmp);
                for (int i = 0; i < count; i++) result[i] = (float)tmp[i];
                return result;
            }
            case DtInt16:
            {
                for (int i = 0; i < count; i++)
                    result[i] = GetInt16(raw, (int)offset + i * 2, le);
                return result;
            }
            case DtUint16:
            {
                for (int i = 0; i < count; i++)
                    result[i] = GetUInt16(raw, (int)offset + i * 2, le);
                return result;
            }
            case DtInt32:
            {
                for (int i = 0; i < count; i++)
                    result[i] = GetInt32(raw, (int)offset + i * 4, le);
                return result;
            }
            case DtUint32:
            {
                for (int i = 0; i < count; i++)
                    result[i] = GetUInt32(raw, (int)offset + i * 4, le);
                return result;
            }
            case DtInt8:
            {
                for (int i = 0; i < count; i++)
                    result[i] = (sbyte)raw[offset + i];
                return result;
            }
            case DtUint8:
            {
                for (int i = 0; i < count; i++)
                    result[i] = raw[offset + i];
                return result;
            }
            default:
                throw new NotSupportedException($"数据类型 {datatype}");
        }
    }

    private static void ReverseFloatEndianness(float[] a)
    {
        var bytes = new byte[4];
        for (int i = 0; i < a.Length; i++)
        {
            Buffer.BlockCopy(a, i * 4, bytes, 0, 4);
            Array.Reverse(bytes);
            a[i] = BitConverter.ToSingle(bytes, 0);
        }
    }

    private static void ReverseDoubleEndianness(double[] a)
    {
        var bytes = new byte[8];
        for (int i = 0; i < a.Length; i++)
        {
            Buffer.BlockCopy(a, i * 8, bytes, 0, 8);
            Array.Reverse(bytes);
            a[i] = BitConverter.ToDouble(bytes, 0);
        }
    }

    private static short GetInt16(byte[] b, int o, bool le) =>
        le ? BitConverter.ToInt16(b, o) : BitConverter.ToInt16(b.Reverse().ToArray(), b.Length - o - 2);

    private static ushort GetUInt16(byte[] b, int o, bool le) =>
        le ? BitConverter.ToUInt16(b, o) : BitConverter.ToUInt16(b.Reverse().ToArray(), b.Length - o - 2);

    private static int GetInt32(byte[] b, int o, bool le) =>
        le ? BitConverter.ToInt32(b, o) : BitConverter.ToInt32(b.Reverse().ToArray(), b.Length - o - 4);

    private static uint GetUInt32(byte[] b, int o, bool le) =>
        le ? BitConverter.ToUInt32(b, o) : BitConverter.ToUInt32(b.Reverse().ToArray(), b.Length - o - 4);

    private static float GetSingle(byte[] b, int o, bool le) =>
        le ? BitConverter.ToSingle(b, o) : BitConverter.ToSingle(b.Reverse().ToArray(), b.Length - o - 4);
}
