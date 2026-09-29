using System.Text;

namespace Radar.Tests.Support;

/// <summary>
/// 极简 .npy v1.0 读取器（仅支持数值 dtype + C 顺序），供黄金基线对拍使用。
/// </summary>
public static class NpyReader
{
    public sealed record NpyArray(Array Data, Type ElementType, int[] Shape);

    public static NpyArray Load(string path)
    {
        using var fs = File.OpenRead(path);
        Span<byte> magic = stackalloc byte[6];
        if (fs.Read(magic) != 6 || magic[0] != 0x93 || Encoding.ASCII.GetString(magic[1..6]) != "NUMPY")
            throw new InvalidDataException($"不是 npy 文件: {path}");

        int major = fs.ReadByte();
        fs.ReadByte(); // minor
        int headerLen = major == 1 ? ReadU16(fs) : ReadU32(fs);
        var headerBytes = new byte[headerLen];
        fs.ReadExactly(headerBytes);
        var header = Encoding.Latin1.GetString(headerBytes);

        var descr = Extract(header, "'descr': '", "'");
        bool fortran = header.Contains("'fortran_order': True");
        var shapeStr = Extract(header, "'shape': (", ")");
        var shape = shapeStr.Split(',')
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .Select(int.Parse)
            .ToArray();

        var elemType = descr switch
        {
            "<f4" or "|f4" => typeof(float),
            "<f8" or "|f8" => typeof(double),
            "<i2" or "|i2" => typeof(short),
            "<i8" => typeof(long),
            "|i1" => typeof(sbyte),
            "|u1" => typeof(byte),
            "<u2" => typeof(ushort),
            "<i4" => typeof(int),
            _ => throw new NotSupportedException($"npy dtype {descr}"),
        };

        int elemSize = SizeOf(descr);
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var raw = new byte[count * elemSize];
        fs.ReadExactly(raw);

        Array data;
        switch (descr)
        {
            case "<f4":
            {
                var arr = new float[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "<f8":
            {
                var arr = new double[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "<i2":
            {
                var arr = new short[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "<i8":
            {
                var arr = new long[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "|i1":
            {
                var arr = new sbyte[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "|u1":
            {
                var arr = new byte[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            case "<i4":
            {
                var arr = new int[count];
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                data = arr;
                break;
            }
            default:
                throw new NotSupportedException($"npy dtype {descr}");
        }

        if (fortran)
            throw new NotSupportedException("fortran 序 npy 暂不支持（golden 均为 C 序）");
        return new NpyArray(data, elemType, shape);
    }

    public static float[] LoadFloat32(string path, out int[] shape)
    {
        var a = Load(path);
        if (a.ElementType != typeof(float))
            throw new InvalidDataException($"期望 float32, 实际 {a.ElementType}");
        shape = a.Shape;
        return (float[])a.Data;
    }

    private static int SizeOf(string descr) => descr switch
    {
        "|i1" or "|u1" => 1,
        "<i2" or "|u2" or "<f2" => 2,
        "<i4" or "<u4" or "<f4" => 4,
        "<i8" or "<u8" or "<f8" => 8,
        _ => throw new NotSupportedException(descr),
    };

    private static string Extract(string header, string begin, string end)
    {
        int i = header.IndexOf(begin, StringComparison.Ordinal);
        if (i < 0) throw new InvalidDataException($"npy header 缺少 {begin}");
        i += begin.Length;
        int j = header.IndexOf(end, i, StringComparison.Ordinal);
        return header[i..j];
    }

    private static int ReadU16(FileStream fs)
    {
        Span<byte> b = stackalloc byte[2];
        fs.ReadExactly(b);
        return b[0] | (b[1] << 8);
    }

    private static int ReadU32(FileStream fs)
    {
        Span<byte> b = stackalloc byte[4];
        fs.ReadExactly(b);
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
    }
}

/// <summary>定位仓库根（含 PLAN.md 的目录）。</summary>
public static class RepoRoot
{
    public static string Find()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "PLAN.md")))
            dir = dir.Parent;
        if (dir == null)
            throw new InvalidOperationException("未找到仓库根（PLAN.md）");
        return dir.FullName;
    }
}
