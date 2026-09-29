using System.IO.Compression;
using System.Text;

namespace Radar.Core.Support;

/// <summary>
/// .npz / .npy 读取器（仅支持 C 序数值 dtype，够用即可）。
/// npz = zip 容器，每个 entry 是一个 .npy 文件。
/// </summary>
public static class NpyReader
{
    public sealed record NpyArray(Array Data, Type ElementType, int[] Shape);

    /// <summary>读取 .npz，返回 {去后缀键名: 数组}。</summary>
    public static Dictionary<string, NpyArray> LoadNpz(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var result = new Dictionary<string, NpyArray>();
        foreach (var entry in zip.Entries)
        {
            using var ms = new MemoryStream();
            entry.Open().CopyTo(ms);
            var name = entry.Name.EndsWith(".npy") ? entry.Name[..^4] : entry.Name;
            result[name] = Parse(ms.ToArray());
        }
        return result;
    }

    public static NpyArray Load(string path)
    {
        var raw = File.ReadAllBytes(path);
        return Parse(raw);
    }

    public static float[] LoadFloat32(string path, out int[] shape)
    {
        var a = Load(path);
        if (a.ElementType != typeof(float))
            throw new InvalidDataException($"期望 float32, 实际 {a.ElementType}");
        shape = a.Shape;
        return (float[])a.Data;
    }

    public static NpyArray Parse(byte[] raw)
    {
        if (raw.Length < 10 || raw[0] != 0x93 || Encoding.ASCII.GetString(raw, 1, 5) != "NUMPY")
            throw new InvalidDataException("不是 npy 数据");
        int major = raw[6];
        int headerLen = major == 1
            ? raw[8] | (raw[9] << 8)
            : (int)(raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24));
        int dataOffset = (major == 1 ? 10 : 12) + headerLen;
        var header = Encoding.Latin1.GetString(raw, major == 1 ? 10 : 12, headerLen);

        var descr = Extract(header, "'descr': '", "'");
        if (header.Contains("'fortran_order': True"))
            throw new NotSupportedException("fortran 序 npy 暂不支持");
        var shapeStr = Extract(header, "'shape': (", ")");
        var shape = shapeStr.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).Select(int.Parse).ToArray();

        long count = shape.Aggregate(1L, (a, b) => a * b);
        int elem = SizeOf(descr);
        if (raw.Length < dataOffset + count * elem)
            throw new InvalidDataException("npy 数据段不完整");

        Array data = descr switch
        {
            "<f4" => To1D<float>(raw, dataOffset, count),
            "<f8" => To1D<double>(raw, dataOffset, count),
            "<i2" => To1D<short>(raw, dataOffset, count),
            "<i4" => To1D<int>(raw, dataOffset, count),
            "<i8" => To1D<long>(raw, dataOffset, count),
            "|i1" => To1D<sbyte>(raw, dataOffset, count),
            "|u1" => To1D<byte>(raw, dataOffset, count),
            "<u2" => To1D<ushort>(raw, dataOffset, count),
            _ => throw new NotSupportedException($"npy dtype {descr}"),
        };
        return new NpyArray(data, data.GetType().GetElementType()!, shape);
    }

    private static Array To1D<T>(byte[] raw, int offset, long count) where T : struct
    {
        var arr = new T[count];
        var elemSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        Buffer.BlockCopy(raw, offset, arr, 0, checked((int)(count * elemSize)));
        return arr;
    }

    private static int SizeOf(string descr) => descr switch
    {
        "|i1" or "|u1" => 1,
        "<i2" or "|u2" => 2,
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
}
