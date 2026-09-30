using System.IO.Compression;

namespace Radar.Core.Visualization;

/// <summary>
/// 零依赖 PNG 编码器（真彩色 RGB8、无隔行、无调色板）。
/// 用 <see cref="ZLibStream"/> 产生 IDAT 所需的 zlib 流，避免引入图像库依赖。
/// </summary>
public static class PngEncoder
{
    private static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    private static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>编码为 PNG 字节。<paramref name="rgb"/> 长度须为 width*height*3（行优先）。</summary>
    public static byte[] Encode(int width, int height, byte[] rgb)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"非法尺寸: {width}x{height}");
        if (rgb.Length != (long)width * height * 3)
            throw new ArgumentException(
                $"RGB 缓冲长度不符: 期望 {width * height * 3}，实际 {rgb.Length}");

        using var ms = new MemoryStream();
        ms.Write(Signature, 0, Signature.Length);

        var ihdr = new byte[13];
        WriteBigEndian(ihdr, 0, width);
        WriteBigEndian(ihdr, 4, height);
        ihdr[8] = 8;    // 位深
        ihdr[9] = 2;    // 颜色类型 2 = 真彩色
        ihdr[10] = 0;   // 压缩方法
        ihdr[11] = 0;   // 滤波方法
        ihdr[12] = 0;   // 隔行方式
        WriteChunk(ms, "IHDR", ihdr);

        // IDAT：每条扫描线前置 1 字节滤波类型（0 = None）
        int stride = width * 3;
        var raw = new byte[height * (1 + stride)];
        for (int y = 0; y < height; y++)
        {
            raw[y * (1 + stride)] = 0;
            Buffer.BlockCopy(rgb, y * stride, raw, y * (1 + stride) + 1, stride);
        }

        using (var comp = new MemoryStream())
        {
            using (var z = new ZLibStream(comp, CompressionLevel.Optimal, leaveOpen: true))
                z.Write(raw, 0, raw.Length);
            WriteChunk(ms, "IDAT", comp.ToArray());
        }
        WriteChunk(ms, "IEND", []);

        return ms.ToArray();
    }

    private static void WriteChunk(Stream s, string type, byte[] data)
    {
        Span<byte> len = stackalloc byte[4];
        WriteBigEndian(len, 0, data.Length);
        s.Write(len);

        var typeBytes = new byte[4];
        for (int i = 0; i < 4; i++) typeBytes[i] = (byte)type[i];
        s.Write(typeBytes, 0, 4);
        s.Write(data, 0, data.Length);

        uint crc = 0xFFFFFFFFu;
        foreach (byte b in typeBytes) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        foreach (byte b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        crc ^= 0xFFFFFFFFu;

        Span<byte> crcBytes = stackalloc byte[4];
        WriteBigEndian(crcBytes, 0, (int)crc);
        s.Write(crcBytes);
    }

    private static void WriteBigEndian(Span<byte> dst, int offset, int value)
    {
        dst[offset] = (byte)(value >> 24);
        dst[offset + 1] = (byte)(value >> 16);
        dst[offset + 2] = (byte)(value >> 8);
        dst[offset + 3] = (byte)value;
    }

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return table;
    }
}
