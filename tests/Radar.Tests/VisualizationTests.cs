using System.IO.Compression;
using System.Text.Json;
using Radar.Core.Support;
using Radar.Core.Visualization;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// PNG 编码器与分割掩码透叠图（实测差距 #2）：
/// PLAN §3.3 承诺的「分割掩码可视化」落地为 <see cref="MaskOverlayPng"/>。
/// <para>
/// 编码器的正确性用<b>独立预言机</b>校验：<c>testdata/golden/png_ref_5x3.json</c>
/// 由 Python 标准库 zlib 按 PNG 规范手工拼出（export/gen_png_ref.py），
/// 逐字节提供 IHDR / IEND 块与 IDAT 解压后的原始扫描线。
/// </para>
/// </summary>
public class VisualizationTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    // ---------- PNG 结构 ----------

    private sealed record Chunk(string Type, byte[] Data, byte[] Full);
    private sealed record RefPng(int Width, int Height, byte[] Pixels,
        byte[] IhdrChunk, byte[] IendChunk, byte[] IdatRaw);

    private static RefPng LoadRef()
    {
        string p = Path.Combine(Root, "testdata", "golden", "png_ref_5x3.json");
        var d = JsonDocument.Parse(File.ReadAllText(p)).RootElement;
        byte[] H(string name) => Convert.FromHexString(d.GetProperty(name).GetString()!);
        return new RefPng(d.GetProperty("width").GetInt32(), d.GetProperty("height").GetInt32(),
            H("pixels_rgb_hex"), H("ihdr_chunk_hex"), H("iend_chunk_hex"), H("idat_raw_hex"));
    }

    /// <summary>解析 PNG 块序列（含长度/类型/数据/CRC 的完整字节）。</summary>
    private static List<Chunk> ParseChunks(byte[] png)
    {
        Assert.True(png.Length > 8, "PNG 太短");
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, png[..8]);

        var list = new List<Chunk>();
        int pos = 8;
        while (pos + 12 <= png.Length)
        {
            int len = (png[pos] << 24) | (png[pos + 1] << 16) | (png[pos + 2] << 8) | png[pos + 3];
            var full = png[pos..(pos + 12 + len)];
            list.Add(new Chunk(System.Text.Encoding.ASCII.GetString(png, pos + 4, 4),
                png[(pos + 8)..(pos + 8 + len)], full));
            pos += 12 + len;
        }
        Assert.Equal(png.Length, pos);          // 块序列必须完整覆盖整个文件
        return list;
    }

    [Fact]
    public void Encode_MatchesPythonOracle_ByteForByte()
    {
        var r = LoadRef();
        byte[] png = PngEncoder.Encode(r.Width, r.Height, r.Pixels);
        var chunks = ParseChunks(png);

        output.WriteLine($"编码 {png.Length} 字节，块序列: {string.Join(" -> ", chunks.Select(c => c.Type))}");

        Assert.Equal(new[] { "IHDR", "IDAT", "IEND" }, chunks.Select(c => c.Type));
        Assert.Equal(r.IhdrChunk, chunks[0].Full);      // IHDR 与压缩实现无关：必须逐字节相同
        Assert.Equal(r.IendChunk, chunks[2].Full);      // 同理 IEND

        // IDAT：绕过压缩级别差异，比解压后的语义内容
        byte[] raw = Inflate(chunks[1].Data);
        Assert.Equal(r.IdatRaw, raw);
        output.WriteLine($"IDAT 解压 {raw.Length} 字节，与 Python 参考一致");
    }

    [Fact]
    public void Encode_IhdrCarriesRequestedGeometry()
    {
        byte[] png = PngEncoder.Encode(7, 4, new byte[7 * 4 * 3]);
        var ihdr = ParseChunks(png)[0].Data;

        int W = (ihdr[0] << 24) | (ihdr[1] << 16) | (ihdr[2] << 8) | ihdr[3];
        int H = (ihdr[4] << 24) | (ihdr[5] << 16) | (ihdr[6] << 8) | ihdr[7];
        Assert.Equal(7, W);
        Assert.Equal(4, H);
        Assert.Equal(8, ihdr[8]);    // 位深
        Assert.Equal(2, ihdr[9]);    // 颜色类型：真彩色 RGB
        Assert.Equal(0, ihdr[10]);
        Assert.Equal(0, ihdr[11]);
        Assert.Equal(0, ihdr[12]);
    }

    [Fact]
    public void Encode_RowFilterBytesAreAllNone()
    {
        const int w = 6, h = 3;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i * 7);

        byte[] raw = Inflate(ParseChunks(PngEncoder.Encode(w, h, rgb))[1].Data);
        Assert.Equal(h * (1 + w * 3), raw.Length);

        int stride = w * 3;
        for (int y = 0; y < h; y++)
        {
            Assert.Equal(0, raw[y * (1 + stride)]);                                   // 滤波类型 0
            Assert.Equal(rgb[(y * stride)..((y + 1) * stride)],
                         raw[(y * (1 + stride) + 1)..((y + 1) * (1 + stride))]);       // 像素原样
        }
    }

    [Fact]
    public void Encode_IsDeterministic()
    {
        var rgb = new byte[9 * 5 * 3];
        new Random(7).NextBytes(rgb);
        Assert.Equal(PngEncoder.Encode(9, 5, rgb), PngEncoder.Encode(9, 5, rgb));
    }

    [Fact]
    public void Encode_RejectsBadArguments()
    {
        Assert.Throws<ArgumentException>(() => PngEncoder.Encode(0, 4, []));
        Assert.Throws<ArgumentException>(() => PngEncoder.Encode(-1, 4, new byte[0]));
        Assert.Throws<ArgumentException>(() => PngEncoder.Encode(2, 2, new byte[2 * 2 * 3 - 1]));
        Assert.Throws<ArgumentException>(() => PngEncoder.Encode(2, 2, new byte[2 * 2 * 3 + 1]));
    }

    private static byte[] Inflate(byte[] zlibData)
    {
        using var src = new MemoryStream(zlibData);
        using var z = new ZLibStream(src, CompressionMode.Decompress);
        using var dst = new MemoryStream();
        z.CopyTo(dst);
        return dst.ToArray();
    }

    // ---------- 掩码透叠图 ----------

    [Fact]
    public void MaskOverlay_ProducesValidPngWithThreeViews()
    {
        const int d = 4, h = 6, w = 5;
        var vol = new float[d * h * w];
        for (int i = 0; i < vol.Length; i++) vol[i] = (i % 10) / 10f;
        var mask = new byte[d * h * w];
        mask[0] = 1;                                  // 造一些前景标签
        for (int i = 0; i < 5; i++) mask[i] = 3;

        byte[] png = MaskOverlayPng.Render(vol, d, h, w, mask);
        var chunks = ParseChunks(png);
        Assert.Equal(new[] { "IHDR", "IDAT", "IEND" }, chunks.Select(c => c.Type));

        // 三个正交视图横向拼：宽 = w + w + h + 2*Gap，高 = max(h, d, d)
        var ihdr = chunks[0].Data;
        int W = (ihdr[0] << 24) | (ihdr[1] << 16) | (ihdr[2] << 8) | ihdr[3];
        int H = (ihdr[4] << 24) | (ihdr[5] << 16) | (ihdr[6] << 8) | ihdr[7];
        Assert.Equal(w + w + h + 8 * 2, W);
        Assert.Equal(Math.Max(h, d), H);
        output.WriteLine($"掩码图 {W}x{H}，{png.Length} 字节");

        // 拼缝留白必须全黑
        byte[] raw = Inflate(chunks[1].Data);
        int stride = W * 3;
        int gapStart = w;                                                     // 第一个 Gap 起点
        for (int y = 0; y < H; y++)
            for (int x = gapStart; x < gapStart + 8; x++)
                Assert.Equal(0, raw[y * (1 + stride) + 1 + x * 3]);
    }

    [Fact]
    public void MaskOverlay_LabelChangesColor_BackgroundIsGray()
    {
        const int d = 1, h = 1, w = 2;
        // 两像素：左=背景，右=器官 1
        var vol = new float[] { 0.5f, 0.5f };
        var mask = new byte[] { 0, 1 };

        byte[] raw = Inflate(ParseChunks(MaskOverlayPng.Render(vol, d, h, w, mask))[1].Data);

        // 视图 0（axis 0）尺寸 w x h = 2x1，位于画布左上角
        int p0 = 1;
        byte bgR = raw[p0], bgG = raw[p0 + 1], bgB = raw[p0 + 2];
        byte fgR = raw[p0 + 3], fgG = raw[p0 + 4], fgB = raw[p0 + 5];

        Assert.Equal(bgR, bgG);                                  // 背景 = 灰度
        Assert.Equal(bgG, bgB);
        Assert.Equal((byte)(0.5 * 255), bgR);

        Assert.True(fgR != bgR || fgG != bgG || fgB != bgB, "标签像素必须与背景可区分");
        Assert.NotEqual(fgR, fgG);                               // 器官 1 是彩色的
    }

    [Fact]
    public void MaskOverlay_IsDeterministic()
    {
        const int d = 3, h = 5, w = 4;
        var vol = new float[d * h * w];
        var mask = new byte[d * h * w];
        for (int i = 0; i < mask.Length; i++) { vol[i] = i / (float)mask.Length; mask[i] = (byte)(i % 5); }
        Assert.Equal(MaskOverlayPng.Render(vol, d, h, w, mask),
                     MaskOverlayPng.Render(vol, d, h, w, mask));
    }

    [Fact]
    public void MaskOverlay_RejectsShortBuffers()
    {
        Assert.Throws<ArgumentException>(() => MaskOverlayPng.Render(new float[7], 2, 2, 2, new byte[8]));
        Assert.Throws<ArgumentException>(() => MaskOverlayPng.Render(new float[8], 2, 2, 2, new byte[7]));
    }

    // ---------- 真实 golden 数据 ----------

    [Fact]
    public void MaskOverlay_RendersRealGoldenVolume()
    {
        string volPath = Path.Combine(Root, "testdata", "golden", "input_window.npy");
        string maskPath = Path.Combine(Root, "testdata", "golden", "stitched_mask.npy");
        if (!File.Exists(volPath) || !File.Exists(maskPath)) return;

        var volArr = NpyReader.Load(volPath);
        var vol = (float[])volArr.Data;
        int d = volArr.Shape[2], h = volArr.Shape[3], w = volArr.Shape[4];

        var maskArr = NpyReader.Parse(File.ReadAllBytes(maskPath));    // int16 (1,d,h,w)
        short[] ms = (short[])maskArr.Data;
        var mask = new byte[d * h * w];
        int maxLabel = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = (byte)Math.Clamp((int)ms[i], 0, 255);
            if (mask[i] > maxLabel) maxLabel = mask[i];
        }
        Assert.True(maxLabel > 0 && maxLabel <= 36, $"golden 掩码标签范围异常: {maxLabel}");

        byte[] png = MaskOverlayPng.Render(vol, d, h, w, mask);
        var chunks = ParseChunks(png);
        Assert.Equal(new[] { "IHDR", "IDAT", "IEND" }, chunks.Select(c => c.Type));

        var ihdr = chunks[0].Data;
        int W = (ihdr[0] << 24) | (ihdr[1] << 16) | (ihdr[2] << 8) | ihdr[3];
        int H = (ihdr[4] << 24) | (ihdr[5] << 16) | (ihdr[6] << 8) | ihdr[7];
        Assert.Equal(w + w + h + 16, W);
        Assert.Equal(Math.Max(h, d), H);
        output.WriteLine($"golden 体 {d}x{h}x{w} → 掩码图 {W}x{H}，{png.Length} 字节，最大标签 {maxLabel}");

        // 掩码前景占比：至少要有器官像素，否则可视化没意义
        int fg = mask.Count(v => v != 0);
        double ratio = (double)fg / mask.Length;
        output.WriteLine($"前景体素 {fg}/{mask.Length} = {ratio:P2}");
        Assert.True(ratio > 0.01, $"前景占比过低 ({ratio:P2})，不像是有效分割");
    }
}
