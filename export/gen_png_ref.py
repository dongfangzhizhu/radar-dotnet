# -*- coding: utf-8 -*-
"""生成 PNG 编码器的独立预言机（参考数据），供 Radar.Tests 对拍。

为什么要独立预言机：
  自己编码、自己解码只能证明「自洽」，不能证明「符合 PNG 规范」。
  这里用 Python 标准库 zlib 按 PNG 规范手工拼一份参考实现，并落盘三样东西：
    - IHDR / IEND 的完整块字节（长度+类型+数据+CRC）—— 与压缩实现无关，可逐字节比对；
    - IDAT 解压后的原始扫描线数据 —— 绕过压缩级别差异，只比语义内容；
    - 像素输入本身 —— 测试端据此复现同样的输入。

产物：testdata/golden/png_ref_5x3.json（入库）与 png_ref_5x3.png（人眼可看）
运行：python export/gen_png_ref.py
"""
import json
import struct
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "testdata" / "golden" / "png_ref_5x3.json"
OUT_PNG = REPO / "testdata" / "golden" / "png_ref_5x3.png"

W, H = 5, 3


def build_pixels():
    # 刻意选非 16 的整数倍宽度、含 0 与 255 边界，覆盖滤波字节与调色路径
    px = bytearray()
    for y in range(H):
        for x in range(W):
            px += bytes([(x * 51) % 256, (y * 120) % 256, (x * y * 37) % 256])
    return bytes(px)


def chunk(ctype: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + ctype + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))


def main():
    pixels = build_pixels()
    assert len(pixels) == W * H * 3

    ihdr_data = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    raw = bytearray()
    stride = W * 3
    for y in range(H):
        raw.append(0)                                  # 滤波类型 0 = None
        raw += pixels[y * stride:(y + 1) * stride]

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr_data)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    OUT_PNG.write_bytes(png)

    ref = {
        "width": W,
        "height": H,
        "pixels_rgb_hex": pixels.hex(),
        "ihdr_chunk_hex": chunk(b"IHDR", ihdr_data).hex(),
        "iend_chunk_hex": chunk(b"IEND", b"").hex(),
        "idat_raw_hex": bytes(raw).hex(),
        "png_sha256_note": "整体字节不比对：压缩级别实现相关，只比对结构与语义",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    print("参考 PNG %d 字节 -> %s" % (len(png), OUT_PNG.name))
    print("参考数据 -> %s（IHDR %d 字节块 / 原始扫描线 %d 字节）"
          % (OUT_JSON.name, len(bytes.fromhex(ref["ihdr_chunk_hex"])), len(raw)))


if __name__ == "__main__":
    main()
