"""의존성 없는 유틸 (stdlib + numpy 만 사용).

torchvision/PIL 없이 흑백 PNG를 저장하기 위한 최소 인코더.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def save_gray_png(arr: np.ndarray, path: str) -> None:
    """arr: (H, W) float[0,1] 또는 uint8 흑백 이미지를 PNG로 저장."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = arr.shape
    # 각 행 앞에 filter byte(0) 추가
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))  # 8bit grayscale
    png += _png_chunk(b"IDAT", zlib.compress(raw, 6))
    png += _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def make_grid(imgs: np.ndarray, nrow: int, pad: int = 2, pad_val: float = 1.0) -> np.ndarray:
    """imgs: (N, H, W) 흑백 또는 (N, H, W, 3) RGB -> 타일 격자. nrow 개씩 한 줄."""
    n, h, w = imgs.shape[:3]
    rgb = imgs.ndim == 4
    ncol = (n + nrow - 1) // nrow
    hg = ncol * (h + pad) + pad
    wg = nrow * (w + pad) + pad
    shape = (hg, wg, 3) if rgb else (hg, wg)
    grid = np.full(shape, pad_val, dtype=np.float32)
    for i in range(n):
        r, c = i // nrow, i % nrow
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        grid[y:y + h, x:x + w] = imgs[i]
    return grid


def save_rgb_png(arr: np.ndarray, path: str) -> None:
    """arr: (H, W, 3) float[0,1] 또는 uint8 RGB 이미지를 PNG로 저장 (stdlib만)."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))  # filter byte + RGB row
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  # 8bit truecolor
    png += _png_chunk(b"IDAT", zlib.compress(raw, 6))
    png += _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
