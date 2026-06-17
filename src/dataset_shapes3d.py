"""3D Shapes 데이터셋 로더.

3D Shapes (DeepMind): 64x64x3 RGB 이미지 480,000장. 6개 독립 생성 인자(fully crossed):
  floor_hue(10) · wall_hue(10) · object_hue(10) · scale(8) · shape(4) · orientation(15).
파일(.h5)이 없으면 자동 다운로드(~255MB). h5py 필요.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
SHAPES3D_FILE = "3dshapes.h5"

# labels 컬럼별 인자 이름과 클래스 수 (연속값 라벨이지만 균등 그리드라 이산 인덱스로 변환 가능)
FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
FACTOR_SIZES = [10, 10, 10, 8, 4, 15]


def download_shapes3d(data_dir: str) -> str:
    """3dshapes.h5 경로 반환. 없으면 다운로드."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, SHAPES3D_FILE)
    if not os.path.exists(path):
        print(f"[3dshapes] downloading (~255MB) -> {path}")
        urllib.request.urlretrieve(SHAPES3D_URL, path)
        print("[3dshapes] download done")
    return path


def _labels_to_indices(labels: np.ndarray) -> np.ndarray:
    """연속 라벨(각 인자 균등 그리드)을 0..(size-1) 정수 인덱스로 변환."""
    idx = np.empty_like(labels, dtype=np.int64)
    for j, size in enumerate(FACTOR_SIZES):
        col = labels[:, j]
        lo, hi = col.min(), col.max()
        if size == 1 or hi == lo:
            idx[:, j] = 0
        else:
            # 균등 그리드: round((v-lo)/(hi-lo)*(size-1))
            idx[:, j] = np.rint((col - lo) / (hi - lo) * (size - 1)).astype(np.int64)
    return idx


class Shapes3D(Dataset):
    """이미지를 (3, 64, 64) float32 [0,1] 텐서로 반환.

    return_factors=True 면 (img, factor_indices[int64, 6]) 튜플 반환(disentanglement 평가용).
    전체 이미지(uint8, ~5.5GB)를 RAM 에 적재한다 — DataLoader fork 시 COW 로 공유돼
    멀티프로세싱 안전(h5py 워커 파일락 데드락 회피)하고 HDF5 랜덤리드보다 빠르다.
    """

    def __init__(self, data_dir: str = "data", return_factors: bool = False,
                 uint8: bool = False):
        import h5py  # 지연 import (의존성 격리)

        self.path = download_shapes3d(data_dir)
        self.return_factors = return_factors
        self.uint8 = uint8                    # True면 (64,64,3) uint8 반환 → GPU 배치 변환(빠름)

        with h5py.File(self.path, "r") as f:
            self.images = f["images"][:]      # (N,64,64,3) uint8 — RAM 적재
            labels = f["labels"][:]           # (N, 6) float
        self.factor_indices = _labels_to_indices(labels)  # (N, 6) int64

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        if self.uint8:
            img = torch.from_numpy(self.images[idx])               # (64,64,3) uint8, 변환 없음
        else:
            arr = self.images[idx].astype(np.float32) / 255.0      # (64,64,3) [0,1]
            img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,64,64)
        if self.return_factors:
            factors = torch.from_numpy(self.factor_indices[idx])   # (6,) int64
            return img, factors
        return img


def prep_batch(x: torch.Tensor, device, uint8: bool) -> torch.Tensor:
    """배치를 device 로. uint8(B,64,64,3) 이면 GPU 에서 (B,3,64,64) float[0,1] 로 변환."""
    x = x.to(device, non_blocking=True)
    if uint8:
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return x
