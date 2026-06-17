"""dSprites 데이터셋 로더.

dSprites: 64x64 이진 이미지 737,280장. 6개 생성 인자(factor):
  - color(1), shape(3), scale(6), orientation(40), posX(32), posY(32).
파일이 없으면 자동 다운로드한다.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

DSPRITES_FILE = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_URL = (
    "https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/"
    "master/" + DSPRITES_FILE
)

# latents_classes 컬럼별 인자 이름과 클래스 수
FACTOR_NAMES = ["color", "shape", "scale", "orientation", "posX", "posY"]
FACTOR_SIZES = [1, 3, 6, 40, 32, 32]


def download_dsprites(data_dir: str) -> str:
    """dSprites npz 경로를 반환. 없으면 다운로드."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, DSPRITES_FILE)
    if not os.path.exists(path):
        print(f"[dsprites] downloading -> {path}")
        urllib.request.urlretrieve(DSPRITES_URL, path)
        print("[dsprites] download done")
    return path


class DSprites(Dataset):
    """이미지를 (1, 64, 64) float32 [0,1] 텐서로 반환.

    return_factors=True 면 (img, latents_classes) 튜플 반환(disentanglement 평가용).
    """

    def __init__(self, data_dir: str = "data", return_factors: bool = False):
        path = download_dsprites(data_dir)
        # mmap 으로 로드해 메모리 절약 (전체 ~2.8GB).
        data = np.load(path, allow_pickle=True, encoding="latin1", mmap_mode="r")
        self.imgs = data["imgs"]                     # (737280, 64, 64) uint8 {0,1}
        self.latents_classes = data["latents_classes"]  # (737280, 6) int
        self.return_factors = return_factors

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx: int):
        img = self.imgs[idx].astype(np.float32)          # (64, 64), {0,1}
        img = torch.from_numpy(img).unsqueeze(0)         # (1, 64, 64)
        if self.return_factors:
            factors = torch.from_numpy(self.latents_classes[idx].astype(np.int64))
            return img, factors
        return img
