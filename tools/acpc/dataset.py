from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

try:
    from .common import crop_square, load_gray, read_json, resize_patch
except ImportError:
    from common import crop_square, load_gray, read_json, resize_patch


class SarPatchDataset(Dataset):
    def __init__(
        self,
        samples_json: Union[str, Path],
        patch_size: int = 96,
        augment: bool = False,
        return_pair: bool = False,
    ):
        payload = read_json(samples_json)
        if isinstance(payload, dict):
            self.samples = payload.get("samples", payload.get("regions", []))
        else:
            self.samples = payload
        self.patch_size = patch_size
        self.augment = augment
        self.return_pair = return_pair
        self._image_cache = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        if path not in self._image_cache:
            self._image_cache[path] = load_gray(path)
        return self._image_cache[path]

    def _patch(self, sample: dict) -> np.ndarray:
        image = self._load_image(sample["image_path"])
        crop_size = int(max(sample.get("crop_size", self.patch_size), self.patch_size))
        patch = crop_square(image, sample["cx"], sample["cy"], crop_size)
        if crop_size != self.patch_size:
            patch = resize_patch(patch, self.patch_size)
        return patch

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        patch_u8 = np.clip(patch * 255, 0, 255).astype(np.uint8)
        mode = "RGB" if patch_u8.ndim == 3 and patch_u8.shape[-1] >= 3 else "L"
        if mode == "RGB":
            patch_u8 = patch_u8[..., :3]
        image = Image.fromarray(patch_u8, mode=mode)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.uniform(-12.0, 12.0)
            image = image.rotate(angle, resample=Image.BILINEAR)
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if random.random() < 0.8:
            scale = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.08, 0.08)
            gamma = random.uniform(0.8, 1.25)
            arr = np.clip((arr * scale + bias), 0, 1) ** gamma
        if random.random() < 0.5:
            noise = np.random.normal(0.0, random.uniform(0.01, 0.04), size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)
        if random.random() < 0.25:
            side = random.randint(max(4, self.patch_size // 12), max(5, self.patch_size // 5))
            x = random.randint(0, self.patch_size - side)
            y = random.randint(0, self.patch_size - side)
            arr[y : y + side, x : x + side] = arr.mean(axis=(0, 1))
        return arr.astype(np.float32)

    @staticmethod
    def _to_tensor(patch: np.ndarray) -> torch.Tensor:
        patch = patch.astype(np.float32)
        if patch.ndim == 2:
            patch = patch[..., None]
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))

    def __getitem__(self, idx: int):
        patch = self._patch(self.samples[idx])
        if self.return_pair:
            q = self._augment(patch) if self.augment else patch
            k = self._augment(patch) if self.augment else patch
            return self._to_tensor(q), self._to_tensor(k), idx
        if self.augment:
            patch = self._augment(patch)
        return self._to_tensor(patch), idx
