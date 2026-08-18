import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: PathLike, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def list_images(image_dirs: Sequence[PathLike]) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for image_dir in image_dirs:
        root = Path(image_dir)
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images[path.stem] = path
    return images


def load_gray(path: PathLike) -> np.ndarray:
    """Backward-compatible loader that preserves Xpol channels when present."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if np.iscomplexobj(array):
            array = np.abs(array)
        array = np.asarray(array, dtype=np.float32)
        lo, hi = np.percentile(array[np.isfinite(array)], [2, 98])
        return np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)
    image = Image.open(path)
    if image.mode in {"RGB", "RGBA"}:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def resize_patch(patch: np.ndarray, size: int) -> np.ndarray:
    patch_u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    mode = "RGB" if patch_u8.ndim == 3 and patch_u8.shape[-1] >= 3 else "L"
    if mode == "RGB":
        patch_u8 = patch_u8[..., :3]
    image = Image.fromarray(patch_u8, mode=mode).resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_square(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    if h >= size:
        y1 = min(max(0, y1), h - size)
        y2 = y1 + size
    else:
        y1, y2 = 0, h
    if w >= size:
        x1 = min(max(0, x1), w - size)
        x2 = x1 + size
    else:
        x1, x2 = 0, w

    patch = image[y1:y2, x1:x2]
    if patch.shape[0] == size and patch.shape[1] == size:
        return patch

    pad_h = max(0, size - patch.shape[0])
    pad_w = max(0, size - patch.shape[1])
    pad_spec = ((pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2))
    if patch.ndim == 3:
        pad_spec += ((0, 0),)
    return np.pad(patch, pad_spec, mode="edge")


def has_full_square_crop(image_shape: Tuple[int, int], cx: float, cy: float, size: int) -> bool:
    h, w = image_shape[:2]
    if h < size or w < size:
        return False
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    return x1 >= 0 and y1 >= 0 and x1 + size <= w and y1 + size <= h


def parse_dota_txt(path: PathLike) -> List[dict]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    anns: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                poly = np.asarray(parts[:8], dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            label = parts[8] if len(parts) > 8 else "object"
            difficulty = int(float(parts[9])) if len(parts) > 9 else 0
            xs = poly[:, 0]
            ys = poly[:, 1]
            cx = float(xs.mean())
            cy = float(ys.mean())
            width = float(max(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])))
            height = float(max(np.linalg.norm(poly[1] - poly[2]), np.linalg.norm(poly[3] - poly[0])))
            anns.append(
                {
                    "polygon": poly.reshape(-1).tolist(),
                    "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    "center": [cx, cy],
                    "width": width,
                    "height": height,
                    "label": label,
                    "difficulty": difficulty,
                }
            )
    return anns


def min_distance_to_points(cx: float, cy: float, points: Sequence[Sequence[float]]) -> float:
    if not points:
        return float("inf")
    arr = np.asarray(points, dtype=np.float32)
    d = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
    return float(d.min())


def patch_stats(patch: np.ndarray) -> List[float]:
    patch = np.asarray(patch, dtype=np.float32)
    mean = float(patch.mean())
    std = float(patch.std())
    p95 = float(np.percentile(patch, 95))
    p99 = float(np.percentile(patch, 99))
    contrast = p95 - float(np.percentile(patch, 5))
    center = patch[patch.shape[0] // 4 : 3 * patch.shape[0] // 4, patch.shape[1] // 4 : 3 * patch.shape[1] // 4]
    center_mean = float(center.mean()) if center.size else mean
    bright = float((patch > max(mean + 2.0 * std, p95)).mean())
    gy, gx = np.gradient(patch)
    edge = float(np.sqrt(gx * gx + gy * gy).mean())
    return [mean, std, p95, p99, contrast, center_mean, bright, edge]


def normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def cosine_topk(query: np.ndarray, keys: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return np.zeros((len(query), 0), dtype=np.int64), np.zeros((len(query), 0), dtype=np.float32)
    sims = query @ keys.T
    k = min(topk, sims.shape[1])
    if k == sims.shape[1]:
        idx = np.argsort(-sims, axis=1)[:, :k]
    else:
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        row = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        idx = idx[row, order]
    row = np.arange(sims.shape[0])[:, None]
    vals = sims[row, idx]
    return idx.astype(np.int64), vals.astype(np.float32)


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def load_optional_map(map_dir: Optional[PathLike], image_id: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if map_dir is None:
        return None
    root = Path(map_dir)
    for suffix in IMAGE_SUFFIXES | {".npy"}:
        path = root / f"{image_id}{suffix}"
        if path.exists():
            if suffix == ".npy":
                arr = np.load(path).astype(np.float32)
                if arr.shape != shape:
                    arr = np.asarray(Image.fromarray(arr).resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)
                arr_min, arr_max = float(arr.min()), float(arr.max())
                return (arr - arr_min) / max(arr_max - arr_min, 1e-6)
            return load_gray(path)
    return None


def read_teacher_json(path: Optional[PathLike]) -> Dict[str, list]:
    if path is None:
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        return data
    by_image: Dict[str, list] = {}
    for item in data:
        image_id = item.get("image_id") or Path(item.get("image", "")).stem
        by_image.setdefault(image_id, []).append(item)
    return by_image


def teacher_item_to_sample(item: dict) -> Optional[Tuple[float, float, float, float, float]]:
    if "bbox_xyxy" in item:
        x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy"]]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2), max(x2 - x1, 1.0), max(y2 - y1, 1.0), float(item.get("score", 1.0)))
    if "bbox" in item:
        vals = [float(v) for v in item["bbox"]]
        if len(vals) >= 4:
            x, y, w, h = vals[:4]
            score = float(item.get("score", vals[4] if len(vals) > 4 else 1.0))
            return (x + 0.5 * w, y + 0.5 * h, max(w, 1.0), max(h, 1.0), score)
    if "center" in item:
        cx, cy = [float(v) for v in item["center"][:2]]
        w = float(item.get("width", item.get("size", 32)))
        h = float(item.get("height", item.get("size", 32)))
        return (cx, cy, w, h, float(item.get("score", 1.0)))
    return None
