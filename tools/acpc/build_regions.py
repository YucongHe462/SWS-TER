from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_inputs(paths: Sequence[str]) -> Dict[str, Path]:
    items: Dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            items[path.stem] = path
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    items[child.stem] = child
    return items


def percentile_normalize(arr: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, lo_q))
    hi = float(np.percentile(finite, hi_q))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


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
    pad_spec = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    if patch.ndim == 3:
        pad_spec = pad_spec + ((0, 0),)
    return np.pad(patch, pad_spec, mode="edge")


def load_display_image(path: Path) -> np.ndarray:
    feature, _ = load_polsar_like(path)
    if feature.ndim == 2 or feature.shape[-1] == 1:
        return feature[..., 0] if feature.ndim == 3 else feature
    return feature[..., :3]


def load_polsar_like(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image-like or PolSAR-like array.

    Returns:
        feature_image: HxWxC real-valued channels for segmentation/display.
        covariances: HxWxqxq Hermitian covariance-like matrices.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[-1] == arr.shape[-2]:
            cov = arr.astype(np.complex64)
            diag = np.real(np.diagonal(cov, axis1=-2, axis2=-1))
            feature = percentile_normalize(diag)
            if feature.shape[-1] == 1:
                feature = np.repeat(feature, 3, axis=-1)
            elif feature.shape[-1] > 3:
                feature = feature[..., :3]
            return feature.astype(np.float32), cov
        if arr.ndim == 2:
            feature = percentile_normalize(arr)[..., None]
        elif arr.ndim == 3:
            feature = percentile_normalize(np.real(arr))
        else:
            raise ValueError(f"Unsupported npy shape for {path}: {arr.shape}")
    else:
        image = Image.open(path)
        if image.mode in {"RGB", "RGBA"}:
            feature = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            feature = np.asarray(image.convert("L"), dtype=np.float32)[..., None] / 255.0

    if feature.ndim == 2:
        feature = feature[..., None]
    if feature.shape[-1] == 1:
        seg_feature = np.repeat(feature, 3, axis=-1)
    else:
        seg_feature = feature[..., :3]

    vectors = feature.astype(np.float32)
    q = vectors.shape[-1]
    cov = vectors[..., :, None] * vectors[..., None, :]
    eye = np.eye(q, dtype=np.float32)[None, None, :, :] * 1e-4
    cov = (cov + eye).astype(np.complex64)
    return seg_feature.astype(np.float32), cov


def build_pol_slic_labels(feature: np.ndarray, region_size: int, ruler: float, iterations: int) -> np.ndarray:
    h, w = feature.shape[:2]
    image_u8 = np.clip(feature * 255.0, 0, 255).astype(np.uint8)
    try:
        import cv2

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
            if image_u8.ndim == 2 or image_u8.shape[-1] == 1:
                image_u8 = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
            slic = cv2.ximgproc.createSuperpixelSLIC(
                image_u8,
                algorithm=cv2.ximgproc.SLICO,
                region_size=int(region_size),
                ruler=float(ruler),
            )
            slic.iterate(int(iterations))
            return slic.getLabels().astype(np.int32)
    except Exception:
        pass

    return simple_slic_labels(feature, region_size, ruler, iterations)


def simple_slic_labels(feature: np.ndarray, region_size: int, ruler: float, iterations: int) -> np.ndarray:
    """Small dependency-free SLIC fallback.

    It is less feature-rich than OpenCV ximgproc SLICO, but it adapts cluster
    boundaries to image intensity/polarimetric channels instead of returning a
    plain grid.
    """
    h, w = feature.shape[:2]
    step = max(int(region_size), 2)
    feat = feature.astype(np.float32)
    if feat.ndim == 2:
        feat = feat[..., None]
    feat = feat[..., : min(feat.shape[-1], 3)]
    feat_work = feat * 40.0
    yy, xx = np.indices((h, w), dtype=np.float32)

    centers = []
    for cy in range(step // 2, h, step):
        for cx in range(step // 2, w, step):
            centers.append([float(cy), float(cx), *feat_work[cy, cx].tolist()])
    if not centers:
        return np.zeros((h, w), dtype=np.int32)
    centers_arr = np.asarray(centers, dtype=np.float32)
    n_centers = centers_arr.shape[0]
    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)
    spatial_weight = (float(ruler) / float(step)) ** 2

    for _ in range(max(int(iterations), 1)):
        distances.fill(np.inf)
        labels.fill(-1)
        for cid, center in enumerate(centers_arr):
            cy, cx = center[:2]
            y1 = max(int(cy - step), 0)
            y2 = min(int(cy + step) + 1, h)
            x1 = max(int(cx - step), 0)
            x2 = min(int(cx + step) + 1, w)
            local_feat = feat_work[y1:y2, x1:x2]
            color_dist = ((local_feat - center[2:]) ** 2).sum(axis=-1)
            spatial_dist = (yy[y1:y2, x1:x2] - cy) ** 2 + (xx[y1:y2, x1:x2] - cx) ** 2
            dist = color_dist + spatial_weight * spatial_dist
            update = dist < distances[y1:y2, x1:x2]
            distances[y1:y2, x1:x2][update] = dist[update]
            labels[y1:y2, x1:x2][update] = cid

        for cid in range(n_centers):
            mask = labels == cid
            if not mask.any():
                continue
            centers_arr[cid, 0] = yy[mask].mean()
            centers_arr[cid, 1] = xx[mask].mean()
            centers_arr[cid, 2:] = feat_work[mask].mean(axis=0)

    missing = labels < 0
    if missing.any():
        labels[missing] = 0
    return labels.astype(np.int32)


def covariance_mean(cov: np.ndarray, mask: np.ndarray, eps: float) -> np.ndarray:
    mat = cov[mask].mean(axis=0)
    q = mat.shape[0]
    return mat + np.eye(q, dtype=np.complex64) * eps


def wishart_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    q = c1.shape[0]
    inv1 = np.linalg.pinv(c1)
    inv2 = np.linalg.pinv(c2)
    val = 0.5 * (np.trace(inv1 @ c2) + np.trace(inv2 @ c1)) - q
    val = float(np.real(val))
    return max(val, 0.0)


def normalize_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def region_adjacency(labels: np.ndarray) -> Dict[int, set]:
    adj: Dict[int, set] = defaultdict(set)
    left = labels[:, :-1]
    right = labels[:, 1:]
    up = labels[:-1, :]
    down = labels[1:, :]
    for a, b in zip(left[left != right].ravel(), right[left != right].ravel()):
        ia, ib = int(a), int(b)
        adj[ia].add(ib)
        adj[ib].add(ia)
    for a, b in zip(up[up != down].ravel(), down[up != down].ravel()):
        ia, ib = int(a), int(b)
        adj[ia].add(ib)
        adj[ib].add(ia)
    return adj


def expand_neighbors(sp_id: int, adj: Dict[int, set], order: int) -> List[int]:
    seen = {sp_id}
    frontier = {sp_id}
    for _ in range(max(order, 1)):
        nxt = set()
        for item in frontier:
            nxt.update(adj.get(item, set()))
        nxt.difference_update(seen)
        seen.update(nxt)
        frontier = nxt
    seen.discard(sp_id)
    return sorted(seen)


def collect_regions(feature: np.ndarray, cov: np.ndarray, labels: np.ndarray, args) -> List[dict]:
    regions: List[dict] = []
    h, w = labels.shape
    intensity = feature.mean(axis=-1).astype(np.float32)
    for sp_id in np.unique(labels):
        mask = labels == sp_id
        area = int(mask.sum())
        if area < args.min_area or area > args.max_area:
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        bbox_area = float(bw * bh)
        aspect = float(max(bw / max(bh, 1), bh / max(bw, 1)))
        compactness = float(area / max(bbox_area, 1.0))
        vals = intensity[mask]
        region_pixels = feature[mask].reshape(area, -1)
        descriptor = np.concatenate((region_pixels.mean(axis=0),
                                     region_pixels.std(axis=0))).astype(np.float32)
        regions.append(
            {
                "superpixel_id": int(sp_id),
                "mask": mask,
                "area": area,
                "bbox": [x1, y1, x2, y2],
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
                "aspect": aspect,
                "compactness": compactness,
                "region_mean": float(vals.mean()),
                "region_std": float(vals.std()),
                "descriptor": descriptor,
                "cov": covariance_mean(cov, mask, args.cov_eps),
            }
        )
    return regions


def compute_superpixel_responses(regions: List[dict], adj: Dict[int, set], args) -> None:
    by_id = {r["superpixel_id"]: r for r in regions}
    raw_sal = np.zeros(len(regions), dtype=np.float32)

    for idx, region in enumerate(regions):
        neighbors = [by_id[n] for n in expand_neighbors(region["superpixel_id"], adj, args.neighbor_order) if n in by_id]
        if not neighbors:
            raw_sal[idx] = 0.0
            continue
        total = 0.0
        for nb in neighbors:
            spatial = (region["cx"] - nb["cx"]) ** 2 + (region["cy"] - nb["cy"]) ** 2
            weight = math.exp(-spatial / max(args.spatial_sigma * args.spatial_sigma, 1e-6))
            dist = wishart_distance(region["cov"], nb["cov"])
            total += weight * (1.0 - math.exp(-dist / max(args.superpixel_scale * args.superpixel_scale, 1e-6)))
        raw_sal[idx] = total / max(len(neighbors), 1)

    d_sal = normalize_values(raw_sal)
    for idx, region in enumerate(regions):
        region["d_sal"] = float(d_sal[idx])

    # Eq. (5): reliable local background must be both low-saliency and
    # low-variance.
    sea_pool = [r for r in regions
                if r["d_sal"] < args.saliency_threshold
                and r["region_std"] < args.variance_threshold]
    if not sea_pool:
        sea_pool = sorted(regions, key=lambda x: x["d_sal"])[: max(1, len(regions) // 4)]

    raw_sep = np.zeros(len(regions), dtype=np.float32)
    for idx, region in enumerate(regions):
        neighbors = [by_id[n] for n in expand_neighbors(region["superpixel_id"], adj, args.neighbor_order) if n in by_id]
        sea_neighbors = [nb for nb in neighbors
                         if nb["d_sal"] < args.saliency_threshold
                         and nb["region_std"] < args.variance_threshold]
        reference = sea_neighbors or sea_pool
        if reference:
            descriptors = np.stack([item["descriptor"] for item in reference])
            background_prototype = descriptors.mean(axis=0)
            denominator = float(np.linalg.norm(descriptors.std(axis=0)))
            raw_sep[idx] = float(np.linalg.norm(
                region["descriptor"] - background_prototype)
                / max(denominator, args.cov_eps))
        else:
            raw_sep[idx] = 0.0

    d_sep = normalize_values(raw_sep)
    continuity_raw = np.asarray(
        [
            0.5 * min(r["aspect"] / max(args.continuity_aspect, 1e-6), 1.0)
            + 0.5 * min(r["area"] / max(args.continuity_area, 1), 1.0)
            for r in regions
        ],
        dtype=np.float32,
    )
    continuity = normalize_values(continuity_raw)
    std = normalize_values(np.asarray([r["region_std"] for r in regions], dtype=np.float32))

    for idx, region in enumerate(regions):
        region["d_sep"] = float(d_sep[idx])
        region["continuity"] = float(continuity[idx])
        region["homogeneity"] = float((1.0 - std[idx]) * (1.0 - min(region["d_sal"], 1.0)))
        region["isolation"] = float(region["d_sal"] * (1.0 - min(region["area"] / max(args.continuity_area, 1), 1.0)))


def assign_region_scores(regions: List[dict], args) -> None:
    a_tar = np.zeros(len(regions), dtype=np.float32)
    for idx, region in enumerate(regions):
        d_sal = region["d_sal"]
        d_sep = region["d_sep"]
        h = region["homogeneity"]
        uncertainty = math.exp(-((d_sal * d_sep - args.uncertainty_center) ** 2) / (2.0 * args.uncertainty_sigma**2))
        region["uncertainty"] = float(uncertainty)
        region["a_tar"] = float(d_sal * d_sep)
        region["a_bg"] = float((1.0 - d_sal) * (1.0 - d_sep) * h)
        region["a_hard"] = float(d_sal * (1.0 - d_sep) + args.uncertainty_weight * uncertainty * (1.0 - d_sep))
        a_tar[idx] = region["a_tar"]

    for key in ("a_tar", "a_bg", "a_hard"):
        values = np.asarray([r[key] for r in regions], dtype=np.float32)
        normed = normalize_values(values)
        for idx, region in enumerate(regions):
            region[key] = float(normed[idx])

    for region in regions:
        scores = {
            "target_like": region["a_tar"],
            "background_like": region["a_bg"],
            "hard_background": region["a_hard"],
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ordered[0]
        second_score = ordered[1][1]
        if best_score - second_score < args.assignment_margin:
            label = "uncertain"
        else:
            label = best_name
        region["partition"] = label
        if label == "target_like":
            region["target"] = 1
            region["loss_weight"] = args.target_weight
        elif label == "background_like":
            region["target"] = 0
            region["loss_weight"] = args.background_weight
        elif label == "hard_background":
            region["target"] = 0
            region["loss_weight"] = args.hard_weight
        else:
            region["target"] = -1
            region["loss_weight"] = args.uncertain_weight


def save_label_map(labels: np.ndarray, regions: List[dict], out_path: Path) -> None:
    code = {"uncertain": 0, "background_like": 1, "hard_background": 2, "target_like": 3}
    out = np.zeros(labels.shape, dtype=np.uint8)
    for region in regions:
        out[labels == region["superpixel_id"]] = code.get(region["partition"], 0)
    ensure_dir(out_path.parent)
    Image.fromarray(out, mode="L").save(out_path)


def save_overlay(feature: np.ndarray, labels: np.ndarray, regions: List[dict], out_path: Path) -> None:
    color = {
        "uncertain": np.asarray([180, 180, 180], dtype=np.float32),
        "background_like": np.asarray([60, 120, 255], dtype=np.float32),
        "hard_background": np.asarray([255, 180, 40], dtype=np.float32),
        "target_like": np.asarray([255, 50, 50], dtype=np.float32),
    }
    base = feature
    if base.ndim == 2 or base.shape[-1] == 1:
        base = np.repeat(base[..., :1], 3, axis=-1)
    base = np.clip(base[..., :3] * 255.0, 0, 255).astype(np.float32)
    overlay = base.copy()
    for region in regions:
        mask = labels == region["superpixel_id"]
        overlay[mask] = 0.45 * base[mask] + 0.55 * color.get(region["partition"], color["uncertain"])
    ensure_dir(out_path.parent)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(out_path)


def save_superpixel_boundaries(feature: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    base = feature
    if base.ndim == 2 or base.shape[-1] == 1:
        base = np.repeat(base[..., :1], 3, axis=-1)
    base = np.clip(base[..., :3] * 255.0, 0, 255).astype(np.uint8)

    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]

    out = base.copy()
    out[boundary] = np.asarray([255, 40, 40], dtype=np.uint8)
    ensure_dir(out_path.parent)
    Image.fromarray(out, mode="RGB").save(out_path)


def save_patch_galleries(regions: List[dict], out_dir: Path, per_class: int, patch_size: int, no_text: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for --save-galleries") from exc

    colors = {
        "target_like": "#d62728",
        "background_like": "#1f77b4",
        "hard_background": "#ff9f1a",
        "uncertain": "#7f7f7f",
    }
    score_key = {
        "target_like": "A_tar",
        "background_like": "A_bg",
        "hard_background": "A_hard",
        "uncertain": "U",
    }
    gallery_dir = ensure_dir(out_dir / "patch_galleries")
    image_cache: Dict[str, np.ndarray] = {}

    for label in ("target_like", "background_like", "hard_background", "uncertain"):
        items = [item for item in regions if item["partition"] == label]
        if not items:
            continue
        key = score_key[label]
        items = sorted(items, key=lambda x: x.get("scores", {}).get(key, 0.0), reverse=True)[:per_class]
        cols = min(6, len(items))
        rows = int(math.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.45), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, item in zip(axes, items):
            image_path = item["image_path"]
            if image_path not in image_cache:
                image_cache[image_path] = load_display_image(Path(image_path))
            patch = crop_square(
                image_cache[image_path],
                float(item["cx"]),
                float(item["cy"]),
                max(int(item.get("crop_size", patch_size)), patch_size),
            )
            if patch.ndim == 2:
                ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(patch, 0, 1))
            if not no_text:
                scores = item.get("scores", {})
                ax.set_title(
                    f"{item['image_id']}\n"
                    f"tar={scores.get('A_tar', 0.0):.2f} bg={scores.get('A_bg', 0.0):.2f} "
                    f"hard={scores.get('A_hard', 0.0):.2f}",
                    fontsize=7,
                    color=colors[label],
                )
            ax.set_axis_off()
        for ax in axes[len(items):]:
            ax.set_axis_off()
        if not no_text:
            fig.suptitle(label, color=colors[label], fontsize=12)
        fig.savefig(gallery_dir / f"{label}_gallery.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def process_one(image_id: str, path: Path, args,
                covariance_path: Optional[Path] = None) -> dict:
    feature, cov = load_polsar_like(path)
    if covariance_path is not None:
        _, cov = load_polsar_like(covariance_path)
        if cov.shape[:2] != feature.shape[:2]:
            raise ValueError(
                f"covariance/image shape mismatch for {image_id}: "
                f"{cov.shape[:2]} versus {feature.shape[:2]}")
    labels = build_pol_slic_labels(feature, args.region_size, args.slico_ruler, args.slico_iterations)
    regions = collect_regions(feature, cov, labels, args)
    adj = region_adjacency(labels)
    compute_superpixel_responses(regions, adj, args)
    assign_region_scores(regions, args)

    # SALRP retrieves the prior of the superpixel that actually owns each
    # detector location.  Persist integer IDs instead of approximating a
    # region with a circle around its centroid.
    superpixel_map = (Path(args.out_dir) / "superpixel_labels"
                      / f"{image_id}.npy").resolve()
    ensure_dir(superpixel_map.parent)
    np.save(superpixel_map, labels.astype(np.int32))

    records = []
    for region in regions:
        item = {
            "sample_id": f"{image_id}_{region['superpixel_id']}",
            "image_id": image_id,
            "image_path": str(path),
            "superpixel_map": str(superpixel_map),
            "superpixel_id": int(region["superpixel_id"]),
            "cx": float(region["cx"]),
            "cy": float(region["cy"]),
            "crop_size": int(max(args.patch_size, args.region_context * max(region["bbox"][2] - region["bbox"][0], region["bbox"][3] - region["bbox"][1]))),
            "bbox_xyxy": [int(v) for v in region["bbox"]],
            "initial_partition": region["partition"],
            "partition": region["partition"],
            "source_type": region["partition"] + "_seed",
            "target": int(region["target"]),
            "loss_weight": float(region["loss_weight"]),
            "scores": {
                "D_sal": float(region["d_sal"]),
                "D_sep": float(region["d_sep"]),
                "A_tar": float(region["a_tar"]),
                "A_bg": float(region["a_bg"]),
                "A_hard": float(region["a_hard"]),
                "H": float(region["homogeneity"]),
                "U": float(region["uncertainty"]),
            },
            "region_stats": {
                "area": float(region["area"]),
                "aspect": float(region["aspect"]),
                "compactness": float(region["compactness"]),
                "isolation": float(region["isolation"]),
                "continuity": float(region["continuity"]),
                "region_mean": float(region["region_mean"]),
                "region_std": float(region["region_std"]),
            },
            "stat_feature": [
                float(region["d_sal"]), float(region["d_sep"]),
                float(region["a_tar"]), float(region["a_bg"]),
                float(region["a_hard"]), float(region["homogeneity"]),
                float(region["uncertainty"]), float(region["compactness"]),
                float(region["continuity"]), float(region["region_mean"]),
                float(region["region_std"]),
            ],
        }
        records.append(item)

    if args.save_maps:
        save_label_map(labels, regions, Path(args.out_dir) / "maps" / f"{image_id}_partition.png")
        save_overlay(feature, labels, regions, Path(args.out_dir) / "overlays" / f"{image_id}_overlay.png")
        save_superpixel_boundaries(feature, labels, Path(args.out_dir) / "superpixels" / f"{image_id}_superpixel.png")

    return {
        "image_id": image_id,
        "image_path": str(path),
        "height": int(labels.shape[0]),
        "width": int(labels.shape[1]),
        "regions": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PolSAR superpixel-level SAL priors.")
    parser.add_argument("--image-dirs", nargs="+", required=True, help="Input image files or directories.")
    parser.add_argument(
        "--covariance-dirs", nargs="+", default=None,
        help=("Optional matching HxWxqxq complex .npy covariance matrices. "
              "Required for the exact revised-Wishart path; if omitted, an "
              "outer-product covariance is derived from Xpol."))
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--out-json", default="polsar_superpixel_sal.json")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--region-size", type=int, default=24)
    parser.add_argument("--slico-ruler", type=float, default=10.0)
    parser.add_argument("--slico-iterations", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--region-context", type=float, default=2.0)
    parser.add_argument("--cov-eps", type=float, default=1e-3)
    parser.add_argument("--neighbor-order", type=int, default=1)
    parser.add_argument("--spatial-sigma", type=float, default=64.0)
    parser.add_argument("--superpixel-scale", type=float, default=24.0)
    parser.add_argument("--saliency-threshold", type=float, default=0.35,
                        help="tau_sal in Eq. (5), after robust normalization.")
    parser.add_argument("--variance-threshold", type=float, default=0.08,
                        help="tau_var in Eq. (5), for normalized Xpol.")
    parser.add_argument("--continuity-aspect", type=float, default=6.0)
    parser.add_argument("--continuity-area", type=int, default=9216)
    parser.add_argument("--uncertainty-center", type=float, default=0.45)
    parser.add_argument("--uncertainty-sigma", type=float, default=0.18)
    parser.add_argument("--uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--assignment-margin", type=float, default=0.08)
    parser.add_argument("--target-weight", type=float, default=1.15)
    parser.add_argument("--background-weight", type=float, default=0.75)
    parser.add_argument("--hard-weight", type=float, default=0.50)
    parser.add_argument("--uncertain-weight", type=float, default=0.0)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--save-galleries", action="store_true", help="Save class-wise patch gallery images.")
    parser.add_argument("--gallery-per-class", type=int, default=24)
    parser.add_argument("--gallery-no-text", action="store_true", help="Do not draw titles or score text on galleries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    inputs = sorted(list_inputs(args.image_dirs).items())
    covariance_inputs = (list_inputs(args.covariance_dirs)
                         if args.covariance_dirs else {})
    if args.max_images is not None:
        inputs = inputs[: args.max_images]
    if not inputs:
        raise FileNotFoundError("No input images were found.")
    if args.covariance_dirs:
        missing = [image_id for image_id, _ in inputs
                   if image_id not in covariance_inputs]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing covariance matrices for {len(missing)} images: {preview}")
    else:
        print("[warning] --covariance-dirs omitted; using the documented "
              "Xpol outer-product fallback")

    images = []
    all_regions = []
    for idx, (image_id, path) in enumerate(inputs, start=1):
        print(f"[{idx}/{len(inputs)}] processing {image_id}")
        payload = process_one(image_id, path, args,
                              covariance_inputs.get(image_id))
        images.append({k: payload[k] for k in ("image_id", "image_path", "height", "width")})
        all_regions.extend(payload["regions"])

    summary = defaultdict(int)
    for item in all_regions:
        summary[item["partition"]] += 1

    output = {
        "meta": {
            "method": "polsar_superpixel_sal",
            "image_count": len(images),
            "region_count": len(all_regions),
            "summary": dict(summary),
            "region_size": args.region_size,
            "assignment_margin": args.assignment_margin,
        },
        "images": images,
        "regions": all_regions,
    }
    out_json = out_dir / args.out_json
    write_json(output, out_json)
    if args.save_galleries:
        save_patch_galleries(all_regions, out_dir, args.gallery_per_class, args.patch_size, args.gallery_no_text)
    print(f"wrote {len(all_regions)} regions to {out_json}")
    print(dict(summary))


if __name__ == "__main__":
    main()
