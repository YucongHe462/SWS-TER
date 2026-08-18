from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from common import normalize_features, read_json, write_json


def simple_kmeans(features: np.ndarray, k: int, max_iter: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    k = min(k, n)
    centers = features[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = features[mask].mean(axis=0)
            else:
                centers[cluster_id] = features[rng.integers(0, n)]
    return labels


def kmeans_labels(features: np.ndarray, k: int, max_iter: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(n_clusters=min(k, len(features)), max_iter=max_iter, random_state=seed, n_init="auto")
        return model.fit_predict(features).astype(np.int64)
    except Exception:
        return simple_kmeans(features, k, max_iter, seed)


def prune_cluster(indices: np.ndarray, features: np.ndarray, retain: int, gamma: float, seed: int) -> List[int]:
    if len(indices) <= retain:
        return indices.tolist()
    rng = random.Random(seed)
    keep = indices.tolist()
    cluster_features = features[keep]
    dist = ((cluster_features[:, None, :] - cluster_features[None, :, :]) ** 2).sum(axis=2)
    affinity = np.exp(-dist / (2.0 * gamma * gamma))
    np.fill_diagonal(affinity, -np.inf)
    active = np.ones(len(keep), dtype=bool)
    while int(active.sum()) > retain:
        masked = affinity.copy()
        masked[~active, :] = -np.inf
        masked[:, ~active] = -np.inf
        flat = int(np.argmax(masked))
        p, q = divmod(flat, masked.shape[1])
        if not np.isfinite(masked[p, q]):
            remaining = np.where(active)[0].tolist()
            active[rng.choice(remaining)] = False
            continue
        active[rng.choice([p, q])] = False
    return [keep[i] for i in np.where(active)[0].tolist()]


def parse_args():
    parser = argparse.ArgumentParser(description="PCLNet-style diversity stimulation using SAR statistics + KMeans.")
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--num-clusters", type=int, default=128)
    parser.add_argument("--retain-per-cluster", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--keep-positive-anchors", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = read_json(args.samples_json)
    if isinstance(payload, dict):
        samples = payload.get("samples", payload.get("regions", []))
    else:
        samples = payload
    features = np.asarray([s.get("stat_feature", []) for s in samples], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(samples) or features.shape[1] == 0:
        raise ValueError("samples must contain non-empty stat_feature vectors")
    features = normalize_features(features)
    labels = kmeans_labels(features, args.num_clusters, args.max_iter, args.seed)

    selected = set()
    for cluster_id in sorted(set(labels.tolist())):
        idx = np.where(labels == cluster_id)[0]
        retain = args.retain_per_cluster
        chosen = prune_cluster(idx, features, retain, args.gamma, args.seed + int(cluster_id))
        selected.update(chosen)

    if args.keep_positive_anchors:
        for i, sample in enumerate(samples):
            if sample.get("source_type") in {
                    "target_like_seed", "background_like_seed",
                    "hard_background_seed"}:
                selected.add(i)

    selected_idx = sorted(selected)
    new_samples = []
    for new_i, old_i in enumerate(selected_idx):
        sample = dict(samples[old_i])
        sample["diversity_cluster"] = int(labels[old_i])
        sample["original_sample_index"] = int(old_i)
        sample["diversity_index"] = new_i
        new_samples.append(sample)

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    meta.update(
        {
            "source_samples_json": str(args.samples_json),
            "diversity_stimulation": {
                "num_clusters": args.num_clusters,
                "retain_per_cluster": args.retain_per_cluster,
                "gamma": args.gamma,
                "input_count": len(samples),
                "output_count": len(new_samples),
                "cluster_histogram": dict(Counter(labels.tolist())),
            },
        }
    )
    write_json({"meta": meta, "samples": new_samples}, args.out_json)
    print(f"selected {len(new_samples)}/{len(samples)} samples -> {args.out_json}")


if __name__ == "__main__":
    main()
