from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import l2_normalize, read_json, write_json


TARGET_TYPES = {"target_like_seed"}
BACKGROUND_TYPES = {"background_like_seed", "background_seed"}
HARD_BACKGROUND_TYPES = {"hard_background_seed"}


def mean_topk_similarity(emb: np.ndarray, anchor_idx: np.ndarray, topk: int, chunk_size: int = 2048) -> np.ndarray:
    out = np.zeros((emb.shape[0],), dtype=np.float32)
    if len(anchor_idx) == 0:
        return out
    anchors = emb[anchor_idx]
    k = min(topk, len(anchor_idx))
    for start in range(0, emb.shape[0], chunk_size):
        end = min(emb.shape[0], start + chunk_size)
        sims = emb[start:end] @ anchors.T
        if k == sims.shape[1]:
            vals = np.sort(sims, axis=1)[:, -k:]
        else:
            vals = np.partition(sims, kth=sims.shape[1] - k, axis=1)[:, -k:]
        out[start:end] = vals.mean(axis=1)
    return out


def q(values: np.ndarray, quantile: float, default: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.quantile(values, quantile))


def parse_args():
    parser = argparse.ArgumentParser(description="Partition unsupervised superpixel samples using contrastive SAL scores.")
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--anchor-topk", type=int, default=8)
    parser.add_argument("--q-target", type=float, default=0.78)
    parser.add_argument("--q-background", type=float, default=0.72)
    parser.add_argument("--q-hard", type=float, default=0.72)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--sal-low", type=float, default=0.30)
    parser.add_argument("--w-sal", type=float, default=0.30)
    parser.add_argument("--w-target", type=float, default=0.45)
    parser.add_argument("--w-background", type=float, default=0.35)
    parser.add_argument("--w-hard", type=float, default=0.45)
    parser.add_argument("--target-like-weight", type=float, default=1.15)
    parser.add_argument("--background-like-weight", type=float, default=0.75)
    parser.add_argument("--hard-background-weight", type=float, default=0.5)
    parser.add_argument("--uncertain-weight", type=float, default=0.85)
    parser.add_argument("--probability-temperature", type=float, default=0.25)
    parser.add_argument("--lambda-target", type=float, default=1.05)
    parser.add_argument("--lambda-background", type=float, default=0.95)
    parser.add_argument("--lambda-hard", type=float, default=0.85)
    parser.add_argument("--weight-min", type=float, default=0.05)
    parser.add_argument("--weight-max", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = read_json(args.samples_json)
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    emb = l2_normalize(np.load(args.embeddings).astype(np.float32))
    if emb.shape[0] != len(samples):
        raise ValueError(f"embedding count {emb.shape[0]} does not match sample count {len(samples)}")

    target_anchor = np.asarray([i for i, s in enumerate(samples) if s.get("source_type") in TARGET_TYPES], dtype=np.int64)
    background_anchor = np.asarray([i for i, s in enumerate(samples) if s.get("source_type") in BACKGROUND_TYPES], dtype=np.int64)
    hard_anchor = np.asarray([i for i, s in enumerate(samples) if s.get("source_type") in HARD_BACKGROUND_TYPES], dtype=np.int64)
    target_rel = mean_topk_similarity(emb, target_anchor, args.anchor_topk)
    background_rel = mean_topk_similarity(emb, background_anchor, args.anchor_topk)
    hard_rel = mean_topk_similarity(emb, hard_anchor, args.anchor_topk)
    sal = np.asarray([float(s.get("scores", {}).get("D_sal", 0.0)) for s in samples], dtype=np.float32)
    compactness = np.asarray([float(s.get("region_stats", {}).get("compactness", 0.0)) for s in samples], dtype=np.float32)
    isolation = np.asarray([float(s.get("region_stats", {}).get("isolation", 0.0)) for s in samples], dtype=np.float32)
    continuity = np.asarray([float(s.get("region_stats", {}).get("continuity", 0.0)) for s in samples], dtype=np.float32)

    target_score = (
        args.w_sal * sal
        + args.w_target * target_rel
        + 0.15 * compactness
        + 0.15 * isolation
        - args.w_background * np.maximum(background_rel, hard_rel)
        - 0.10 * continuity
    )
    background_score = (
        args.w_background * background_rel
        + args.w_sal * (1.0 - sal)
        - args.w_target * target_rel
        - 0.10 * continuity
    )
    hard_score = (
        args.w_hard * hard_rel
        + 0.45 * sal
        + 0.25 * continuity
        - 0.25 * target_rel
    )

    non_anchor = np.ones(len(samples), dtype=bool)
    non_anchor[target_anchor] = False
    t_target = q(target_score[non_anchor], args.q_target, float(target_score.mean()))
    t_background = q(background_score[non_anchor], args.q_background, float(background_score.mean()))
    t_hard = q(hard_score[non_anchor], args.q_hard, float(hard_score.mean()))

    # Convert the three integrated response/similarity scores into continuous
    # region priors.  These are the P_tar/P_bg/P_hard terms consumed by Eq. (32).
    score_matrix = np.stack((target_score, background_score, hard_score), axis=1)
    shifted = (score_matrix - score_matrix.max(axis=1, keepdims=True)) / max(
        args.probability_temperature, 1e-6)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)

    results = []
    for i, sample in enumerate(samples):
        source = sample.get("source_type")
        p_target, p_background, p_hard = [float(value) for value in probabilities[i]]
        reliability = max(p_target, p_background, p_hard)
        names = ("target_like", "background_like", "hard_background")
        order = np.argsort(-probabilities[i])
        if probabilities[i, order[0]] - probabilities[i, order[1]] < args.margin:
            label, target = "uncertain", -1
        else:
            label = names[int(order[0])]
            target = 1 if label == "target_like" else 0
        eq32_weight = np.clip(
            1.0 + args.lambda_background * p_background
            + args.lambda_hard * p_hard - args.lambda_target * p_target,
            args.weight_min, args.weight_max)
        item = {
            "sample_id": sample["sample_id"],
            "image_id": sample["image_id"],
            "superpixel_id": sample.get("superpixel_id", -1),
            "superpixel_map": sample.get("superpixel_map"),
            "cx": sample["cx"],
            "cy": sample["cy"],
            "crop_size": sample["crop_size"],
            "source_type": source,
            "partition": label,
            "target": target,
            "loss_weight": float(eq32_weight),
            "reliability": reliability,
            "priors": {
                "P_tar": p_target,
                "P_bg": p_background,
                "P_hard": p_hard,
            },
            "target_like_score": float(target_score[i]),
            "background_like_score": float(background_score[i]),
            "hard_background_score": float(hard_score[i]),
            "sim_target": float(target_rel[i]),
            "sim_background": float(background_rel[i]),
            "sim_hard_background": float(hard_rel[i]),
            "pos_rel": float(target_rel[i]),
            "neg_rel": float(background_rel[i]),
            "sal": float(sal[i]),
            "compactness_score": float(compactness[i]),
            "isolation_score": float(isolation[i]),
            "continuity_score": float(continuity[i]),
        }
        results.append(item)

    summary = Counter(r["partition"] for r in results)
    out = {
        "meta": {
            "samples_json": str(args.samples_json),
            "embeddings": str(args.embeddings),
            "target_like_prototypes": int(len(target_anchor)),
            "background_like_prototypes": int(len(background_anchor)),
            "hard_background_prototypes": int(len(hard_anchor)),
            "thresholds": {"target_like": t_target, "background_like": t_background, "hard_background": t_hard},
            "summary": dict(summary),
        },
        "partitions": results,
    }
    write_json(out, args.out_json)
    print(f"wrote partition to {args.out_json}")
    print(dict(summary))


if __name__ == "__main__":
    main()
