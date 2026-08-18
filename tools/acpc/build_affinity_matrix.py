from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ensure_dir, l2_normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Build top-k cosine affinity matrix from embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--save-dense", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    emb = l2_normalize(np.load(args.embeddings).astype(np.float32))
    n = emb.shape[0]
    topk = min(args.topk, n)
    indices = np.zeros((n, topk), dtype=np.int64)
    similarities = np.zeros((n, topk), dtype=np.float32)
    affinities = np.zeros((n, topk), dtype=np.float32)
    dense = np.zeros((n, n), dtype=np.float32) if args.save_dense else None
    for start in range(0, n, args.chunk_size):
        end = min(n, start + args.chunk_size)
        sims = emb[start:end] @ emb.T
        if dense is not None:
            dense[start:end] = sims
        if topk == n:
            idx = np.argsort(-sims, axis=1)[:, :topk]
        else:
            idx = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
            row = np.arange(end - start)[:, None]
            order = np.argsort(-sims[row, idx], axis=1)
            idx = idx[row, order]
        row = np.arange(end - start)[:, None]
        vals = sims[row, idx]
        dist = 1.0 - vals
        aff = np.exp(-(dist * dist) / (2.0 * args.gamma * args.gamma))
        indices[start:end] = idx
        similarities[start:end] = vals.astype(np.float32)
        affinities[start:end] = aff.astype(np.float32)
        print(f"processed {end}/{n}")
    ensure_dir(Path(args.out_npz).parent)
    payload = {
        "indices": indices,
        "similarities": similarities,
        "affinities": affinities,
        "gamma": np.asarray(args.gamma, dtype=np.float32),
    }
    if dense is not None:
        payload["dense_similarity"] = dense
    np.savez_compressed(args.out_npz, **payload)
    print(f"wrote topk affinity matrix to {args.out_npz}")


if __name__ == "__main__":
    main()

