from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from common import ensure_dir, l2_normalize, read_json, write_json
from dataset import SarPatchDataset
from pclnet_encoder import ContrastiveModel


def parse_args():
    parser = argparse.ArgumentParser(description="Extract normalized embeddings for contrastive SAL samples.")
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-npy", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_args = ckpt.get("args", {})
    model = ContrastiveModel(
        in_channels=int(model_args.get("in_channels", 3)),
        feat_dim=int(model_args.get("feat_dim", 128)),
        projection_dim=int(model_args.get("projection_dim", 64)),
        queue_size=int(model_args.get("queue_size", 8192)),
        momentum=float(model_args.get("momentum", 0.999)),
        temperature=float(model_args.get("temperature", 0.07)),
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dataset = SarPatchDataset(args.samples_json, patch_size=args.patch_size, augment=False, return_pair=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    embeddings = np.zeros((len(dataset), int(model_args.get("feat_dim", 128))), dtype=np.float32)
    with torch.no_grad():
        for patch, idx in loader:
            patch = patch.to(args.device, non_blocking=True)
            emb = model.encoder_q.forward_features(patch).cpu().numpy().astype(np.float32)
            embeddings[idx.numpy()] = emb
    embeddings = l2_normalize(embeddings).astype(np.float32)
    ensure_dir(Path(args.out_npy).parent)
    np.save(args.out_npy, embeddings)
    print(f"wrote embeddings {embeddings.shape} to {args.out_npy}")

    if args.out_json:
        payload = read_json(args.samples_json)
        if isinstance(payload, dict):
            samples = payload.get("samples", payload.get("regions", []))
        else:
            samples = payload
        for i, sample in enumerate(samples):
            sample["embedding_index"] = i
        if isinstance(payload, dict):
            payload.pop("regions", None)
            payload["samples"] = samples
            payload.setdefault("meta", {})["embedding_file"] = str(args.out_npy)
        else:
            payload = {"meta": {"embedding_file": str(args.out_npy)}, "samples": samples}
        write_json(payload, args.out_json)
        print(f"wrote indexed samples to {args.out_json}")


if __name__ == "__main__":
    main()
