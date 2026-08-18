from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from common import ensure_dir
from dataset import SarPatchDataset
from pclnet_encoder import build_model


def save_checkpoint(obj, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        torch.save(obj, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCLNet-style contrastive encoder for SAR patches.")
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=8192)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    dataset = SarPatchDataset(args.samples_json, patch_size=args.patch_size, augment=True, return_pair=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(model.encoder_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for q, k, _ in loader:
            q = q.to(args.device, non_blocking=True)
            k = k.to(args.device, non_blocking=True)
            loss = model(q, k)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * q.shape[0]
            count += q.shape[0]
        avg = total / max(count, 1)
        print(f"epoch {epoch:03d}/{args.epochs:03d} loss={avg:.5f}")
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
        }
        save_checkpoint(ckpt, out_dir / "latest.pth")
    save_checkpoint({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs}, out_dir / "contrastive_encoder.pth")
    print(f"saved checkpoint to {out_dir / 'contrastive_encoder.pth'}")


if __name__ == "__main__":
    main()
