from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from common import crop_square, ensure_dir, load_gray, read_json


COLORS = {
    "target_like": "#2ca02c",
    "background_like": "#1f77b4",
    "hard_background": "#d62728",
    "uncertain": "#8a8a8a",
    "positive_anchor": "#2ca02c",
    "positive": "#1f9e4b",
    "negative_anchor": "#1f77b4",
    "negative": "#5dade2",
    "hard_negative": "#d62728",
    "ignore": "#8a8a8a",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offline contrastive SAL partition results.")
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--gallery-per-class", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=96)
    return parser.parse_args()


def draw_overlays(partitions: List[dict], sample_by_id: Dict[str, dict], out_dir: Path, max_images: int) -> None:
    by_image = defaultdict(list)
    for item in partitions:
        by_image[item["image_id"]].append(item)

    overlay_dir = ensure_dir(out_dir / "overlays")
    for image_idx, (image_id, items) in enumerate(sorted(by_image.items())):
        if image_idx >= max_images:
            break
        sample = sample_by_id[items[0]["sample_id"]]
        image = load_gray(sample["image_path"])
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        for label, color in COLORS.items():
            xs = [x["cx"] for x in items if x["partition"] == label]
            ys = [x["cy"] for x in items if x["partition"] == label]
            if xs:
                ax.scatter(xs, ys, s=46 if label not in {"ignore", "uncertain"} else 22, c=color, label=f"{label} ({len(xs)})",
                           marker="x" if label in {"hard_negative", "hard_background"} else "o", linewidths=1.4, alpha=0.9)
        ax.set_title(f"{image_id} partition overlay")
        ax.set_axis_off()
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.savefig(overlay_dir / f"{image_id}_overlay.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def draw_gallery(partitions: List[dict], sample_by_id: Dict[str, dict], out_dir: Path, per_class: int, patch_size: int) -> None:
    gallery_dir = ensure_dir(out_dir / "patch_galleries")
    for label, color in COLORS.items():
        items = [x for x in partitions if x["partition"] == label]
        if not items:
            continue
        items = sorted(items, key=lambda x: x.get("hard_background_score", x.get("hard_negative_score", 0.0)), reverse=True)[:per_class]
        cols = min(6, len(items))
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.45), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, item in zip(axes, items):
            sample = sample_by_id[item["sample_id"]]
            image = load_gray(sample["image_path"])
            patch = crop_square(image, item["cx"], item["cy"], max(int(item["crop_size"]), patch_size))
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"{item['image_id']} {item['source_type']}\n"
                f"ts={item.get('target_like_score', item.get('positive_score', 0.0)):.2f} "
                f"hs={item.get('hard_background_score', item.get('hard_negative_score', 0.0)):.2f}",
                fontsize=7,
                color=color,
            )
            ax.set_axis_off()
        for ax in axes[len(items):]:
            ax.set_axis_off()
        fig.suptitle(label, color=color, fontsize=12)
        fig.savefig(gallery_dir / f"{label}_gallery.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def draw_score_plots(partitions: List[dict], out_dir: Path) -> None:
    score_dir = ensure_dir(out_dir / "scores")
    labels = list(COLORS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    score_names = ["target_like_score", "background_like_score", "hard_background_score"]
    legacy_score_names = ["positive_score", "negative_score", "hard_negative_score"]
    for ax, score_name, legacy_name in zip(axes, score_names, legacy_score_names):
        key = score_name if any(score_name in x for x in partitions) else legacy_name
        data = [[x[key] for x in partitions if x["partition"] == label and key in x] for label in labels]
        ax.boxplot(data, labels=[l.replace("_", "\n") for l in labels], showfliers=False)
        ax.set_title(key)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(score_dir / "score_boxplots.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for label, color in COLORS.items():
        xs = [x.get("sim_target", x.get("pos_rel", 0.0)) for x in partitions if x["partition"] == label]
        ys = [x.get("sim_background", x.get("neg_rel", 0.0)) for x in partitions if x["partition"] == label]
        if xs:
            ax.scatter(xs, ys, s=24, c=color, label=label, alpha=0.8)
    ax.set_xlabel("sim_target")
    ax.set_ylabel("sim_background")
    ax.set_title("Contrastive relation separation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(score_dir / "pos_neg_relation_scatter.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    partition_payload = read_json(args.partition_json)
    sample_payload = read_json(args.samples_json)
    partitions = partition_payload["partitions"]
    samples = sample_payload["samples"] if isinstance(sample_payload, dict) and "samples" in sample_payload else sample_payload
    sample_by_id = {sample["sample_id"]: sample for sample in samples}

    draw_overlays(partitions, sample_by_id, out_dir, args.max_images)
    draw_gallery(partitions, sample_by_id, out_dir, args.gallery_per_class, args.patch_size)
    draw_score_plots(partitions, out_dir)
    print(f"wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
