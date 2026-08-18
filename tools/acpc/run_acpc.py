"""Run the complete annotation-free contrastive prior construction stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(arguments):
    print('+', ' '.join(str(item) for item in arguments), flush=True)
    subprocess.run([str(item) for item in arguments], check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PSRPG -> diversity sampling -> SCFE -> continuous priors')
    parser.add_argument('--xpol-dirs', nargs='+', required=True)
    parser.add_argument(
        '--covariance-dirs', nargs='+', default=None,
        help='Matching HxWxqxq complex covariance .npy files for Eq. (4).')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--in-channels', type=int, choices=(1, 3), default=1,
                        help='SCFE input channels; grayscale SAR defaults to 1.')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--region-size', type=int, default=24)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--num-clusters', type=int, default=128)
    parser.add_argument('--retain-per-cluster', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    regions = work / 'regions.json'
    diverse = work / 'regions_diverse.json'
    checkpoint_dir = work / 'scfe'
    checkpoint = checkpoint_dir / 'contrastive_encoder.pth'
    embeddings = work / 'embeddings.npy'
    indexed = work / 'regions_indexed.json'
    priors = work / 'acpc_priors.json'

    command = [sys.executable, here / 'build_regions.py', '--image-dirs',
               *args.xpol_dirs, '--out-dir', work, '--out-json', regions.name,
               '--region-size', args.region_size, '--patch-size',
               args.patch_size, '--save-maps']
    if args.covariance_dirs:
        command.extend(['--covariance-dirs', *args.covariance_dirs])
    if args.max_images is not None:
        command.extend(['--max-images', args.max_images])
    run(command)
    run([sys.executable, here / 'diversity_stimulation.py',
         '--samples-json', regions, '--out-json', diverse,
         '--num-clusters', args.num_clusters, '--retain-per-cluster',
         args.retain_per_cluster])
    run([sys.executable, here / 'train_contrastive.py',
         '--samples-json', diverse, '--out-dir', checkpoint_dir,
         '--patch-size', args.patch_size, '--epochs', args.epochs,
         '--batch-size', args.batch_size, '--num-workers', args.num_workers,
         '--in-channels', args.in_channels, '--device', args.device])
    # The frozen SCFE is evaluated on every region, not only the diverse subset.
    run([sys.executable, here / 'extract_embeddings.py',
         '--samples-json', regions, '--checkpoint', checkpoint,
         '--out-npy', embeddings, '--out-json', indexed,
         '--patch-size', args.patch_size, '--batch-size', args.batch_size,
         '--num-workers', args.num_workers, '--device', args.device])
    run([sys.executable, here / 'sal_threshold_partition.py',
         '--samples-json', indexed, '--embeddings', embeddings,
         '--out-json', priors])
    print(f'ACPC priors: {priors}')


if __name__ == '__main__':
    main()
