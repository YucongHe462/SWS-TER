"""Construct Xpol = [Pd_Y4O, Pv_Y4O, Delta_Sani] (paper Eqs. 1--3).

The manuscript obtains the three physical components with PolSARpro/MATLAB.
This script performs the reproducible alignment, robust normalization and
channel stacking step.  Inputs may be TIFF/PNG images or NumPy arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SUFFIXES = ('.npy', '.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg')


def discover(path: str):
    root = Path(path)
    if root.is_file():
        return {root.stem: root}
    return {item.stem: item for item in root.iterdir()
            if item.is_file() and item.suffix.lower() in SUFFIXES}


def read_component(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.npy':
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path))
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[..., 0]
    if np.iscomplexobj(array):
        array = np.abs(array)
    return array.astype(np.float32)


def normalize(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    return np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description='Build the three-channel Xpol representation.')
    parser.add_argument('--pd', required=True, help='Pd_Y4O file or directory.')
    parser.add_argument('--pv', required=True, help='Pv_Y4O file or directory.')
    parser.add_argument('--delta-sani', required=True, help='Delta_Sani file or directory.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--lower-percentile', type=float, default=2.0)
    parser.add_argument('--upper-percentile', type=float, default=98.0)
    parser.add_argument('--save-float', action='store_true',
                        help='Also save normalized float32 HxWx3 arrays.')
    return parser.parse_args()


def main():
    args = parse_args()
    sources = [discover(args.pd), discover(args.pv), discover(args.delta_sani)]
    identifiers = sorted(set(sources[0]) & set(sources[1]) & set(sources[2]))
    if not identifiers:
        raise FileNotFoundError('No matching Pd/Pv/Delta_Sani stems were found.')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for identifier in identifiers:
        components = [normalize(read_component(source[identifier]),
                                args.lower_percentile,
                                args.upper_percentile) for source in sources]
        if len({component.shape for component in components}) != 1:
            raise ValueError(f'component shape mismatch for {identifier}: '
                             f'{[item.shape for item in components]}')
        xpol = np.stack(components, axis=-1).astype(np.float32)
        Image.fromarray((xpol * 255).round().astype(np.uint8), mode='RGB').save(
            output / f'{identifier}.png')
        if args.save_float:
            np.save(output / f'{identifier}.npy', xpol)
    print(f'wrote {len(identifiers)} Xpol images to {output}')


if __name__ == '__main__':
    main()

