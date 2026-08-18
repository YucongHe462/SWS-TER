"""Build per-pixel Pauli coherency matrices for ACPC Eq. (4).

The input arrays contain complex single-look scattering coefficients with a
shared stem and shape.  Monostatic reciprocity is assumed, so one cross-pol
channel (HV) is sufficient.  The Pauli vector is

    k = [S_hh + S_vv, S_hh - S_vv, 2 S_hv] / sqrt(2)

and the saved matrix is ``k k^H`` at every pixel.  Region averaging and the
diagonal numerical regularizer are applied later by ``build_regions.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def discover(path: str):
    root = Path(path)
    if root.is_file():
        return {root.stem: root}
    return {item.stem: item for item in root.glob('*.npy') if item.is_file()}


def pauli_coherency(hh: np.ndarray, hv: np.ndarray,
                    vv: np.ndarray) -> np.ndarray:
    if hh.shape != hv.shape or hh.shape != vv.shape:
        raise ValueError(
            f'scattering channel shapes differ: {hh.shape}, {hv.shape}, {vv.shape}')
    if hh.ndim != 2:
        raise ValueError(f'each scattering channel must be HxW, got {hh.shape}')
    scale = np.sqrt(2.0)
    pauli = np.stack(((hh + vv) / scale,
                      (hh - vv) / scale,
                      (2.0 * hv) / scale), axis=-1).astype(np.complex64)
    return (pauli[..., :, None]
            * pauli[..., None, :].conj()).astype(np.complex64)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build HxWx3x3 complex Pauli coherency arrays.')
    parser.add_argument('--hh', required=True,
                        help='Complex S_hh .npy file or directory.')
    parser.add_argument('--hv', required=True,
                        help='Complex S_hv .npy file or directory.')
    parser.add_argument('--vv', required=True,
                        help='Complex S_vv .npy file or directory.')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sources = [discover(args.hh), discover(args.hv), discover(args.vv)]
    identifiers = sorted(set(sources[0]) & set(sources[1]) & set(sources[2]))
    if not identifiers:
        raise FileNotFoundError('No matching HH/HV/VV NumPy stems were found.')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for identifier in identifiers:
        hh, hv, vv = (np.load(source[identifier]) for source in sources)
        np.save(output / f'{identifier}.npy', pauli_coherency(hh, hv, vv))
    print(f'wrote {len(identifiers)} covariance arrays to {output}')


if __name__ == '__main__':
    main()
