"""Validate a Pd_Pv_Sa sparse semi-supervised split without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SPLITS = {
    'labeled': (
        'semi_ratio_20/sparse_ratio_20/label_image',
        'semi_ratio_20/sparse_ratio_20/label_annotation'),
    'unlabeled': (
        'semi_ratio_20/sparse_ratio_20/unlabel_image',
        'semi_ratio_20/sparse_ratio_20/unlabel_annotation'),
    'test': ('test_image', 'test_annotation'),
}


def parse_annotation(path: Path, expected_class: str) -> int:
    count = 0
    for line_number, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1):
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(
                f'{path}:{line_number}: expected 10 fields, got {len(fields)}')
        try:
            [float(value) for value in fields[:8]]
            int(float(fields[9]))
        except ValueError as error:
            raise ValueError(
                f'{path}:{line_number}: invalid DOTA annotation') from error
        if fields[8] != expected_class:
            raise ValueError(
                f'{path}:{line_number}: expected class {expected_class!r}, '
                f'got {fields[8]!r}')
        count += 1
    return count


def inspect_split(root: Path, name: str, image_rel: str, annotation_rel: str,
                  expected_class: str) -> dict:
    image_dir = root / image_rel
    annotation_dir = root / annotation_rel
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f'{name}: missing {image_dir} or {annotation_dir}')

    images = {item.stem: item for item in image_dir.iterdir()
              if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES}
    annotations = {item.stem: item for item in annotation_dir.glob('*.txt')}
    if not images:
        raise FileNotFoundError(f'{name}: no images in {image_dir}')
    if set(images) != set(annotations):
        missing_annotations = sorted(set(images) - set(annotations))[:5]
        missing_images = sorted(set(annotations) - set(images))[:5]
        raise ValueError(
            f'{name}: image/annotation stems differ; '
            f'missing annotations={missing_annotations}, '
            f'missing images={missing_images}')

    object_count = sum(parse_annotation(path, expected_class)
                       for path in annotations.values())
    empty_count = sum(path.stat().st_size == 0 for path in annotations.values())
    with Image.open(next(iter(images.values()))) as image:
        sample_mode = image.mode
        sample_size = image.size

    if name == 'unlabeled' and empty_count != len(annotations):
        raise ValueError(
            f'unlabeled: expected all annotations to be empty, '
            f'got {len(annotations) - empty_count} non-empty files')
    return {
        'images': len(images),
        'annotations': len(annotations),
        'objects': object_count,
        'empty_annotations': empty_count,
        'sample_mode': sample_mode,
        'sample_size': sample_size,
        'stems': set(images),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate Pd_Pv_Sa images and DOTA-style annotations.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--class-name', default='ship')
    parser.add_argument(
        '--expect-counts', action='store_true',
        help='Require the released 452/1810/565 split counts.')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'data root does not exist: {root}')

    summaries = {
        name: inspect_split(root, name, *paths, args.class_name)
        for name, paths in SPLITS.items()
    }
    overlap = ((summaries['labeled']['stems']
                & summaries['unlabeled']['stems'])
               | (summaries['labeled']['stems']
                  & summaries['test']['stems'])
               | (summaries['unlabeled']['stems']
                  & summaries['test']['stems']))
    if overlap:
        raise ValueError(f'split leakage detected: {sorted(overlap)[:5]}')

    if args.expect_counts:
        expected = {'labeled': 452, 'unlabeled': 1810, 'test': 565}
        actual = {name: summary['images']
                  for name, summary in summaries.items()}
        if actual != expected:
            raise ValueError(f'expected split counts {expected}, got {actual}')

    print(f'Pd_Pv_Sa root: {root}')
    for name, summary in summaries.items():
        print(
            f"{name:9s}: images={summary['images']}, "
            f"objects={summary['objects']}, "
            f"empty_annotations={summary['empty_annotations']}, "
            f"sample={summary['sample_size']} {summary['sample_mode']}")
    print('Pd_Pv_Sa validation passed')


if __name__ == '__main__':
    main()
