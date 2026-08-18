"""Create deterministic sparse-image/sparse-instance G-R dataset splits.

Pascal VOC XML annotations are converted to the DOTA-style text layout read by
the bundled SARDataset.  Rotated objects can be supplied through either a
``robndbox`` element (cx, cy, w, h, angle in radians) or an ordinary VOC
``bndbox``; the latter is exported with angle zero.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def polygon(cx, cy, width, height, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    corners = [(-width / 2, -height / 2), (width / 2, -height / 2),
               (width / 2, height / 2), (-width / 2, height / 2)]
    return [(cx + x * cosine - y * sine, cy + x * sine + y * cosine)
            for x, y in corners]


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    output = []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='ship')
        rotated = obj.find('robndbox')
        if rotated is not None:
            values = [float(rotated.findtext(key))
                      for key in ('cx', 'cy', 'w', 'h', 'angle')]
        else:
            box = obj.find('bndbox')
            if box is None:
                continue
            xmin, ymin, xmax, ymax = [float(box.findtext(key))
                                     for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            values = [(xmin + xmax) / 2, (ymin + ymax) / 2,
                      xmax - xmin, ymax - ymin, 0.0]
        points = polygon(*values)
        output.append(' '.join(f'{coordinate:.3f}' for point in points
                               for coordinate in point) + f' {name} 0')
    return output


def copy_pair(image: Path, xml: Path, image_dir: Path, annotation_dir: Path,
              annotations):
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / image.name)
    (annotation_dir / f'{image.stem}.txt').write_text(
        '\n'.join(annotations) + ('\n' if annotations else ''), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare G-R sparse subsets.')
    parser.add_argument('--images', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--image-ratios', type=float, nargs='+', default=(0.1, 0.2, 0.3))
    parser.add_argument('--instance-ratios', type=float, nargs='+', default=(0.1, 0.2, 0.3))
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    image_root, xml_root, output = Path(args.images), Path(args.annotations), Path(args.out_dir)
    images = sorted(item for item in image_root.iterdir()
                    if item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'})
    pairs = [(image, xml_root / f'{image.stem}.xml') for image in images
             if (xml_root / f'{image.stem}.xml').exists()]
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    test_count = round(len(pairs) * args.test_ratio)
    test, train = pairs[:test_count], pairs[test_count:]
    for image, xml in test:
        copy_pair(image, xml, output / 'test_image', output / 'test_annotation', parse_xml(xml))
    for image_ratio in args.image_ratios:
        labeled_count = round(len(train) * image_ratio)
        labeled = set(index for index in range(labeled_count))
        for instance_ratio in args.instance_ratios:
            root = output / f'semi_ratio_{round(image_ratio * 100)}' / f'sparse_ratio_{round(instance_ratio * 100)}'
            for index, (image, xml) in enumerate(train):
                annotations = parse_xml(xml)
                if index in labeled:
                    keep = max(1, round(len(annotations) * instance_ratio)) if annotations else 0
                    local = random.Random(f'{args.seed}:{image.stem}:{instance_ratio}')
                    selected = local.sample(annotations, keep) if keep else []
                    copy_pair(image, xml, root / 'label_image', root / 'label_annotation', selected)
                else:
                    copy_pair(image, xml, root / 'unlabel_image', root / 'unlabel_annotation', [])
    print(f'prepared {len(train)} train and {len(test)} test images in {output}')


if __name__ == '__main__':
    main()

