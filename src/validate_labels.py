"""Строгая проверка YOLO detection, без автоматического исправления."""
import math
from pathlib import Path

from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_labels(text, num_classes):
    boxes, errors = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('expected 5 fields')
            cls, x, y, w, h = map(float, fields)
            if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
                raise ValueError('non-finite value')
            if not cls.is_integer() or not 0 <= cls < num_classes:
                raise ValueError('unknown class')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('coordinates outside YOLO range')
            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                raise ValueError('box crosses image boundary')
            box = (int(cls), x, y, w, h)
            if box in boxes:
                raise ValueError('duplicate box')
            boxes.append(box)
        except ValueError as exc:
            errors.append(f'line {number}: {exc}')
    return boxes, errors


def inspect_dataset(root, names):
    root = Path(root)
    records = []
    images = sorted(p for p in (root / 'images').iterdir() if p.suffix.lower() in EXTENSIONS)
    stems = set()
    for image in images:
        errors = []
        if image.stem in stems:
            errors.append('duplicate image stem')
        stems.add(image.stem)
        try:
            with Image.open(image) as im:
                im.verify()
            with Image.open(image) as im:
                im.load()
        except (OSError, ValueError) as exc:
            errors.append(f'broken image: {exc}')
        label = root / 'labels' / (image.stem + '.txt')
        boxes = []
        if label.exists():
            try:
                boxes, label_errors = parse_labels(label.read_text(encoding='utf-8-sig'), len(names))
                errors.extend(label_errors)
            except UnicodeError as exc:
                errors.append(f'invalid label encoding: {exc}')
        else:
            errors.append('missing annotation')
        records.append({'image': image.name, 'boxes': boxes, 'errors': errors})
    orphans = sorted(p.name for p in (root / 'labels').glob('*.txt') if p.stem not in stems)
    return records, orphans
