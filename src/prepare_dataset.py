import argparse
import random
import shutil
from collections import Counter
from pathlib import Path
import json

import yaml
from PIL import Image, ImageDraw

from .utils import ROOT, dump_json, sha256
from .validate_labels import inspect_dataset, parse_labels


def split_groups(groups, seed):
    """Greedy 70/15/15 split; frames from one camera scene stay together."""
    buckets = {}
    for name, group in sorted(groups.items()):
        buckets.setdefault(group, []).append(name)
    items = list(buckets.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    target = dict(zip(('train', 'val', 'test'), (0.7, 0.15, 0.15)))
    result = {key: [] for key in target}
    total = len(groups)
    for _, names in items:
        split = max(target, key=lambda key: target[key] * total - len(result[key]))
        result[split].extend(names)
    return {key: sorted(value) for key, value in result.items()}


def visualize(image, boxes, names, output):
    with Image.open(image) as im:
        im = im.convert('RGB')
        im.thumbnail((1280, 960))
        w, h = im.size
        draw = ImageDraw.Draw(im)
        for cls, x, y, bw, bh in boxes:
            color = ('#ff5544', '#33cc66', '#55aaff', '#ffcc33')[cls % 4]
            rect = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
            draw.rectangle(rect, outline=color, width=2)
            draw.text(rect[:2], names[cls], fill=color, stroke_width=1, stroke_fill='black')
        im.save(output)


def prepare(input_dir, output, seed=42, reports=Path('reports')):
    input_dir, output, reports = map(Path, (input_dir, output, reports))
    if output.exists() and any(p.name != '.gitkeep' for p in output.iterdir()):
        raise ValueError(f'Output is not empty: {output}. Choose a new directory.')
    names = (input_dir / 'classes.txt').read_text(encoding='utf-8-sig').splitlines()
    raw, orphans = inspect_dataset(input_dir, names)
    if not raw:
        raise ValueError('No images found')
    fixes = json.loads((ROOT / 'configs/annotation_fixes.json').read_text(encoding='utf-8'))
    groups = json.loads((ROOT / 'configs/scene_groups.json').read_text(encoding='utf-8'))
    if set(groups) != {r['image'] for r in raw}:
        raise ValueError('scene_groups.json must describe every input image exactly once')
    prepared = {}
    for record in raw:
        image = input_dir / 'images' / record['image']
        if any('broken image' in error or 'duplicate image stem' in error for error in record['errors']):
            raise ValueError(f'Cannot prepare {image.name}: {record["errors"]}')
        if image.stem in fixes:
            text = '\n'.join(' '.join(map(str, row)) for row in fixes[image.stem]['boxes']) + '\n'
        else:
            label = input_dir / 'labels' / (image.stem + '.txt')
            if not label.exists():
                raise ValueError(f'Missing reviewed annotation: {image.name}')
            text = label.read_text(encoding='utf-8-sig')
        boxes, errors = parse_labels(text, len(names))
        if errors:
            raise ValueError(f'{image.name}: {errors}')
        prepared[image.name] = (text, boxes)
    # Identical image bytes must not silently enter different splits.
    hashes = {}
    for filename in prepared:
        digest = sha256(input_dir / 'images' / filename)
        if digest in hashes and groups[filename] != groups[hashes[digest]]:
            raise ValueError('Duplicate images assigned to different scene groups')
        hashes[digest] = filename
    splits = split_groups(groups, seed)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'samples').mkdir(exist_ok=True)
    counts = {}
    for split, filenames in splits.items():
        for folder in ('images', 'labels'):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
        count = Counter()
        for filename in filenames:
            image = input_dir / 'images' / filename
            text, boxes = prepared[filename]
            shutil.copyfile(image, output / 'images' / split / filename)
            (output / 'labels' / split / (image.stem + '.txt')).write_text(text, encoding='utf-8')
            visualize(image, boxes, names, reports / 'samples' / (image.stem + '.jpg'))
            count.update(box[0] for box in boxes)
        counts[split] = count
    # Relative to dataset YAML, resolved explicitly by the experiment runner.
    (output / 'dataset.yaml').write_text(yaml.safe_dump({
        'path': '.', 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'names': dict(enumerate(names)),
    }, allow_unicode=True, sort_keys=False), encoding='utf-8')
    dump_json(output / 'split.json', {'seed': seed, 'splits': splits, 'groups': groups})
    dump_json(reports / 'validation.json', {'raw': raw, 'orphan_labels': orphans})
    manifest = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != '.gitkeep':
            manifest.append(f'{sha256(path)}  {path.relative_to(output).as_posix()}')
    (reports / 'manifest.sha256').write_text('\n'.join(manifest) + '\n', encoding='utf-8')
    missing = sum('missing annotation' in r['errors'] for r in raw)
    lines = ['# Отчёт по выборке', '', f'Seed: {seed}. Изображений: {len(raw)}. Отсутствующих исходных аннотаций: {missing}.',
             f'Битых изображений: 0. Осиротевших аннотаций: {len(orphans)}.', '',
             'Все 40 изображений просмотрены. Исправления и причины: `configs/annotation_fixes.json`.',
             'Исходники не менялись. Боксы на границе обрезаны по изображению, неизвестный ID 11 заменён после визуальной проверки.',
             'Ошибки исходной разметки с номерами строк сохранены в `validation.json`; после исправления ошибок валидатора нет.', '',
             '## Разбиение', '', '| Split | Изображения | Объекты |', '| --- | ---: | ---: |']
    for split in splits:
        lines.append(f'| {split} | {len(splits[split])} | {sum(counts[split].values())} |')
    lines += ['', 'Связанные кадры одной сцены объединены до разбиения. Список групп и состав split сохранены в `split.json`.',
              'Пропорции целевые 70/15/15; стратификация по классам не выполняется.', '',
              '## Классы', '', '| ID | Класс | Train | Val | Test |', '| --- | --- | ---: | ---: | ---: |']
    for cls, name in enumerate(names):
        lines.append(f'| {cls} | {name} | ' + ' | '.join(str(counts[s][cls]) for s in splits) + ' |')
    lines += ['', '## Ограничения', '',
              'Разметка добавлена вручную по видимым границам. Далёкие неразличимые силуэты и объекты за масками не размечались.',
              'Для велосипедистов отмечены и велосипед, и человек; пикап относится к car, прицеп отдельно к trailer.',
              'Существующие классы спецтехники сохранены, кроме явно исправленных случаев. Правила truck/lorry в исходной инструкции не уточнены.',
              'Повторная независимая проверка плотных сцен желательна: форматная валидация не доказывает полноту боксов.',
              'Выборка мала, редкие классы могут отсутствовать в val/test. Метрики служат проверкой пайплайна, а не оценкой качества модели.', '',
              '## Воспроизведение', '',
              '`python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42`',
              'Манифест содержит SHA-256 всех файлов результата относительно выходного каталога. Визуализации всех кадров: `reports/samples/`.', '']
    (reports / 'dataset_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Prepared {len(raw)} images: ' + ', '.join(f'{s}={len(v)}' for s, v in splits.items()))


def main():
    parser = argparse.ArgumentParser(description='Validate, apply reviewed annotations and split traffic data')
    parser.add_argument('--input', type=Path, default=Path('test_data/traffic_v5_mini'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.seed, args.reports)


if __name__ == '__main__':
    main()
