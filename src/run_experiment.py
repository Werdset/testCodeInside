import argparse
import importlib.metadata
import json
import platform
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from .tensorboard_logger import ExperimentLogger
from .utils import ROOT, code_version, dump_json, sha256, source_hash
from .validate_labels import EXTENSIONS, parse_labels


def check_data(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    base = (path.parent / data.get('path', '.')).resolve()
    names = data['names']
    if isinstance(names, dict) and set(names) != set(range(len(names))):
        raise ValueError('Class IDs must be consecutive, starting at zero')
    digest = __import__('hashlib').sha256()
    digest.update(path.read_bytes())
    counts = {}
    all_hashes = {}
    for split in ('train', 'val', 'test'):
        folder = base / data[split]
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'Empty split: {split}')
        for image in images:
            label = base / 'labels' / split / (image.stem + '.txt')
            _, errors = parse_labels(label.read_text(encoding='utf-8-sig'), len(names))
            if errors:
                raise ValueError(f'{label.name}: {errors}')
            image_hash = sha256(image)
            if image_hash in all_hashes and all_hashes[image_hash] != split:
                raise ValueError('Image leakage across splits')
            all_hashes[image_hash] = split
            for item in (image, label):
                digest.update(item.relative_to(base).as_posix().encode('utf-8'))
                digest.update(bytes.fromhex(sha256(item)))
        counts[split] = len(images)
    return data, base, digest.hexdigest(), counts


def run(args):
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.device is not None:
        config['device'] = args.device
    if int(config['epochs']) < 1:
        raise ValueError('epochs must be positive')
    for required in ('save', 'val', 'plots'):
        if not config.get(required, True):
            raise ValueError(f'{required} must remain enabled for this experiment')
    data, base, dataset_hash, counts = check_data(args.data)
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory = args.runs / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    directory.mkdir(parents=True)
    metadata = {
        'mode': 'dry-run' if args.dry_run else 'train', 'status': 'running',
        'config': config, 'dataset_sha256': dataset_hash, 'images': counts,
        'code': code_version(), 'source_sha256': source_hash(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'torchvision', 'ultralytics', 'tensorboard', 'numpy', 'Pillow', 'PyYAML')},
    }
    dump_json(directory / 'metadata.json', metadata)
    (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (directory / 'dataset.yaml').write_text(args.data.read_text(encoding='utf-8'), encoding='utf-8')
    logger = ExperimentLogger(directory / 'tensorboard')
    logger.text('run/metadata', '```json\n' + json.dumps(metadata, ensure_ascii=False, indent=2) + '\n```')
    try:
        logger.scalars({f'data/{split}_images': count for split, count in counts.items()}, 0)
        if args.dry_run:
            # Infrastructure check only: no invented losses, metrics or predictions.
            logger.text('run/status', 'Dry-run: dataset and environment verified; training was not executed.')
        else:
            os.environ.setdefault('YOLO_CONFIG_DIR', str((args.runs / '.ultralytics').resolve()))
            os.environ.setdefault('MPLCONFIGDIR', str((args.runs / '.matplotlib').resolve()))
            from ultralytics import YOLO, settings
            settings.update({'tensorboard': False, 'wandb': False, 'mlflow': False, 'comet': False})
            train_config = dict(config)
            model_config = train_config.pop('model')
            model = YOLO(model_config)
            model.add_callback('on_fit_epoch_end', logger.epoch_end)
            # Runtime absolute paths never enter versioned configs or dataset YAML.
            resolved = dict(data)
            resolved['path'] = str(base)
            with tempfile.TemporaryDirectory(prefix='sts_yolo_') as temp:
                runtime_yaml = Path(temp) / 'dataset.yaml'
                runtime_yaml.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding='utf-8')
                model.train(data=str(runtime_yaml), project=str(directory.resolve()), name='train',
                            exist_ok=False, **train_config)
                training_dir = directory / 'train'
                logger.images(training_dir, 'train', int(config['epochs']))
                best = training_dir / 'weights' / 'best.pt'
                if not best.exists():
                    raise RuntimeError('Training did not produce best.pt')
                best_model = YOLO(str(best))
                metrics = best_model.val(data=str(runtime_yaml), split='test',
                                         imgsz=config['imgsz'], batch=config['batch'],
                                         device=config['device'], workers=0, plots=True,
                                         project=str(directory.resolve()), name='test')
                test_metrics = {f'test/{k}': float(v) for k, v in metrics.results_dict.items()}
                logger.scalars(test_metrics, int(config['epochs']))
                dump_json(directory / 'test_metrics.json', test_metrics)
                prediction_dir = directory / 'predictions'
                prediction_dir.mkdir()
                for image in sorted((base / data['test']).iterdir()):
                    if image.suffix.lower() in EXTENSIONS:
                        result = best_model.predict(str(image), imgsz=config['imgsz'],
                                                    device=config['device'], conf=0.001, verbose=False)[0]
                        result.save(filename=str(prediction_dir / (image.stem + '.jpg')))
                logger.images(directory / 'test', 'test', int(config['epochs']))
                logger.images(prediction_dir, 'predictions', int(config['epochs']))
        metadata['status'] = 'completed'
        logger.text('run/status', metadata['mode'] + ' completed')
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        logger.text('run/error', metadata['error'])
        raise
    finally:
        dump_json(directory / 'metadata.json', metadata)
        logger.close()
        print(f'Run artifacts: {directory}')
    return directory


def main():
    parser = argparse.ArgumentParser(description='Local YOLO experiment with TensorBoard')
    parser.add_argument('--config', type=Path, default=Path('configs/yolo.yaml'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/dataset.yaml'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
