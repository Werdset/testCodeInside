# Тестовое задание: CV-эксперимент для STS

## Цель

Подготовить небольшую выборку данных и реализовать воспроизводимый эксперимент детекции объектов на базе Ultralytics YOLO с логированием в TensorBoard.

Оцениваются подготовка данных, инженерная аккуратность, воспроизводимость и структура эксперимента. Итоговый `mAP` не является главным критерием.

Входная мини-выборка находится в `test_data/traffic_v5_mini/`. В ней есть корректные, неполные и отсутствующие аннотации; нужно проверить все изображения и подготовить исправленную копию.

## 1. Подготовка данных

Строго по приложенной инструкции:

- проверить изображения и YOLO-разметку;
- выявить битые файлы, ошибки координат и неизвестные классы;
- сформировать `train/val/test`;
- создать `dataset.yaml`;
- подготовить краткий отчёт со статистикой;
- сохранить изображения с визуализацией bounding box.

Для визуальной проверки и доразметки можно использовать [CVAT](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/): импортировать выборку в формате Ultralytics YOLO, проверить или исправить боксы и экспортировать результат обратно. После экспорта обязательно повторно запустить собственную проверку разметки и сверить соответствие ID классов.

Пример запуска:

```bash
python -m src.prepare_dataset \
  --input test_data/traffic_v5_mini \
  --output data/processed \
  --seed 42
```

Результаты:

```text
data/processed/
reports/dataset_report.md
reports/samples/
reports/manifest.sha256
```

## 2. Эксперимент YOLO + TensorBoard

Написать запускаемый код эксперимента на базе Ultralytics YOLO.

Обязательно:

- запуск из командной строки;
- конфигурация через YAML или аргументы;
- создание отдельной папки запуска в `runs/`;
- логирование параметров, версии кода и датасета;
- логирование `loss`, `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95`;
- confusion matrix и примеры предсказаний;
- сохранение checkpoint при обучении;
- возможность открыть результаты через TensorBoard.

Полное обучение не требуется. Достаточно dry-run и короткого запуска на несколько эпох, если позволяет железо.

```bash
python -m src.run_experiment \
  --config configs/yolo.yaml \
  --data data/processed/dataset.yaml \
  --dry-run
```

Сервер или аккаунт не требуются. Логи должны сохраняться локально через `SummaryWriter` в папку `runs/`.

В README указать команду просмотра результатов:

```bash
tensorboard --logdir runs
```

## Артефакты

- Git-репозиторий;
- обработанная выборка или manifest для её восстановления;
- отчёт по данным;
- запускаемый код;
- `README.md`;
- папка `runs/` с логами эксперимента;
- описание состава логов TensorBoard.

## Регламент Git

- полный датасет, веса, логи и секреты не коммитить; предоставленную мини-выборку `test_data/traffic_v5_mini/` включить в проект;
- не использовать абсолютные пути;
- зафиксировать версии зависимостей;
- сохранять логичную историю коммитов;
- использовать сообщения в формате:

```text
feat: add dataset validation
feat: add tensorboard experiment
docs: add run instructions
```

В `README.md` должны быть команды установки, запуска TensorBoard, подготовки данных и запуска эксперимента из чистого окружения.

## Структура репозитория

```text
sts-cv-test/
├── README.md
├── requirements.txt
├── .gitignore
├── test_data/
│   └── traffic_v5_mini/
│       ├── README.md
│       ├── classes.txt
│       ├── images/
│       └── labels/
├── configs/
│   └── yolo.yaml
├── src/
│   ├── __init__.py
│   ├── prepare_dataset.py
│   ├── validate_labels.py
│   ├── tensorboard_logger.py
│   ├── run_experiment.py
│   └── utils.py
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── reports/
│   ├── dataset_report.md
│   ├── manifest.sha256
│   └── samples/
│       └── .gitkeep
├── runs/
│   └── .gitkeep
└── tests/
    ├── test_labels.py
    └── test_dataset_split.py
```
