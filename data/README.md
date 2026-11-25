# Данные

`raw/` зарезервирован для внешних данных и не входит в Git.

Мини-выборка находится в `test_data/traffic_v5_mini/`. Восстановление обработанной копии:

```bash
python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42
```

`processed/` не коммитится. Контрольные суммы: `reports/manifest.sha256`; строки содержат пути относительно `processed/`.
