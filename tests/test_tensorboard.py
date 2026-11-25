from types import SimpleNamespace

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.tensorboard_logger import ExperimentLogger


def test_epoch_metrics_written_once(tmp_path):
    logger = ExperimentLogger(tmp_path)
    trainer = SimpleNamespace(
        epoch=0, metrics={'metrics/mAP50(B)': 0.25, 'val/box_loss': 1.5},
        tloss=[2.0], lr={'lr/pg0': 0.001},
        label_loss_items=lambda values, prefix: {prefix + '/box_loss': values[0]},
    )
    logger.epoch_end(trainer)
    logger.epoch_end(trainer)
    logger.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    values = events.Scalars('metrics/mAP50(B)')
    assert len(values) == 1
    assert values[0].step == 1
    assert values[0].value == pytest.approx(0.25)
    assert events.Scalars('train/box_loss')[0].value == 2.0


def test_nonfinite_metric_fails(tmp_path):
    logger = ExperimentLogger(tmp_path)
    try:
        with pytest.raises(ValueError, match='Non-finite'):
            logger.scalars({'loss': float('nan')}, 1)
    finally:
        logger.close()
