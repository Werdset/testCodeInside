import math
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, directory):
        self.writer = SummaryWriter(str(directory))
        self.last_epoch = -1

    def text(self, tag, value):
        self.writer.add_text(tag, value, 0)

    def scalars(self, values, step):
        for key, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'Non-finite metric {key}: {value}')
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def epoch_end(self, trainer):
        # Ultralytics calls this again when validating best.pt after training.
        step = trainer.epoch + 1
        if step <= self.last_epoch:
            return
        values = dict(trainer.metrics)
        values.update(trainer.label_loss_items(trainer.tloss, prefix='train'))
        values.update(trainer.lr)
        self.scalars(values, step)
        self.last_epoch = step

    def images(self, directory, prefix, step=0):
        for path in sorted(Path(directory).glob('*.jpg')) + sorted(Path(directory).glob('*.png')):
            with Image.open(path) as im:
                im = im.convert('RGB')
                im.thumbnail((1280, 960))
                self.writer.add_image(f'{prefix}/{path.stem}', np.asarray(im), step, dataformats='HWC')
        self.writer.flush()

    def close(self):
        self.writer.close()
