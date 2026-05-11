from __future__ import annotations

import numpy as np
from fastai.vision.augment import RandTransform
from fastai.vision.core import PILImage
from fastcore.basics import store_attr


class AlbumentationsTransform(RandTransform):
    """Wraps an albumentations pipeline as a fastai batch transform."""

    split_idx, order = None, 2

    def __init__(self, train_aug=None, valid_aug=None):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0 and self.train_aug is not None:
            aug_img = self.train_aug(image=np.array(img))["image"]
        elif self.valid_aug is not None:
            aug_img = self.valid_aug(image=np.array(img))["image"]
        else:
            aug_img = np.array(img)
        return PILImage.create(aug_img)


def acc_camvid(inp, targ):
    background_code = 0
    targ = targ.squeeze(1)
    mask = targ != background_code
    if not mask.any():
        return inp.new_tensor(float("nan"))
    return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()
