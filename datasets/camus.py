"""
CAMUS dataset — loads PNG frames and masks from the CAMUS echocardiography
dataset.

Expected directory layout::

    camus-.../
        frames/
            patient001_frame01.png
            ...
        masks/
            mask_patient001_frame01.png
            ...

Mask pixel values: 0, 85, 170, 255 → scaled to 0, 1, 2, 3.
Classes: 0:BG, 2:MYO, 3:LV  (class 1 is unused).
"""

import os

import numpy as np
from PIL import Image

from datasets.base import BaseSegmentationDataset


class CAMUSDataset(BaseSegmentationDataset):
    """CAMUS PNG frame/mask dataset for cardiac segmentation."""

    def __init__(self, frames_dir, masks_dir, transform=None):
        """
        Args:
            frames_dir: Path to the ``frames/`` directory containing PNG images.
            masks_dir:  Path to the ``masks/``  directory containing PNG masks.
            transform:  Optional Albumentations augmentation.
        """
        super().__init__(transform=transform)
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self._build_index()

    def _build_index(self):
        self.frame_files = sorted(
            [f for f in os.listdir(self.frames_dir) if f.endswith(".png")]
        )
        self.mask_dict = {
            f.replace("mask_", "frame_"): f
            for f in os.listdir(self.masks_dir)
            if f.endswith(".png")
        }

    def __len__(self):
        return len(self.frame_files)

    def _load_sample(self, idx):
        img_filename = self.frame_files[idx]
        img_path = os.path.join(self.frames_dir, img_filename)

        mask_filename = self.mask_dict.get(img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        if mask.max() > 3:
            mask = (mask / 85).astype(np.uint8)

        return image, mask
