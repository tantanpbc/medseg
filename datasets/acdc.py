"""
ACDC dataset — loads preprocessed HDF5 slices from the Automated Cardiac
Diagnosis Challenge.

Expected directory layout::

    ACDC_preprocessed/
        ACDC_training_slices/
            patient001_slice00.h5
            patient001_slice01.h5
            ...

Each HDF5 file contains:
    * ``image`` — 2-D grayscale array (float32)
    * ``label`` — 2-D integer mask (0:BG, 1:RV, 2:MYO, 3:LV)
"""

import os

import h5py
import numpy as np

from datasets.base import BaseSegmentationDataset


class ACDCDataset(BaseSegmentationDataset):
    """ACDC HDF5 slice dataset for cardiac segmentation."""

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir:  Path to ``ACDC_training_slices/`` containing .h5 files.
            transform:  Optional Albumentations augmentation.
        """
        super().__init__(transform=transform)
        self.image_dir = image_dir
        self._build_index()

    def _build_index(self):
        self.samples = sorted(
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(".h5")
        )

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        fpath = self.samples[idx]
        with h5py.File(fpath, "r") as h5f:
            image = h5f["image"][:].astype(np.float32)
            label = h5f["label"][:].astype(np.int64)
        if image.ndim == 3:
            image = image.squeeze()
        if label.ndim == 3:
            label = label.squeeze()
        return image, label
