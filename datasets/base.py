"""
Base dataset class for medical image segmentation.

Provides shared preprocessing (resize, augmentation, normalization, tensor
conversion) so that each dataset subclass only needs to implement
``_load_sample(idx) -> (image, label)``.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import IMAGE_SIZE


class BaseSegmentationDataset(Dataset):
    """
    Abstract base for medical segmentation datasets.

    Subclasses must:
        * Build an index of samples in ``__init__``
        * Implement ``_load_sample(idx) -> (np.ndarray, np.ndarray)``
          returning float32 image and int64 label arrays
        * Implement ``__len__``

    Shared preprocessing handled here:
        1. Resize to ``IMAGE_SIZE``
        2. Apply Albumentations transform (if provided)
        3. Per-sample z-score normalization
        4. Convert to torch tensors
    """

    def __init__(self, transform=None):
        self.transform = transform

    # ── Subclass protocol ──

    def _load_sample(self, idx):
        """Return (image: np.float32, label: np.int64) for sample *idx*."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # ── Shared preprocessing ──

    def __getitem__(self, idx):
        image, label = self._load_sample(idx)

        # Resize
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # Augmentation (operates on NumPy arrays)
        if self.transform is not None:
            result = self.transform(image=image, mask=label)
            image = result["image"]
            label = result["mask"]

        # Per-sample normalisation (zero mean, unit std)
        mean_val = image.mean()
        std_val = image.std()
        if std_val > 1e-8:
            image = (image - mean_val) / std_val

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)   # (1, H, W)
        label = torch.from_numpy(label).long()                  # (H, W)
        return image, label
