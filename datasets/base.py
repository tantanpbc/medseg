"""
Base dataset class for medical image segmentation.

Provides shared preprocessing (resize, augmentation, normalization, tensor
conversion) so that each dataset subclass only needs to implement
``_load_sample(idx) -> (image, label)``.

Supports both:
    * Grayscale inputs  (in_channels=1) — image shape (H, W),  scalar z-score norm
    * RGB inputs        (in_channels=3) — image shape (H, W, 3), per-channel z-score norm
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
          returning float32 image and int64 label arrays.
          Grayscale: image shape (H, W).
          RGB:       image shape (H, W, 3).
        * Implement ``__len__``

    Shared preprocessing handled here:
        1. Resize to ``IMAGE_SIZE``
        2. Apply Albumentations transform (if provided)
        3. Per-sample z-score normalization
           - Grayscale: single scalar mean/std across all pixels
           - RGB:       per-channel mean/std (preserves colour relationships)
        4. Convert to torch tensors
           - Grayscale: (1, H, W)
           - RGB:       (3, H, W)
    """

    def __init__(self, transform=None, in_channels=1):
        """
        Args:
            transform:   Optional Albumentations augmentation pipeline.
            in_channels: 1 for grayscale datasets, 3 for RGB datasets.
        """
        self.transform   = transform
        self.in_channels = in_channels
        # Subclasses must populate self.samples with a list of file paths.
        # This is read by the patient-level splitter in datasets/splitter.py.
        self.samples = []

    # ── Subclass protocol ──

    def _load_sample(self, idx):
        """Return (image: np.float32, label: np.int64) for sample *idx*."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # ── Shared preprocessing ──

    def __getitem__(self, idx):
        image, label = self._load_sample(idx)

        # Resize — handle both (H, W) and (H, W, 3)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # Augmentation (operates on NumPy arrays)
        if self.transform is not None:
            result = self.transform(image=image, mask=label)
            image  = result["image"]
            label  = result["mask"]

        # Per-sample normalisation
        if self.in_channels == 1:
            # Scalar z-score across all pixels
            mean_val = image.mean()
            std_val  = image.std()
            if std_val > 1e-8:
                image = (image - mean_val) / std_val
        else:
            # Per-channel z-score — preserves colour relationships
            # image shape: (H, W, C)
            mean_val = image.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
            std_val  = image.std(axis=(0, 1),  keepdims=True)   # (1, 1, C)
            std_val  = np.where(std_val > 1e-8, std_val, 1.0)   # avoid div-by-zero
            image    = (image - mean_val) / std_val

        # Convert to tensors
        if self.in_channels == 1:
            image = torch.from_numpy(image).float().unsqueeze(0)        # (1, H, W)
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1)   # (3, H, W)

        label = torch.from_numpy(label).long()                          # (H, W)
        return image, label