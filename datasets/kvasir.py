"""
Kvasir-SEG dataset — loads polyp images and binary segmentation masks
from the Kvasir-SEG benchmark.

Expected directory layout::

    kvasir-seg/
        Kvasir-SEG/
            images/
                cju0qkwl9lqbf0799zr69er7j.jpg
                ...
            masks/
                cju0qkwl9lqbf0799zr69er7j.jpg
                ...
            kavsir_bboxes.json   (unused; bounding boxes only)

Images are RGB JPEGs; colour carries diagnostic signal (tissue tone,
redness) and is preserved. Masks are grayscale JPEGs where pixel
value 255 indicates polyp (foreground) and 0 indicates background.

Label mapping (binary segmentation):
    0 — Background
    1 — Polyp

Image resolutions vary from 332×487 to 1920×1072 pixels; all samples
are resized to ``IMAGE_SIZE`` in the shared base ``__getitem__``.

References:
    Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset",
    MMM 2020. https://datasets.simula.no/kvasir-seg/
"""

import os

import cv2
import numpy as np

from datasets.base import BaseSegmentationDataset


class KvasirDataset(BaseSegmentationDataset):
    """Kvasir-SEG polyp segmentation dataset (RGB input, binary mask)."""

    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir: Path to ``Kvasir-SEG/images/`` containing JPEG images.
            masks_dir:  Path to ``Kvasir-SEG/masks/`` containing JPEG masks.
            transform:  Optional Albumentations augmentation pipeline.
        """
        super().__init__(transform=transform, in_channels=3)
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self._build_index()

    def _build_index(self):
        """Collect all image filenames that have a matching mask."""
        self.samples = sorted(
            fname
            for fname in os.listdir(self.images_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.exists(os.path.join(self.masks_dir, fname))
        )

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        """
        Load one image–mask pair.

        Returns:
            image: float32 RGB array of shape (H, W, 3), values in [0, 255]
            label: int64 binary mask of shape (H, W) with values in {0, 1}
        """
        fname = self.samples[idx]

        # Load as RGB — cv2 reads BGR by default, convert to RGB
        image_path = os.path.join(self.images_dir, fname)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Load mask — grayscale JPEG (255 = polyp, 0 = background)
        mask_path = os.path.join(self.masks_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarise: JPEG compression can shift pure 255 slightly
        label = (mask >= 128).astype(np.int64)

        return image, label