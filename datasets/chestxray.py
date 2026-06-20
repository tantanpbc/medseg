"""
Chest X-Ray Masks and Defect Detection dataset.

Source: https://www.kaggle.com/datasets/azkihimmawan/chest-xray-masks-and-defect-detection

Expected directory layout::

    dataset_root/
        train_images/
            <case_id>/
                images/
                    <case_id>.png
                masks/
                    <case_id>.png
        test_images/
            <case_id>/
                images/
                    <case_id>.png
                masks/
                    <case_id>.png

Images are grayscale chest X-rays (PA view). No colour information is
present so images are loaded as single-channel (in_channels=1).

Masks are binary:
    0 -- Background
    1 -- Lung field (both lungs treated as one class)

Masks may be stored as pixel values {0, 255} or {0, 1} depending on
the dataset version. Both are handled by thresholding at 128.

The Kaggle pre-split (train_images / test_images) is ignored by default
in favour of the patient-level 70/10/20 splitter used by all other
datasets in this repo. Set use_kaggle_split=True to restore it.

References:
    Chest X-Ray Masks and Defect Detection (Kaggle)
    Based on Montgomery County and Shenzhen Hospital datasets (NIH/NLM)
"""

import os

import cv2
import numpy as np

from datasets.base import BaseSegmentationDataset


class ChestXRayDataset(BaseSegmentationDataset):
    """
    Chest X-Ray lung segmentation dataset (grayscale input, binary mask).

    Args:
        root_dir:         Path to dataset root (contains train_images/ and test_images/).
        split:            'train' or 'test' -- which Kaggle split folder to load.
                          Only used when use_kaggle_split=True.
        use_kaggle_split: If True, load only the specified split folder.
                          If False (default), load both folders and let the
                          patient-level splitter handle the split.
        transform:        Optional Albumentations augmentation pipeline.
    """

    _SPLIT_DIRS = {
        "train": "train_images",
        "test":  "test_images",
    }

    def __init__(self, root_dir, split="train", use_kaggle_split=False, transform=None):
        super().__init__(transform=transform, in_channels=1)
        self.root_dir         = root_dir
        self.split            = split
        self.use_kaggle_split = use_kaggle_split
        self._build_index()

    def _build_index(self):
        """
        Walk the nested per-case structure and collect (image_path, mask_path) pairs.

        Layout per case:
            <split_dir>/<case_id>/images/<case_id>.png
            <split_dir>/<case_id>/masks/<case_id>.png
        """
        pairs = []

        split_dirs = (
            [self._SPLIT_DIRS[self.split]]
            if self.use_kaggle_split
            else list(self._SPLIT_DIRS.values())
        )

        for split_dir in split_dirs:
            split_path = os.path.join(self.root_dir, split_dir)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(
                    f"Split directory not found: {split_path}\n"
                    f"Expected: {self.root_dir}/train_images/<case_id>/images/*.png"
                )

            for case_id in sorted(os.listdir(split_path)):
                case_dir = os.path.join(split_path, case_id)
                if not os.path.isdir(case_dir):
                    continue

                images_dir = os.path.join(case_dir, "images")
                masks_dir  = os.path.join(case_dir, "masks")
                if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
                    continue

                image_files = sorted(
                    f for f in os.listdir(images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )
                mask_files = sorted(
                    f for f in os.listdir(masks_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )
                if not image_files or not mask_files:
                    continue

                pairs.append((
                    os.path.join(images_dir, image_files[0]),
                    os.path.join(masks_dir,  mask_files[0]),
                ))

        if not pairs:
            raise RuntimeError(
                f"No valid image/mask pairs found under {self.root_dir}. "
                f"Check the directory structure matches the expected layout."
            )

        self._pairs  = pairs
        self.samples = [p[0] for p in pairs]   # image paths used by patient-level splitter

    def __len__(self):
        return len(self._pairs)

    def _load_sample(self, idx):
        """
        Load one grayscale X-ray and its binary lung mask.

        PERFORMANCE NOTE (applies to the IMAGE only):
            Chest X-ray PNGs in this dataset are typically exported at very
            high native resolution (often 2000-4000px per side) — roughly
            100x more pixels than the 256x256 IMAGE_SIZE the model actually
            trains on. Decoding the IMAGE at full resolution and then
            immediately throwing away ~99% of those pixels via cv2.resize()
            in BaseSegmentationDataset.__getitem__() wastes most of the
            decode time on every single sample, every epoch.

            cv2.IMREAD_REDUCED_GRAYSCALE_4 asks libpng/libjpeg to decode
            directly at 1/4 resolution (a cheap operation built into the
            decoder itself), so a 3000x3000 source image decodes as if it
            were ~750x750 — still comfortably above the 256x256 target, so
            the final cv2.resize() in the base class loses no information
            that would have survived the eventual downsize anyway, but the
            decode itself is roughly 16x cheaper.

        CORRECTNESS NOTE (why the MASK is NOT reduced-decoded):
            Reduced-resolution decoding performs an internal box-filter-like
            downsample as part of the decode itself. For a continuous-tone
            photograph (the X-ray) this is harmless. For a BINARY mask
            ({0,255} or {0,1}) it is not: averaging neighbouring pixels at
            a lung boundary produces intermediate gray values (e.g. ~127)
            that then get rounded essentially arbitrarily by the `>= 128`
            threshold below. This silently shifts the ground-truth boundary
            by a pixel or two at effectively random locations on every mask,
            every epoch -- directly hurting Dice/IoU, which are boundary-
            sensitive metrics. Masks are always decoded at full resolution;
            the later cv2.resize(..., INTER_NEAREST) in the base class is
            safe because nearest-neighbour resize never invents intermediate
            values, unlike the reduced decode path.

        Returns:
            image: float32 array shape (H, W), pixel values in [0, 255]
            label: int64 binary array shape (H, W), values in {0, 1}
                   0 = background, 1 = lung field
        """
        image_path, mask_path = self._pairs[idx]

        # Decode the IMAGE at reduced resolution — cheap libpng/libjpeg-native
        # downscale, avoids paying for pixels that get discarded by resize anyway.
        from config import IMAGE_SIZE
        min_dim = min(IMAGE_SIZE)

        image = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        # Safety: if the source file was already small, a 1/4 reduced decode
        # could undersize below the model's target resolution. Fall back to
        # full decode in that case so cv2.resize() in the base class always
        # has at least as much detail as IMAGE_SIZE requires.
        if image is None or min(image.shape[:2]) < min_dim:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = image.astype(np.float32)

        # Decode the MASK at FULL resolution, always — see correctness note above.
        # Nearest-neighbour resize happens later in BaseSegmentationDataset and
        # is safe; reduced decode is not, so it is never used for masks.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarise robustly: handle both {0,1} and {0,255} encodings.
        # If max pixel value <= 1 the mask is already binary {0,1}.
        # Otherwise threshold at 128 to convert {0,255} to {0,1}.
        if mask.max() <= 1:
            label = mask.astype(np.int64)
        else:
            label = (mask >= 128).astype(np.int64)

        return image, label