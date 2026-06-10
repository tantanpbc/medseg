"""
Dataset registry — factory function for building dataset instances by name.

To add a new dataset:
    1. Create ``datasets/your_dataset.py`` with a class inheriting
       from ``BaseSegmentationDataset``.
    2. Add an entry to ``build_dataset()`` below.
    3. Add a corresponding entry in ``config.DATASET_CONFIGS``.
"""

import os

from datasets.acdc import ACDCDataset
from datasets.camus import CAMUSDataset
from datasets.kvasir import KvasirDataset


def build_dataset(dataset_name, transform=None, **kwargs):
    """
    Instantiate the correct dataset class by name.

    Args:
        dataset_name: One of ``"ACDC"``, ``"CAMUS"``, ``"KVASIR"``.
        transform:    Optional Albumentations augmentation.
        **kwargs:     Dataset-specific arguments (paths, etc.).

    Returns:
        A ``BaseSegmentationDataset`` subclass instance.
    """
    if dataset_name == "ACDC":
        return ACDCDataset(image_dir=kwargs["image_dir"], transform=transform)
    elif dataset_name == "CAMUS":
        return CAMUSDataset(
            frames_dir=kwargs["frames_dir"],
            masks_dir=kwargs["masks_dir"],
            transform=transform,
        )
    elif dataset_name == "KVASIR":
        return KvasirDataset(
            images_dir=kwargs["images_dir"],
            masks_dir=kwargs["masks_dir"],
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: 'ACDC', 'CAMUS', 'KVASIR'."
        )


def get_num_classes(dataset_name):
    """Return the number of segmentation classes for a given dataset."""
    _NUM_CLASSES = {"ACDC": 4, "CAMUS": 4, "KVASIR": 2}
    return _NUM_CLASSES.get(dataset_name, 4)


def get_dataset_kwargs(dataset_name, args):
    """
    Build the dataset-specific keyword arguments from parsed CLI args.

    Returns a dict suitable for passing to ``build_dataset()``.
    """
    if dataset_name == "ACDC":
        # args.acdc_dir points to ACDC_preprocessed; .h5 files are in ACDC_training_slices/
        image_dir = os.path.join(args.acdc_dir, "ACDC_training_slices")
        return {"image_dir": image_dir}
    elif dataset_name == "CAMUS":
        return {
            "frames_dir": args.camus_frames_dir,
            "masks_dir": args.camus_masks_dir,
        }
    elif dataset_name == "KVASIR":
        return {
            "images_dir": os.path.join(args.kvasir_dir, "images"),
            "masks_dir":  os.path.join(args.kvasir_dir, "masks"),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
