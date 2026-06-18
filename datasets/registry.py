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
from datasets.chestxray import ChestXRayDataset


def build_dataset(dataset_name, transform=None, **kwargs):
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
    elif dataset_name == "CHESTXRAY":
        return ChestXRayDataset(
            root_dir         = kwargs["root_dir"],
            split            = kwargs.get("split", "train"),
            use_kaggle_split = kwargs.get("use_kaggle_split", False),
            transform        = transform,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: 'ACDC', 'CAMUS', 'KVASIR', 'CHESTXRAY'."
        )


def get_num_classes(dataset_name):
    _NUM_CLASSES = {"ACDC": 4, "CAMUS": 4, "KVASIR": 2, "CHESTXRAY": 2}
    return _NUM_CLASSES.get(dataset_name, 4)


def get_dataset_kwargs(dataset_name, args):
    if dataset_name == "ACDC":
        image_dir = os.path.join(args.acdc_dir, "ACDC_training_slices")
        return {"image_dir": image_dir}
    elif dataset_name == "CAMUS":
        return {
            "frames_dir": args.camus_frames_dir,
            "masks_dir":  args.camus_masks_dir,
        }
    elif dataset_name == "KVASIR":
        return {
            "images_dir": os.path.join(args.kvasir_dir, "images"),
            "masks_dir":  os.path.join(args.kvasir_dir, "masks"),
        }
    elif dataset_name == "CHESTXRAY":
        return {
            "root_dir":         args.chestxray_dir,
            "use_kaggle_split": False,   # use patient-level splitter for fair 70/10/20 split
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")