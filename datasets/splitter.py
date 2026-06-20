"""
Deterministic patient-level train / val / test splitter.

WHY PATIENT-LEVEL:
    ACDC (and most medical datasets) contain multiple 2-D slices per patient.
    A slice-level random split leaks data — the model sees slices from the
    same patient in both train and val/test, inflating reported performance.
    Patient-level splitting ensures the test set contains ONLY unseen patients.

SPLIT RATIOS (default):
    Train : 70%   of patients  → used for gradient updates
    Val   : 10%   of patients  → used during training for early stopping / checkpointing
    Test  : 20%   of patients  → held out until final evaluation, never used for decisions

The split is seeded so it is deterministic across all runs and all models,
making comparisons fair.

Usage:
    from datasets.splitter import split_dataset_indices

    train_idx, val_idx, test_idx = split_dataset_indices(
        samples,           # list of file paths / any list of items
        patient_id_fn,     # callable: sample → patient_id string
        train_ratio=0.70,
        val_ratio=0.10,
        seed=42,
    )
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, List, Tuple


def split_dataset_indices(
    samples: List,
    patient_id_fn: Callable,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split sample indices by patient ID into train / val / test.

    Args:
        samples:        Ordered list of dataset items (file paths, dicts, etc.).
        patient_id_fn:  Function mapping a sample to its patient ID string.
                        e.g. lambda p: os.path.basename(p).split("_")[0]
                        yields "patient001" from "patient001_slice00.h5"
        train_ratio:    Fraction of patients for training   (default 0.70)
        val_ratio:      Fraction of patients for validation (default 0.10)
        seed:           RNG seed for reproducibility        (default 42)

    Returns:
        (train_indices, val_indices, test_indices) — lists of integer indices
        into ``samples``.

    Raises:
        ValueError: if ratios don't sum to ≤ 1 or there are too few patients.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"train_ratio + val_ratio must be ≤ 1.0, "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio:.3f}"
        )

    # Group sample indices by patient ID
    patient_to_indices: dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        pid = patient_id_fn(sample)
        patient_to_indices[pid].append(idx)

    # Sort patients deterministically, then shuffle with fixed seed
    patients = sorted(patient_to_indices.keys())
    n = len(patients)
    if n < 3:
        raise ValueError(
            f"Need at least 3 patients for a 3-way split, got {n}."
        )

    # Use a seeded hash-based shuffle (avoids numpy dependency here)
    def _stable_hash(s: str) -> int:
        return int(hashlib.md5((s + str(seed)).encode()).hexdigest(), 16)

    patients_shuffled = sorted(patients, key=_stable_hash)

    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))
    # Give all remainder to test to avoid rounding errors swallowing patients
    n_train = min(n_train, n - 2)
    n_val   = min(n_val,   n - n_train - 1)

    train_patients = set(patients_shuffled[:n_train])
    val_patients   = set(patients_shuffled[n_train:n_train + n_val])
    test_patients  = set(patients_shuffled[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for pid, indices in patient_to_indices.items():
        if pid in train_patients:
            train_idx.extend(indices)
        elif pid in val_patients:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    print(
        f"Patient split (seed={seed}): "
        f"{len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test patients"
    )
    print(
        f"Slice split: "
        f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test slices"
    )

    return train_idx, val_idx, test_idx
