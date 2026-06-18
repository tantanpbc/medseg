"""
Shared loss functions used by all trainers.

Having one place for loss definitions ensures every model is trained with
identical objectives — critical for fair comparison.

All models use:    combined_loss = weighted_CE + 0.5 * soft_Dice (foreground only)

Weighted CE handles per-pixel class imbalance.
Soft Dice directly optimises the evaluation metric and is robust to
background dominance without needing explicit class weighting.

Dataset-specific class weights are looked up from DATASET_CONFIGS so the
loss automatically adapts when switching between ACDC / CAMUS / KVASIR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Per-dataset inverse-frequency CE weights.
# Approximate pixel distributions:
#   ACDC  : BG~90%, RV~3%, MYO~4%, LV~3%   → down-weight BG heavily
#   CAMUS : BG~85%, RV~5%, MYO~5%, LV~5%   → similar to ACDC
#   KVASIR: BG~75%, Polyp~25%               → mild imbalance
_CE_WEIGHTS = {
    "ACDC":      [0.1, 1.5, 1.2, 1.5],
    "CAMUS":     [0.1, 1.3, 1.3, 1.3],
    "KVASIR":    [0.3, 1.0],
    # Chest X-Ray: lung fields cover ~60% of image, background ~40%
    # Very mild imbalance — slight down-weight on background
    "CHESTXRAY": [0.4, 1.2],
}


def soft_dice_loss(pred, target, num_classes, ignore_bg=True, smooth=1e-6):
    """
    Soft multi-class Dice loss.

    Args:
        pred:        (B, C, H, W) raw logits
        target:      (B, H, W)   integer class labels
        num_classes: total number of classes (including background)
        ignore_bg:   if True, skip class 0 (background) — default True
        smooth:      Laplace smoothing constant

    Returns:
        scalar Dice loss averaged over foreground classes
    """
    pred_soft = torch.softmax(pred, dim=1)
    start_cls = 1 if ignore_bg else 0
    n_classes  = num_classes - start_cls

    loss = 0.0
    for cls in range(start_cls, num_classes):
        p = pred_soft[:, cls]
        t = (target == cls).float()
        intersection = (p * t).sum()
        loss += 1.0 - (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)

    return loss / max(n_classes, 1)


def make_loss(dataset_name, device, num_classes=None, dice_weight=0.5):
    """
    Build the combined CE + Dice loss for a given dataset.

    Args:
        dataset_name: key into _CE_WEIGHTS (e.g. "ACDC")
        device:       torch device for weight tensor
        num_classes:  override if dataset_name not in _CE_WEIGHTS
        dice_weight:  weight applied to the Dice term (default 0.5)

    Returns:
        callable  loss_fn(pred, target) → scalar
    """
    raw_weights = _CE_WEIGHTS.get(dataset_name)
    if raw_weights is None:
        # Unknown dataset — fall back to uniform CE + Dice
        ce_fn = nn.CrossEntropyLoss()
        n_cls = num_classes or 2
    else:
        ce_weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        ce_fn = nn.CrossEntropyLoss(weight=ce_weights)
        n_cls = num_classes or len(raw_weights)

    def combined_loss(pred, target):
        ce   = ce_fn(pred, target)
        dice = soft_dice_loss(pred, target, num_classes=n_cls, ignore_bg=True)
        return ce + dice_weight * dice

    return combined_loss