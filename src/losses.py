"""Training losses for reconstruction: L1, L2, and L1+SSIM, each combining an
amplitude term and a phase term.

Per the assignment, evaluation metrics (metrics.py) must differ from whatever
loss trained the model, so PSNR/SSIM used for evaluation are handled
separately even when SSIM also appears here as a training loss.
"""
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


def get_criterion(name, device):
    """Returns fn(amp_pred, amp_gt, pha_pred, pha_gt) -> scalar loss."""
    if name == "l1":
        base = nn.L1Loss()
        return lambda ap, ag, pp, pg: base(ap, ag) + base(pp, pg)

    if name == "l2":
        base = nn.MSELoss()
        return lambda ap, ag, pp, pg: base(ap, ag) + base(pp, pg)

    if name == "l1+ssim":
        l1 = nn.L1Loss()
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # SSIM only applied to amplitude (it expects a bounded, image-like signal;
        # phase is mean-centered and can be negative/unbounded, so it stays L1-only).
        return lambda ap, ag, pp, pg: l1(ap, ag) + (1 - ssim(ap, ag)) + l1(pp, pg)

    raise ValueError(f"unknown loss name: {name}")
