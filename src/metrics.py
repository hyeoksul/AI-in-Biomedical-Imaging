"""Evaluation metrics: PSNR and SSIM on reconstructed amplitude.

Kept deliberately separate from losses.py: the assignment requires evaluation
metrics that differ from whatever loss trained the model (e.g. an L1-trained
model must still be judged by PSNR/SSIM), so metrics are never imported by the
training loop's loss computation, only by evaluate.py.
"""
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class ReconstructionMetrics:
    def __init__(self, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def compute(self, amp_pred, amp_gt):
        """amp_pred/amp_gt: [B, 1, H, W] in [0, 1]. Returns (psnr, ssim) as floats."""
        return self.psnr(amp_pred, amp_gt).item(), self.ssim(amp_pred, amp_gt).item()


def summarize(values):
    values = np.array(values)
    return {"mean": float(values.mean()), "std": float(values.std()), "n": len(values)}
