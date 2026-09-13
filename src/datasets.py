"""Paired (hologram, complex field) dataset for holographic reconstruction.

Expects the KAIST course dataset layout:
    root/<split>/field/*.mat   (key "field", complex amplitude+phase)
    root/<split>/inten/*.mat   (key "inten", real intensity / hologram)
"""
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class HologramFieldDataset(Dataset):
    """Returns (hologram, amplitude, phase), each a [1, H, W] float32 tensor.

    Amplitude is min-max normalized to [0, 1] per-sample. Phase is only
    mean-centered here (not scaled) -- its raw range is small and unstable to
    regress directly, so scaling is applied at training time via a
    `phase_scale` factor, not baked into the dataset.
    """

    def __init__(self, root, split):
        self.field_dir = Path(root) / split / "field"
        self.inten_dir = Path(root) / split / "inten"
        self.field_files = sorted(os.listdir(self.field_dir))
        self.inten_files = sorted(os.listdir(self.inten_dir))
        assert len(self.field_files) == len(self.inten_files), (
            f"field/inten count mismatch in {root}/{split}"
        )

    def __len__(self):
        return len(self.field_files)

    def __getitem__(self, idx):
        field = sio.loadmat(self.field_dir / self.field_files[idx])["field"]
        inten = sio.loadmat(self.inten_dir / self.inten_files[idx])["inten"]

        amp = np.abs(field).astype(np.float32)
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
        pha = np.angle(field).astype(np.float32)
        pha = pha - pha.mean()

        holo = inten.astype(np.float32)
        holo = (holo - holo.min()) / (holo.max() - holo.min() + 1e-8)

        holo = torch.from_numpy(holo).unsqueeze(0)
        amp = torch.from_numpy(amp).unsqueeze(0)
        pha = torch.from_numpy(pha).unsqueeze(0)
        return holo, amp, pha


class FieldOnlyDataset(Dataset):
    """Returns just (amplitude, phase) -- for self-supervised training, where
    holograms are synthesized on the fly from the field via the physics
    forward model instead of being loaded from disk."""

    def __init__(self, root, split):
        self.field_dir = Path(root) / split / "field"
        self.field_files = sorted(os.listdir(self.field_dir))

    def __len__(self):
        return len(self.field_files)

    def __getitem__(self, idx):
        field = sio.loadmat(self.field_dir / self.field_files[idx])["field"]
        amp = np.abs(field).astype(np.float32)
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
        pha = np.angle(field).astype(np.float32)
        pha = pha - pha.mean()
        return torch.from_numpy(amp).unsqueeze(0), torch.from_numpy(pha).unsqueeze(0)
