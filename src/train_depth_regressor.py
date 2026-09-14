"""Assignment task 2-2: estimate propagation distance directly from a hologram
(no reconstruction involved) -- a proxy for camera auto-focusing.

Trained self-supervised, same idea as train_selfsupervised.py: synthesize a
hologram from a ground-truth field at a random distance, but here the
network's target is the distance itself (regression), not the field.
Evaluated on real test holograms at fixed distances (5/10/15/20mm) using MAE/MSE.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import FieldOnlyDataset, HologramFieldDataset
from physics import field_to_hologram


class DepthEstimator(nn.Module):
    """Small CNN -> global average pool -> single scalar (predicted distance, mm)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        return self.regressor(x).squeeze(1)


def train_epoch(net, loader, criterion, device, dist_min, dist_max, optimizer):
    net.train()
    total_loss = 0.0
    for amp_gt, pha_gt in loader:
        amp_gt, pha_gt = amp_gt.to(device), pha_gt.to(device)
        b = amp_gt.shape[0]
        dist = torch.empty(b, device=device).uniform_(dist_min, dist_max)

        holo = field_to_hologram(amp_gt, pha_gt, dist)
        pred = net(holo)
        loss = criterion(pred, dist)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_on_real_distances(net, data_root, distances, device):
    """Real holograms at fixed distances -> MAE/MSE of predicted vs true distance."""
    results = {}
    for d in distances:
        loader = DataLoader(HologramFieldDataset(data_root, f"test/Distance{d}mm"),
                             batch_size=1, shuffle=False)
        errs = []
        for holo, _, _ in loader:
            pred = net(holo.to(device)).item()
            errs.append(pred - d)
        errs = torch.tensor(errs)
        results[d] = {
            "mae": errs.abs().mean().item(),
            "mse": (errs ** 2).mean().item(),
            "mean_predicted": (errs.mean().item() + d),
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=r"D:\Graduate Study\AI in biomed imaging\Bio_image\MicroDegree\Data")
    p.add_argument("--run_name", default="depth_5_20")
    p.add_argument("--dist_min", type=float, default=5)
    p.add_argument("--dist_max", type=float, default=20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(FieldOnlyDataset(args.data_root, "train"),
                               batch_size=args.batch_size, shuffle=True, drop_last=True)

    net = DepthEstimator().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    run_dir = Path(args.out_dir) / "depth_regressor" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(net, train_loader, criterion, device, args.dist_min, args.dist_max, optimizer)
        train_losses.append(loss)
        print(f"[{args.run_name}] epoch {epoch}/{args.epochs}  train_mse={loss:.4f}")

    torch.save(net.state_dict(), run_dir / "best.pth")

    eval_results = evaluate_on_real_distances(net, args.data_root, [5, 10, 15, 20], device)
    for d, r in eval_results.items():
        print(f"  {d}mm -> MAE={r['mae']:.2f}  MSE={r['mse']:.2f}  mean_pred={r['mean_predicted']:.2f}mm")

    with open(run_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "train_losses": train_losses, "eval": eval_results}, f, indent=2)


if __name__ == "__main__":
    main()
