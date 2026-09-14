"""Supervised training: hologram (fixed distance, e.g. 10mm) -> complex field.

Covers assignment tasks 1-1 (architecture), 1-2 (epochs), 1-3 (loss function) --
all three are just different (model, epochs, loss) combos of the same loop, so
one script + CLI args replaces three near-duplicate notebook cells.

Example:
    python src/train_supervised.py --run_name deep_l1ssim_e50 \
        --arch deep --loss l1+ssim --epochs 50
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import HologramFieldDataset
from losses import get_criterion
from models import UNET_CONFIGS, UNet

PHASE_SCALE = 2 * torch.pi  # phase targets are stored mean-centered in radians;
                            # the network's raw output is scaled up by this factor
                            # before comparing to the target, for stabler gradients
                            # early in training (small init outputs, large target range).


def run_epoch(net, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    net.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for holo, amp_gt, pha_gt in loader:
            holo, amp_gt, pha_gt = holo.to(device), amp_gt.to(device), pha_gt.to(device)

            out = net(holo)
            amp_pred, pha_pred = out[:, 0:1], out[:, 1:2] * PHASE_SCALE

            loss = criterion(amp_pred, amp_gt, pha_pred, pha_gt)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=r"D:\Graduate Study\AI in biomed imaging\Bio_image\MicroDegree\Data")
    p.add_argument("--run_name", required=True)
    p.add_argument("--arch", choices=UNET_CONFIGS.keys(), default="baseline")
    p.add_argument("--loss", choices=["l1", "l2", "l1+ssim"], default="l1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        HologramFieldDataset(args.data_root, "train"),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        HologramFieldDataset(args.data_root, "valid"),
        batch_size=args.batch_size, shuffle=False,
    )

    net = UNet(channels=UNET_CONFIGS[args.arch]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    criterion = get_criterion(args.loss, device)

    run_dir = Path(args.out_dir) / "supervised" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(net, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(net, valid_loader, criterion, device, optimizer=None)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{args.run_name}] epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), run_dir / "best.pth")

    torch.save(net.state_dict(), run_dir / "last.pth")
    with open(run_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), **history}, f, indent=2)

    print(f"done. best val loss = {best_val:.4f}. saved to {run_dir}")


if __name__ == "__main__":
    main()
