"""Self-supervised training (assignment task 2-1): no real holograms are used.
Instead, ground-truth complex fields are propagated by the ASM forward model
(src/physics.py) to synthesize a hologram at a randomly sampled distance each
batch, and the network is trained to invert that synthetic hologram back to
the original field. Validation still uses real (field, hologram) pairs.

Example:
    python src/train_selfsupervised.py --run_name ss_5_20 --dist_min 5 --dist_max 20
    python src/train_selfsupervised.py --run_name ss_10_10 --dist_min 10 --dist_max 10
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import FieldOnlyDataset, HologramFieldDataset
from models import UNET_CONFIGS, UNet
from physics import field_to_hologram
from train_supervised import PHASE_SCALE


def train_epoch(net, loader, criterion, device, dist_min, dist_max, optimizer):
    net.train()
    total_loss = 0.0
    for amp_gt, pha_gt in loader:
        amp_gt, pha_gt = amp_gt.to(device), pha_gt.to(device)
        b = amp_gt.shape[0]

        # a fresh random distance per batch -- this is what makes training
        # "diverse-distance" rather than single-distance
        dist = torch.empty(b, device=device).uniform_(dist_min, dist_max)
        # pha_gt from the dataset is already true (mean-centered) phase in
        # radians -- PHASE_SCALE only reinterprets the network's *output*
        # later, it plays no role in synthesizing the hologram here.
        holo = field_to_hologram(amp_gt, pha_gt, dist)

        out = net(holo)
        amp_pred, pha_pred = out[:, 0:1], out[:, 1:2] * PHASE_SCALE

        loss = criterion(amp_pred, amp_gt) + criterion(pha_pred, pha_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(net, loader, criterion, device):
    net.eval()
    total_loss = 0.0
    for holo, amp_gt, pha_gt in loader:
        holo, amp_gt, pha_gt = holo.to(device), amp_gt.to(device), pha_gt.to(device)
        out = net(holo)
        amp_pred, pha_pred = out[:, 0:1], out[:, 1:2] * PHASE_SCALE
        loss = criterion(amp_pred, amp_gt) + criterion(pha_pred, pha_gt)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=r"D:\Graduate Study\AI in biomed imaging\Bio_image\MicroDegree\Data")
    p.add_argument("--run_name", required=True)
    p.add_argument("--arch", choices=UNET_CONFIGS.keys(), default="baseline")
    p.add_argument("--dist_min", type=float, required=True)
    p.add_argument("--dist_max", type=float, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_interval", type=int, default=2)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        FieldOnlyDataset(args.data_root, "train"),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        HologramFieldDataset(args.data_root, "valid"),
        batch_size=args.batch_size, shuffle=False,
    )

    net = UNet(channels=UNET_CONFIGS[args.arch]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    criterion = torch.nn.L1Loss()

    run_dir = Path(args.out_dir) / "selfsupervised" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(net, train_loader, criterion, device,
                                  args.dist_min, args.dist_max, optimizer)
        scheduler.step()
        history["train_loss"].append(train_loss)
        msg = f"[{args.run_name}] epoch {epoch}/{args.epochs}  train={train_loss:.4f}"

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_loss = validate(net, valid_loader, criterion, device)
            history["val_loss"].append((epoch, val_loss))
            msg += f"  val={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(net.state_dict(), run_dir / "best.pth")

        print(msg)

    torch.save(net.state_dict(), run_dir / "last.pth")
    with open(run_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), **history}, f, indent=2)

    print(f"done. best val loss = {best_val:.4f}. saved to {run_dir}")


if __name__ == "__main__":
    main()
