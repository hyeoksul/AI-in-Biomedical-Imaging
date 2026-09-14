"""Evaluate a trained checkpoint on a test-distance folder, e.g.:

    python src/evaluate.py --checkpoint results/supervised/deep_l1ssim_e50/best.pth \
        --arch deep --test_split test/Distance10mm --run_name deep_l1ssim_e50

Writes results/tables/<run_name>__<test_split>.json with mean PSNR/SSIM, and
saves one qualitative comparison figure (input hologram / GT / reconstruction,
amplitude + phase) to results/figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import HologramFieldDataset
from metrics import ReconstructionMetrics, summarize
from models import UNET_CONFIGS, UNet
from train_supervised import PHASE_SCALE


def load_model(checkpoint, arch, device):
    net = UNet(channels=UNET_CONFIGS[arch]).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, loader, device):
    metrics = ReconstructionMetrics(device)
    psnr_vals, ssim_vals = [], []
    first_batch = None

    for i, (holo, amp_gt, pha_gt) in enumerate(loader):
        holo, amp_gt = holo.to(device), amp_gt.to(device)
        out = net(holo)
        amp_pred, pha_pred = out[:, 0:1], out[:, 1:2] * PHASE_SCALE

        psnr, ssim = metrics.compute(amp_pred, amp_gt)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

        if i == 0:
            first_batch = (holo[0, 0].cpu(), amp_gt[0, 0].cpu(), pha_gt[0, 0].cpu(),
                            amp_pred[0, 0].cpu(), pha_pred[0, 0].cpu())

    return summarize(psnr_vals), summarize(ssim_vals), first_batch


def save_figure(sample, out_path, title):
    holo, amp_gt, pha_gt, amp_pred, pha_pred = sample
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes[0, 0].imshow(holo, cmap="gray"); axes[0, 0].set_title("Input Hologram")
    axes[0, 1].imshow(amp_gt, cmap="gray"); axes[0, 1].set_title("GT Amplitude")
    axes[0, 2].imshow(amp_pred, cmap="gray"); axes[0, 2].set_title("Reconstructed Amplitude")
    axes[1, 1].imshow(pha_gt, cmap="gray"); axes[1, 1].set_title("GT Phase")
    axes[1, 2].imshow(pha_pred, cmap="gray"); axes[1, 2].set_title("Reconstructed Phase")
    axes[1, 0].axis("off")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=r"D:\Graduate Study\AI in biomed imaging\Bio_image\MicroDegree\Data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arch", choices=UNET_CONFIGS.keys(), required=True)
    p.add_argument("--test_split", required=True, help="e.g. test/Distance10mm")
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.checkpoint, args.arch, device)

    loader = DataLoader(HologramFieldDataset(args.data_root, args.test_split), batch_size=1, shuffle=False)
    psnr_summary, ssim_summary, sample = evaluate(net, loader, device)

    tag = f"{args.run_name}__{args.test_split.replace('/', '_')}"
    tables_dir = Path(args.out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / f"{tag}.json", "w") as f:
        json.dump({"psnr": psnr_summary, "ssim": ssim_summary}, f, indent=2)

    save_figure(sample, Path(args.out_dir) / "figures" / f"{tag}.png", tag)

    print(f"{tag}: PSNR={psnr_summary['mean']:.3f}  SSIM={ssim_summary['mean']:.3f}  (n={psnr_summary['n']})")


if __name__ == "__main__":
    main()
