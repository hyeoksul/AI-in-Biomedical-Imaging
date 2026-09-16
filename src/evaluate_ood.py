"""Extra credit: Out-of-Distribution (OOD) comparison.

No real hologram data exists at 2mm/25mm (the dataset only has 5/10/15/20mm),
so ground-truth fields (identical across all test/Distance*/field folders --
verified: same 50 tissue objects, only the sensor distance differs) are
re-propagated with the physics forward model to synthesize holograms at these
unseen distances. Compares a supervised model (trained only at 10mm) against
a self-supervised model (trained across the continuous 5-20mm range) to see
which one generalizes better outside its training distribution.

Example:
    python src/evaluate_ood.py \
        --supervised_ckpt results/supervised/deep_l1ssim_e20/best.pth --supervised_arch deep \
        --selfsupervised_ckpt results/selfsupervised/ss_5_20/best.pth --selfsupervised_arch baseline
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
from physics import field_to_hologram
from train_supervised import PHASE_SCALE


def load_model(checkpoint, arch, device):
    net = UNet(channels=UNET_CONFIGS[arch]).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    return net


@torch.no_grad()
def evaluate_at_distance(net, loader, distance_mm, device):
    metrics = ReconstructionMetrics(device)
    psnr_vals, ssim_vals = [], []
    first_sample = None

    for i, (_, amp_gt, pha_gt) in enumerate(loader):
        amp_gt, pha_gt = amp_gt.to(device), pha_gt.to(device)
        b = amp_gt.shape[0]
        dist = torch.full((b,), float(distance_mm), device=device)

        holo = field_to_hologram(amp_gt, pha_gt, dist)
        out = net(holo)
        amp_pred, pha_pred = out[:, 0:1], out[:, 1:2] * PHASE_SCALE

        psnr, ssim = metrics.compute(amp_pred, amp_gt)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

        if i == 0:
            first_sample = (holo[0, 0].cpu(), amp_gt[0, 0].cpu(), amp_pred[0, 0].cpu(),
                             pha_gt[0, 0].cpu(), pha_pred[0, 0].cpu())

    return summarize(psnr_vals), summarize(ssim_vals), first_sample


def save_figure(sample, out_path, title):
    """Same colormap convention as evaluate.py: viridis for intensity-like
    panels, twilight (cyclic) for phase."""
    holo, amp_gt, amp_pred, pha_gt, pha_pred = sample
    pha_abs_max = max(pha_gt.abs().max().item(), pha_pred.abs().max().item(), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    panels = [
        (axes[0, 0], holo, "Synthesized Hologram", "viridis", None),
        (axes[0, 1], amp_gt, "GT Amplitude", "viridis", None),
        (axes[0, 2], amp_pred, "Reconstructed Amplitude", "viridis", None),
        (axes[1, 1], pha_gt, "GT Phase", "twilight", pha_abs_max),
        (axes[1, 2], pha_pred, "Reconstructed Phase", "twilight", pha_abs_max),
    ]
    for ax, img, panel_title, cmap, sym_range in panels:
        kwargs = {"vmin": -sym_range, "vmax": sym_range} if sym_range else {}
        im = ax.imshow(img, cmap=cmap, **kwargs)
        ax.set_title(panel_title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    p.add_argument("--field_split", default="test/Distance10mm",
                    help="any test distance folder works -- ground-truth field is identical across them")
    p.add_argument("--supervised_ckpt", required=True)
    p.add_argument("--supervised_arch", choices=UNET_CONFIGS.keys(), required=True)
    p.add_argument("--selfsupervised_ckpt", required=True)
    p.add_argument("--selfsupervised_arch", choices=UNET_CONFIGS.keys(), required=True)
    p.add_argument("--ood_distances", type=float, nargs="+", default=[2, 25])
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(HologramFieldDataset(args.data_root, args.field_split), batch_size=1, shuffle=False)

    supervised = load_model(args.supervised_ckpt, args.supervised_arch, device)
    selfsupervised = load_model(args.selfsupervised_ckpt, args.selfsupervised_arch, device)

    results = {}
    for dist in args.ood_distances:
        for tag, net in [("supervised", supervised), ("selfsupervised", selfsupervised)]:
            psnr, ssim, sample = evaluate_at_distance(net, loader, dist, device)
            results[f"{tag}_{dist}mm"] = {"psnr": psnr, "ssim": ssim}
            save_figure(sample, Path(args.out_dir) / "figures" / f"ood_{tag}_{dist}mm.png",
                        f"{tag} @ {dist}mm (OOD)")
            print(f"{tag} @ {dist}mm: PSNR={psnr['mean']:.3f}  SSIM={ssim['mean']:.3f}")

    tables_dir = Path(args.out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "ood_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
