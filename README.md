# Deep Learning for Digital Holographic Reconstruction

Reconstructing the amplitude and phase of light from a single intensity-only
hologram, for label-free imaging of tissue histology slides — comparing a
supervised approach against a physics-informed self-supervised approach that
uses no real hologram data at all.

<!-- TODO once results land: one-sentence headline result, e.g.
"The self-supervised model generalizes to distances 2.5x outside its training
range with X% less PSNR degradation than the single-distance supervised model." -->

## Why this problem is hard

A camera can only measure light *intensity*; the *phase* of the field —
which encodes the tissue's refractive-index structure — is destroyed at the
moment of measurement, and the light further diffracts as it travels from the
sample to the sensor. Recovering the original complex field from a single
diffracted intensity pattern is a classic ill-posed inverse problem in
computational imaging. Full derivation and the physics behind the forward
model used here: [`docs/background.md`](docs/background.md).

## Two approaches, compared

| | Supervised | Self-supervised |
|---|---|---|
| Trains on | Real (hologram, ground-truth field) pairs, single fixed distance (10mm) | Only ground-truth fields; holograms synthesized on the fly at a randomly sampled distance via a differentiable physics forward model (Angular Spectrum Method) |
| Learns | A direct mapping for the one distance it saw | A mapping expected to hold across a *range* of distances |
| Risk | May not generalize to other sensor distances | Depends on how faithful the physics simulator is |

The physics forward model (`src/physics.py`) was validated against real
measured 10mm holograms before being trusted for training: **Pearson
correlation r = 0.997** between simulated and real holograms of the same
object.

## Results

<!-- TODO: fill in after scripts/run_all_experiments.sh completes.
Structure to fill, matching the assignment's task numbering:
-->

### 1-1: Network architecture (baseline / wide / deep U-Net)
_Training/validation loss curves, PSNR/SSIM table @ 10mm, qualitative comparison._

### 1-2: Training length (10 / 20 / 50 epochs)
_Loss curves, PSNR/SSIM table, qualitative comparison._

### 1-3: Loss function (L1 / L2 / L1+SSIM)
_PSNR/SSIM table (evaluated with metrics different from the training loss), qualitative comparison._

### 1-4: Generalization to unseen distances (5 / 15 / 20mm)
_PSNR/SSIM vs. distance, discussion of failure modes._

### 2-1: Self-supervised training across distance configurations
_Fixed-distance vs. continuous-range training, cross-distance evaluation table._

### 2-2: Distance estimation (auto-focus proxy)
_MAE/MSE per test distance, discussion of what the errors reveal about the network's behavior._

### Extra credit: Out-of-distribution robustness (2mm / 25mm)
_Supervised vs. self-supervised comparison at distances outside the training range._

## What I learned

<!-- TODO: write after seeing final results -- in your own words. -->

## Repository structure

```
holographic-reconstruction/
  docs/background.md          background on holography, the phase problem, and the ASM forward model
  src/
    datasets.py                paired and field-only .mat dataset loaders
    physics.py                  differentiable Angular Spectrum Method forward model
    models.py                    parametrized U-Net (baseline/wide/deep via config)
    losses.py                     L1 / L2 / L1+SSIM training losses
    metrics.py                     PSNR / SSIM evaluation (kept separate from training losses)
    train_supervised.py            tasks 1-1, 1-2, 1-3
    train_selfsupervised.py        task 2-1
    train_depth_regressor.py       task 2-2
    evaluate.py                     runs a checkpoint against a test-distance split
    evaluate_ood.py                  extra credit: supervised vs. self-supervised at 2mm/25mm
  scripts/run_all_experiments.sh   reproduces the entire experiment matrix
  notebooks/                        data exploration + final results figures
  results/                          metrics tables, figures, training logs (checkpoints gitignored)
```

## Reproducing this

Dataset (not included -- KAIST course material, not redistributed):
```
Data/
  train/{field,inten}/*.mat        # 1500 pairs, 10mm
  valid/{field,inten}/*.mat        # 20 pairs, 10mm
  test/Distance{5,10,15,20}mm/{field,inten}/*.mat   # 50 pairs each
```
`.mat` files use keys `"field"` (complex64, ground-truth amplitude+phase) and
`"inten"` (real, measured hologram intensity).

```bash
pip install -r requirements.txt
bash scripts/run_all_experiments.sh   # full matrix, ~6h on one GPU
```

Or run individual pieces, e.g.:
```bash
python src/train_supervised.py --run_name deep_l1ssim --arch deep --loss l1+ssim --epochs 20
python src/evaluate.py --checkpoint results/supervised/deep_l1ssim/best.pth --arch deep --test_split test/Distance10mm --run_name deep_l1ssim
```

## Acknowledgments

Dataset and assignment framing from KAIST's "AI in Biomedical Imaging" course
(Instructor: Mooseok Jang). This repository is an independent, from-scratch
reimplementation done to build a deeper understanding of the material, not a
copy of the original course submission.
