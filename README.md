# To Skip or Not to Skip: Investigating U-Net Skip-Connections and Task Complexity

![MICCAI 2023](https://img.shields.io/badge/Conference-MICCAI%202023-blue) ![CIBM 2025](https://img.shields.io/badge/Journal-CIBM%202025-green) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/amithjkamath/to_skip_or_not) ![Testing](https://github.com/amithjkamath/to_skip_or_not/actions/workflows/test.yml/badge.svg)

This repository accompanies **two papers**, both by
*[Amith Kamath](https://amithjkamath.github.io), [Jonas Willmann](https://scholar.google.com/citations?user=smUWeEgAAAAJ&hl=en), [Nicolaus Andratschke](https://scholar.google.com/citations?user=n0oz878AAAAJ&hl=en), [Mauricio Reyes](https://mauricioreyes.me/)*:

- **MICCAI 2023** — *Do we really need that skip-connection? Understanding its
  interplay with task complexity*
- **CIBM 2025** — *The impact of U-Net architecture choices and skip connections
  on the robustness of segmentation across texture variations*
  (journal extension: 6 architectures, 4 datasets, 5 perturbation levels, both
  synthetic blending directions)

Which results belong to which paper is set out in the tables below. Input data, model
weights and per-image metrics are archived outside this repository and shared on
request — see [Configuring where the data lives](#configuring-where-the-data-lives).
Known caveats about the published numbers are recorded [below](#known-caveats).

See a short video description of this work here:

[<img src="https://i.ytimg.com/vi/YreG6vC64aw/maxresdefault.jpg" width="50%">](https://youtu.be/YreG6vC64aw "To Skip or Not to Skip")

🔗 [Project Website](https://amithjkamath.github.io/to_skip_or_not/)  
---

## Overview

This project explores the **necessity and effect of skip-connections in U-Net architectures** under varying task complexities in medical image segmentation. The study examines how the usefulness of skip-connections depends on the **textural similarity** between foreground and background, using:

- Controlled **synthetic texture experiments**, blending in both directions
- Evaluations on real **medical imaging modalities**: Ultrasound (US), Histology,
  CT, and MRI
- Comparison across six U-Net variants in three groups by skip-connection
  density: **No-Skip**, **Standard**, and **Enhanced**

---

## Demonstration videos

`docs/videos/` holds one video loop per dataset showing how each architecture's
segmentation reacts as texture disparity is perturbed. Each frame is 4 columns ×
6 rows — one row per architecture, and per row: the clean image, the ground
truth, the perturbed image, and that model's prediction with the ground truth
outlined for comparison. Column 3 is identical down the whole column, so at any
instant every model sees the same input and column 4 isolates the effect of
architecture. The time axis sweeps texture disparity and returns, so the clips
loop.

| Video | Content |
| --- | --- |
| [`robustness_busi.mp4`](docs/videos/robustness_busi.mp4) | Breast ultrasound |
| [`robustness_glas.mp4`](docs/videos/robustness_glas.mp4) | Colon histology |
| [`robustness_heart.mp4`](docs/videos/robustness_heart.mp4) | Heart MRI |
| [`robustness_spleen.mp4`](docs/videos/robustness_spleen.mp4) | Spleen CT |
| [`robustness_synthetic_background.mp4`](docs/videos/robustness_synthetic_background.mp4) | Foreground texture bleeding into the background |
| [`robustness_synthetic_foreground.mp4`](docs/videos/robustness_synthetic_foreground.mp4) | Background texture bleeding into the foreground |

Rebuild them with:

```bash
TOSKIPORNOT_ARCHIVE=/path/to/2024-11-CIBM/artifacts scripts/fetch_artifacts.sh
scripts/make_videos.sh
```

See [toskipornot/visualization/README-videos.md](toskipornot/visualization/README-videos.md)
for the layout, perturbation semantics and caveats.

---

## Key Contributions

- **Novel robustness evaluation pipeline** using texture-based task complexity via LBP histograms
- Evidence that **skip-connections may reduce robustness** in out-of-domain (OOD) scenarios
- Finding that **attention-gated skips** help only under high-complexity conditions
- Demonstrated **failure modes** of skip-connections where performance gains come at the cost of generalizability

---

## Architecture Variants

| Group | Model | Description | Papers |
|-------|-------|-------------|--------|
| No-Skip | `NoSkipU-Net` | U-Net with all skip-connections removed | both |
| No-Skip | `NoSkipV-Net` | V-Net with all skip-connections removed | CIBM |
| Standard | `U-Net` | Identity skip-connections (concatenation) | both |
| Standard | `V-Net` | Identity skips via element-wise addition | CIBM |
| Enhanced | `AGU-Net` | Attention gating on skips | both |
| Enhanced | `UNet++` | Dense skip-connections | CIBM |

Implemented using [MONAI](https://monai.io/) and PyTorch.

---

## Datasets

### Synthetic
- Texture-based foreground/background blending, in both directions
- 9 complexity levels via α ∈ {0.1, ..., 0.9} as reported; the archived data
  covers 15 levels up to α = 0.98
- Higher α means the two textures are **more** similar, i.e. a harder task

### Medical
- **Breast Ultrasound**: Benign vs malignant tumors, from the [BUSI data set](https://www.sciencedirect.com/science/article/pii/S2352340919312181). 147 test images.
- **Colon Histology**: gland segmentation from the [GLaS challenge](https://arxiv.org/abs/1603.00275) (CIBM only). 80 test images.
- **Heart MRI**: left atrium segmentation from the [MSD challenge](https://www.nature.com/articles/s41467-022-30695-9). 120 test images.
- **Spleen CT**: binary organ segmentation, also from the [MSD challenge](https://www.nature.com/articles/s41467-022-30695-9). 216 test images.

Each medical test set exists at five perturbation levels, from speckle noise
(harder) through unperturbed to background blur (easier). See
the table above for the mapping to the paper labels.

---

## Getting Started

### Requirements
- Python 3.11.9
- PyTorch
- MONAI >= 1.3
- CUDA-enabled GPU (24GB VRAM recommended)

```bash
git clone https://github.com/amithjkamath/to_skip_or_not.git
cd to_skip_or_not
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### Configuring where the data lives

No paths are hardcoded. Copy `.env.example` to `.env` and point it at your copy
of the artifacts (or set the same variables in the environment):

```ini
TOSKIPORNOT_ARCHIVE=/path/to/2024-11-CIBM/artifacts   # the shared zip files
#TOSKIPORNOT_DATA=/mnt/big-disk/toskipornot/data      # if already unpacked
#TOSKIPORNOT_CHECKPOINTS=/mnt/big-disk/toskipornot/checkpoints
#TOSKIPORNOT_RESULTS=/mnt/big-disk/toskipornot/results
```

Everything falls back to directories inside the repository, so the defaults work
once `scripts/fetch_artifacts.sh` has unpacked the archives. Verify with:

```bash
python -m toskipornot.config
```

Only code and the demonstration videos are committed here. Input data, model
weights, per-image metrics and rendered figures live in the artifact archive and are
shared on request; the archive carries its own manifest describing the layout. Fetch
from it with:

```bash
WHAT=results scripts/fetch_artifacts.sh   # ~16 MB, all metrics + figures
scripts/fetch_artifacts.sh                # adds inputs and weights (~1 GB)
```

---

## Known caveats

Four things a reader of either paper should know. The first three are recomputable
from the archived metrics; the fourth is required to reproduce anything at all.

### MICCAI Table 1's Spleen CT *Easier* column is superseded

Of MICCAI Table 1's 27 values (3 architectures × 3 datasets × 3 perturbation levels),
24 agree with the archived per-image metrics to ≤ 0.008 Dice. The three exceptions are
all Spleen CT *Easier*:

| Spleen CT, *Easier* | MICCAI Table 1 | Archived metrics | Seeds 1 / 2 / 3 |
| --- | --- | --- | --- |
| AGU-Net | 0.809 | **0.299** | 0.012 / 0.258 / 0.628 |
| U-Net | 0.394 | **0.107** | 0.050 / 0.027 / 0.244 |
| NoSkipU-Net | 0.486 | **0.271** | 0.082 / 0.384 / 0.349 |

Spleen *Harder* and *In-domain* match MICCAI to within 0.004, so the checkpoints are
the same ones; breast ultrasound and heart *Easier* match to within 0.001. Sweeping
the blur strength shows MICCAI's 0.809 corresponds to a much weaker blur (`sigma` ≈
1.0–1.7) than the `sigma=3.0` the released test set uses, so that column was most
likely computed before the spleen blurred set settled and never refreshed. **CIBM does
not inherit the error** — it reports spleen degrading sharply under *Easier* /
*Easiest*, which is what the artifacts show. Anyone reusing MICCAI Table 1 should take
the spleen *Easier* row from CIBM instead.

### The blur parameter is a standard deviation, not a variance

Both papers describe the *Easier* / *Easiest* perturbation as a "Gaussian kernel of
variance 3.0" (and 7.0). The code passes those numbers to `skimage.filters.gaussian`
as `sigma`, so the variances are 9.0 and 49.0.

### The synthetic α convention is stated inconsistently

Both papers print the blend as `I = αI_fg + (1−α)I_bg`, which reads as "higher α is
easier". The code does `Image.blend(fg, bg, α)` = `fg·(1−α) + bg·α`, so **higher α is
harder**, and the data follows the code: α=0.10 is the in-domain easy end, α=0.98 the
hardest. Confirmed empirically — models trained at α=0.10 score ~0.99 there and
collapse as α rises. CIBM's prose and its Figure 2 agree with the code; MICCAI's prose
reads the opposite way, so MICCAI's α axis is inverted relative to the released data.

### One detail is required to reproduce any metric

MONAI's `PILReader` swaps the first two array axes relative to PIL, so the original
training and evaluation pipeline fed the networks **transposed** images. Inference has
to transpose to match. Without it the numbers are silently wrong but plausible —
breast ultrasound `UNet` in-domain reads 0.598 instead of 0.748. See `predict()` in
[`toskipornot/visualization/make_robustness_video.py`](toskipornot/visualization/make_robustness_video.py).

### A degenerate checkpoint in the released weights

GLaS V-Net predicts all-foreground for any input, at every seed. It is visible in the
metrics without loading a model: `GLaS_stats_VNet_*.csv` is identical across all five
perturbation levels (mean Dice 0.7134, std 0.2083). GLaS appears only in CIBM.

If this is useful in your research, please consider citing:

    @article{kamath2025impact,
      title={The impact of U-Net architecture choices and skip connections on the robustness of segmentation across texture variations},
      author={Kamath, Amith and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
      journal={Computers in Biology and Medicine},
      year={2025},
      publisher={Elsevier}
    }

    @inproceedings{kamath2023we,
      title={Do we really need that skip-connection? understanding its interplay with task complexity},
      author={Kamath, Amith and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={302--311},
      year={2023},
      organization={Springer}
    }
