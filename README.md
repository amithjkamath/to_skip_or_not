# To Skip or Not to Skip: Investigating U-Net Skip-Connections and Task Complexity

![MICCAI 2023](https://img.shields.io/badge/Conference-MICCAI%202023-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/amithjkamath/to_skip_or_not) ![Testing](https://github.com/amithjkamath/to_skip_or_not/actions/workflows/test.yml/badge.svg)

This repository accompanies the MICCAI 2023 paper:
**"Do we really need that skip-connection? Understanding its interplay with task complexity"**  
by *[Amith Kamath](https://amithjkamath.github.io), [Jonas Willmann](https://scholar.google.com/citations?user=smUWeEgAAAAJ&hl=en), [Nicolaus Andratschke](https://scholar.google.com/citations?user=n0oz878AAAAJ&hl=en), [Mauricio Reyes](https://mauricioreyes.me/)*.

See a short video description of this work here:

[<img src="https://i.ytimg.com/vi/YreG6vC64aw/maxresdefault.jpg" width="50%">](https://youtu.be/YreG6vC64aw "To Skip or Not to Skip")

🔗 [Project Website](https://amithjkamath.github.io/projects/2023-miccai-skip-connections/)  
---

## Overview

This project explores the **necessity and effect of skip-connections in U-Net architectures** under varying task complexities in medical image segmentation. The study examines how the usefulness of skip-connections depends on the **textural similarity** between foreground and background, using:

- Controlled **synthetic texture experiments**
- Evaluations on real **medical imaging modalities**: Ultrasound (US), CT, and MRI
- Comparison across three U-Net variants:
  - **Standard U-Net**
  - **NoSkip U-Net** (no skip-connections)
  - **AGU-Net** (Attention-Gated U-Net)

---

## Key Contributions

- **Novel robustness evaluation pipeline** using texture-based task complexity via LBP histograms
- Evidence that **skip-connections may reduce robustness** in out-of-domain (OOD) scenarios
- Finding that **attention-gated skips** help only under high-complexity conditions
- Demonstrated **failure modes** of skip-connections where performance gains come at the cost of generalizability

---

## Architecture Variants

| Model        | Description                              |
|--------------|------------------------------------------|
| `U-Net`      | Standard U-Net with identity skip-connections |
| `NoSkipU-Net`| U-Net with all skip-connections removed  |
| `AGU-Net`    | U-Net with attention gating on skips     |

Implemented using [MONAI](https://monai.io/) and PyTorch.

---

## Datasets

### Synthetic
- Texture-based foreground/background blending
- 9 complexity levels via α ∈ {0.1, ..., 0.9}

### Medical
- **Breast Ultrasound**: Benign vs malignant tumors, from the [BUSI data set](https://www.sciencedirect.com/science/article/pii/S2352340919312181).
- **Spleen CT**: binary organ segmentation from the [MSD challenge](https://www.nature.com/articles/s41467-022-30695-9).
- **Heart MRI**: left atrium segmentation, also from the [MSD challenge](https://www.nature.com/articles/s41467-022-30695-9).

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

If this is useful in your research, please consider citing:

    @inproceedings{kamath2023we,
      title={Do we really need that skip-connection? understanding its interplay with task complexity},
      author={Kamath, Amith and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={302--311},
      year={2023},
      organization={Springer}
    }
