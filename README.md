# To Skip or Not to Skip: Investigating U-Net Skip-Connections and Task Complexity

![MICCAI 2023](https://img.shields.io/badge/Conference-MICCAI%202023-blue)

This repository accompanies the MICCAI 2023 paper:
**"Do we really need that skip-connection? Understanding its interplay with task complexity"**  
by *Amith Kamath, Jonas Willmann, Nicolaus Andratschke, Mauricio Reyes*.

🔗 [Project Website](https://amithjkamath.github.io/projects/2023-miccai-skip-connections/)  
---

## 🧠 Overview

This project explores the **necessity and effect of skip-connections in U-Net architectures** under varying task complexities in medical image segmentation. The study examines how the usefulness of skip-connections depends on the **textural similarity** between foreground and background, using:

- Controlled **synthetic texture experiments**
- Evaluations on real **medical imaging modalities**: Ultrasound (US), CT, and MRI
- Comparison across three U-Net variants:
  - **Standard U-Net**
  - **NoSkip U-Net** (no skip-connections)
  - **AGU-Net** (Attention-Gated U-Net)

---

## 🔍 Key Contributions

- 🧪 **Novel robustness evaluation pipeline** using texture-based task complexity via LBP histograms
- ⚖️ Evidence that **skip-connections may reduce robustness** in out-of-domain (OOD) scenarios
- 💡 Finding that **attention-gated skips** help only under high-complexity conditions
- 📉 Demonstrated **failure modes** of skip-connections where performance gains come at the cost of generalizability

---

## 🏗️ Architecture Variants

| Model        | Description                              |
|--------------|------------------------------------------|
| `U-Net`      | Standard U-Net with identity skip-connections |
| `NoSkipU-Net`| U-Net with all skip-connections removed  |
| `AGU-Net`    | U-Net with attention gating on skips     |

Implemented using [MONAI](https://monai.io/) and PyTorch.

---

## 🗂️ Datasets

### Synthetic
- Texture-based foreground/background blending
- 9 complexity levels via α ∈ {0.1, ..., 0.9}

### Medical
- 🩺 **Breast Ultrasound**: Benign vs malignant tumors  
- 🧠 **Spleen CT**: Organ segmentation  
- ❤️ **Heart MRI**: Cardiac structure segmentation

---

## 🚀 Getting Started

### Requirements
- Python 3.10
- PyTorch
- MONAI >= 1.1
- CUDA-enabled GPU (24GB VRAM recommended)

```bash
pip install -r requirements.txt
