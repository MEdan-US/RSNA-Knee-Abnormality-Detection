# RSNA Knee Abnormality Detection Solution

![Competition](https://img.shields.io/badge/Competition-RSNA%20Knee%20Abnormality%20Detection-1f6feb?style=flat-square)
![Score](https://img.shields.io/badge/Leaderboard-0.917%20Macro%20ROC--AUC-2ea043?style=flat-square)
![5-Fold CV](https://img.shields.io/badge/5--Fold%20CV-0.9227-2ea043?style=flat-square)
![Framework](https://img.shields.io/badge/PyTorch-timm-ee4c2c?style=flat-square)

## Problem Statement

The **RSNA Knee Abnormality Detection** competition requires multi-label classification of **12 distinct knee pathologies** from multi-series, multi-plane knee MRI DICOM studies (**Sagittal**, **Coronal**, and **Axial**), evaluated by **Macro ROC-AUC**:

> `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

---

## Key Challenges & Difficulties

1. **Severe Scanner & Protocol Heterogeneity:** Studies originate from diverse MRI vendors (Siemens, GE, Philips) and field strengths (1.5T / 3.0T) with inconsistent slice thickness, field-of-view (FOV), and contrast protocols (FS, T1, PD/T2) — alongside non-diagnostic scout/localizer and motion-corrupted series.
2. **3D Coordinate & Laterality Ambiguity:** Scanners store slices in arbitrary directions (causing 180° normal vector inversions), while Left vs. Right knee mirror symmetry flips Medial and Lateral compartments if not normalized.
3. **Noisy Weak Supervision:** Out of **4,407** training studies, only **58** cases have radiologist-verified Gold Standard labels. The vast majority rely on noisy, incomplete LLM-extracted report labels with frequent missing-mention false negatives.
4. **Multi-Scale & Plane-Specific Pathologies:** Abnormalities range from millimeter-scale focal tears (ACL, meniscal horns) to diffuse joint-wide findings (effusion, osteoarthritis), each visible only within specific anatomical planes and depth bands.
5. **Hidden Test Scale & Strict Memory Ceilings:** Kaggle's hidden test set contains thousands of uncompressed DICOM volumes that must be processed 100% offline within tight host RAM, GPU VRAM, and runtime limits without OOM crashes.

---

## High-Level Solution Overview

- **Physics-Standardized Preprocessing:** Filters scout/corrupted series, sorts slices via 3D LPS normal projection, applies a `130 mm` physical FOV crop, builds 21 true 2.5D triplets (`3.5 mm` step, `224×224`), and aligns Right knees to Left orientation on GPU.
- **5-Fold ConvNeXt-Tiny 2.5D:** Combines multi-scale 2D FPN spatial fusion, trainable GeM pooling, and anatomy-aware plane attention with warm-start Gaussian depth priors.
- **Two-Stage Cross-Modal Self-Correction:** Stage 1 pretrains on confident labels (masking uncertain `w <= 0.30` positions), followed by selective self-correction (100% protecting the 58 Gold Standards) and Stage 2 SWA retraining.
- **Zero-RAM Streaming Inference:** Preloads all 5 folds in `564 MB` VRAM and streams test studies in small batches (`B=8`, `<100 MB` RAM) with 3-pass physical depth slice-jittering TTA and rank normalization.
- **Results:** **0.917** Kaggle Leaderboard Macro ROC-AUC (**0.9227** 5-Fold Mean CV | **0.9193** on the 58 Gold Standards).

---

## Repository Structure

| Notebook | Description |
| :--- | :--- |
| [`rsna-data-preprocessed.ipynb`](rsna-data-preprocessed.ipynb) | Physics-driven DICOM preprocessing (`21` 2.5D slices @ `224×224`). |
| [`rsna-2-stage.ipynb`](rsna-2-stage.ipynb) | 5-Fold Two-Stage ConvNeXt-Tiny training & cross-modal self-correction. |
| [`rsna-convnext-inference.ipynb`](rsna-convnext-inference.ipynb) | Offline zero-RAM streaming 5-fold inference & submission generation. |

### Quick Start
1. Run `rsna-data-preprocessed.ipynb` to generate `knee_train_21slices_224.npy` and `train_metadata_21slices.csv`.
2. Run `rsna-2-stage.ipynb` to train the 5-fold checkpoints (`best_convnext_tiny_kaggle_fold{0..4}.pth`).
3. Attach the checkpoints to `rsna-convnext-inference.ipynb` and run offline on Kaggle to produce `submission.csv`.
