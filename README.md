# Transfer Learning for Chest X-ray Disease Classification

This project evaluates **Frozen Backbone** vs **Fine-Tuning** strategies for multi-label chest X-ray disease classification using **ResNet50** and **DenseNet121**.  
Dataset used: **NIH ChestX-ray14**.

---

## Project Summary

We compare how freezing the pretrained backbone vs fine-tuning it affects performance, training cost, and generalization.  
Models evaluated:

- **ResNet50 (Frozen)**
- **ResNet50 (Fine-Tuned)**
- **DenseNet121 (Frozen)**
- **DenseNet121 (Fine-Tuned)**

Main metric: **AUROC (macro + per-class)**.

All experiments follow a consistent training pipeline for fair comparison.

---

## Dataset

- **NIH ChestX-ray14**, ~112k frontal chest X-rays  
- **14 disease labels** (multi-label)  
- Images resized to **224×224**, converted to **3-channel RGB**, normalized with ImageNet stats  
- Train/val/test split stored after preprocessing  

> **Dataset note:** To avoid re-running preprocessing locally, the project uses a preprocessed dataset stored on Google Drive. Download `processed_dataset.zip` and extract it to `data/processed/` to obtain the train/val/test folders and label files required by the notebooks. The processed dataset is **not** tracked in this repository (large files); see [`data/README.md`](./data/README.md) for details and the download link.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Experiments

Run **all four training notebooks**

- `resnet_frozen.ipynb`
- `resnet_finetuned.ipynb`
- `densenet_frozen.ipynb`
- `densenet_finetuned.ipynb`

Each notebook saves:

- model weights
- metrics CSVs

These saved metrics are required for the visualization notebook.

---

## Visualizations and Plots

All plots are generated in:  
📁 **`notebooks/result_plots.ipynb`**

Includes:

- Training curves
- Macro ROC curve
- Per-class AUROC bar plots
- Per-class AUROC heatmap

---

## Final Results

### Final Results (after training)

| Model | Frozen / FT | Macro AUROC |
|--------------|---------------|-------------|
| ResNet50 | Frozen | 0.6946 |
| ResNet50 | Fine-Tuned | 0.7801 |
| DenseNet121 | Frozen | 0.6981 |
| DenseNet121 | Fine-Tuned | 0.7882 |

---

## References

1. NIH ChestX-ray14 dataset paper: Wang, Xiaosong, et al. ''ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases''.

---

## Course

**Foundations of Machine Learning (CS725)**
M.Tech CSE, IIT Bombay — Oct 2025

---
