# Transfer Learning for Chest X-ray Disease Classification

> ⚠️ **TENTATIVE**
> All details below (dataset, preprocessing, experiments, setup, results, reproducibility, references, contributors) are tentative.

---

## 🩺 Project Summary

This repository compares **Frozen Backbone** vs **Fine-Tuning** strategies for multi-label chest disease classification using **ResNet50** and **DenseNet121** on the **NIH ChestX-ray14** dataset.

---

## 🎯 Objective

Compare performance and tradeoffs between:

* Training only the final classifier layers on pretrained backbones (**Frozen Backbone**)
* Unfreezing and fine-tuning pretrained backbone weights (**Fine-Tuning**)

**Models Used**

* ResNet50 (pretrained on ImageNet)
* DenseNet121 (pretrained on ImageNet)

**Task:** Multi-label classification for 14 thoracic diseases in the NIH ChestX-ray14 dataset.

---

## 📁 Dataset (TENTATIVE)

* **Source:** NIH ChestX-ray14 (~112,000 frontal chest X-rays).
* **Original image size:** 1024 × 1024 (grayscale)
* **Preprocessing plan:**

  * Resize to **224 × 224** (or 256 → center crop to 224)
  * Convert grayscale → 3-channel RGB (repeat same channel)
  * Normalize with ImageNet mean and std
  * Tentative split: 70% train / 15% validation / 15% test
  * Optionally use a smaller subset for faster experiments (document final subset used)

---

## 🧹 Preprocessing Notes

* Pretrained CNNs (ResNet, DenseNet) expect **3-channel 224×224 inputs**.
* Preprocessing (resizing, conversion, normalization) is performed **once** and saved to a smaller processed dataset.
* The original NIH dataset (~45GB) can be reduced significantly after resizing and optional subsetting.

---

## 🧪 Planned Experiments

1. **ResNet50 – Frozen Backbone**
2. **ResNet50 – Fine-Tuned**
3. **DenseNet121 – Frozen Backbone**
4. **DenseNet121 – Fine-Tuned**

**Possible extensions (optional):**

* Partial fine-tuning (last N blocks)
* Data augmentation effects
* Different learning rates for backbone vs head

---

## 📊 Evaluation Metrics

* AUROC (per class + macro average) — **primary metric**
* F1-score (per class + macro average)
* Accuracy (less relevant for imbalanced multi-label tasks)
* Training time, GPU usage, and parameter count (for resource comparison)

---

## 🏫 Course

**Foundations of Machine Learning (CS725)**
M.Tech CSE, IIT Bombay — Oct 2025 *(Tentative project)*

---
