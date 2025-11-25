# 📦 Model Download & Usage Guide

This directory contains links to pretrained model checkpoints used in our experiments.
To keep the repository lightweight, model weights are hosted on Google Drive instead of GitHub.

---

## 📁 Available Models

| Architecture | Setting         | Download Link              | Filename Example             |
| ------------ | --------------- | -------------------------- | ---------------------------- |
| ResNet50     | Frozen Backbone | [Download](https://drive.google.com/file/d/1ez4II56Bg_pvLEPmUcdUP6vjN8ahPxZW/view?usp=sharing) | `resnet_frozen.pt`      |
| ResNet50     | Fine-Tuned      | [Download](https://drive.google.com/file/d/1Jx_e9D4jEK4JbZpXe1cm_asF1SgvxNoO/view?usp=sharing) | `resnet_finetuned.pt`   |
| DenseNet121  | Frozen Backbone | [Download](https://drive.google.com/file/d/1hZq1SGmrL9Ukz8_wRwu7ta55UBkvwzYK/view?usp=sharing) | `densenet_frozen.pt`    |
| DenseNet121  | Fine-Tuned      | [Download](https://drive.google.com/file/d/1pxsuY1yICfYDgCTCVizx8q0tDX7t52PM/view?usp=sharing) | `densenet_finetuned.pt` |

---

## 📥 Download Instructions

Each Google Drive link provides:

* ✅ Model file
* ✅ Direct `.pt` checkpoint

---

## 📂 Directory Structure (after download)

Place downloaded checkpoints here:

```
models/
├── resnet_frozen.pt
│── resnet_finetuned.pt
├── densenet_frozen.pt
└── densenet_finetuned.pt
```

---

If you encounter issues accessing files, open a GitHub issue in this repository.
