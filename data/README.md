# 📂 Dataset Storage

This project uses the **NIH ChestX-ray14** dataset.

To avoid preprocessing every time, the processed dataset is stored in Google Drive.

---

## 📦 Contents

This folder is **NOT** tracked in Git (large files).  
Expected structure after download:

```
data/
 ├── raw/                 # Original NIH dataset (not uploaded)
 ├── processed/           # Preprocessed train/val/test images
 └── processed_dataset.zip # Zipped processed dataset for Colab use
```

---

## 🔗 Download Processed Dataset

Processed dataset (ready to train):  
👉 **Google Drive:** [processed_dataset.zip folder](https://drive.google.com/drive/folders/1HLOxnhiwPN7Pj3-XISS6Co4JjtBhyrTD)

After downloading:

```
unzip data/processed_dataset.zip -d data/processed
```

This will create:

```
data/processed/train/
data/processed/val/
data/processed/test/
```

---

## 📝 Notes

- Original raw NIH dataset is too large for Git — download from Kaggle instead.
- `processed_dataset.zip` includes:
  - resized 224x224 images
  - train/val/test split
  - labels CSV & txt files

If you want to re-generate processing:  
See `notebooks/prepare_dataset.ipynb` or `scripts/prepare_dataset.py`.
