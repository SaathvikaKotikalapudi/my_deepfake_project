# Deepfake Detection & Generation Project

A machine learning framework developed to analyze, generate, and detect deepfake media. This project implements state-of-the-art computer vision models to distinguish between manipulated imagery and authentic digital media.

## 📌 Project Overview
With the rapid advancements in generative adversarial networks (GANs) and diffusion models, media forensics has become crucial. This project focuses on building an end-to-end pipeline capable of processing image/video datasets, extracting frame-level biological or digital artifacts, and classifying media as `REAL` or `FAKE`.

## 🗂️ Repository Structure
```text
├── dataset/                  # Local directory (Excluded from GitHub)
│   ├── train/
│   └── test/
│       ├── FAKE/
│       └── REAL/
├── models/                   # Saved model architectures and weights (.h5, .pt)
├── notebooks/                # Jupyter Notebooks for experimentation
├── src/                      # Source code scripts
│   ├── preprocessing.py      # Face extraction, alignment, and normalization
│   ├── train.py              # Model training loop
│   └── detect.py             # Inference script for testing new media
├── FINAL_DATASET.csv         # Metadata tracking file
├── .gitignore                # Prevents heavy media tracking
└── README.md
## 📊 Dataset Configuration
⚠️ Note: The dataset/ directory and FINAL_DATASET.csv are omitted from this repository due to GitHub storage limitations.

To reproduce the training results, download your image/video corpus and structure it locally as follows:

- Place training samples in dataset/train/ split into REAL and FAKE subdirectories.

- Place validation samples in dataset/test/ split into REAL and FAKE subdirectories.
## 🛠️ Tech Stack & Dependencies
Languages: Python

Frameworks: PyTorch / TensorFlow (Keras)

Computer Vision: OpenCV, Face Recognition API

Data Handling: Pandas, NumPy, Scikit-learn
