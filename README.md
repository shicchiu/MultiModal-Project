# MultiModal-Project
 
Team project for **INF 385T: Deep Learning and Multimodal Systems**
 
## Overview
 
This project develops a multimodal classification system that predicts the presence of **Pneumonia** and **Pneumothorax** from chest X-ray images paired with corresponding radiology reports (Findings and Impression sections). Labels are sourced from CheXbert annotations, where `1` indicates a confirmed condition and `0` indicates no finding. Uncertain labels (`-1`) are excluded from training.
 
We experiment with six label combinations for the two conditions: `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)`, `(1,NaN)`, and `(NaN,1)`. Our multimodal fusion model combines a ResNet50 image encoder with a BERT text encoder, and is compared against image-only and text-only baselines.
 
## Repository Structure
 
```
MultiModal-Project/
├── Cross-Modal_Alignment_MIMIC-CXR.ipynb       # Main notebook: full pipeline
├── huggingface data/
│   └── mimic_cxr_with_chexbert_labels.ipynb    # How CheXbert labels were added to MIMIC-CXR
└── models/
    ├── image_only_baseline_HF.ipynb             # Image-only baseline
    └── text_only_baseline_HF.ipynb              # Text-only baseline
```
 
## Dataset
 
The dataset used in this project is publicly available on Hugging Face:
 
**[cchitse/mimic-cxr-with-chexbert-labels](https://huggingface.co/datasets/cchitse/mimic-cxr-with-chexbert-labels)**
 
It combines MIMIC-CXR chest X-ray images with structured CheXbert labels. No access token is required to download it.
 
The notebook `huggingface data/mimic_cxr_with_chexbert_labels.ipynb` documents how we constructed this dataset by appending CheXbert labels to the original MIMIC-CXR records.
 
## How to Run
 
All notebooks are designed to run on **Google Colab**. No local setup is required.
 
1. Open the desired notebook in Google Colab.
2. Run all cells sequentially. The dataset will be downloaded directly from Hugging Face at runtime — no manual data download or Drive mounting is needed.
3. Start with `Cross-Modal_Alignment_MIMIC-CXR.ipynb` for the complete analysis, which covers data preprocessing, model training, evaluation, and Grad-CAM visualization.
The baseline notebooks under `models/` were used for early-stage testing and can be run independently.
 
## Dependencies
 
The following packages are required and can be installed within Colab:
 
- `torch`, `torchvision`
- `transformers`
- `datasets` (Hugging Face)
- `scikit-learn`
- `matplotlib`, `numpy`, `opencv-python`
