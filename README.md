# 🩺 Medical Image Segmentation for Pathology Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-Deep_Learning-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Computer_vision)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

## 📌 Project Overview
This project features an end-to-end deep learning pipeline designed to automate the detection of localized anomalies and pathologies in medical scans. Specifically, it implements a **U-Net architecture** built in PyTorch to perform semantic segmentation on colonoscopy frames, accurately isolating and highlighting colon polyps. 

Automated pathology detection serves as a critical secondary measure for gastroenterologists, reducing human error and improving early intervention rates for colorectal diseases.

## 🧠 Architecture & Methodology
Rather than relying on traditional 2D signal processing or manual feature engineering, this project utilizes **U-Net**, the industry-standard architecture for biomedical image segmentation. 

* **Encoder (Contracting Path):** Acts as a traditional Convolutional Neural Network (CNN), progressively downsampling the image to extract deep feature representations (textures, edges, shapes).
* **Decoder (Expanding Path):** Upsamples the compressed feature map back to the original 256x256 image resolution.
* **Skip Connections:** Copies high-resolution spatial coordinates directly from the Encoder to the Decoder. This crucial step allows the network to combine its deep understanding of *what* a polyp looks like with the exact pixel coordinates of *where* it is located.

### Loss Function & Evaluation Metrics
The model is optimized using `BCEWithLogitsLoss` for superior numerical stability. 

To evaluate model accuracy, **Intersection over Union (IoU)** is used rather than standard pixel accuracy. Because healthy background tissue occupies significantly more space than a localized anomaly, IoU provides a rigorous evaluation by isolating how perfectly the model's *predicted* polyp mask overlaps with the *actual* ground-truth polyp mask.

## 📊 The Dataset
The model was trained on the **CVC-ClinicDB** dataset, which consists of 612 high-resolution frames extracted from colonoscopy videos alongside expertly annotated ground-truth masks. The dataset was obtained via Kaggle.

**Dataset Citation:**
> Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., & Vilariño, F. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. *Computerized Medical Imaging and Graphics*, 43, 99-111.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python installed and are working in the project's root directory, then run:
```bash
pip install torch torchvision opencv-python matplotlib numpy scikit-learn
python dataset.py
python model.py
python train.py
