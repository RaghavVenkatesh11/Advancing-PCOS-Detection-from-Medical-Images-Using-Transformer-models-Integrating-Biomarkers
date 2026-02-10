# Advancing PCOS Detection: Multimodal Transformer & Biomarker Integration

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)

## 📌 Project Overview
This project implements a **Hybrid Multimodal Artificial Intelligence Framework** for the detection of Polycystic Ovary Syndrome (PCOS). Unlike traditional single-stream diagnostic tools, this system utilizes a **Dual-Stream Late Fusion Architecture** that integrates:

1.  **Unstructured Visual Data:** Ovarian ultrasound images processed via a **Swin Transformer** (Tiny Variant) to capture global contextual features and texture patterns.
2.  **Structured Clinical Data:** The "Golden 8" clinical biomarkers processed via a **CatBoost Classifier** to handle non-linear associations.

By combining these modalities through a Multi-Layer Perceptron (MLP), the system reduces false positives and improves diagnostic reliability compared to using imaging or biomarkers alone.

## 🚀 Key Features
* **Multimodal Fusion:** Integrates ultrasound imaging with hormonal and metabolic indicators.
* **Swin Transformer Backbone:** Utilizes hierarchical vision transformers with shifted windows for robust image feature extraction.
* **CatBoost Clinical Encoder:** Handles categorical and numerical data with high resistance to overfitting.
* **High Performance:** The proposed model achieves a Recall of **98.5%** and an F1-Score of **99.09%**.
* **Interactive UI:** Powered by **Streamlit** for real-time diagnostic support.

## 📂 Dataset
The model was developed using a dataset comprising:
* **Ultrasound Images:** 1,924 transvaginal ultrasound images, augmented to approximately 9,800 samples (resized to 224x224 pixels).
* **Clinical Data:** Records from 541 patients, utilizing **8 key biomarkers**:
    * Age, BMI, Cycle Length, Waist-to-Hip Ratio
    * FSH (Follicle-Stimulating Hormone), LH (Luteinizing Hormone)
    * AMH (Anti-Müllerian Hormone)
    * Volumetric parameters

> **⚠️ Dataset Access:**
> The dataset is not included in this repository. Please download it from the source below and place it in the `data/` directory.
>
> (https://www.kaggle.com/datasets/rvenkateswarareddy/dataset)

## 🛠️ Methodology
The pipeline consists of four distinct stages:
1.  **Preprocessing:**
    * *Images:* Z-score normalization and augmentation (rotation, flipping, brightness jitter).
    * *Clinical:* Median imputation for missing values and feature selection.
2.  **Feature Extraction:** Independent processing via Swin Transformer (Visual) and CatBoost (Clinical).
3.  **Fusion:** Concatenation of probability outputs from both streams.
4.  **Classification:** A fully connected MLP with a sigmoid activation function outputs a probability score (Threshold: 0.5).

## 💻 Installation

### Prerequisites
* Python 3.8+
* pip

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/RaghavVenkatesh11/pcos-multimodal-detection.git](https://github.com/RaghavVenkatesh11/pcos-multimodal-detection.git)
    cd pcos-multimodal-detection
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage
To launch the web interface, run the following command in your terminal:

```bash
streamlit run app.py
