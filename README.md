# Advancing PCOS Detection: Multimodal Transformer & Biomarker Integration

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)

## 📌 Project Overview
[cite_start]This project implements a **Hybrid Multimodal Artificial Intelligence Framework** for the detection of Polycystic Ovary Syndrome (PCOS)[cite: 118]. Unlike traditional single-stream diagnostic tools, this system utilizes a **Dual-Stream Late Fusion Architecture** that integrates:

1.  [cite_start]**Unstructured Visual Data:** Ovarian ultrasound images processed via a **Swin Transformer** (Tiny Variant) to capture global contextual features and texture patterns[cite: 119, 131].
2.  [cite_start]**Structured Clinical Data:** The "Golden 8" clinical biomarkers processed via a **CatBoost Classifier** to handle non-linear associations[cite: 126, 142].

[cite_start]By combining these modalities through a Multi-Layer Perceptron (MLP), the system reduces false positives and improves diagnostic reliability compared to using imaging or biomarkers alone[cite: 147, 163].

## 🚀 Key Features
* [cite_start]**Multimodal Fusion:** Integrates ultrasound imaging with hormonal and metabolic indicators[cite: 30].
* [cite_start]**Swin Transformer Backbone:** Utilizes hierarchical vision transformers with shifted windows for robust image feature extraction[cite: 138].
* [cite_start]**CatBoost Clinical Encoder:** Handles categorical and numerical data with high resistance to overfitting[cite: 142].
* [cite_start]**High Performance:** The proposed model achieves a Recall of **98.5%** and an F1-Score of **99.09%**[cite: 199, 202].
* **Interactive UI:** Powered by **Streamlit** for real-time diagnostic support.

## 📂 Dataset
The model was developed using a dataset comprising:
* [cite_start]**Ultrasound Images:** 1,924 transvaginal ultrasound images, augmented to approximately 9,800 samples (resized to 224x224 pixels)[cite: 122, 172, 193].
* [cite_start]**Clinical Data:** Records from 541 patients, utilizing **8 key biomarkers**[cite: 122, 126]:
    * Age, BMI, Cycle Length, Waist-to-Hip Ratio
    * FSH (Follicle-Stimulating Hormone), LH (Luteinizing Hormone)
    * AMH (Anti-Müllerian Hormone)
    * Volumetric parameters

> **⚠️ Dataset Access:**
> The dataset is not included in this repository. Please download it from the source below and place it in the `data/` directory.
>
> (https://www.kaggle.com/datasets/rvenkateswarareddy/dataset)

## 🛠️ Methodology
[cite_start]The pipeline consists of four distinct stages[cite: 120]:
1.  **Preprocessing:**
    * [cite_start]*Images:* Z-score normalization and augmentation (rotation, flipping, brightness jitter)[cite: 129, 130].
    * [cite_start]*Clinical:* Median imputation for missing values and feature selection[cite: 125].
2.  [cite_start]**Feature Extraction:** Independent processing via Swin Transformer (Visual) and CatBoost (Clinical)[cite: 131, 142].
3.  [cite_start]**Fusion:** Concatenation of probability outputs from both streams[cite: 146].
4.  [cite_start]**Classification:** A fully connected MLP with a sigmoid activation function outputs a probability score (Threshold: 0.5)[cite: 147, 153].

## 💻 Installation

### Prerequisites
* Python 3.8+
* pip

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/pcos-multimodal-detection.git](https://github.com/yourusername/pcos-multimodal-detection.git)
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
