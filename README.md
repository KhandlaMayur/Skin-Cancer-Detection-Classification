# DermaVision AI

AI-assisted dermoscopic skin lesion classification using YOLO26M-CLS trained on the HAM10000 dataset.

> **Research prototype.** This project is an academic AI system developed for research and educational purposes. It is not a medical diagnostic device and must not be used as a substitute for clinical evaluation by a qualified healthcare professional.

---

## Overview

DermaVision AI classifies dermoscopic images of skin lesions across seven diagnostic categories from the HAM10000 benchmark dataset. The system is built on a YOLO26M-CLS architecture, trained with a lesion-aware data split and moderate class oversampling to partially address the severe class imbalance present in the dataset.

The inference pipeline is deployed as an interactive Streamlit web application. Users upload a dermoscopic image and receive the predicted lesion class, model confidence, a full probability distribution across all seven classes, and Top-3 ranked predictions.

---

## Problem Statement

HAM10000 contains 10,015 dermoscopic images spanning seven lesion types with a strongly imbalanced distribution. Melanocytic nevi (nv) account for approximately 67% of all images, while rare classes such as dermatofibroma and vascular lesions represent less than 2% each. This imbalance causes standard classifiers to under-perform on minority classes, particularly on melanoma — the most clinically important category — which shares significant visual features with benign nevi.

The primary modelling challenge is melanoma-to-nevus confusion: the visual similarity between early-stage melanoma and benign moles means even human experts require dermoscopy and histopathology for reliable discrimination.

---

## Dataset

**Source:** HAM10000 — Human Against Machine with 10000 training images  
**Total images:** 10,015  
**Input resolution:** 224 × 224 pixels

### Diagnostic Classes

| Code | Condition |
|------|-----------|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma (Bowen's Disease) |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions (Seborrheic Keratoses) |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesions |

### Split Strategy

| Split | Images |
|-------|-------:|
| Train (original) | 6,981 |
| Train (oversampled) | 10,683 |
| Validation | 1,532 |
| Test | 1,502 |

A **lesion-aware split** was applied to ensure that images originating from the same physical lesion remain within a single partition. This prevents data leakage between train, validation, and test sets — a critical requirement given that HAM10000 contains multiple images per lesion.

Moderate oversampling was applied to the training set only to partially compensate for class imbalance without introducing excessive synthetic duplication.

---

## Model

**Architecture:** YOLO26M-CLS (YOLO26 medium classification variant)  
**Input size:** 224 × 224  
**Classes:** 7  
**Best configuration:** Oversampled_YOLO26M_CLS

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Pretrained weights | ImageNet |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Batch size | 32 |
| LR schedule | Cosine annealing |
| Mixed precision (AMP) | Enabled |

---

## Experimental Results

Results on the held-out test set (1,502 images):

| Metric | Baseline | Oversampled |
|--------|--------:|------------:|
| Accuracy | 81.76% | **83.16%** |
| Balanced Accuracy | 67.82% | **68.98%** |
| Macro F1-score | 69.14% | **71.77%** |
| Macro ROC-AUC | 95.93% | **96.33%** |
| Melanoma Recall | 43.11% | **49.10%** |

Moderate oversampling improved performance across all metrics:

- Accuracy: **+1.40 pp**
- Balanced Accuracy: **+1.16 pp**
- Macro F1: **+2.63 pp**
- Melanoma Recall: **+5.99 pp**

The improvement in melanoma recall is particularly notable given the clinical importance of that class.

---

## Error Analysis

| Metric | Value |
|--------|------:|
| Total test images | 1,502 |
| Correctly classified | 1,249 |
| Misclassified | 253 |
| Overall accuracy | 83.16% |
| Melanoma test images | 167 |
| Melanoma correctly classified | 82 |
| Melanoma missed | 85 |
| Melanoma predicted as nevus | 65 |
| Mel → Nv confusion rate | 38.92% |
| Mean confidence (wrong nv prediction) | 0.8813 |

**Primary limitation:** Melanoma-to-nevus confusion. Of the 85 missed melanoma cases, 65 (76.5%) were misclassified as melanocytic nevus, often with high model confidence (mean: 0.8813). This reflects the genuine visual similarity between these two classes under dermoscopy and underscores why this system must not be used as a clinical screening tool.

---

## Streamlit Application

The deployed application provides:

- **Image upload** — JPG, JPEG, and PNG dermoscopic images
- **Predicted class** — top-1 classification with full class name and HAM10000 code
- **Confidence score** — model softmax probability for the predicted class
- **Probability distribution** — horizontal bar chart across all seven diagnostic categories
- **Top-3 predictions** — ranked by confidence
- **Detailed table** — full probability breakdown for all classes (expandable)
- **Research disclaimer** — visible at all times
- **Model information** — architecture, dataset, and test metrics in the sidebar

---

## Project Workflow & Architecture

The system follows a research-to-deployment pipeline designed for dermoscopic skin lesion classification:

1. Data preparation
   - HAM10000 images and metadata are loaded from the dataset folders.
   - A lesion-aware split is applied so multiple images from the same lesion stay in the same partition.
   - Class imbalance is handled with a moderate oversampling strategy during training.

2. Model training
   - The project trains a YOLO26M-CLS classifier for 7 lesion categories.
   - Baseline and oversampled model variants are trained and compared.
   - Training runs are logged under the `runs/` directory with metrics and weights.

3. Evaluation and analysis
   - Model performance is assessed on a held-out test set using accuracy, balanced accuracy, macro F1, macro ROC-AUC, and per-class recall.
   - Misclassification patterns and error analysis are stored in `research_outputs/` and publication tables.

4. Deployment
   - The trained model is exported to `final_model/YOLO26M_HAM10000_FINAL_best.pt`.
   - A Streamlit app loads the model and classifies uploaded dermoscopic images in real time.

### High-level architecture

- Frontend: Streamlit web interface (`app.py`)
- Model: YOLO26M-CLS image classifier
- Data layer: HAM10000 metadata + lesion-aware train/validation/test split
- Training outputs: model checkpoints, metrics, confusion matrices, plots
- Deployment layer: local and cloud-ready Streamlit inference app

---

## Key Features

- Multi-class skin lesion classification across 7 HAM10000 classes
- YOLO26M-CLS based model optimized for dermoscopic image analysis
- Lesion-aware train/validation/test split to reduce data leakage
- Oversampling support to improve minority-class recognition
- Real-time image upload and prediction through Streamlit
- Probability distribution and Top-3 ranked predictions for each image
- Research-ready outputs including confusion matrices, metrics, and comparison reports
- Model weights and evaluation artifacts included for reproducibility
- Safe research disclaimer and non-clinical usage guidance

---

## How to Run the Project

### 1) Clone the repository

```bash
git clone https://github.com/KhandlaMayur/Skin-Cancer-Detection-Classification
cd Skin-Cancer-Detection-Classification
```

### 2) Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4) Start the app

```bash
streamlit run app.py
```

This launches the web app locally at:

```text
http://localhost:8501
```

### 5) Use the app

- Upload a skin lesion image in JPG, JPEG, or PNG format
- View the predicted class, confidence score, and probability chart
- Inspect the Top-3 predictions and full class distribution

### Notes

- The trained model is already present in `final_model/YOLO26M_HAM10000_FINAL_best.pt`.
- If CUDA is available, inference will use GPU automatically; otherwise it falls back to CPU.
- This is a research prototype and should not be used as a medical diagnostic tool.

---

## Project Structure

```
DermaVision-AI/
├── app.py                                  # Streamlit application
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Repository exclusion rules
├── README.md                               # Project documentation
├── HAM10000_images/                        # Raw HAM10000 image folder
├── HAM10000_YOLO/                          # Prepared YOLO-style dataset
├── HAM10000_YOLO_OVERSAMPLED/              # Oversampled training dataset
├── final_model/                            # Final trained weights and summaries
│   ├── YOLO26M_HAM10000_FINAL_best.pt
│   ├── final_model_comparison.csv
│   ├── final_project_summary.txt
│   └── final_research_summary.json
├── runs/                                   # Training logs and checkpoints
├── research_outputs/                       # Evaluation and error-analysis artifacts
├── publication_figures/                    # Plots for reports
├── publication_tables/                     # Publication-ready metrics tables
├── weights/                                # Supporting model weights
├── canser_classification.ipynb             # Notebook for experimentation
├── HAM10000_lesion_aware_split.csv         # Lesion-aware split metadata
├── HAM10000_metadata.csv                   # HAM10000 dataset metadata
└── yolo26m-cls.pt                          # YOLO classification pretrained weights
```

---

## Installation

```bash
git clone https://github.com/KhandlaMayur/Skin-Cancer-Detection-Classification
cd DermaVision-AI
python -m pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended. A virtual environment is strongly advised.

---

## Run Locally

```bash
python -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

The model is loaded once at startup using `@st.cache_resource`. Inference runs on GPU if CUDA is available, and falls back to CPU automatically.

---

## Deployment

This repository is structured for deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to GitHub
2. Sign in at [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select this repository, branch `main`, and entrypoint `app.py`
5. Click **Deploy**

All dependencies are resolved from `requirements.txt`. The trained model file is included in the repository (19.93 MB — within GitHub's standard file limit).

---

## Technologies

| Component | Library / Framework |
|-----------|-------------------|
| Application framework | Streamlit |
| Model architecture | Ultralytics YOLO26 |
| Deep learning backend | PyTorch |
| Data handling | Pandas, NumPy |
| Image processing | Pillow, OpenCV Headless |
| Visualization | Altair |
| Language | Python 3.10+ |

---

## Research Limitations

This model has the following known limitations:

- **Class imbalance:** Despite oversampling, the training distribution remains imbalanced. Minority-class performance (df, vasc, akiec) is lower than majority-class performance.
- **Melanoma-nevus confusion:** Visual similarity between melanoma and benign nevi produces the highest error rate (mel → nv: 38.92%), often with high confidence. This is the critical failure mode.
- **Dataset-specific generalization:** The model was trained and evaluated exclusively on HAM10000. Generalization to images from different scanners, clinical settings, or patient populations has not been evaluated.
- **No external clinical validation:** Results are reported on the HAM10000 test split only. No prospective clinical study or external validation has been conducted.
- **No multimodal features:** The model uses image pixels only. Clinical metadata (patient age, anatomical site, lesion history) is not incorporated.
- **Research prototype:** This system has not undergone the regulatory review required for use as a medical device.

---

## Future Work

Research directions that could meaningfully improve the system:

- **Loss function:** Class-balanced cross-entropy or focal loss to further address minority-class under-representation
- **Augmentation:** Melanoma-targeted augmentation strategies (colour jitter, dermoscopy-specific transforms) to improve melanoma recall
- **Resolution:** Experiments at higher input resolution (384 × 384, 448 × 448) to capture finer lesion detail
- **External validation:** Evaluation on independent datasets (ISIC 2019, ISIC 2020, Derm7pt) to assess real-world generalization
- **Calibration:** Temperature scaling or Platt calibration to align confidence scores with actual accuracy
- **Explainability:** Grad-CAM or SHAP visualization to understand which image regions drive predictions
- **Multimodal fusion:** Integration of structured clinical metadata to reduce melanoma-nevus ambiguity

---

## Disclaimer

This project is an academic research prototype and is **not a medical diagnostic device**.

Model predictions must not replace examination, dermoscopy assessment, histopathological diagnosis, or advice from a qualified healthcare professional. The system has not been validated for clinical use and carries no regulatory approval for diagnostic applications.
