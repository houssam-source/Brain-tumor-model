# Brain‑Tumor Classification

> Multi‑class classification of brain MRI scans with >95 % accuracy.  
> Uses **PCA** for dimensionality reduction and an **XGBoost + Logistic‑Regression** ensemble.

---  

## Table of Contents
1. [Problem Statement](#problem-statement)  
2. [Dataset](#dataset)  
3. [Methodology](#methodology)  
4. [Results](#results)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Reproducing the Experiments](#reproducing-the-experiments)  
8. [License](#license)  
9. [Contributing](#contributing)  
10. [References](#references)  

---  

## Problem Statement
Automatic detection of brain tumor type (meningioma, glioma, pituitary, etc.) from T1‑weighted MRI slices.  
A reliable, fast classifier can assist radiologists in triaging cases and reducing diagnostic time.

## Dataset
* **Source** – Kaggle “Brain MRI Images for Brain Tumor Detection” (5 200 labeled images).  
* **Classes** – 4 tumor types + “no tumor”.  
* **Pre‑processing** –  
  * Resized to 224 × 224, grayscale → 3‑channel (to keep compatibility with pretrained nets).  
  * Normalised to `[0, 1]`.  
  * Train/validation split: **80 % / 20 %** (stratified).  

## Methodology
1. **Feature extraction** –  
   * Flattened pixel values → **PCA** retaining **95 % variance** (≈ 120 components).  
2. **Modeling** –  
   * **XGBoost** (max_depth = 6, learning_rate = 0.1, 200 trees).  
   * **Logistic Regression** (L2 regularisation).  
   * **Ensemble** – weighted average (XGBoost 70 %, LR 30 %).  
3. **Evaluation** – Accuracy, macro‑F1, ROC‑AUC for each class, precision‑recall curves.  

## Results
| Metric                | Value |
|-----------------------|-------|
| Overall Accuracy      | **95.4 %** |
| Macro‑F1              | **0.93** |
| ROC‑AUC (average)     | **0.98** |
| Training time (CPU)   | ~ 3 min on i5‑8250U |
| Model size (disk)     | 12 MB (XGBoost) + 1 MB (LR) |

> The ensemble consistently outperformed each base model, especially on the minority “pituitary” class.

## Installation
```bash
# Clone the repo
git clone https://github.com/houssam-lamsatfi/brain-tumor-classification.git
cd brain-tumor-classification

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

