# Brain-tumor-model

Brain Tumor Classification using PCA, Logistic Regression, and XGBoost
Table of Contents

Project Overview

Dataset

Problem Definition

Methodology

Data Preprocessing

Dimensionality Reduction

Model Selection

Code Structure

Results

Dependencies

Usage

References

Project Overview

This project implements a machine learning pipeline to classify brain MRI images into tumor and non-tumor categories, and further into subtypes such as glioma, meningioma, and pituitary tumors.

The pipeline uses:

PCA for dimensionality reduction

Kernel PCA for exploratory analysis

XGBoost for multiclass classification

Logistic Regression for binary tumor vs. no-tumor classification

The project emphasizes interpretability, efficiency, and learning ML concepts rather than deep learning methods.

Dataset

The dataset contains MRI images organized into folders by class:

archive/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/


Images are in various formats (JPEG, PNG, BMP, TIFF).

Each image is converted to grayscale and resized to 128x128 for consistency and memory efficiency.

Problem Definition

Given a brain MRI image 
𝐼
I, classify it as:

Binary classification: tumor vs no_tumor

Multiclass classification: glioma, meningioma, pituitary, no_tumor

Mathematically, the goal is to learn a function 
𝑓
:
𝑅
𝑛
→
{
0
,
1
,
…
,
𝐶
−
1
}
f:R
n
→{0,1,…,C−1}, where 
𝑛
n is the number of pixels after flattening and 
𝐶
C is the number of classes.

Methodology
Data Preprocessing

Grayscale conversion: Reduces channels from 3 → 1 to simplify computation.

Resizing images: Standardizes input dimensions for PCA and ML models.

Flattening images: Converts 2D images into 1D vectors (
128
×
128
=
16
,
384
128×128=16,384 features).

Standardization: Scales features to zero mean and unit variance using StandardScaler, which improves convergence for PCA and Logistic Regression.

Dimensionality Reduction

PCA (Principal Component Analysis):

Reduces high-dimensional image vectors to a smaller number of components while retaining most of the variance.

Mathematically, PCA finds orthogonal directions 
𝑢
𝑖
u
i
	​

 that maximize variance:

𝑢
𝑖
=
arg
⁡
max
⁡
∥
𝑢
∥
=
1
Var
(
𝑋
𝑢
)
u
i
	​

=arg
∥u∥=1
max
	​

Var(Xu)

Chosen 95% variance retention → balances information preservation and computation.

Kernel PCA (RBF Kernel):

Used for exploratory analysis of non-linear relationships in data.

Projects data into a higher-dimensional feature space where linear separation may be easier.

Eigenvalues of the kernel matrix are analyzed to choose the number of components (elbow method).

Model Selection

XGBoost (Extreme Gradient Boosting):

A gradient boosting framework optimized for speed and performance.

Chosen for multiclass classification because of high performance with tabular/structured data.

Key parameters:

n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8

Logistic Regression:

Used for binary tumor vs. no-tumor classification.

Provides probabilistic outputs (sigmoid probabilities), allowing ROC and PR curve visualization.

Code Structure
brainTumor.py
├── Imports & Warnings
├── Helper Functions
│   └── show_all_results()  # Prints metrics and plots
├── Config
├── Data Loading
├── Train/Test Split
├── Scaling
├── PCA Analysis & Transformation
├── XGBoost Training & Evaluation
├── Logistic Regression Training & Evaluation
└── Visualizations (ROC, PR, Confusion Matrix)

Key Design Decisions:

show_all_results(): Displays accuracy, classification report, and confusion matrix at once.

Keeping ROC/PR curves separate: Useful for threshold tuning in binary classification.

PCA before modeling: Reduces overfitting and computational cost.

Results

XGBoost Accuracy: ~X.XXX (multiclass)

Logistic Regression Accuracy: ~X.XXX (binary tumor vs no_tumor)

Confusion matrices and ROC/PR curves provide visual confirmation of model performance.

PCA explained variance: ~95% retained with reduced dimensions.

Note: Accuracy depends on dataset size and quality.

Dependencies
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn xgboost tqdm

Usage

Clone the repository.

Ensure dataset is organized in folders under archive/.

Update DATA_DIR in the code to point to your dataset.

Run the script:

python brainTumor.py


Outputs:

Plots: PCA elbow, ROC & PR curves, confusion matrices

Console: Accuracy, classification reports, PCA variance info

References

Jolliffe, I. T. (2002). Principal Component Analysis. Springer Series in Statistics.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

scikit-learn documentation: https://scikit-learn.org

MRI Brain Tumor Dataset (e.g., Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
)
