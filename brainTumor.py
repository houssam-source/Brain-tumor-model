import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


def show_all_results(model, X_test, y_test, classes, pca=None, title="Model Results"):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    print(f"\n{'=' * 60}")
    print(f"📊 {title}")
    print(f"{'=' * 60}")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Optional PCA info
    if pca is not None:
        explained = np.cumsum(pca.explained_variance_ratio_)
        print("\n🧠 PCA Variance Retained:")
        print(f"{len(pca.explained_variance_ratio_)} components → {explained[-1] * 100:.2f}% variance retained")


# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = r"C:\Users\ROG\Desktop\archive (4)"
IMG_SIZE = (128, 128)   # Reduce if memory is tight, e.g., (96, 96) or (64, 64)
PCA_VARIANCE_TARGET = 0.95
RANDOM_STATE = 42

# ---------------------------
# DATA LOADING
# ---------------------------
def load_images(data_dir, img_size=(128, 128)):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_to_idx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        image_paths = []
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(cls_dir, ext)))
        for p in tqdm(image_paths, desc=f"Loading {cls}", unit="img"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            X.append(img.flatten())
            y.append(label_to_idx[cls])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, classes

print("Loading dataset...")
X, y, classes = load_images(DATA_DIR, IMG_SIZE)
print(f"Data shape: {X.shape}, Labels: {Counter(y)}, Classes: {classes}")

# ---------------------------
# TRAIN/TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# PCA + ELBOW (EXPLAINED VARIANCE)
# ---------------------------
print("Fitting PCA for elbow analysis...")
pca_full = PCA(n_components=min(512, X_train_scaled.shape[1]), svd_solver="randomized", random_state=RANDOM_STATE)
pca_full.fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7,4))
plt.plot(np.arange(1, len(cum_var)+1), cum_var, marker='o', lw=1)
plt.axhline(PCA_VARIANCE_TARGET, color='r', ls='--', label=f'{int(PCA_VARIANCE_TARGET*100)}% variance')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Elbow Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

n_components_95 = np.searchsorted(cum_var, PCA_VARIANCE_TARGET) + 1
print(f"Components to reach {int(PCA_VARIANCE_TARGET*100)}% variance: {n_components_95}")

# Fit PCA with chosen number of components
pca = PCA(n_components=n_components_95, svd_solver="randomized", random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA reduced shape: {X_train_pca.shape}")

# ---------------------------
# KERNEL PCA (RBF) ANALYSIS
# ---------------------------
# Note: KernelPCA has no explained_variance_ratio_; use normalized eigenvalues (lambdas_) as a proxy.
print("Fitting KernelPCA (RBF) for eigenvalue elbow proxy...")
kpca_probe = KernelPCA(n_components=min(256, X_train_scaled.shape[0]-1), kernel="rbf", gamma=1.0/(2*(IMG_SIZE[0]*IMG_SIZE[1])), fit_inverse_transform=False, random_state=RANDOM_STATE)
X_train_kpca_probe = kpca_probe.fit_transform(X_train_scaled)
lambdas = kpca_probe.eigenvalues_
lambdas_sorted = np.sort(lambdas)[::-1]
cum_lam = np.cumsum(lambdas_sorted) / np.sum(lambdas_sorted)

plt.figure(figsize=(7,4))
plt.plot(np.arange(1, len(cum_lam)+1), cum_lam, marker='o', lw=1)
plt.xlabel("Number of components")
plt.ylabel("Cumulative normalized eigenvalues")
plt.title("KernelPCA (RBF) 'Elbow' Proxy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------
# XGBOOST (MULTICLASS) ON PCA FEATURES
# ---------------------------
print("Training XGBoost (multiclass) on PCA features...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(classes),
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(
    X_train_pca, y_train,
    eval_set=[(X_test_pca, y_test)],
    verbose=False
)
y_pred_xgb = xgb.predict(X_test_pca)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=classes))

show_all_results(xgb, X_test_pca, y_test, classes, pca, title="XGBoost (PCA Features)")


# ---------------------------
# LOGISTIC REGRESSION (BINARY: tumor vs no_tumor) + SIGMOID + THRESHOLD VISUALIZATION
# ---------------------------
# Map labels to binary: 1 = tumor (any of glioma/meningioma/pituitary), 0 = no_tumor
if "no_tumor" in classes:
    no_tumor_idx = classes.index("no_tumor")
    y_train_bin = (y_train != no_tumor_idx).astype(int)
    y_test_bin = (y_test != no_tumor_idx).astype(int)

    print("Training Logistic Regression (binary: tumor vs no_tumor) on PCA features...")
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    logreg.fit(X_train_pca, y_train_bin)

    # Sigmoid probabilities
    y_proba = logreg.predict_proba(X_test_pca)[:, 1]

    # Threshold visualization
    thresholds = np.linspace(0.0, 1.0, 201)
    accs, tprs, fprs, precisions, recalls = [], [], [], [], []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_bin, y_hat).ravel()
        accs.append((tp+tn)/(tp+tn+fp+fn))
        tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)   # recall
        fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0.0)
        precisions.append(tp/(tp+fp) if (tp+fp)>0 else 0.0)
        recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)

    # Curves vs threshold
    plt.figure(figsize=(7,4))
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall (TPR)")
    plt.axvline(0.5, color="k", ls="--", label="Threshold=0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Logistic Regression: Metrics vs Threshold (tumor=1)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ROC & PR curves
    fpr, tpr, roc_thr = roc_curve(y_test_bin, y_proba)
    prec, rec, pr_thr = precision_recall_curve(y_test_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(5.3,4.3))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (LogReg, tumor vs no_tumor)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    show_all_results(logreg, X_test_pca, y_test_bin, ["no_tumor", "tumor"], pca, title="Logistic Regression (Binary)")

    plt.figure(figsize=(5.3, 4.3))
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (LogReg, tumor vs no_tumor)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("Class 'no_tumor' not found. Skipping binary logistic regression threshold visualization.")

print("Done.")