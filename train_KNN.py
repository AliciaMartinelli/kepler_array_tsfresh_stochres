import os
import sys
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if len(sys.argv) < 2:
    print("Usage: python train_KNN.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

FEATURE_FILE = os.path.join("aug_dataset", f"X_kepler_augmented_sigma_{sigma:.5f}.npy")
LABEL_FILE = os.path.join("dataset", "y_kepler.npy")
RESULT_CSV = os.path.join("results", "knn", f"results_knn_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

def load_data():
    X = np.load(FEATURE_FILE)
    y = np.load(LABEL_FILE)
    return X, y

results = []

print(f"Training KNN for sigma = {sigma:.5f} ...")

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

for run in range(50):
    X, y = load_data()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED + run)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=SEED + run)

    knn_model.fit(X_train, y_train)

    y_val_proba = knn_model.predict_proba(X_val)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_idx = np.argmax(f1_vals) if thresholds.size > 0 else 0
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

    y_test_proba = knn_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    p_misclass = cm[1, 0] / cm[1].sum() * 100 if cm[1].sum() != 0 else 0
    n_misclass = cm[0, 1] / cm[0].sum() * 100 if cm[0].sum() != 0 else 0

    results.append({
        "run": run + 1,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p_misclass": p_misclass,
        "n_misclass": n_misclass
    })

    print(f"Run {run+1}/50 - AUC: {auc:.4f}, F1: {f1:.4f}")

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\nAll results saved to {RESULT_CSV}")