#!/usr/bin/env python3
"""
Fast Cross-Dataset Meta-Training for Detector Weighting

Script trains a meta-model on dataset metrics --> optimal detector weighting split

Pipeline:

1. Load datasets (iris, wine, breast_cancer, digits)
2. Compute dataset-level descriptors:
        - num classes
        - class imbalance
        - label entropy
        - feature dimensionality
        - PCA explained variance (mean, max, min)
        - mean pairwise distance
3. For multiple noise levels:
        • inject label noise
        • compute detector AUROCs (RF, kNN, LOF)
        → these 3 AUROCs form the target weight vector

4. Meta-model = small MLP mapping dataset_descriptor → weights
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os
import random

RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)
random.seed(RNG)


# ---------------------------------------------------------------------
# Base Detector Functions
# ---------------------------------------------------------------------
def rf_oof_suspicion(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG)
    N = len(y)
    classes = np.unique(y)
    C = len(classes)
    oof_probs = np.zeros((N, C))

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(n_estimators=200, random_state=fold, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        try:
            oof_probs[te] = clf.predict_proba(X[te])
        except Exception:
            oof_probs[te] = np.ones((len(te), C)) / C

    class_index = {c:i for i, c in enumerate(classes)}
    cols = np.array([class_index[c] for c in y])
    self_conf = oof_probs[np.arange(N), cols]
    return 1 - self_conf

def knn_disagreement(X, y, k=10):
    k = min(k, max(1, len(y)-1))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    preds = clf.predict(X)
    return (preds != y).astype(float)

def lof_score(X, y):
    n_neighbors = min(20, max(2, len(y)-1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    pred = lof.fit_predict(X)
    return (pred == -1).astype(float)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def safe_auc(y, scores):
    try:
        if len(np.unique(y)) < 2:
            return 0.5
        return roc_auc_score(y, scores)
    except:
        return 0.5

def inject_symmetric_noise(y, noise):
    y2 = y.copy()
    n = int(len(y)*noise)
    idx = np.random.choice(len(y), n, replace=False)
    classes = np.unique(y)
    for i in idx:
        c = [z for z in classes if z != y[i]]
        y2[i] = np.random.choice(c)
    mask = (y2 != y).astype(int)
    return y2, mask

# ---------------------------------------------------------------------
# Dataset-Level Feature Extraction
# ---------------------------------------------------------------------
def compute_dataset_descriptor(X, y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    # PCA stats
    pca = PCA(n_components=min(10, X.shape[1]), random_state=RNG)
    pca.fit(X)
    ev = pca.explained_variance_ratio_

    # Pairwise distances
    try:
        d = pdist(X)
        avg_dist = np.mean(d)
    except:
        avg_dist = 0.0

    desc = np.array([
        len(classes),                # number of classes
        np.max(probs),               # majority class proportion (imbalance)
        -np.sum(probs * np.log(probs + 1e-12)),   # label entropy
        X.shape[1],                  # dimensionality
        np.mean(ev),                 # PCA mean variance ratio
        np.max(ev),                  # PCA largest component
        np.min(ev),                  # PCA smallest component
        avg_dist,                    # mean pairwise distance
    ], dtype=float)

    return desc

# ---------------------------------------------------------------------
# Load Datasets
# ---------------------------------------------------------------------
def load_all_datasets():
    loaders = [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]

    datasets = []

    for name, fn in loaders:
        data = fn()
        X = data.data.astype(float)
        y = data.target.astype(int)

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        datasets.append({
            "name": name,
            "X": Xs,
            "y": y
        })

    return datasets

# ---------------------------------------------------------------------
# Meta-Model
# ---------------------------------------------------------------------
class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------
# Cross-Dataset Meta-Training
# ---------------------------------------------------------------------
def train_meta(datasets, noise_levels=(0.05, 0.1, 0.2), epochs=200, lr=1e-3):
    descriptors = []
    targets = []

    # Build training table
    for ds in datasets:
        X, y = ds["X"], ds["y"]
        base_desc = compute_dataset_descriptor(X, y)

        for noise in noise_levels:
            y_noisy, mask = inject_symmetric_noise(y, noise)

            rf = rf_oof_suspicion(X, y_noisy)
            kn = knn_disagreement(X, y_noisy)
            lf = lof_score(X, y_noisy)

            w_rf = safe_auc(mask, rf)
            w_kn = safe_auc(mask, kn)
            w_lf = safe_auc(mask, lf)

            descriptors.append(base_desc)
            targets.append([w_rf, w_kn, w_lf])

    Xtrain = torch.tensor(np.vstack(descriptors), dtype=torch.float32)
    Ytrain = torch.tensor(np.vstack(targets), dtype=torch.float32)

    model = MetaMLP(in_dim=Xtrain.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        pred = model(Xtrain)
        loss = loss_fn(pred, Ytrain)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(f"Epoch {ep}/{epochs} | loss={loss.item():.5f}")

    return model

# ---------------------------------------------------------------------
# Evaluate on a held-out dataset
# ---------------------------------------------------------------------
def evaluate_with_cleanlab(model, ds, noise_rate=0.05, top_k=15):
    from cleanlab.classification import CleanLearning
    from cleanlab.filter import find_label_issues
    X, y = ds["X"], ds["y"]

    # -----------------------
    # Meta-model evaluation
    # -----------------------
    desc = torch.tensor(compute_dataset_descriptor(X, y), dtype=torch.float32).unsqueeze(0)
    w = model(desc).detach().numpy()[0]
    w_rf, w_kn, w_lf = w

    print(f"\n[Meta-ensemble] Predicted weights for '{ds['name']}':")
    print(f" RF={w_rf:.3f}, kNN={w_kn:.3f}, LOF={w_lf:.3f}")

    y_noisy, mask = inject_symmetric_noise(y, noise_rate)
    rf = rf_oof_suspicion(X, y_noisy)
    kn = knn_disagreement(X, y_noisy)
    lf = lof_score(X, y_noisy)

    ensemble = w_rf*rf + w_kn*kn + w_lf*lf
    topk_idx_meta = np.argsort(-ensemble)[:mask.sum()]
    hit_meta = len(set(np.where(mask==1)[0]).intersection(topk_idx_meta))
    print(f"Injected flips={mask.sum()}, (meta)={hit_meta}")

    # -----------------------
    # CleanLab evaluation
    # -----------------------
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    cl_model = CleanLearning(clf=clf)
    label_issues_info = cl_model.find_label_issues(X, y_noisy)
    detected_issues_df = label_issues_info[label_issues_info["is_label_issue"] == True]
    topk_idx_cl = detected_issues_df.index.values
    hit_cl = len(set(np.where(mask == 1)[0]).intersection(topk_idx_cl))
    print(f"Injected flips={mask.sum()}, (CleanLab)={hit_cl}")

    # -----------------------
    # PCA visualization
    # -----------------------
    try:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(8,6))
        plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="tab10", alpha=0.5, s=30, label="original points")

        # flips
        flipped_idx = np.where(mask==1)[0]
        if len(flipped_idx) > 0:
            plt.scatter(X_2d[flipped_idx,0], X_2d[flipped_idx,1], facecolor="red", edgecolor="black", s=100, label="true flips")

        # Meta-model top-k
        plt.scatter(X_2d[topk_idx_meta,0], X_2d[topk_idx_meta,1], marker="*", facecolor="yellow", edgecolor="black", s=150, label="meta top detections")

        # CleanLab detections
        plt.scatter(X_2d[topk_idx_cl,0], X_2d[topk_idx_cl,1], marker="^", facecolor="cyan", edgecolor="black", s=120, label="CleanLab detections")

        plt.title(f"Meta-ensemble vs CleanLab — {ds['name']}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("PCA plot failed:", e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    datasets = load_all_datasets()
    for ds in datasets:
        print(f"{ds['name']}: n={ds['X'].shape[0]}, d={ds['X'].shape[1]}, classes={len(np.unique(ds['y']))}")

    model = train_meta(datasets, noise_levels=(0.05,0.1,0.2), epochs=200, lr=1e-3)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/meta_dataset_mlp.pth")
    print("\nSaved → models/meta_dataset_mlp.pth")

    evaluate_with_cleanlab(model, datasets[-1], noise_rate=0.10)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()


