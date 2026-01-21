#!/usr/bin/env python3
"""
Dataset-level Meta-Learning for Label Noise Detector Selection (with Cross Validation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import openml

from pathlib import Path
from typing import List, Dict, Callable
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

from cleanlab.classification import CleanLearning

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)
random.seed(RNG)

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Detector Registry
# ---------------------------------------------------------------------
@dataclass
class DetectorConfig:
    name: str
    func: Callable
    enabled: bool = True


class DetectorRegistry:
    def __init__(self):
        self.detectors: Dict[str, DetectorConfig] = {}

    def register(self, name: str, enabled: bool = True):
        def decorator(func: Callable):
            self.detectors[name] = DetectorConfig(name, func, enabled)
            return func
        return decorator

    def get_enabled(self) -> List[DetectorConfig]:
        return [d for d in self.detectors.values() if d.enabled]

    def run_all(self, X, y):
        out = {}
        for det in self.get_enabled():
            try:
                out[det.name] = det.func(X, y)
            except Exception as e:
                print(f"⚠ {det.name} failed: {e}")
                out[det.name] = np.zeros(len(y))
        return out


registry = DetectorRegistry()

# ---------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------
@registry.register("rf_oof")
def rf_oof(X, y, n_folds=3):
    n_folds = min(n_folds, np.min(np.bincount(y)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG)
    probs = np.zeros((len(y), len(np.unique(y))))

    for f, (tr, te) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(
            n_estimators=80, max_depth=10, n_jobs=1, random_state=f
        )
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])

    return 1.0 - probs[np.arange(len(y)), y]


@registry.register("gb_conf")
def gradient_boost_confidence(X, y):
    clf = HistGradientBoostingClassifier(max_depth=6, random_state=RNG)
    clf.fit(X, y)
    p = clf.predict_proba(X)
    return 1.0 - p[np.arange(len(y)), y]


@registry.register("logit_margin")
def logistic_margin(X, y):
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X, y)
    p = clf.predict_proba(X)
    return 1.0 - p[np.arange(len(y)), y]


@registry.register("knn")
def knn_disagreement(X, y, k=15):
    k = min(k, len(y) - 1)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    return (clf.predict(X) != y).astype(float)


@registry.register("lof")
def lof(X, y):
    n = min(25, len(y) - 1)
    lof = LocalOutlierFactor(n_neighbors=n, contamination=0.1)
    return (lof.fit_predict(X) == -1).astype(float)


@registry.register("iforest")
def isolation_forest(X, y):
    clf = IsolationForest(contamination=0.1, random_state=RNG)
    clf.fit(X)
    return (clf.predict(X) == -1).astype(float)


# ---------------------------------------------------------------------
# Dataset Descriptor Computation
# ---------------------------------------------------------------------
def compute_dataset_descriptor(X, y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    feats = [
        len(classes),
        np.max(probs),
        -np.sum(probs * np.log(probs + 1e-12)),
        np.std(counts),
        X.shape[1],
        np.log1p(len(X)),
    ]

    try:
        pca = PCA(n_components=min(5, X.shape[1]))
        ev = pca.fit(X).explained_variance_ratio_
        feats += [ev.mean(), ev.max(), ev[:3].sum()]
    except:
        feats += [0, 0, 0]

    try:
        idx = np.random.choice(len(X), min(200, len(X)), replace=False)
        d = pdist(X[idx])
        feats += [d.mean(), d.std()]
    except:
        feats += [0, 0]

    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------
# Meta Model
# ---------------------------------------------------------------------
class DatasetMetaModel(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, d_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def inject_symmetric_noise(y, noise):
    y2 = y.copy()
    n = int(len(y) * noise)
    idx = np.random.choice(len(y), n, replace=False)
    classes = np.unique(y)
    for i in idx:
        y2[i] = np.random.choice(classes[classes != y[i]])
    return y2, idx


def inject_fringe_noise(X, y, noise, random_state=42):
    """
    Feature-dependent label noise:
    - flips labels of boundary (low-confidence) points
    - flips to closest alternative class in feature space

    Returns:
        y_noisy : noisy labels
        idx     : indices flipped
    """
    rng = np.random.default_rng(random_state)

    Xs = StandardScaler().fit_transform(X)
    y2 = y.copy()
    n_flip = int(len(y) * noise)

    clf = LogisticRegression(
        max_iter=1000, multi_class="auto", n_jobs=1
    )
    clf.fit(Xs, y)

    probs = clf.predict_proba(Xs)
    true_class_probs = probs[np.arange(len(y)), y]
    fringe_idx = np.argsort(true_class_probs)[:n_flip]
    classes = np.unique(y)

    for i in fringe_idx:
        x = Xs[i:i+1]

        # candidate classes except true
        others = classes[classes != y[i]]

        # centroids of other classes
        centroids = np.array([
            Xs[y == c].mean(axis=0) for c in others
        ])

        # nearest class centroid
        dists = cdist(x, centroids)[0]
        y2[i] = others[np.argmin(dists)]

    return y2, fringe_idx


def topk_hit_rate(scores, flips):
    if len(flips) == 0:
        return 0.0
    topk = np.argsort(-scores)[: len(flips)]
    return len(set(topk) & set(flips)) / len(flips)

def auc_flip_detection(scores, flips, n):
    """
    ROC-AUC for detecting flipped labels.
    scores : anomaly / noise scores (higher = more suspicious)
    flips  : indices of corrupted points
    n      : total number of samples
    """
    if len(flips) == 0 or len(flips) == n:
        return 0.5  # undefined AUC → neutral

    y_true = np.zeros(n, dtype=int)
    y_true[flips] = 1

    # AUC requires both classes present
    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------
# Pairwise Ranking Loss
# ---------------------------------------------------------------------
def pairwise_ranking_loss(pred, target):
    loss, cnt = 0.0, 0
    for i in range(pred.size(1)):
        for j in range(pred.size(1)):
            if i == j:
                continue
            s = torch.sign(target[:, i] - target[:, j])
            m = s * (pred[:, i] - pred[:, j])
            loss += torch.mean(torch.relu(1.0 - m))
            cnt += 1
    return loss / cnt


# ---------------------------------------------------------------------
# Sample Generation
# ---------------------------------------------------------------------
def generate_samples(datasets, noise_levels):
    Xd, Yp = [], []
    names = [d.name for d in registry.get_enabled()]

    for ds in datasets:
        X, y = ds["X"], ds["y"]
        desc = compute_dataset_descriptor(X, y)

        for noise in noise_levels:
            y_n, flips = inject_fringe_noise(X, y, noise)
            scores = registry.run_all(X, y_n)
            Yp.append([
                auc_flip_detection(scores[n], flips, len(y))
                for n in names
            ])
            Xd.append(desc)

    return (
        torch.tensor(Xd, dtype=torch.float32),
        torch.tensor(Yp, dtype=torch.float32),
        names,
    )

# ---------------------------------------------------------------------
# Cache dataset descriptors
# ---------------------------------------------------------------------
def build_meta_dataset(datasets, noise_levels):
    X_meta = []
    Y_meta = []
    dataset_ids = []
    detector_names = [d.name for d in registry.get_enabled()]

    for ds_idx, ds in enumerate(datasets):
        X, y = ds["X"], ds["y"]
        desc = compute_dataset_descriptor(X, y)

        for noise in noise_levels:
            y_n, flips = inject_fringe_noise(X, y, noise)
            scores = registry.run_all(X, y_n)

            aucs = [
                auc_flip_detection(scores[name], flips, len(y))
                for name in detector_names
            ]

            X_meta.append(desc)
            Y_meta.append(aucs)
            dataset_ids.append(ds_idx)

    return (
        torch.tensor(X_meta, dtype=torch.float32),
        torch.tensor(Y_meta, dtype=torch.float32),
        np.array(dataset_ids),
        detector_names,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_meta_model_loocv_fast(
    X_meta,
    Y_meta,
    dataset_ids,
    detector_names,
    num_datasets,
    epochs=150,
    lr=1e-3,
    patience=20,
    min_delta=1e-4,
):
    models = []

    for held_out in range(num_datasets):
        print(f"\n=== LOOCV fold (held out dataset {held_out}) ===")

        train_mask = dataset_ids != held_out
        val_mask = dataset_ids == held_out

        Xtr, Ytr = X_meta[train_mask], Y_meta[train_mask]
        Xva, Yva = X_meta[val_mask], Y_meta[val_mask]

        model = DatasetMetaModel(Xtr.shape[1], Ytr.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best = float("inf")
        patience_ctr = 0

        for e in range(epochs):
            model.train()
            opt.zero_grad()
            loss = pairwise_ranking_loss(
                model(Xtr.to(DEVICE)),
                Ytr.to(DEVICE)
            )
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val = pairwise_ranking_loss(
                    model(Xva.to(DEVICE)),
                    Yva.to(DEVICE)
                )

            if val < best - min_delta:
                best = val
                patience_ctr = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "detectors": detector_names,
                        "held_out": held_out,
                    },
                    f"models/meta_ranker_loocv_{held_out}.pth",
                )
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                break

        models.append(model)

    return models




# ---------------------------------------------------------------------
# Benchmark (vs Cleanlab)
# ---------------------------------------------------------------------
def benchmark(model, datasets, noise_levels):
    rows = []
    names = [d.name for d in registry.get_enabled()]

    for ds in datasets:
        X, y = ds["X"], ds["y"]
        desc = compute_dataset_descriptor(X, y)

        for noise in noise_levels:
            y_n, flips = inject_fringe_noise(X, y, noise)
            scores = registry.run_all(X, y_n)

            with torch.no_grad():
                w = torch.softmax(
                    model(torch.tensor(desc).float().unsqueeze(0).to(DEVICE)),
                    dim=1,
                )[0].cpu().numpy()

            ensemble = sum(w[i] * scores[n] for i, n in enumerate(names))

            # Cleanlab
            clf = RandomForestClassifier(n_estimators=150, random_state=RNG)
            cl = CleanLearning(clf=clf)
            issues = cl.find_label_issues(X, y_n)
            cl_idx = issues[issues.is_label_issue].index.values

            row = {
                "dataset": ds["name"],
                "noise": noise,
                "meta": auc_flip_detection(ensemble, flips, len(y)),
                "cleanlab": auc_flip_detection(
                    np.isin(np.arange(len(y)), cl_idx).astype(float),
                    flips,
                    len(y),
                ),
            }

            for i, n in enumerate(names):
                row[n] = auc_flip_detection(scores[n], flips, len(y))
                row[f"w_{n}"] = w[i]

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("results/meta_ranker_results_v4.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Dataset Loader (Stratified Subsample)
# ---------------------------------------------------------------------
def load_datasets_fast(max_tasks=300, max_samples=1200):
    suite = openml.study.get_suite("OpenML-CC18")
    datasets = []
    print(f"CC18 total tasks available: {len(suite.tasks)}")
    for tid in suite.tasks[:max_tasks]:
        try:
            task = openml.tasks.get_task(tid)
            ds = task.get_dataset()
            X, y, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")

            X, y = X.dropna(), y.loc[X.dropna().index]
            if y.dtype.kind in {"f", "c"}:
                continue

            y = LabelEncoder().fit_transform(y)
            if len(np.unique(y)) < 2:
                continue

            X = pd.get_dummies(X, drop_first=True)
            X = StandardScaler().fit_transform(X)

            if len(y) > max_samples:
                sss = StratifiedShuffleSplit(
                    n_splits=1, train_size=max_samples, random_state=RNG
                )
                idx, _ = next(sss.split(X, y))
                X, y = X[idx], y[idx]

            datasets.append({"name": ds.name, "X": X, "y": y})
            print(f"✓ {ds.name}")

        except:
            continue

    return datasets


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    datasets = load_datasets_fast(max_tasks=200)

    noise_levels = (0.05, 0.1, 0.2)

    print("\nPrecomputing meta-dataset...")
    X_meta, Y_meta, dataset_ids, detector_names = build_meta_dataset(
        datasets, noise_levels
    )

    print("Training LOOCV meta-models...")
    models = train_meta_model_loocv_fast(
        X_meta,
        Y_meta,
        dataset_ids,
        detector_names,
        num_datasets=len(datasets),
    )

    # Example: evaluate using last model
    df = benchmark(models[-1], datasets, noise_levels=(0.1, 0.2))

    print(df.groupby("noise")[["meta", "cleanlab"]].mean())



if __name__ == "__main__":
    main()
