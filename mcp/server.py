#!/usr/bin/env python3
"""
MCP Server for Dataset-Level Meta-Learning Noise Detector
Loads model from models/ and runs inference on datasets from test_data/
"""

from mcp.server.fastmcp import FastMCP
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import skew
import torch.serialization
from typing import Optional
from typing import List, Dict, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning
from cleanlab.count import estimate_cv_predicted_probabilities


RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)
random.seed(RNG)

# Create MCP server
mcp = FastMCP("Noise Detector Inference Server")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model cache
MODEL_CACHE = {}

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
            except Exception:
                out[det.name] = np.zeros(len(y))
        return out


registry = DetectorRegistry()

@registry.register("logreg_conf")
def logreg_conf(X, y):
    le = LabelEncoder()
    y_mapped = le.fit_transform(y)
    min_class_count = np.min(np.bincount(y_mapped))
    n_splits = min(3, min_class_count)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RNG
    )
    n_classes = len(np.unique(y_mapped))
    probs = np.zeros((len(y_mapped), n_classes))
    for train_idx, val_idx in skf.split(X, y_mapped):
        clf = LogisticRegression(
            C=0.5,  # slight regularization
            max_iter=100,
            solver="lbfgs"
        )
        clf.fit(X[train_idx], y_mapped[train_idx])
        probs[val_idx] = clf.predict_proba(X[val_idx])
    return 1.0 - probs[np.arange(len(y_mapped)), y_mapped]


@registry.register("rf_oof")
def rf_oof(X, y):
    le = LabelEncoder()
    y_mapped = le.fit_transform(y)
    min_class_count = np.min(np.bincount(y_mapped))
    n_splits = min(3, min_class_count)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RNG
    )
    n_classes = len(np.unique(y_mapped))
    probs = np.zeros((len(y_mapped), n_classes))
    for train_idx, val_idx in skf.split(X, y_mapped):
        clf = RandomForestClassifier(
            n_estimators=100,  # stronger
            max_depth=None,  # allow flexibility
            min_samples_leaf=3,  # prevent overfit
            max_features="sqrt",
            n_jobs=-1,
            random_state=RNG
        )
        clf.fit(X[train_idx], y_mapped[train_idx])
        probs[val_idx] = clf.predict_proba(X[val_idx])
    return 1.0 - probs[np.arange(len(y_mapped)), y_mapped]


@registry.register("gb_conf")
def gb_conf(X, y):
    le = LabelEncoder()
    y_mapped = le.fit_transform(y)
    min_class_count = np.min(np.bincount(y_mapped))
    n_splits = min(3, min_class_count)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RNG
    )
    n_classes = len(np.unique(y_mapped))
    probs = np.zeros((len(y_mapped), n_classes))
    for train_idx, val_idx in skf.split(X, y_mapped):
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=None,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RNG
        )
        clf.fit(X[train_idx], y_mapped[train_idx])
        probs[val_idx] = clf.predict_proba(X[val_idx])
    return 1.0 - probs[np.arange(len(y_mapped)), y_mapped]


@registry.register("knn")
def knn_dis(X, y):
    clf = KNeighborsClassifier(n_neighbors=min(15, len(y) - 1)).fit(X, y)
    return (clf.predict(X) != y).astype(float)

@registry.register("iforest")
def iforest(X, y):
    return (IsolationForest(random_state=RNG).fit_predict(X) == -1).astype(float)

@registry.register("lof")
def lof_det(X, y):
    lof = LocalOutlierFactor(n_neighbors=min(20, len(y)-1))
    preds = lof.fit_predict(X)
    return (preds == -1).astype(float)

@registry.register("cleanlab_margin")
def cleanlab_margin(X, y):
    return cleanlab_paper_scores_new(X, y)

@registry.register("label_prop")
def label_propagation_detector(X, y):
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Fit label propagation model
    lp = LabelSpreading(
        kernel="knn",
        n_neighbors=min(15, len(y) - 1),
        alpha=0.2  # less clamping = more smoothing
    )

    lp.fit(X, y_enc)

    # Get propagated probabilities
    probs = lp.label_distributions_

    # Score = disagreement with original label
    scores = 1.0 - probs[np.arange(len(y_enc)), y_enc]

    return scores

def cleanlab_paper_scores_new(X, y_noisy):
    """
    Cleanlab v2 faithful confident learning implementation.
    Returns per-sample score (higher = more likely mislabeled).
    """

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from cleanlab.count import estimate_cv_predicted_probabilities
    from cleanlab.rank import get_label_quality_scores

    X_arr = np.asarray(X)
    s = np.asarray(y_noisy)

    # 1. Estimate out-of-sample predicted probabilities P(s^ | x)
    psx = estimate_cv_predicted_probabilities(
        X_arr,
        s,
        clf=LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs"
        )
    )

    # 2. Compute label quality scores via confident learning
    quality_scores = get_label_quality_scores(
        labels=s,
        pred_probs=psx,
        method="normalized_margin"   # closest to original paper ranking
    )

    # Convert to "higher = more likely mislabeled"
    scores = 1.0 - quality_scores

    return scores



DETECTOR_NAMES = [
    d.name for d in registry.get_enabled()
]


class DatasetMetaModel(nn.Module):
    """Meta-model architecture from training script"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, d_out)
        )

    def forward(self, x):
        return self.net(x)


def compute_dataset_descriptor(X, y):
    """
    Compute dataset descriptor features.
    Returns a 19-dimensional feature vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Basic shape
    n_samples, n_features = X.shape

    # CLASS DISTRIBUTION
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / (counts.sum() + 1e-12)

    n_classes = float(len(classes))
    majority_prop = float(np.max(probs)) if probs.size > 0 else 0.0
    class_entropy = float(-np.nansum(probs * np.log(probs + 1e-12)))
    imbalance_ratio = float(np.max(counts) / (np.min(counts) + 1e-12)) if counts.size > 0 else 1.0

    # BASIC STRUCTURE
    log_n = float(np.log1p(n_samples))
    sample_feature_ratio = float(n_samples / (n_features + 1e-12))

    # FEATURE STATISTICS
    variances = np.var(X, axis=0)
    mean_variance = float(np.nanmean(variances)) if variances.size > 0 else 0.0
    near_constant_frac = float(np.mean(variances < 1e-5)) if variances.size > 0 else 0.0

    # Sparsity (fraction of zeros)
    sparsity = float(np.mean(X == 0)) if X.size > 0 else 0.0

    # Skewness (mean absolute skew across features)
    if n_features > 0:
        mean_skew = float(np.nanmean(np.abs(skew(X, axis=0, bias=False))))
    else:
        mean_skew = 0.0

    # CORRELATION STRUCTURE
    if n_features > 1:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = corr[np.triu_indices(n_features, k=1)]
        abs_upper = np.abs(upper)

        mean_abs_corr = float(np.mean(abs_upper)) if abs_upper.size > 0 else 0.0
        max_abs_corr = float(np.max(abs_upper)) if abs_upper.size > 0 else 0.0
        high_corr_frac = float(np.mean(abs_upper > 0.8)) if abs_upper.size > 0 else 0.0
    else:
        mean_abs_corr = 0.0
        max_abs_corr = 0.0
        high_corr_frac = 0.0

    # SIGNAL-TO-NOISE PROXY
    if n_classes > 1 and n_samples > 1:
        try:
            class_means = np.array([X[y == c].mean(axis=0) for c in classes])
            between_class_var = np.var(class_means, axis=0).mean()

            within_class_var = np.mean([
                np.var(X[y == c], axis=0).mean() for c in classes
            ])

            snr_proxy = float(between_class_var / (within_class_var + 1e-12))
        except Exception:
            snr_proxy = 0.0
    else:
        snr_proxy = 0.0

    # PCA GEOMETRY
    n_pca = min(3, n_features) if n_features > 0 else 0
    if n_pca > 0 and n_samples > 1:
        try:
            pca = PCA(n_components=n_pca)
            pca.fit(X)
            ev = pca.explained_variance_ratio_
        except Exception:
            ev = np.zeros(n_pca, dtype=float)
    else:
        ev = np.zeros(n_pca, dtype=float)

    # Pad to length 3
    ev = list(ev) + [0.0] * (3 - len(ev))
    ev = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in ev]
    pca_cumulative = float(np.sum(ev))

    # FINAL FEATURE VECTOR (19 features)
    feats = np.array([
        n_classes,
        majority_prop,
        class_entropy,
        imbalance_ratio,
        float(n_features),
        log_n,
        sample_feature_ratio,
        mean_variance,
        near_constant_frac,
        sparsity,
        mean_abs_corr,
        max_abs_corr,
        high_corr_frac,
        mean_skew,
        snr_proxy,
        ev[0],
        ev[1],
        ev[2],
        pca_cumulative
    ], dtype=float)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def load_model():
    if "model" in MODEL_CACHE:
        return MODEL_CACHE["model"]

    models_dir = Path("models")
    pth_files = list(models_dir.glob("*.pth"))

    if len(pth_files) == 0:
        raise FileNotFoundError("No .pth model file found")

    if len(pth_files) > 1:
        raise ValueError("Multiple .pth files found")

    model_path = pth_files[0]

    try:
        # Reconstruct architecture
        model = DatasetMetaModel(d_in=19, d_out=len(DETECTOR_NAMES)).to(DEVICE)

        # Load state_dict (THIS matches training)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

        model.eval()

        MODEL_CACHE["model"] = model
        MODEL_CACHE["model_path"] = str(model_path)
        return model

    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


def load_dataset(filename: str):
    """
    Load dataset from test_data/ directory.
    Supports CSV files with tab separator.
    """
    # Try different possible paths
    possible_paths = [
        Path("test_data") / filename,
        Path("test_data/medical") / filename,
        Path(filename)  # Direct path
    ]

    dataset_path = None
    for p in possible_paths:
        if p.exists():
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError(f"Dataset '{filename}' not found in test_data/ or test_data/medical/")

    # Load CSV
    df = pd.read_csv(dataset_path)

    # Detect label column
    if "Label" in df.columns:
        label_col = "Label"
    elif "LabelNew" in df.columns:
        label_col = "LabelNew"
    elif "LabelOld" in df.columns:
        label_col = "LabelOld"
    else:
        label_col = df.columns[-1]

    # Extract labels
    y = LabelEncoder().fit_transform(df[label_col].values)

    # Extract features
    X_df = (
        df.drop(columns=[label_col])
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
        .fillna(0)
    )

    X = X_df.values

    # Standardize
    X = StandardScaler().fit_transform(X)

    return X, y, df.shape[0], X_df.shape[1]


@mcp.tool()
def predict_detector_weights(dataset_filename: str) -> str:
    """
    Load a dataset and predict optimal detector weights.

    Args:
        dataset_filename: Name of dataset file in test_data/ (e.g., "RNA0.csv")

    Returns:
        JSON with detector weights and dataset info
    """
    try:
        # Load dataset
        X, y, n_samples, n_features = load_dataset(dataset_filename)

        # Compute descriptor
        descriptor = compute_dataset_descriptor(X, y)

        # Load model
        model = load_model()

        # Predict weights
        with torch.no_grad():
            descriptor_tensor = torch.tensor(descriptor).float().unsqueeze(0).to(DEVICE)
            logits = model(descriptor_tensor)
            weights = torch.softmax(logits / 0.5, dim=1).cpu().numpy()[0]

        # Create results
        result = {
            "success": True,
            "dataset": dataset_filename,
            "dataset_info": {
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_classes": int(descriptor[0])
            },
            "detector_weights": {
                name: float(weight)
                for name, weight in zip(DETECTOR_NAMES, weights)
            },
            "recommended_detector": DETECTOR_NAMES[int(np.argmax(weights))],
            "top_3_detectors": [
                {
                    "name": DETECTOR_NAMES[idx],
                    "weight": float(weights[idx])
                }
                for idx in np.argsort(weights)[-3:][::-1]
            ],
            "dataset_descriptor": {
                "n_classes": float(descriptor[0]),
                "majority_prop": float(descriptor[1]),
                "class_entropy": float(descriptor[2]),
                "imbalance_ratio": float(descriptor[3]),
                "n_features": float(descriptor[4]),
                "sample_feature_ratio": float(descriptor[6]),
                "mean_variance": float(descriptor[7]),
                "sparsity": float(descriptor[9]),
                "mean_abs_corr": float(descriptor[10]),
                "snr_proxy": float(descriptor[14])
            }
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.tool()
def get_dataset_quality_score(dataset_filename: str) -> str:
    """
    Compute an objective dataset quality score (0-1 scale).
    Higher score = cleaner/higher quality data.

    Args:
        dataset_filename: Name of dataset file in test_data/

    Returns:
        JSON with quality score and interpretation
    """
    try:
        # Load dataset
        X, y, _, _ = load_dataset(dataset_filename)

        # Compute descriptor
        descriptor = compute_dataset_descriptor(X, y)

        # Load model
        model = load_model()

        # Predict weights
        with torch.no_grad():
            # 1. Get detector weights from meta-model
            descriptor_tensor = torch.tensor(descriptor).float().unsqueeze(0).to(DEVICE)
            logits = model(descriptor_tensor)
            weights = torch.softmax(logits / 0.5, dim=1).cpu().numpy()[0]

            # 2. Run all detectors
            detector_outputs = registry.run_all(X, y)

            # 3. Align outputs with DETECTOR_NAMES
            scores_matrix = np.column_stack([
                detector_outputs[name] for name in DETECTOR_NAMES
            ])

            # 4. Ensemble score per sample
            ensemble_scores = np.sum(weights * scores_matrix, axis=1)

            # 5. Final dataset quality
            quality_score = float(1.0 - np.mean(ensemble_scores))

        interpretation = (
            "Excellent" if quality_score > 0.8 else
            "Good" if quality_score > 0.6 else
            "Fair" if quality_score > 0.4 else
            "Poor"
        )

        result = {
            "success": True,
            "dataset": dataset_filename,
            "quality_score": quality_score,
            "interpretation": interpretation,
            "estimated_noise_level": float(1.0 - quality_score)
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.resource("datasets://available")
def list_available_datasets() -> str:
    """List all datasets in test_data/ directory"""
    datasets = []

    test_data_dir = Path("test_data")
    if test_data_dir.exists():
        for f in test_data_dir.rglob("*.csv"):
            datasets.append({
                "name": f.name,
                "path": str(f.relative_to(test_data_dir)),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })

    return json.dumps({
        "datasets": datasets,
        "count": len(datasets)
    }, indent=2)


@mcp.resource("models://info")
def get_model_info() -> str:
    """Get information about the loaded model"""
    try:
        model = load_model()
        model_path = MODEL_CACHE.get("model_path", "unknown")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        info = {
            "model_path": model_path,
            "architecture": "DatasetMetaModel",
            "input_features": 19,
            "output_detectors": len(DETECTOR_NAMES),
            "detector_names": DETECTOR_NAMES,
            "total_parameters": n_params,
            "device": str(DEVICE)
        }

        return json.dumps(info, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


if __name__ == "__main__":
    print("Starting Noise Detector Inference Server...")
    print(f" Looking for model in: {Path('models').absolute()}")
    print(f" Looking for datasets in: {Path('test_data').absolute()}")
    print("\nAvailable tools:")
    print("  - predict_detector_weights: Get optimal detector weights for a dataset")
    print("  - get_dataset_quality_score: Estimate dataset quality (0-1)")
    print("  - analyze_dataset: Complete dataset analysis")
    print("\nAvailable resources:")
    print("  - datasets://available: List all datasets")
    print("  - models://info: Model information")

    mcp.run(transport="stdio")
