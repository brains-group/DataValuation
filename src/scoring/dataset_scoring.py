#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json
from glob import glob
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTPUT_CSV = "dataset_quality_report.csv"
DATA_DIR = "test_data/"


# ----------------- Metric Functions ----------------- #
def compute_clustering_metrics(num_df, y=None):
    """Compute clustering quality metrics using KMeans label approximation."""
    from sklearn.cluster import KMeans

    if num_df.shape[1] < 2 or num_df.shape[0] < 5:
        return np.nan, np.nan, np.nan

    X = StandardScaler().fit_transform(num_df)

    # Clusters = # of classes if provided, else 2
    k = len(np.unique(y)) if y is not None and len(np.unique(y)) > 1 else 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return sil, ch, db


def plot_pca(csv_path, num_df, y=None):
    """Save PCA 2D scatter plot to disk."""
    if num_df.shape[1] < 2 or num_df.shape[0] < 5:
        return None

    X = StandardScaler().fit_transform(num_df)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    if y is not None:
        if isinstance(y, pd.Series) and y.dtype == object:
            y = LabelEncoder().fit_transform(y)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.7)
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

    plt.title(f"PCA Projection: {os.path.basename(csv_path)}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    out_path = f"{os.path.splitext(csv_path)[0]}_pca.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def compute_completeness(df):
    missing_total = df.isna().sum().sum()
    total_values = df.size
    missing_ratio = missing_total / total_values
    row_integrity = (df.isna().sum(axis=1) == 0).mean()
    col_integrity = (df.isna().sum() == 0).mean()
    return missing_ratio, row_integrity, col_integrity


def compute_accuracy(df):
    duplicate_ratio = df.duplicated().mean()
    uniqueness = 1 - duplicate_ratio
    return duplicate_ratio, uniqueness


def compute_outliers(num_df):
    if len(num_df.columns) == 0:
        return np.nan, np.nan, np.nan
    zscores = (num_df - num_df.mean()) / num_df.std(ddof=0)
    outlier_ratio = (np.abs(zscores) > 4).mean().mean()
    mean_skew = num_df.skew().mean()
    mean_kurt = num_df.kurtosis().mean()
    return outlier_ratio, mean_skew, mean_kurt


def compute_class_balance(df, class_col):
    if class_col not in df.columns:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    y = df[class_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    value_counts = pd.Series(y).value_counts()
    num_classes = value_counts.size
    majority_ratio = value_counts.max() / len(df)
    class_ent = entropy(value_counts / len(df))
    imbalance_score = 1 - (class_ent / np.log(num_classes)) if num_classes > 1 else 1.0
    return num_classes, majority_ratio, class_ent, imbalance_score, value_counts


def autoencoder_validity_score(num_df):
    """Detect invalid samples via autoencoder reconstruction distance."""
    if num_df.shape[1] < 2 or num_df.shape[0] < 10:
        return np.nan, np.nan
    X = StandardScaler().fit_transform(num_df)
    input_dim = X.shape[1]
    hidden = max(2, input_dim // 2)
    ae = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=800, random_state=42)
    ae.fit(X, X)
    recon = ae.predict(X)
    dist = np.linalg.norm(X - recon, axis=1)
    norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    threshold = np.quantile(norm_dist, 0.95)
    invalid_ratio = (norm_dist > threshold).mean()
    return invalid_ratio, norm_dist.mean()


# ----------------- Updated Classification Evaluation ----------------- #
def evaluate_classification(df, class_col, drop_cols=None, cv_splits=5):
    """Evaluate multiple simple classifiers using cross-validation."""
    if class_col not in df.columns:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Drop unnecessary columns
    if drop_cols:
        drop_list = [c.strip() for c in drop_cols.split(",")]
        df = df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')

    X = df.drop(columns=[class_col]).select_dtypes(include=[np.number])
    y = df[class_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    if len(X.columns) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers
    classifiers = {
        "MLP": MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
    }

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for name, clf in classifiers.items():
        acc = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy').mean()
        prec = cross_val_score(clf, X_scaled, y, cv=skf, scoring='precision_weighted').mean()
        rec = cross_val_score(clf, X_scaled, y, cv=skf, scoring='recall_weighted').mean()
        f1 = cross_val_score(clf, X_scaled, y, cv=skf, scoring='f1_weighted').mean()
        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)

    # Return the best performing classifier metrics (based on accuracy)
    best_idx = int(np.argmax(metrics["accuracy"]))
    return metrics["accuracy"][best_idx], metrics["precision"][best_idx], metrics["recall"][best_idx], metrics["f1"][best_idx]


# ----------------- Main Scoring Function ----------------- #
def score_dataset(csv_path, class_col=None, drop_cols=None):
    df = pd.read_csv(csv_path)
    if drop_cols:
        drop_list = [c.strip() for c in drop_cols.split(",")]
        df = df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')

    num_df = df.select_dtypes(include=[np.number])
    n_rows, n_cols = df.shape

    # Completeness
    missing_ratio, row_integrity, col_integrity = compute_completeness(df)

    # Accuracy
    duplicate_ratio, uniqueness = compute_accuracy(df)

    # Outliers
    outlier_ratio, mean_skew, mean_kurt = compute_outliers(num_df)

    # Class-related scores
    if class_col and class_col in df.columns:
        num_classes, majority_ratio, class_ent, imbalance_score, _ = compute_class_balance(df, class_col)
        accuracy, precision, recall, f1 = evaluate_classification(df, class_col, drop_cols)
        y = df[class_col]
    else:
        num_classes = majority_ratio = class_ent = imbalance_score = np.nan
        accuracy = precision = recall = f1 = np.nan
        y = None

    # Autoencoder anomaly score
    invalid_ratio, avg_distance = autoencoder_validity_score(num_df)

    # --- clustering metrics ---
    sil, ch, db = compute_clustering_metrics(num_df, y)

    # --- PCA visualization ---
    pca_path = plot_pca(csv_path, num_df, y)

    metrics = {
        "file": csv_path,
        "rows": n_rows,
        "columns": n_cols,
        "numeric_columns": len(num_df.columns),
        "missing_ratio": round(missing_ratio, 4),
        "record_integrity": round(row_integrity, 4),
        "element_integrity": round(col_integrity, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "uniqueness": round(uniqueness, 4),
        "outlier_ratio": round(outlier_ratio, 4) if outlier_ratio == outlier_ratio else np.nan,
        "mean_skewness": round(mean_skew, 4) if mean_skew == mean_skew else np.nan,
        "mean_kurtosis": round(mean_kurt, 4) if mean_kurt == mean_kurt else np.nan,
        "num_classes": num_classes,
        "majority_class_ratio": round(majority_ratio, 4) if majority_ratio == majority_ratio else np.nan,
        "class_entropy": round(class_ent, 4) if class_ent == class_ent else np.nan,
        "imbalance_score": round(imbalance_score, 4) if imbalance_score == imbalance_score else np.nan,
        "invalid_ratio_autoencoder": round(invalid_ratio, 4) if invalid_ratio == invalid_ratio else np.nan,
        "avg_reconstruction_dist": round(avg_distance, 4) if avg_distance == avg_distance else np.nan,
        "silhouette_score": round(sil, 4) if sil == sil else np.nan,
        "calinski_harabasz": round(ch, 4) if ch == ch else np.nan,
        "davies_bouldin": round(db, 4) if db == db else np.nan,
        "pca_plot": pca_path,
        "accuracy": round(accuracy, 4) if accuracy == accuracy else np.nan,
        "precision": round(precision, 4) if precision == precision else np.nan,
        "recall": round(recall, 4) if recall == recall else np.nan,
        "f1_score": round(f1, 4) if f1 == f1 else np.nan,
    }

    return metrics


# ----------------- Batch Processing ----------------- #
def process_all_datasets(data_dir=DATA_DIR):
    csv_files = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    all_metrics = []
    for csv_path in csv_files:
        base_name = os.path.splitext(csv_path)[0]
        json_path = f"{base_name}.json"
        class_col = drop_cols = None
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
                class_col = meta.get("classcol")
                drop_cols = meta.get("dropcols")
        print(f"\nProcessing: {csv_path}")
        metrics = score_dataset(csv_path, class_col, drop_cols)
        all_metrics.append(metrics)

    header_needed = not os.path.exists(OUTPUT_CSV)
    pd.DataFrame(all_metrics).to_csv(OUTPUT_CSV, mode="a", header=header_needed, index=False)
    print(f"\n All dataset metrics saved to {OUTPUT_CSV}")


# ----------------- CLI ----------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset scoring tool for all CSVs in a folder")
    parser.add_argument("--data_dir", required=False, default=DATA_DIR, help="Root folder containing CSV datasets")
    args = parser.parse_args()
    process_all_datasets(args.data_dir)
