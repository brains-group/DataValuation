#!/usr/bin/env python3
"""
Confident-Learning-like label error detection
(using cross-validated RandomForest probabilities)
+ PCA visualization with:
  - Unique color per TRUE class
  - Dual-color rings for suspected errors (actual + predicted)

Usage:
    python dataset_error_analysis.py --csv data.csv --classcol label --dropcols id
"""

import argparse
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def run_confident_learning(df, class_col, drop_cols=None, n_folds=5):
    """Return dataframe + arrays needed for visualization + CV metrics."""

    # drop unwanted columns
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    # Separate features and target
    y_raw = df[class_col]
    X = df.drop(columns=[class_col])
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    N, C = len(y), len(classes)
    oof_probs = np.zeros((N, C))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    print("\nRunning cross-validated predictions...")

    # CROSS VALIDATION LOOP
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):

        print(f"\n==================== Fold {fold+1}/{n_folds} ====================")

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=fold,
            n_jobs=-1
        )
        clf.fit(X_scaled[train_idx], y[train_idx])
        probs = clf.predict_proba(X_scaled[test_idx])
        preds = np.argmax(probs, axis=1)
        oof_probs[test_idx] = probs

        acc = (preds == y[test_idx]).mean()
        fold_accuracies.append(acc)

        print(f"Fold {fold+1} accuracy: {acc:.4f}")

        # Per-class accuracy
        for c in range(C):
            class_mask = (y[test_idx] == c)
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == c).mean()
                print(f"  Class '{classes[c]}' accuracy: {class_acc:.4f}")

        cm = pd.crosstab(y[test_idx], preds, rownames=["True"], colnames=["Pred"])
        print("\nConfusion Matrix:")
        print(cm)

        print("========================================================")

    # OVERALL CV METRICS
    print("\n==================== Cross-Validation Summary ====================")
    print(f"Mean CV accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std CV accuracy:  {np.std(fold_accuracies):.4f}")
    print("==============================================================\n")

    # self confidence
    self_conf = oof_probs[np.arange(N), y]
    predicted = np.argmax(oof_probs, axis=1)
    # Suspicion
    suspicion_score = 1 - self_conf
    threshold = np.quantile(suspicion_score, 0.90)
    likely_error = suspicion_score >= threshold

    result_df = pd.DataFrame({
        "index": np.arange(N),
        "true_label": y_raw.values,
        "predicted_label": [classes[p] for p in predicted],
        "self_confidence": self_conf,
        "suspicion_score": suspicion_score,
        "likely_error": likely_error
    }).sort_values("suspicion_score", ascending=False).reset_index(drop=True)

    return result_df, X_scaled, y, predicted, likely_error, classes



def plot_pca(X_scaled, y, predicted, likely_error, classes):
    """PCA plot with:
       - unique TRUE class colors
       - for errors: inner ring (actual), outer ring (predicted)
    """
    print("\nGenerating PCA visualization...")

    pca = PCA(n_components=2)
    pts = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))

    # Build a color map for true classes
    cmap = plt.get_cmap("viridis", len(classes))
    true_colors = {c: cmap(c) for c in range(len(classes))}
    pred_colors = true_colors  # use same colormap for predicted classes

    # Plot normal points
    for c in range(len(classes)):
        idx = np.where((y == c) & (~likely_error))[0]
        if len(idx) > 0:
            plt.scatter(
                pts[idx, 0],
                pts[idx, 1],
                s=25,
                color=true_colors[c],
                alpha=0.45,
                label=f"Class {classes[c]}" if c == 0 else None
            )

    # Plot high-suspicion points with dual-color rings
    err_idx = np.where(likely_error)[0]
    for i in err_idx:
        actual = y[i]
        pred = predicted[i]

        # Outer ring = predicted class
        plt.scatter(
            pts[i, 0],
            pts[i, 1],
            s=180,
            edgecolors=pred_colors[pred],
            facecolors='none',
            linewidths=2.5,
        )

        # Inner ring = actual class
        plt.scatter(
            pts[i, 0],
            pts[i, 1],
            s=70,
            edgecolors=true_colors[actual],
            facecolors='none',
            linewidths=2,
        )

        plt.text(
            pts[i, 0],
            pts[i, 1],
            str(i),
            fontsize=7,
            color="black"
        )

    plt.title("2D PCA – Unique Class Colors + Dual-Rings for Suspected Errors")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Custom legend
    leg_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=true_colors[c], markersize=10,
                   label=f"Class {classes[c]}")
        for c in range(len(classes))
    ] + [
        plt.Line2D([0], [0], marker='o', color='red',
                   markerfacecolor='none', markersize=15,
                   label="Error outer ring = predicted"),
        plt.Line2D([0], [0], marker='o', color='black',
                   markerfacecolor='none', markersize=8,
                   label="Error inner ring = actual")
    ]

    plt.legend(handles=leg_elements, loc="best", fontsize=8)
    plt.tight_layout()

    out_path = "pca_suspicion_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"PCA plot saved to: {out_path}")

def plot_umap(X_scaled, y, predicted, likely_error, classes):
    """UMAP plot with:
       - unique TRUE class colors
       - dual-color rings for suspected label errors
    """
    print("\nGenerating UMAP visualization...")

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    pts = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))

    # Color map for classes
    cmap = plt.get_cmap("viridis", len(classes))
    true_colors = {c: cmap(c) for c in range(len(classes))}
    pred_colors = true_colors

    # Plot clean points
    for c in range(len(classes)):
        idx = np.where((y == c) & (~likely_error))[0]
        if len(idx) > 0:
            plt.scatter(
                pts[idx, 0],
                pts[idx, 1],
                s=25,
                color=true_colors[c],
                alpha=0.45,
                label=f"Class {classes[c]}" if c == 0 else None
            )

    # Plot suspicious / error points
    err_idx = np.where(likely_error)[0]
    for i in err_idx:
        actual = y[i]
        pred = predicted[i]

        # Outer ring = predicted class
        plt.scatter(
            pts[i, 0],
            pts[i, 1],
            s=180,
            edgecolors=pred_colors[pred],
            facecolors='none',
            linewidths=2.5,
        )

        # Inner ring = actual class
        plt.scatter(
            pts[i, 0],
            pts[i, 1],
            s=70,
            edgecolors=true_colors[actual],
            facecolors='none',
            linewidths=2,
        )

        plt.text(
            pts[i, 0],
            pts[i, 1],
            str(i),
            fontsize=7,
            color="black"
        )

    plt.title("UMAP – True Class Colors + Dual Rings for Suspected Label Errors")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Custom legend
    leg_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=true_colors[c], markersize=10,
                   label=f"Class {classes[c]}")
        for c in range(len(classes))
    ] + [
        plt.Line2D([0], [0], marker='o', color='red',
                   markerfacecolor='none', markersize=15,
                   label="Error outer ring = predicted"),
        plt.Line2D([0], [0], marker='o', color='black',
                   markerfacecolor='none', markersize=8,
                   label="Error inner ring = actual")
    ]

    plt.legend(handles=leg_elements, loc="best", fontsize=8)
    plt.tight_layout()

    out_path = "umap_suspicion_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"UMAP plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Confident-Learning-like label error detection + PCA")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--classcol", required=True)
    parser.add_argument("--dropcols", required=False, help="Comma-separated cols to drop")

    args = parser.parse_args()
    drop_cols = args.dropcols.split(",") if args.dropcols else None

    df = pd.read_csv(args.csv)
    if args.classcol not in df.columns:
        raise SystemExit(f"ERROR: class column '{args.classcol}' not in CSV.")

    result, X_scaled, y, predicted, likely_error, classes = run_confident_learning(
        df, args.classcol, drop_cols
    )

    out_csv = "label_errors_detected.csv"
    result.to_csv(out_csv, index=False)

    print("\n=== Top Suspicious Samples ===")
    print(result.head(20))
    print(f"\nSaved ranked suspicious sample list to: {out_csv}")

    plot_pca(X_scaled, y, predicted, likely_error, classes)
    plot_umap(X_scaled, y, predicted, likely_error, classes)


if __name__ == "__main__":
    main()
