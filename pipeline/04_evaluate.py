# AMRGuard - Script 04: Evaluate Models
# Generates detailed evaluation — ROC curves, confusion matrix, SHAP

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import shap

# Configuration
FEATURES_FILE = "data/processed/features_X.csv"
LABELS_FILE   = "data/processed/labels_y.csv"
MODELS_DIR    = "models"
OUTPUT_DIR    = "data/processed/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.2

MODEL_FILES = {
    "Logistic Regression" : "logistic_regression.pkl",
    "Random Forest"       : "random_forest.pkl",
    "SVM"                 : "svm.pkl",
    "XGBoost"             : "xgboost.pkl"
}

def load_data():
    print("\n📂 Loading data...")
    X = pd.read_csv(FEATURES_FILE)
    y = pd.read_csv(LABELS_FILE).squeeze()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"   ✅ Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def load_models():
    print("\n📦 Loading trained models...")
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"   ✅ Loaded: {name}")
        else:
            print(f"   ⚠️  Not found: {name}")
    return models

def plot_roc_curves(models, X_test, y_test):
    print("\n📈 Plotting ROC curves...")
    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("AMRGuard — ROC Curves (All Models)", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ✅ Saved → {path}")

def plot_confusion_matrices(models, X_test, y_test):
    print("\n🔲 Plotting confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Susceptible", "Resistant"]
        )
        disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
        axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")

    plt.suptitle("AMRGuard — Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ✅ Saved → {path}")

def plot_feature_importance(models, feature_names):
    print("\n🔍 Plotting feature importance (Random Forest)...")

    if "Random Forest" not in models:
        print("   ⚠️  Random Forest not found, skipping.")
        return

    rf = models["Random Forest"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)),
            importances[indices],
            color="steelblue", edgecolor="white")
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right", fontsize=10)
    plt.ylabel("Importance Score", fontsize=12)
    plt.title("AMRGuard — Feature Importance (Random Forest)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ✅ Saved → {path}")

def run_shap(models, X_test, feature_names):
    print("\n🔮 Running SHAP explainability (Random Forest)...")

    if "Random Forest" not in models:
        print("   ⚠️  Random Forest not found, skipping SHAP.")
        return

    try:
        rf = models["Random Forest"]
        X_sample = X_test.iloc[:100]

        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification take class 1 (Resistant)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar"
        )
        plt.title("AMRGuard — SHAP Feature Impact",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved → {path}")

    except Exception as e:
        print(f"   ⚠️  SHAP failed: {e}")

def save_classification_reports(models, X_test, y_test):
    print("\n📋 Saving classification reports...")
    report_path = os.path.join(OUTPUT_DIR, "classification_reports.txt")

    with open(report_path, "w") as f:
        f.write("AMRGuard - Classification Reports\n")
        f.write("=" * 60 + "\n\n")

        for name, model in models.items():
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred,
                target_names=["Susceptible", "Resistant"]
            )
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(report)
            f.write("\n\n")

    print(f"   ✅ Saved → {report_path}")

def main():
    print("=" * 60)
    print("  AMRGuard - Model Evaluation Script")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data()
    models = load_models()

    if not models:
        print("\n❌ No models found! Run 03_train_models.py first.")
        return

    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models, feature_names)
    run_shap(models, X_test, feature_names)
    save_classification_reports(models, X_test, y_test)

    print("\n" + "=" * 60)
    print("  ✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\n  All plots saved to: data/processed/evaluation/")
    print(f"  Files generated:")
    print(f"    - roc_curves.png")
    print(f"    - confusion_matrices.png")
    print(f"    - feature_importance.png")
    print(f"    - shap_summary.png")
    print(f"    - classification_reports.txt")
    print(f"\n  Ready for Script 05 - Predict!")

if __name__ == "__main__":
    main()