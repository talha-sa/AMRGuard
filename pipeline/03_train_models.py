# AMRGuard - Script 03: Train ML Models
# Trains 4 ML models and saves the best one

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report
)

# Configuration
FEATURES_FILE = "data/processed/features_X.csv"
LABELS_FILE   = "data/processed/labels_y.csv"
MODELS_DIR    = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TEST_SIZE    = 0.2
RANDOM_STATE = 42

def load_data():
    print("\n📂 Loading feature matrix and labels...")
    X = pd.read_csv(FEATURES_FILE)
    y = pd.read_csv(LABELS_FILE).squeeze()
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    print(f"   Resistant(1): {y.sum()} | Susceptible(0): {(y==0).sum()}")
    return X, y

def split_data(X, y):
    print("\n✂️  Splitting data 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples:  {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    print("\n🔵 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("   ✅ Done!")
    return model

def train_random_forest(X_train, y_train):
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✅ Done!")
    return model

def train_svm(X_train, y_train):
    print("\n🔴 Training SVM...")
    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("   ✅ Done!")
    return model

def train_xgboost(X_train, y_train):
    print("\n⚡ Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("   ✅ Done!")
    return model

def evaluate_model(name, model, X_test, y_test):
    print(f"\n📊 Evaluating: {name}")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Susceptible", "Resistant"]))

    return {"name": name, "model": model,
            "accuracy": acc, "f1": f1, "auc": auc}

def save_models(results):
    print("\n💾 Saving all models...")
    for r in results:
        safe_name = r["name"].replace(" ", "_").lower()
        path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(r["model"], path)
        print(f"   ✅ Saved → {path}")

def print_summary(results):
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:<25} {r['accuracy']:>10.4f} "
              f"{r['f1']:>10.4f} {r['auc']:>10.4f}")

    best = max(results, key=lambda x: x["auc"])
    print("\n" + "=" * 60)
    print(f"  🏆 BEST MODEL: {best['name']}")
    print(f"     AUC-ROC: {best['auc']:.4f}")
    print("=" * 60)

    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best["model"], best_path)
    print(f"\n  💾 Best model saved → {best_path}")

    summary_df = pd.DataFrame([{
        "model": r["name"],
        "accuracy": r["accuracy"],
        "f1": r["f1"],
        "auc": r["auc"]
    } for r in results])
    summary_df.to_csv(
        os.path.join(MODELS_DIR, "model_comparison.csv"),
        index=False
    )
    print(f"  💾 Comparison table → models/model_comparison.csv")

def main():
    print("=" * 60)
    print("  AMRGuard - Train ML Models Script")
    print("=" * 60)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = [
        ("Logistic Regression", train_logistic_regression(X_train, y_train)),
        ("Random Forest",       train_random_forest(X_train, y_train)),
        ("SVM",                 train_svm(X_train, y_train)),
        ("XGBoost",             train_xgboost(X_train, y_train)),
    ]

    results = []
    for name, model in models:
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    save_models(results)
    print_summary(results)

    print("\n  ✅ TRAINING COMPLETE!")
    print("  Ready for Script 04 - Evaluation!")

if __name__ == "__main__":
    main()