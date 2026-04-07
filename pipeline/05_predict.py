# AMRGuard - Script 05: Predict
# Takes user input and predicts AMR resistance profile

import pandas as pd
import numpy as np
import os
import joblib

# Configuration
MODELS_DIR    = "models"
FEATURES_FILE = "data/processed/features_X.csv"

TARGET_ANTIBIOTICS = [
    "ampicillin",
    "ciprofloxacin",
    "tetracycline",
    "gentamicin",
    "imipenem",
    "trimethoprim"
]

ORGANISMS = {
    "1": "Escherichia coli",
    "2": "Klebsiella pneumoniae",
    "3": "Staphylococcus aureus"
}

ORGANISM_ENCODING = {
    "Escherichia coli"      : 0,
    "Klebsiella pneumoniae" : 1,
    "Staphylococcus aureus" : 2
}

ANTIBIOTIC_ENCODING = {
    "ampicillin"     : 0,
    "ciprofloxacin"  : 1,
    "gentamicin"     : 2,
    "imipenem"       : 3,
    "tetracycline"   : 4,
    "trimethoprim"   : 5
}

def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        print("❌ Best model not found! Run 03_train_models.py first.")
        return None
    model = joblib.load(path)
    print("✅ Best model loaded successfully!")
    return model

def get_feature_names():
    df = pd.read_csv(FEATURES_FILE)
    return df.columns.tolist()

def build_input_row(organism, antibiotic, is_south_asian):
    row = {
        "organism_encoded"       : ORGANISM_ENCODING.get(organism, 0),
        "antibiotic_encoded"     : ANTIBIOTIC_ENCODING.get(antibiotic, 0),
        "is_south_asian"         : is_south_asian,
        "year_normalized"        : 0.5,
        "is_ampicillin"          : 1 if antibiotic == "ampicillin"    else 0,
        "is_ciprofloxacin"       : 1 if antibiotic == "ciprofloxacin" else 0,
        "is_tetracycline"        : 1 if antibiotic == "tetracycline"  else 0,
        "is_gentamicin"          : 1 if antibiotic == "gentamicin"    else 0,
        "is_imipenem"            : 1 if antibiotic == "imipenem"      else 0,
        "is_trimethoprim"        : 1 if antibiotic == "trimethoprim"  else 0,
        "is_escherichia_coli"    : 1 if organism == "Escherichia coli"     else 0,
        "is_klebsiella_pneumoniae": 1 if organism == "Klebsiella pneumoniae" else 0,
        "is_staphylococcus_aureus": 1 if organism == "Staphylococcus aureus" else 0,
    }
    return pd.DataFrame([row])

def predict_single(model, organism, antibiotic, is_south_asian):
    X = build_input_row(organism, antibiotic, is_south_asian)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    result = "RESISTANT" if prediction == 1 else "SUSCEPTIBLE"
    confidence = probability[prediction] * 100

    return result, confidence, probability

def predict_full_profile(model, organism, is_south_asian):
    print(f"\n{'=' * 55}")
    print(f"  AMR RESISTANCE PROFILE")
    print(f"  Organism : {organism}")
    print(f"  Origin   : {'South Asian' if is_south_asian else 'Other/Unknown'}")
    print(f"{'=' * 55}")
    print(f"  {'Antibiotic':<20} {'Result':<15} {'Confidence':>10}")
    print(f"  {'-' * 50}")

    results = []
    for antibiotic in TARGET_ANTIBIOTICS:
        result, confidence, proba = predict_single(
            model, organism, antibiotic, is_south_asian
        )
        icon = "🔴" if result == "RESISTANT" else "🟢"
        print(f"  {antibiotic:<20} {icon} {result:<13} {confidence:>9.1f}%")
        results.append({
            "antibiotic" : antibiotic,
            "result"     : result,
            "confidence" : round(confidence, 2)
        })

    resistant_count = sum(1 for r in results if r["result"] == "RESISTANT")
    print(f"\n  Summary: {resistant_count}/6 antibiotics resistant")
    print(f"{'=' * 55}")
    return results

def save_results(results, organism):
    safe_name = organism.replace(" ", "_").lower()
    path = f"data/processed/{safe_name}_prediction.csv"
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\n  💾 Results saved → {path}")

def main():
    print("=" * 55)
    print("  AMRGuard - AMR Resistance Predictor")
    print("=" * 55)

    # Load model
    model = load_best_model()
    if model is None:
        return

    # Select organism
    print("\n  Select organism:")
    print("  1. Escherichia coli")
    print("  2. Klebsiella pneumoniae")
    print("  3. Staphylococcus aureus")
    choice = input("\n  Enter number (1/2/3): ").strip()

    if choice not in ORGANISMS:
        print("❌ Invalid choice! Enter 1, 2, or 3.")
        return

    organism = ORGANISMS[choice]

    # South Asian origin
    sa_input = input("\n  Is this a South Asian isolate? (y/n): ").strip().lower()
    is_south_asian = 1 if sa_input == "y" else 0

    # Predict full profile
    results = predict_full_profile(model, organism, is_south_asian)

    # Save results
    save_results(results, organism)

    print("\n  ✅ Prediction complete!")
    print("  Pipeline finished — ready to build the web app!")

if __name__ == "__main__":
    main()