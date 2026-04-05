# AMRGuard - Script 02: Feature Extraction
# Converts AMR metadata into ML-ready feature matrix

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_FILE = "data/raw/master_amr_data.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOUTH_ASIA_COUNTRIES = [
    "Pakistan", "India", "Bangladesh",
    "Sri Lanka", "Nepal", "Afghanistan"
]

TARGET_ANTIBIOTICS = [
    "ampicillin",
    "ciprofloxacin",
    "tetracycline",
    "gentamicin",
    "imipenem",
    "trimethoprim"
]

def load_data(filepath):
    print(f"\n📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    return df

def encode_organism(df):
    print("\n🔬 Encoding organisms...")
    le = LabelEncoder()
    df["organism_encoded"] = le.fit_transform(df["organism"])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   Mapping: {mapping}")
    return df, le

def encode_antibiotic(df):
    print("\n💊 Encoding antibiotics...")
    le = LabelEncoder()
    df["antibiotic_encoded"] = le.fit_transform(df["antibiotic"])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   Mapping: {mapping}")
    return df, le

def encode_country(df):
    print("\n🌏 Encoding geographic origin...")
    if "isolation_country" in df.columns:
        df["is_south_asian"] = df["isolation_country"].apply(
            lambda x: 1 if str(x) in SOUTH_ASIA_COUNTRIES else 0
        )
    else:
        df["is_south_asian"] = 0
    sa_count = df["is_south_asian"].sum()
    print(f"   South Asian isolates: {sa_count}")
    print(f"   Other/Unknown: {len(df) - sa_count}")
    return df

def encode_year(df):
    print("\n📅 Encoding collection year...")
    if "collection_year" in df.columns:
        df["collection_year"] = pd.to_numeric(
            df["collection_year"], errors="coerce"
        )
        median_year = df["collection_year"].median()
        df["collection_year"] = df["collection_year"].fillna(median_year)
        min_year = df["collection_year"].min()
        max_year = df["collection_year"].max()
        if max_year > min_year:
            df["year_normalized"] = (
                (df["collection_year"] - min_year) / (max_year - min_year)
            )
        else:
            df["year_normalized"] = 0.5
        print(f"   Year range: {int(min_year)} - {int(max_year)}")
    else:
        df["year_normalized"] = 0.5
        print("   No year column found, using default 0.5")
    return df

def create_antibiotic_dummies(df):
    print("\n🔢 Creating antibiotic dummy columns...")
    for ab in TARGET_ANTIBIOTICS:
        col_name = f"is_{ab}"
        df[col_name] = (df["antibiotic"] == ab).astype(int)
        print(f"   {col_name}: {df[col_name].sum()} records")
    return df

def create_organism_dummies(df):
    print("\n🔢 Creating organism dummy columns...")
    organisms = df["organism"].unique()
    for org in organisms:
        safe_name = org.replace(" ", "_").lower()
        col_name = f"is_{safe_name}"
        df[col_name] = (df["organism"] == org).astype(int)
        print(f"   {col_name}: {df[col_name].sum()} records")
    return df

def build_feature_matrix(df):
    print("\n🏗️  Building feature matrix...")
    feature_cols = (
        ["organism_encoded", "antibiotic_encoded",
         "is_south_asian", "year_normalized"] +
        [f"is_{ab}" for ab in TARGET_ANTIBIOTICS] +
        [col for col in df.columns if
         col.startswith("is_escherichia") or
         col.startswith("is_klebsiella") or
         col.startswith("is_staphylococcus")]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].copy()
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Samples: {len(X)}")
    print(f"   Resistant(1): {y.sum()} | Susceptible(0): {(y==0).sum()}")
    return X, y, feature_cols

def save_outputs(df, X, y, feature_cols):
    print("\n💾 Saving outputs...")
    df.to_csv(os.path.join(OUTPUT_DIR, "processed_amr_data.csv"), index=False)
    X.to_csv(os.path.join(OUTPUT_DIR, "features_X.csv"), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, "labels_y.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        for feat in feature_cols:
            f.write(feat + "\n")
    print("   ✅ All files saved to data/processed/")

def main():
    print("=" * 60)
    print("  AMRGuard - Feature Extraction Script")
    print("=" * 60)
    df = load_data(INPUT_FILE)
    df, org_encoder  = encode_organism(df)
    df, ab_encoder   = encode_antibiotic(df)
    df               = encode_country(df)
    df               = encode_year(df)
    df               = create_antibiotic_dummies(df)
    df               = create_organism_dummies(df)
    X, y, feature_cols = build_feature_matrix(df)
    save_outputs(df, X, y, feature_cols)
    print("\n" + "=" * 60)
    print("  ✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Total samples: {len(y)}")

if __name__ == "__main__":
    main()