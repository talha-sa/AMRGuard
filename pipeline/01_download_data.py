"""
AMRGuard - Script 01: Download AMR Data from PATRIC Database
Downloads genome metadata + AMR phenotype data for 3 target organisms
filtered by South Asian geographic origin.
"""

import requests
import pandas as pd
import os
import time

# ── Configuration ──────────────────────────────────────────────────────────────

PATRIC_API = "https://www.patricbrc.org/api"

TARGET_ORGANISMS = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus"
]

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

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper Functions ───────────────────────────────────────────────────────────

def fetch_amr_data(organism_name, max_records=5000):
    """
    Fetch AMR phenotype data for a given organism from PATRIC.
    """
    print(f"\n📥 Fetching AMR data for: {organism_name}")

    organism_encoded = organism_name.replace(" ", "%20")
    url = (
        f"https://www.patricbrc.org/api/genome_amr/?"
        f"eq(genome_name,{organism_encoded})"
        f"&select(genome_id,genome_name,antibiotic,resistant_phenotype,"
        f"isolation_country,collection_year)"
        f"&limit({max_records})"
    )

    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            records = response.json()
            df = pd.DataFrame(records)
            print(f"   ✅ Retrieved {len(df)} AMR records")
            return df
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()


def filter_south_asian(df):
    """
    Filter records to keep South Asian isolates.
    If too few found, keep all records and flag SA ones.
    """
    if df.empty or "isolation_country" not in df.columns:
        return df

    south_asian = df[df["isolation_country"].isin(SOUTH_ASIA_COUNTRIES)]
    no_country = df[df["isolation_country"].isna()]

    print(f"   🌏 South Asian isolates: {len(south_asian)}")
    print(f"   ❓ No country info: {len(no_country)}")

    if len(south_asian) < 100:
        print(f"   ⚠️  Too few SA records, using full dataset")
        df["south_asian"] = df["isolation_country"].isin(SOUTH_ASIA_COUNTRIES)
        return df
    else:
        combined = pd.concat([south_asian, no_country], ignore_index=True)
        combined["south_asian"] = combined["isolation_country"].isin(
            SOUTH_ASIA_COUNTRIES
        )
        return combined


def filter_target_antibiotics(df):
    """
    Keep only rows matching our 6 target antibiotics.
    """
    if df.empty or "antibiotic" not in df.columns:
        return df

    df = df.copy()
    df["antibiotic"] = df["antibiotic"].str.lower().str.strip()
    filtered = df[df["antibiotic"].isin(TARGET_ANTIBIOTICS)]
    print(f"   💊 Records for target antibiotics: {len(filtered)}")
    return filtered


def clean_phenotype(df):
    """
    Standardize resistance labels → Resistant=1, Susceptible=0
    Drop intermediate (ambiguous).
    """
    if df.empty or "resistant_phenotype" not in df.columns:
        return df

    df = df.copy()
    df["resistant_phenotype"] = df["resistant_phenotype"].str.lower().str.strip()

    mapping = {
        "resistant": 1,
        "susceptible": 0,
        "intermediate": None
    }

    df["label"] = df["resistant_phenotype"].map(mapping)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"   🏷️  After label cleaning: {len(df)} records")
    print(f"      Resistant: {df['label'].sum()} | Susceptible: {(df['label']==0).sum()}")
    return df


# ── Main Execution ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AMRGuard — Data Download Script")
    print("  Source: PATRIC Database (patricbrc.org)")
    print("=" * 60)

    all_data = []

    for organism in TARGET_ORGANISMS:

        # Step 1: Fetch raw data
        df = fetch_amr_data(organism)

        if df.empty:
            print(f"   ⚠️  No data retrieved for {organism}, skipping.")
            continue

        # Step 2: Filter South Asian isolates
        df_sa = filter_south_asian(df)

        # Step 3: Filter target antibiotics
        df_filtered = filter_target_antibiotics(df_sa)

        # Step 4: Clean labels
        df_clean = clean_phenotype(df_filtered)

        if df_clean.empty:
            print(f"   ⚠️  No usable data after cleaning for {organism}")
            continue

        # Step 5: Add organism column
        df_clean["organism"] = organism

        all_data.append(df_clean)

        # Save per-organism CSV
        safe_name = organism.replace(" ", "_").lower()
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_amr.csv")
        df_clean.to_csv(out_path, index=False)
        print(f"   💾 Saved → {out_path}")

        # Pause between requests
        time.sleep(2)

    # Combine all into master file
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)

        print("\n" + "=" * 60)
        print("  ✅ DOWNLOAD COMPLETE — Summary")
        print("=" * 60)
        print(f"  Total records: {len(master_df)}")
        print(f"\n  By organism:")
        print(master_df["organism"].value_counts().to_string())
        print(f"\n  By antibiotic:")
        print(master_df["antibiotic"].value_counts().to_string())
        print(f"\n  By label (1=Resistant, 0=Susceptible):")
        print(master_df["label"].value_counts().to_string())

        master_path = os.path.join(OUTPUT_DIR, "master_amr_data.csv")
        master_df.to_csv(master_path, index=False)
        print(f"\n  💾 Master file saved → {master_path}")

    else:
        print("\n❌ No data downloaded. Check your internet connection.")


if __name__ == "__main__":
    main()