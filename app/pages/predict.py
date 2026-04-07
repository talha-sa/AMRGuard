# AMRGuard - Predict Page

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = "models"

ORGANISMS = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus"
]

TARGET_ANTIBIOTICS = [
    "ampicillin",
    "ciprofloxacin",
    "tetracycline",
    "gentamicin",
    "imipenem",
    "trimethoprim"
]

ORGANISM_ENCODING = {
    "Escherichia coli"      : 0,
    "Klebsiella pneumoniae" : 1,
    "Staphylococcus aureus" : 2
}

ANTIBIOTIC_ENCODING = {
    "ampicillin"    : 0,
    "ciprofloxacin" : 1,
    "gentamicin"    : 2,
    "imipenem"      : 3,
    "tetracycline"  : 4,
    "trimethoprim"  : 5
}

@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def build_input_row(organism, antibiotic, is_south_asian):
    row = {
        "organism_encoded"        : ORGANISM_ENCODING.get(organism, 0),
        "antibiotic_encoded"      : ANTIBIOTIC_ENCODING.get(antibiotic, 0),
        "is_south_asian"          : is_south_asian,
        "year_normalized"         : 0.5,
        "is_ampicillin"           : 1 if antibiotic == "ampicillin"    else 0,
        "is_ciprofloxacin"        : 1 if antibiotic == "ciprofloxacin" else 0,
        "is_tetracycline"         : 1 if antibiotic == "tetracycline"  else 0,
        "is_gentamicin"           : 1 if antibiotic == "gentamicin"    else 0,
        "is_imipenem"             : 1 if antibiotic == "imipenem"      else 0,
        "is_trimethoprim"         : 1 if antibiotic == "trimethoprim"  else 0,
        "is_escherichia_coli"     : 1 if organism == "Escherichia coli"      else 0,
        "is_klebsiella_pneumoniae": 1 if organism == "Klebsiella pneumoniae" else 0,
        "is_staphylococcus_aureus": 1 if organism == "Staphylococcus aureus" else 0,
    }
    return pd.DataFrame([row])

def show():
    st.title("🔬 AMR Resistance Predictor")
    st.markdown("Select an organism and origin to get a full resistance profile.")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.error("Model not found! Please run the training pipeline first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        organism = st.selectbox(
            "🧫 Select Organism",
            ORGANISMS,
            help="Choose the bacterial species to analyze"
        )

    with col2:
        origin = st.radio(
            "🌏 Isolate Origin",
            ["South Asian", "Other / Unknown"],
            help="South Asian isolates may have distinct resistance patterns"
        )

    is_south_asian = 1 if origin == "South Asian" else 0

    st.markdown("---")

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing resistance profile..."):

            results = []
            for antibiotic in TARGET_ANTIBIOTICS:
                X = build_input_row(organism, antibiotic, is_south_asian)
                pred  = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                conf  = proba[pred] * 100
                results.append({
                    "Antibiotic"  : antibiotic.capitalize(),
                    "Result"      : "RESISTANT" if pred == 1 else "SUSCEPTIBLE",
                    "Confidence"  : round(conf, 1),
                    "R_prob"      : round(proba[1] * 100, 1),
                    "S_prob"      : round(proba[0] * 100, 1),
                })

        st.success("✅ Prediction Complete!")

        st.markdown(f"### 🧫 {organism}")
        st.markdown(f"**Origin:** {origin}")
        st.markdown("---")

        resistant   = [r for r in results if r["Result"] == "RESISTANT"]
        susceptible = [r for r in results if r["Result"] == "SUSCEPTIBLE"]

        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Resistant",   len(resistant))
        col2.metric("🟢 Susceptible", len(susceptible))
        col3.metric("📊 Total Tested", len(results))

        st.markdown("### 📋 Full Resistance Profile")

        for r in results:
            col1, col2, col3 = st.columns([3, 2, 5])
            with col1:
                st.markdown(f"**{r['Antibiotic']}**")
            with col2:
                if r["Result"] == "RESISTANT":
                    st.error(f"🔴 {r['Result']}")
                else:
                    st.success(f"🟢 {r['Result']}")
            with col3:
                st.progress(
                    int(r["R_prob"]),
                    text=f"Resistant prob: {r['R_prob']}%"
                )

        st.markdown("---")

        df_results = pd.DataFrame(results)[
            ["Antibiotic", "Result", "Confidence", "R_prob", "S_prob"]
        ]
        df_results.columns = [
            "Antibiotic", "Prediction",
            "Confidence (%)", "Resistant Prob (%)", "Susceptible Prob (%)"
        ]

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name=f"amrguard_{organism.replace(' ','_')}_results.csv",
            mime="text/csv",
            use_container_width=True
        )