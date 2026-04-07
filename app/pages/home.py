# AMRGuard - Home Page

import streamlit as st

def show():
    st.title("🦠 AMRGuard")
    st.subheader("ML-Based Antimicrobial Resistance Predictor")
    st.markdown("*Focused on South Asian Clinical Pathogens*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧫 Organisms Covered", "3")
        st.caption("E. coli, K. pneumoniae, S. aureus")

    with col2:
        st.metric("💊 Antibiotics Predicted", "6")
        st.caption("Ampicillin, Ciprofloxacin & more")

    with col3:
        st.metric("🗃️ Training Records", "2,789")
        st.caption("From PATRIC Database")

    st.markdown("---")

    st.markdown("### 🌏 Why South Asia?")
    st.info(
        "Antimicrobial resistance is a critical public health threat in "
        "Pakistan, India, and Bangladesh. This tool is specifically trained "
        "on South Asian clinical isolates to provide regionally relevant "
        "resistance predictions — a gap no existing tool addresses."
    )

    st.markdown("### 🔬 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("**Step 1**\n\nSelect organism & origin")
    with col2:
        st.success("**Step 2**\n\nML model analyzes features")
    with col3:
        st.success("**Step 3**\n\nGet resistance profile")
    with col4:
        st.success("**Step 4**\n\nDownload results")

    st.markdown("### 🎯 Target Organisms")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.warning("🔴 **Escherichia coli**\nMost common UTI & gut pathogen")
    with col2:
        st.warning("🔴 **Klebsiella pneumoniae**\nHospital-acquired infections")
    with col3:
        st.warning("🔴 **Staphylococcus aureus**\nSkin & bloodstream infections")

    st.markdown("---")
    st.markdown("### 💊 Antibiotics Covered")
    abs = [
        ("Ampicillin", "Penicillin class"),
        ("Ciprofloxacin", "Fluoroquinolone"),
        ("Tetracycline", "Broad spectrum"),
        ("Gentamicin", "Aminoglycoside"),
        ("Imipenem", "Carbapenem — last resort"),
        ("Trimethoprim", "Folate inhibitor"),
    ]
    cols = st.columns(3)
    for i, (ab, cls) in enumerate(abs):
        with cols[i % 3]:
            st.info(f"**{ab}**\n{cls}")

    st.markdown("---")
    st.caption(
        "Data source: PATRIC Database | "
        "Models: Random Forest, XGBoost, SVM, Logistic Regression"
    )