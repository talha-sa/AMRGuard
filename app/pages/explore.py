# AMRGuard - Explore Data Page

import streamlit as st
import pandas as pd
import plotly.express as px
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(ROOT, "data", "raw", "master_amr_data.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

def show():
    st.title("📊 Dataset Explorer")
    st.markdown("Explore the AMR training dataset from PATRIC Database.")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error(f"Data not found at: {DATA_FILE}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",     len(df))
    col2.metric("Organisms",         df["organism"].nunique())
    col3.metric("Antibiotics",       df["antibiotic"].nunique())
    col4.metric("Resistant Records", int(df["label"].sum()))

    st.markdown("---")

    st.markdown("### 🧫 Records by Organism")
    fig1 = px.bar(
        df["organism"].value_counts().reset_index(),
        x="organism", y="count",
        color="organism",
        labels={"organism": "Organism", "count": "Records"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 💊 Records by Antibiotic")
    fig2 = px.bar(
        df["antibiotic"].value_counts().reset_index(),
        x="antibiotic", y="count",
        color="antibiotic",
        labels={"antibiotic": "Antibiotic", "count": "Records"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏷️ Resistance Distribution")
        label_counts = df["label"].value_counts().reset_index()
        label_counts["label"] = label_counts["label"].map(
            {0: "Susceptible", 1: "Resistant"}
        )
        fig3 = px.pie(
            label_counts, values="count", names="label",
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("### 🌏 South Asian vs Other")
        if "isolation_country" in df.columns:
            sa_countries = [
                "Pakistan", "India", "Bangladesh",
                "Sri Lanka", "Nepal", "Afghanistan"
            ]
            df["region"] = df["isolation_country"].apply(
                lambda x: "South Asian"
                if str(x) in sa_countries else "Other/Unknown"
            )
            region_counts = df["region"].value_counts().reset_index()
            fig4 = px.pie(
                region_counts, values="count", names="region",
                color_discrete_sequence=["#3498db", "#95a5a6"]
            )
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🔥 Resistance Heatmap")
    pivot = df.groupby(
        ["organism", "antibiotic"]
    )["label"].mean().reset_index()
    pivot.columns = ["Organism", "Antibiotic", "Resistance Rate"]
    pivot_wide = pivot.pivot(
        index="Organism",
        columns="Antibiotic",
        values="Resistance Rate"
    )
    fig5 = px.imshow(
        pivot_wide,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Resistance Rate (0=Susceptible, 1=Resistant)"
    )
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### 🗃️ Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)