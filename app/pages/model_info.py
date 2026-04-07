# AMRGuard - Model Info Page

import streamlit as st
import pandas as pd
import plotly.express as px
import os

COMPARISON_FILE = "models/model_comparison.csv"
EVAL_DIR        = "data/processed/evaluation"

def show():
    st.title("🤖 Model Performance")
    st.markdown("Comparison of all 4 ML models trained on the AMR dataset.")
    st.markdown("---")

    if os.path.exists(COMPARISON_FILE):
        df = pd.read_csv(COMPARISON_FILE)

        best_idx = df["auc"].idxmax()

        st.markdown("### 🏆 Model Comparison Table")
        st.dataframe(
            df.style.highlight_max(
                subset=["accuracy", "f1", "auc"],
                color="#d4edda"
            ).format({
                "accuracy": "{:.4f}",
                "f1"      : "{:.4f}",
                "auc"     : "{:.4f}"
            }),
            use_container_width=True
        )

        best = df.loc[best_idx]
        st.success(
            f"🏆 Best Model: **{best['model']}** "
            f"with AUC-ROC of **{best['auc']:.4f}**"
        )

        st.markdown("### 📊 AUC-ROC Comparison")
        fig1 = px.bar(
            df, x="model", y="auc",
            color="model",
            text=df["auc"].round(4),
            labels={"model": "Model", "auc": "AUC-ROC Score"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(
            showlegend=False, height=400,
            yaxis_range=[0.5, 1.0]
        )
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Accuracy")
            fig2 = px.bar(
                df, x="model", y="accuracy",
                color="model",
                text=df["accuracy"].round(4),
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig2.update_layout(showlegend=False, height=350,
                               yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("### 📐 F1 Score")
            fig3 = px.bar(
                df, x="model", y="f1",
                color="model",
                text=df["f1"].round(4),
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig3.update_layout(showlegend=False, height=350,
                               yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Model comparison file not found. Run pipeline scripts first.")

    st.markdown("---")
    st.markdown("### 📈 ROC Curves")
    roc_path = os.path.join(EVAL_DIR, "roc_curves.png")
    if os.path.exists(roc_path):
        st.image(roc_path, use_column_width=True)
    else:
        st.info("ROC curve plot not found. Run 04_evaluate.py first.")

    st.markdown("### 🔲 Confusion Matrices")
    cm_path = os.path.join(EVAL_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True)
    else:
        st.info("Confusion matrix plot not found. Run 04_evaluate.py first.")

    st.markdown("### 🔍 Feature Importance")
    fi_path = os.path.join(EVAL_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_column_width=True)
    else:
        st.info("Feature importance plot not found. Run 04_evaluate.py first.")

    st.markdown("### 🔮 SHAP Explainability")
    shap_path = os.path.join(EVAL_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_column_width=True)
    else:
        st.info("SHAP plot not found. Run 04_evaluate.py first.")

    st.markdown("---")
    st.markdown("### ℹ️ About the Models")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Random Forest**\n\n"
            "Ensemble of 100 decision trees. "
            "Handles class imbalance well and provides feature importance."
        )
        st.info(
            "**XGBoost**\n\n"
            "Gradient boosting framework. "
            "Fast, accurate, and excellent on tabular data."
        )
    with col2:
        st.info(
            "**SVM**\n\n"
            "Support Vector Machine with RBF kernel. "
            "Strong performance on small-to-medium datasets."
        )
        st.info(
            "**Logistic Regression**\n\n"
            "Simple linear baseline model. "
            "Fast and interpretable."
        )