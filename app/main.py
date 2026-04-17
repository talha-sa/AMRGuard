# AMRGuard - Main App Entry Point

import streamlit as st
import os
import sys

# Permanent path fix — always point to AMRGuard root folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="AMRGuard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🦠 AMRGuard")
st.sidebar.markdown("*AMR Predictor for South Asian Pathogens*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔬 Predict", "📊 Explore Data", "🤖 Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Talha")
st.sidebar.markdown("**University:** UAF")
st.sidebar.markdown("**Project:** AMRGuard v1.0")

if page == "🏠 Home":
    from app.pages.home import show
    show()
elif page == "🔬 Predict":
    from app.pages.predict import show
    show()
elif page == "📊 Explore Data":
    from app.pages.explore import show
    show()
elif page == "🤖 Model Info":
    from app.pages.model_info import show
    show()