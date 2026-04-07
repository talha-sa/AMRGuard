# AMRGuard - Main App Entry Point

import streamlit as st

st.set_page_config(
    page_title="AMRGuard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
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

# Load pages
if page == "🏠 Home":
    from pages.home import show
    show()

elif page == "🔬 Predict":
    from pages.predict import show
    show()

elif page == "📊 Explore Data":
    from pages.explore import show
    show()

elif page == "🤖 Model Info":
    from pages.model_info import show
    show()