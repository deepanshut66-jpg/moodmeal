import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MoodMeal Analytics Dashboard",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {background-color: #FDFAF6;}
    [data-testid="stSidebar"] {background-color: #FFF8F5;}
    .stMetric {background: white; padding: 12px; border-radius: 10px; border: 1px solid #E8E5DE; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
    h1, h2, h3 {color: #2C2C2A;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

from data_loader import load_and_preprocess, build_feature_matrix
from descriptive import render_descriptive, render_diagnostic
from clustering import render_clustering
from association import render_association
from classification import render_classification
from regression import render_regression
from predictor import render_predictor

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = load_and_preprocess("moodmeal_survey_data_2000.csv")
    df_feat, label_encoders = build_feature_matrix(df)
    return df, df_feat, label_encoders

df, df_feat, label_encoders = load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🍱 MoodMeal")
    st.markdown("**Analytics Dashboard**")
    st.caption("Data-driven decision making for India's first mood-driven food brand")
    st.divider()

    st.markdown("### 🔍 Global Filters")
    st.caption("Filters apply across all tabs")

    # Age filter
    age_options = sorted(df['age_group'].unique())
    selected_ages = st.multiselect("Age Group", age_options, default=age_options)

    # City filter
    city_options = sorted(df['city_tier'].unique())
    selected_cities = st.multiselect("City Tier", city_options, default=city_options)

    # Income filter
    income_options = ['Below 25K', '25K-50K', '50K-1L', '1L-2L', 'Above 2L']
    income_present = [i for i in income_options if i in df['monthly_income'].unique()]
    selected_incomes = st.multiselect("Monthly Income", income_present, default=income_present)

    # Diet filter
    diet_options = sorted(df['dietary_preference'].unique())
    selected_diets = st.multiselect("Dietary Preference", diet_options, default=diet_options)

    st.divider()
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Rows", f"{len(df):,}")
    st.metric("Total Columns", f"{len(df.columns)}")
    st.metric("Survey Questions", "25")

    st.divider()
    st.caption("Built for MoodMeal by Deepanshu")
    st.caption("MBA — Global Marketing Management")

# Apply filters
mask = (
    df['age_group'].isin(selected_ages) &
    df['city_tier'].isin(selected_cities) &
    df['monthly_income'].isin(selected_incomes) &
    df['dietary_preference'].isin(selected_diets)
)
df_filtered = df[mask].copy()
df_feat_filtered = df_feat[mask].copy()

# Show filter status
if len(df_filtered) < len(df):
    st.info(f"🔍 Showing **{len(df_filtered):,}** of {len(df):,} respondents based on sidebar filters")

# ============================================================
# TABS
# ============================================================
st.markdown("# 🍱 MoodMeal — Consumer Analytics Dashboard")
st.caption("Descriptive → Diagnostic → Predictive → Prescriptive | Classification · Clustering · Association · Regression")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🔍 Diagnostic", "👥 Segmentation",
    "🔗 Basket Analysis", "🎯 Classification", "💰 Regression", "🚀 Predict New"
])

with tab1:
    render_descriptive(df_filtered)

with tab2:
    render_diagnostic(df_filtered, df_feat_filtered)

with tab3:
    km_result = render_clustering(df_filtered, df_feat_filtered)
    if km_result:
        km_model, km_scaler, km_features, km_k = km_result

with tab4:
    render_association(df_filtered)

with tab5:
    clf_results, clf_features, clf_best_name = render_classification(df_feat)

with tab6:
    reg_results, reg_features, reg_best_name = render_regression(df_feat)

with tab7:
    if 'km_model' in dir() and km_model is not None:
        render_predictor(df_feat, clf_results, clf_features, clf_best_name,
                         reg_results, reg_features, reg_best_name,
                         km_model, km_scaler, km_features, km_k)
    else:
        st.warning("⚠️ Please visit the **Segmentation** tab first to train the clustering model, then return here.")
        st.info("The prediction engine requires all models to be trained. Click the 'Segmentation' tab, select a K value, then come back.")
