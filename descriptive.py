import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from data_loader import encode_target, get_classification_features, build_feature_matrix

def render_descriptive(df):
    st.header("📊 Executive Overview")
    st.caption("Descriptive snapshot of 2,000 survey respondents across India")

    # KPI Row
    total = len(df)
    interest_rate = (df['likelihood_to_try_moodmeal'].isin(['Definitely yes', 'Probably yes'])).mean() * 100
    top_city = df['city_tier'].mode()[0]
    top_cat_cols = [c for c in df.columns if c.startswith('interest_')]
    top_cat = max(top_cat_cols, key=lambda c: df[c].sum()).replace('interest_', '').replace('_', ' ').title()
    avg_mood = df['mood_food_sensitivity'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Respondents", f"{total:,}")
    c2.metric("Interest Rate", f"{interest_rate:.1f}%")
    c3.metric("Top City Tier", top_city)
    c4.metric("Top Category", top_cat[:18])
    c5.metric("Avg Mood Score", f"{avg_mood:.1f}/5")

    st.divider()

    # Row 1: Demographics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age & Gender Distribution")
        ag = df.groupby(['age_group', 'gender']).size().reset_index(name='count')
        fig = px.sunburst(ag, path=['age_group', 'gender'], values='count',
                          color='count', color_continuous_scale='Tealgrn')
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Interest Level Distribution")
        order = ['Definitely yes', 'Probably yes', 'Neutral', 'Probably no', 'Definitely no']
        vc = df['likelihood_to_try_moodmeal'].value_counts().reindex(order).fillna(0)
        colors = ['#1D9E75', '#5DCAA5', '#FAC775', '#F0997B', '#E24B4A']
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation='h',
                               marker_color=colors, text=vc.values, textposition='auto'))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380,
                          yaxis=dict(categoryorder='array', categoryarray=order[::-1]),
                          xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: City & Occupation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Respondents by City Tier")
        city_interest = pd.crosstab(df['city_tier'], df['likelihood_to_try_moodmeal'])
        order_cols = ['Definitely yes', 'Probably yes', 'Neutral', 'Probably no', 'Definitely no']
        city_interest = city_interest.reindex(columns=order_cols, fill_value=0)
        fig = px.bar(city_interest, barmode='stack',
                     color_discrete_sequence=['#1D9E75', '#5DCAA5', '#FAC775', '#F0997B', '#E24B4A'])
        fig.update_layout(margin=dict(t=10, b=10), height=350, legend_title="Interest Level",
                          xaxis_title="City Tier", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Product Category Interest")
        cat_cols = [c for c in df.columns if c.startswith('interest_')]
        cat_sums = df[cat_cols].sum().sort_values(ascending=True)
        cat_sums.index = [x.replace('interest_', '').replace('_', ' ').title() for x in cat_sums.index]
        fig = go.Figure(go.Bar(x=cat_sums.values, y=cat_sums.index, orientation='h',
                               marker_color='#7F77DD', text=cat_sums.values, textposition='auto'))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350,
                          xaxis_title="Number of Respondents Interested")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Occupation & Income treemap
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Occupation Breakdown")
        occ = df['occupation'].value_counts().reset_index()
        occ.columns = ['occupation', 'count']
        fig = px.pie(occ, values='count', names='occupation', hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Income Distribution by City")
        inc_city = df.groupby(['city_tier', 'monthly_income']).size().reset_index(name='count')
        fig = px.treemap(inc_city, path=['city_tier', 'monthly_income'], values='count',
                         color='count', color_continuous_scale='Purples')
        fig.update_layout(margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_diagnostic(df, df_feat):
    st.header("🔍 Customer Deep-Dive — Diagnostic Analysis")
    st.caption("Understanding WHY some customers are interested and others aren't")

    # Cross-tab selector
    st.subheader("Interactive Cross-Tabulation")
    col1, col2 = st.columns(2)
    cat_options = ['city_tier', 'age_group', 'monthly_income', 'occupation', 'gender',
                   'dietary_preference', 'biggest_frustration', 'social_media_hours',
                   'eco_packaging_willingness', 'subscription_interest']
    with col1:
        dim1 = st.selectbox("Row dimension", cat_options, index=0)
    with col2:
        dim2 = st.selectbox("Column dimension", ['likelihood_to_try_moodmeal'] + cat_options, index=0)

    ct = pd.crosstab(df[dim1], df[dim2], normalize='index').round(3) * 100
    fig = px.imshow(ct, text_auto='.1f', color_continuous_scale='Tealgrn', aspect='auto')
    fig.update_layout(margin=dict(t=30, b=10), height=400, title=f"{dim1} vs {dim2} (row %)")
    st.plotly_chart(fig, use_container_width=True)

    # Chi-square test
    contingency = pd.crosstab(df[dim1], df[dim2])
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    if p_val < 0.05:
        st.success(f"✅ **Statistically significant** association (χ² = {chi2:.1f}, p = {p_val:.4f}, dof = {dof})")
    else:
        st.warning(f"⚠️ **Not significant** at 95% confidence (χ² = {chi2:.1f}, p = {p_val:.4f}, dof = {dof})")

    st.divider()

    # Correlation heatmap
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Feature Correlation Heatmap")
        num_cols = ['income_numeric', 'spend_numeric', 'wtp_numeric', 'mood_food_sensitivity',
                    'clean_label_importance', 'social_media_numeric', 'meals_out_numeric',
                    'order_freq_numeric', 'target_numeric']
        num_cols = [c for c in num_cols if c in df_feat.columns]
        corr = df_feat[num_cols].corr().round(2)
        labels = [c.replace('_numeric', '').replace('_', ' ').title() for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        x=labels, y=labels, zmin=-1, zmax=1)
        fig.update_layout(margin=dict(t=10, b=10), height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Predictive Features")
        st.caption("Random Forest feature importance (diagnostic)")
        feature_cols = get_classification_features(df_feat)
        X = df_feat[feature_cols].fillna(0)
        y = df_feat['target_numeric']
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
        imp.index = [x.replace('_numeric', '').replace('_enc', '').replace('_', ' ').title()[:25] for x in imp.index]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker_color='#D85A30'))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=450, xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Frustration analysis
    st.subheader("Pain Point Analysis by Interest Level")
    frust_int = pd.crosstab(df['biggest_frustration'], df['likelihood_to_try_moodmeal'], normalize='columns').round(3) * 100
    order = ['Definitely yes', 'Probably yes', 'Neutral', 'Probably no', 'Definitely no']
    frust_int = frust_int.reindex(columns=[c for c in order if c in frust_int.columns])
    fig = px.bar(frust_int, barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(t=10, b=10), height=380, xaxis_title="Frustration",
                      yaxis_title="% within Interest Level", legend_title="Interest Level")
    st.plotly_chart(fig, use_container_width=True)
