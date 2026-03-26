import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from data_loader import get_clustering_features

PERSONA_NAMES = {
    0: "Health-first Metro Pro",
    1: "Budget-conscious Explorer",
    2: "Comfort-seeking Traditionalist",
    3: "Digital-native Snacker",
    4: "Premium Wellness Seeker",
    5: "Casual Occasional Buyer"
}

PERSONA_COLORS = ['#1D9E75', '#D85A30', '#7F77DD', '#378ADD', '#D4537E', '#BA7517']

@st.cache_resource
def run_clustering(df_feat, feature_cols, max_k=8):
    X = df_feat[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    sil_scores = []
    models = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled))))
        models[k] = (km, labels)

    return X_scaled, scaler, inertias, sil_scores, models


def render_clustering(df, df_feat):
    st.header("👥 Customer Segmentation — Clustering")
    st.caption("Discovering distinct customer personas for targeted marketing")

    feature_cols = get_clustering_features(df_feat)
    X_scaled, scaler, inertias, sil_scores, models = run_clustering(df_feat, feature_cols)

    # Elbow & Silhouette
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Method")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2, 9)), y=inertias, mode='lines+markers',
                                 marker=dict(size=10, color='#7F77DD'), line=dict(color='#7F77DD', width=2)))
        fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="Number of Clusters (K)",
                          yaxis_title="Inertia (WCSS)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Silhouette Scores")
        colors = ['#1D9E75' if s == max(sil_scores) else '#D3D1C7' for s in sil_scores]
        fig = go.Figure(go.Bar(x=list(range(2, 9)), y=sil_scores, marker_color=colors,
                               text=[f"{s:.3f}" for s in sil_scores], textposition='auto'))
        fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="Number of Clusters (K)",
                          yaxis_title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Select K
    best_k = sil_scores.index(max(sil_scores)) + 2
    selected_k = st.slider("Select number of clusters", 2, 8, best_k, help=f"Best silhouette score at K={best_k}")

    km_model, labels = models[selected_k]
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    df_feat_clustered = df_feat.copy()
    df_feat_clustered['cluster'] = labels

    # Cluster sizes
    st.subheader(f"Cluster Distribution (K={selected_k})")
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    cluster_names = [PERSONA_NAMES.get(i, f"Segment {i}") for i in cluster_counts.index]
    fig = go.Figure(go.Bar(x=cluster_names, y=cluster_counts.values,
                           marker_color=PERSONA_COLORS[:selected_k],
                           text=cluster_counts.values, textposition='auto'))
    fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="Cluster", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Radar chart profiles
    st.subheader("Cluster Profiles — Radar Charts")
    profile_cols = ['income_numeric', 'spend_numeric', 'wtp_numeric', 'mood_food_sensitivity',
                    'clean_label_importance', 'social_media_numeric', 'meals_out_numeric', 'order_freq_numeric']
    profile_cols = [c for c in profile_cols if c in df_feat_clustered.columns]
    profile_labels = [c.replace('_numeric', '').replace('_', ' ').title() for c in profile_cols]

    cluster_means = df_feat_clustered.groupby('cluster')[profile_cols].mean()
    # Normalize to 0-1 for radar
    mins = cluster_means.min()
    maxs = cluster_means.max()
    cluster_norm = (cluster_means - mins) / (maxs - mins + 1e-8)

    fig = go.Figure()
    for i in range(selected_k):
        if i in cluster_norm.index:
            vals = cluster_norm.loc[i].tolist()
            vals.append(vals[0])
            labels_r = profile_labels + [profile_labels[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=labels_r, fill='toself',
                                          name=PERSONA_NAMES.get(i, f"Segment {i}"),
                                          line=dict(color=PERSONA_COLORS[i % len(PERSONA_COLORS)])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      margin=dict(t=30, b=30), height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed profile table
    st.subheader("Segment Profiling — Key Characteristics")
    profile_summary = []
    for i in range(selected_k):
        seg = df_clustered[df_clustered['cluster'] == i]
        profile_summary.append({
            'Segment': PERSONA_NAMES.get(i, f"Segment {i}"),
            'Size': len(seg),
            'Top Age': seg['age_group'].mode()[0] if len(seg) > 0 else '-',
            'Top City': seg['city_tier'].mode()[0] if len(seg) > 0 else '-',
            'Top Income': seg['monthly_income'].mode()[0] if len(seg) > 0 else '-',
            'Top Occupation': seg['occupation'].mode()[0] if len(seg) > 0 else '-',
            'Avg Mood Score': round(seg['mood_food_sensitivity'].mean(), 1) if len(seg) > 0 else 0,
            'Interest Rate %': round((seg['likelihood_to_try_moodmeal'].isin(['Definitely yes', 'Probably yes'])).mean() * 100, 1) if len(seg) > 0 else 0,
            'Top Frustration': seg['biggest_frustration'].mode()[0] if len(seg) > 0 else '-',
        })
    st.dataframe(pd.DataFrame(profile_summary), use_container_width=True, hide_index=True)

    # Scatter 2D visualization
    st.subheader("2D Cluster Visualization (Income vs Spend)")
    fig = px.scatter(df_feat_clustered, x='income_numeric', y='spend_numeric',
                     color=df_feat_clustered['cluster'].astype(str),
                     color_discrete_sequence=PERSONA_COLORS[:selected_k],
                     labels={'income_numeric': 'Monthly Income (₹)', 'spend_numeric': 'Monthly Food Spend (₹)',
                             'color': 'Cluster'},
                     hover_data=['mood_food_sensitivity', 'clean_label_importance'])
    fig.update_layout(margin=dict(t=10, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

    return km_model, scaler, feature_cols, selected_k
