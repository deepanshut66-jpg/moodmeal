import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import get_regression_features

@st.cache_resource
def train_regressors(df_feat):
    feature_cols = get_regression_features(df_feat)
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['spend_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    return results, X_train, X_test, y_train, y_test, feature_cols


def render_regression(df_feat):
    st.header("💰 Spend Prediction — Regression")
    st.caption("Predicting monthly food spend to inform pricing & subscription strategy")

    results, X_train, X_test, y_train, y_test, feature_cols = train_regressors(df_feat)

    # Metrics Table
    st.subheader("Model Performance Comparison")
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            'Model': name,
            'R² Score': f"{res['r2']:.4f}",
            'MAE (₹)': f"₹{res['mae']:,.0f}",
            'RMSE (₹)': f"₹{res['rmse']:,.0f}"
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    best_name = max(results, key=lambda x: results[x]['r2'])
    st.success(f"🏆 **Best Model: {best_name}** (R² = {results[best_name]['r2']:.4f})")

    st.divider()

    # Visual metrics
    st.subheader("Performance Metrics — Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        colors = ['#1D9E75', '#7F77DD', '#D85A30']
        names = list(results.keys())
        r2s = [results[n]['r2'] for n in names]
        fig.add_trace(go.Bar(x=names, y=r2s, marker_color=colors,
                             text=[f"{v:.3f}" for v in r2s], textposition='auto'))
        fig.update_layout(margin=dict(t=30, b=10), height=350, title="R² Score (higher is better)",
                          yaxis_range=[0, max(1, max(r2s) * 1.1)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        maes = [results[n]['mae'] for n in names]
        rmses = [results[n]['rmse'] for n in names]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MAE', x=names, y=maes, marker_color='#7F77DD',
                             text=[f"₹{v:,.0f}" for v in maes], textposition='auto'))
        fig.add_trace(go.Bar(name='RMSE', x=names, y=rmses, marker_color='#D85A30',
                             text=[f"₹{v:,.0f}" for v in rmses], textposition='auto'))
        fig.update_layout(barmode='group', margin=dict(t=30, b=10), height=350,
                          title="MAE & RMSE (lower is better)", yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Actual vs Predicted
    st.subheader("Actual vs Predicted — Scatter Plots")
    cols = st.columns(3)
    colors = ['#1D9E75', '#7F77DD', '#D85A30']
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.values, y=res['y_pred'], mode='markers',
                                     marker=dict(color=colors[idx], opacity=0.5, size=5),
                                     name='Predictions'))
            min_val = min(y_test.min(), res['y_pred'].min())
            max_val = max(y_test.max(), res['y_pred'].max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', line=dict(dash='dash', color='gray'),
                                     name='Perfect Prediction'))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350,
                              xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature Importance (RF only)
    st.subheader("Feature Importance — Random Forest Regressor")
    rf_model = results['Random Forest Regressor']['model']
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
    imp.index = [x.replace('_numeric', '').replace('_enc', '').replace('_', ' ').title()[:30] for x in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                           marker_color='#1D9E75', text=[f"{v:.3f}" for v in imp.values], textposition='auto'))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=450, xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

    # Residual distribution
    st.subheader("Residual Distribution — Best Model")
    best_res = results[best_name]
    residuals = y_test.values - best_res['y_pred']
    fig = px.histogram(residuals, nbins=40, labels={'value': 'Residual (₹)', 'count': 'Frequency'},
                       color_discrete_sequence=['#7F77DD'])
    fig.update_layout(margin=dict(t=10, b=10), height=350, xaxis_title="Residual (₹)", yaxis_title="Count",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    return results, feature_cols, best_name
