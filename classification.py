import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
from data_loader import get_classification_features

@st.cache_resource
def train_classifiers(df_feat):
    feature_cols = get_classification_features(df_feat)
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_split=10),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, min_samples_split=5),
        'Gradient Boosted Tree': GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42, learning_rate=0.1, min_samples_split=5)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        # ROC AUC
        if y_proba is not None:
            classes = sorted(y.unique())
            y_test_bin = label_binarize(y_test, classes=classes)
            if y_test_bin.shape[1] > 1:
                try:
                    results[name]['roc_auc'] = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    results[name]['roc_auc'] = 0.0
            else:
                results[name]['roc_auc'] = 0.0
        else:
            results[name]['roc_auc'] = 0.0

    return results, X_train, X_test, y_train, y_test, feature_cols


def render_classification(df_feat):
    st.header("🎯 Interest Prediction — Classification")
    st.caption("Predicting customer likelihood to try MoodMeal using 3 models")

    results, X_train, X_test, y_train, y_test, feature_cols = train_classifiers(df_feat)

    # Metrics comparison table
    st.subheader("Model Performance Comparison")
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']:.4f}",
            'Precision (W)': f"{res['precision']:.4f}",
            'Recall (W)': f"{res['recall']:.4f}",
            'F1-Score (W)': f"{res['f1']:.4f}",
            'ROC-AUC (W)': f"{res['roc_auc']:.4f}"
        })
    metrics_df = pd.DataFrame(metrics_data)

    # Highlight best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.success(f"🏆 **Best Model: {best_model_name}** (highest weighted F1-Score: {results[best_model_name]['f1']:.4f})")

    st.divider()

    # Visual metrics comparison
    st.subheader("Performance Metrics — Visual Comparison")
    metric_names = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1-Score (W)', 'ROC-AUC (W)']
    fig = go.Figure()
    colors = ['#1D9E75', '#7F77DD', '#D85A30']
    for idx, (name, res) in enumerate(results.items()):
        vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
        fig.add_trace(go.Bar(name=name, x=metric_names, y=vals,
                             marker_color=colors[idx], text=[f"{v:.3f}" for v in vals], textposition='auto'))
    fig.update_layout(barmode='group', margin=dict(t=10, b=10), height=380,
                      yaxis_title="Score", yaxis_range=[0, 1], legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ROC Curves
    st.subheader("ROC Curves — Multi-class (One-vs-Rest)")
    classes = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    target_map = {0: 'Definitely No', 1: 'Probably No', 2: 'Neutral', 3: 'Probably Yes', 4: 'Definitely Yes'}

    # Select which class to show ROC for
    class_names = [target_map.get(c, f"Class {c}") for c in classes]
    selected_class = st.selectbox("Select class for ROC curve", class_names, index=min(3, len(class_names)-1))
    class_idx = class_names.index(selected_class)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                             name='Random (AUC=0.5)', showlegend=True))

    for idx, (name, res) in enumerate(results.items()):
        if res['y_proba'] is not None and y_test_bin.shape[1] > class_idx:
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], res['y_proba'][:, class_idx])
            try:
                auc_val = roc_auc_score(y_test_bin[:, class_idx], res['y_proba'][:, class_idx])
            except Exception:
                auc_val = 0.0
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{name} (AUC={auc_val:.3f})",
                                     line=dict(color=colors[idx], width=2.5)))

    fig.update_layout(margin=dict(t=10, b=10), height=420,
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      legend=dict(x=0.55, y=0.05))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            cm = res['confusion']
            labels = [target_map.get(c, str(c)) for c in classes]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Tealgrn',
                            x=labels, y=labels, aspect='auto')
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                              xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature Importance
    st.subheader("Feature Importance — Best Model")
    best_model = results[best_model_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
        imp.index = [x.replace('_numeric', '').replace('_enc', '').replace('_', ' ').title()[:30] for x in imp.index]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                               marker_color='#D85A30', text=[f"{v:.3f}" for v in imp.values],
                               textposition='auto'))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=550,
                          xaxis_title="Feature Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Per-class classification report
    st.subheader("Per-Class Classification Report")
    report = results[best_model_name]['report']
    report_data = []
    for cls_key in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        cls_name = target_map.get(int(float(cls_key)), cls_key) if cls_key.replace('.', '').isdigit() else cls_key
        report_data.append({
            'Class': cls_name,
            'Precision': f"{report[cls_key]['precision']:.3f}",
            'Recall': f"{report[cls_key]['recall']:.3f}",
            'F1-Score': f"{report[cls_key]['f1-score']:.3f}",
            'Support': int(report[cls_key]['support'])
        })
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    return results, feature_cols, best_model_name
