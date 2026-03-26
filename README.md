---
title: MoodMeal Consumer Analytics Dashboard
emoji: 🍱
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: mit
short_description: Data-driven analytics for India's first mood-driven food brand
tags:
  - streamlit
  - analytics
  - machine-learning
  - classification
  - clustering
  - association-rules
  - regression
  - food-tech
  - india
  - marketing
---

# 🍱 MoodMeal — Consumer Analytics Dashboard

**Data-driven decision making for India's first mood-driven personalised food brand.**

## What is MoodMeal?

MoodMeal is a food startup concept offering personalised meal options, curated food combos, and experience-driven food products to customers across India. This dashboard analyses synthetic survey data from 2,000 Indian respondents to inform product strategy, customer targeting, and pricing decisions.

## Dashboard Tabs

| Tab | Analysis Type | What It Does |
|-----|--------------|--------------|
| 📊 Overview | Descriptive | KPIs, demographic sunburst, interest funnel, category breakdown |
| 🔍 Diagnostic | Diagnostic | Cross-tabs, chi-square tests, correlation heatmap, feature importance |
| 👥 Segmentation | Clustering | K-Means with elbow/silhouette, radar profiles, persona naming |
| 🔗 Basket Analysis | Association | Apriori rules with support, confidence, lift, network graph |
| 🎯 Classification | Predictive | Decision Tree, Random Forest, Gradient Boosted Tree — accuracy, precision, recall, F1, ROC-AUC |
| 💰 Regression | Predictive | Linear, Ridge, RF Regressor — R², MAE, RMSE |
| 🚀 Predict New | Prescriptive | Upload new customer CSV → get interest prediction + spend estimate + segment + recommendations |

## ML Algorithms Used

- **Classification**: Decision Tree, Random Forest, Gradient Boosted Trees
- **Clustering**: K-Means with elbow method and silhouette scoring
- **Association Rule Mining**: Apriori algorithm with support, confidence, and lift
- **Regression**: Linear Regression, Ridge Regression, Random Forest Regressor

## Dataset

- 2,000 synthetic survey respondents
- 73 columns covering demographics, digital behaviour, food preferences, ordering habits, psychographics
- Conditional probability-based generation mirroring real Indian consumer patterns
- ~10% controlled noise and outliers for model robustness testing

## Tech Stack

- **Frontend**: Streamlit
- **Visualisation**: Plotly
- **ML**: scikit-learn, mlxtend
- **Data**: pandas, numpy, scipy

## Built By

**Deepanshu** — MBA, Global Marketing Management
