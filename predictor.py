import streamlit as st
import pandas as pd
import numpy as np
from data_loader import (build_feature_matrix, get_classification_features,
                         get_regression_features, get_clustering_features)

TARGET_MAP = {0: 'Definitely No', 1: 'Probably No', 2: 'Neutral', 3: 'Probably Yes', 4: 'Definitely Yes'}

PERSONA_NAMES = {
    0: "Health-first Metro Pro",
    1: "Budget-conscious Explorer",
    2: "Comfort-seeking Traditionalist",
    3: "Digital-native Snacker",
    4: "Premium Wellness Seeker",
    5: "Casual Occasional Buyer"
}

DISCOUNT_MAP = {
    0: "5% loyalty only",
    1: "20% first order + referral",
    2: "15% combo deal",
    3: "10% subscription discount",
    4: "Free trial + 5%",
    5: "25% welcome offer"
}

BUNDLE_MAP = {
    0: "Power Bowl + Focus Fuel + Protein Bar",
    1: "Starter Snack Box + Budget Beverage",
    2: "Comfort Bowl + Chai + Sweet Bite",
    3: "Millet Chips + Cold Brew + Energy Bar",
    4: "Premium Breakfast Kit + Wellness Beverage",
    5: "Mixed Sampler Box"
}

CHANNEL_MAP = {
    0: "Instagram + Brand App",
    1: "Quick Commerce + Google Ads",
    2: "Offline Retail + WhatsApp",
    3: "Instagram Reels + Influencer",
    4: "LinkedIn + Premium Partnerships",
    5: "Delivery App Ads"
}


def render_predictor(df_feat, clf_results, clf_features, clf_best_name,
                     reg_results, reg_features, reg_best_name,
                     km_model, km_scaler, km_features, km_k):
    st.header("🚀 Predict New Customers — Upload & Score")
    st.caption("Upload a CSV of new survey respondents and get instant predictions + recommendations")

    st.info("""
    **How to use:**
    1. Upload a CSV file with the **same column structure** as the original survey data (minus the target column)
    2. The system will auto-predict: **Interest Level** (Classification), **Monthly Spend** (Regression), and **Customer Segment** (Clustering)
    3. Download the scored results with personalised marketing recommendations
    """)

    # Download template
    st.subheader("Step 1: Download Template")
    template_cols = [c for c in df_feat.columns if c not in ['respondent_id', 'likelihood_to_try_moodmeal',
                                                              'target_numeric', 'income_numeric', 'spend_numeric',
                                                              'wtp_numeric', 'social_media_numeric', 'meals_out_numeric',
                                                              'order_freq_numeric'] and not c.endswith('_enc')]
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button("📥 Download CSV Template", csv_template, "moodmeal_new_customer_template.csv", "text/csv")

    st.divider()

    # Upload
    st.subheader("Step 2: Upload New Customer Data")
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            new_df.columns = new_df.columns.str.strip()
            st.success(f"✅ Uploaded **{len(new_df)} new customers** with {len(new_df.columns)} columns")

            with st.expander("Preview uploaded data"):
                st.dataframe(new_df.head(10), use_container_width=True)

            st.divider()
            st.subheader("Step 3: Generating Predictions...")

            # Build features for new data
            new_feat, _ = build_feature_matrix(new_df)

            # === CLASSIFICATION ===
            st.markdown("#### 🎯 Interest Prediction (Classification)")
            best_clf = clf_results[clf_best_name]['model']
            clf_cols_available = [c for c in clf_features if c in new_feat.columns]
            X_clf = new_feat[clf_cols_available].fillna(0)

            # Handle missing columns
            for c in clf_features:
                if c not in X_clf.columns:
                    X_clf[c] = 0
            X_clf = X_clf[clf_features]

            pred_interest = best_clf.predict(X_clf)
            pred_interest_labels = [TARGET_MAP.get(p, 'Unknown') for p in pred_interest]
            new_df['predicted_interest'] = pred_interest_labels

            proba = best_clf.predict_proba(X_clf)
            new_df['confidence_score'] = [round(max(p) * 100, 1) for p in proba]

            interest_dist = pd.Series(pred_interest_labels).value_counts()
            st.bar_chart(interest_dist)

            # === REGRESSION ===
            st.markdown("#### 💰 Spend Prediction (Regression)")
            best_reg = reg_results[reg_best_name]['model']
            reg_cols_available = [c for c in reg_features if c in new_feat.columns]
            X_reg = new_feat[reg_cols_available].fillna(0)
            for c in reg_features:
                if c not in X_reg.columns:
                    X_reg[c] = 0
            X_reg = X_reg[reg_features]

            pred_spend = best_reg.predict(X_reg)
            new_df['predicted_monthly_spend'] = [f"₹{max(0, round(s)):,}" for s in pred_spend]

            # === CLUSTERING ===
            st.markdown("#### 👥 Segment Assignment (Clustering)")
            clust_cols_available = [c for c in km_features if c in new_feat.columns]
            X_clust = new_feat[clust_cols_available].fillna(0)
            for c in km_features:
                if c not in X_clust.columns:
                    X_clust[c] = 0
            X_clust = X_clust[km_features]

            X_clust_scaled = km_scaler.transform(X_clust)
            pred_cluster = km_model.predict(X_clust_scaled)
            new_df['customer_segment'] = [PERSONA_NAMES.get(c, f"Segment {c}") for c in pred_cluster]

            # === PRESCRIPTIVE RECOMMENDATIONS ===
            st.markdown("#### 💡 Prescriptive Recommendations")
            new_df['recommended_bundle'] = [BUNDLE_MAP.get(c, "Mixed Sampler") for c in pred_cluster]
            new_df['discount_tier'] = [DISCOUNT_MAP.get(c, "10% standard") for c in pred_cluster]
            new_df['best_channel'] = [CHANNEL_MAP.get(c, "Multi-channel") for c in pred_cluster]

            # Priority score
            priority = []
            for idx in range(len(new_df)):
                p = 0
                if pred_interest_labels[idx] in ['Definitely Yes', 'Probably Yes']:
                    p += 40
                elif pred_interest_labels[idx] == 'Neutral':
                    p += 20
                p += min(30, pred_spend[idx] / 500)
                p += new_df['confidence_score'].iloc[idx] * 0.3
                priority.append(round(min(100, p), 1))
            new_df['targeting_priority_score'] = priority

            st.divider()

            # Results preview
            st.subheader("Step 4: Scored Results")
            display_cols = ['predicted_interest', 'confidence_score', 'predicted_monthly_spend',
                            'customer_segment', 'recommended_bundle', 'discount_tier',
                            'best_channel', 'targeting_priority_score']
            available_display = [c for c in display_cols if c in new_df.columns]
            st.dataframe(new_df[available_display].head(20), use_container_width=True, hide_index=True)

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            total_new = len(new_df)
            hot_leads = sum(1 for x in pred_interest_labels if x in ['Definitely Yes', 'Probably Yes'])
            col1.metric("Total Uploaded", total_new)
            col2.metric("Hot Leads", f"{hot_leads} ({hot_leads/total_new*100:.0f}%)")
            col3.metric("Avg Pred. Spend", f"₹{np.mean(pred_spend):,.0f}/mo")
            col4.metric("Top Segment", new_df['customer_segment'].mode()[0][:20] if len(new_df) > 0 else '-')

            # Download scored results
            st.divider()
            st.subheader("Step 5: Download Scored Results")
            csv_out = new_df.to_csv(index=False)
            st.download_button("📥 Download Scored Customer Data", csv_out,
                               "moodmeal_scored_customers.csv", "text/csv",
                               type="primary")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the same column names as the survey dataset.")
