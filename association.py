import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

def prepare_transactions(df):
    txn = pd.DataFrame()

    # Category interests
    cat_cols = [c for c in df.columns if c.startswith('interest_')]
    for c in cat_cols:
        txn[c.replace('interest_', 'Cat:')] = df[c].astype(bool)

    # Add-ons
    addon_cols = [c for c in df.columns if c.startswith('addon_') and c != 'addon_none']
    for c in addon_cols:
        txn[c.replace('addon_', 'Addon:')] = df[c].astype(bool)

    # Occasions
    occ_cols = [c for c in df.columns if c.startswith('occasion_')]
    for c in occ_cols:
        txn[c.replace('occasion_', 'Time:')] = df[c].astype(bool)

    # Flavours
    flav_cols = [c for c in df.columns if c.startswith('flavour_')]
    for c in flav_cols:
        txn[c.replace('flavour_', 'Flav:')] = df[c].astype(bool)

    # Health goals
    health_cols = [c for c in df.columns if c.startswith('health_') and c != 'health_no_specific_goal']
    for c in health_cols:
        txn[c.replace('health_', 'Goal:')] = df[c].astype(bool)

    # Clean column names
    txn.columns = [c.replace('_', ' ').title() for c in txn.columns]

    return txn


def render_association(df):
    st.header("🔗 Market Basket Analysis — Association Rules")
    st.caption("Discovering what goes together: categories, add-ons, occasions, flavours & health goals")

    # Prepare transaction data
    txn = prepare_transactions(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Min Support", 0.05, 0.50, 0.15, 0.05,
                                help="Minimum frequency of itemset in dataset")
    with col2:
        min_confidence = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05,
                                   help="P(B|A) — how often rule is correct")
    with col3:
        min_lift = st.slider("Min Lift", 1.0, 3.0, 1.1, 0.1,
                             help="Lift > 1 means positive association")

    # Run Apriori
    with st.spinner("Mining frequent itemsets..."):
        freq_items = apriori(txn, min_support=min_support, use_colnames=True)

    if len(freq_items) == 0:
        st.warning("No frequent itemsets found. Try lowering the minimum support.")
        return

    st.success(f"Found **{len(freq_items)}** frequent itemsets")

    # Generate rules
    try:
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence, num_itemsets=len(freq_items))
        rules = rules[rules['lift'] >= min_lift]
    except Exception:
        st.warning("Could not generate rules with current thresholds. Try adjusting parameters.")
        return

    if len(rules) == 0:
        st.warning("No rules found with current thresholds. Try lowering confidence or lift.")
        return

    rules = rules.sort_values('lift', ascending=False)
    st.info(f"**{len(rules)}** association rules discovered (confidence ≥ {min_confidence}, lift ≥ {min_lift})")

    st.divider()

    # Top Rules Table
    st.subheader("Top Association Rules")
    display_rules = rules.head(20).copy()
    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    display_rules = display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
    display_rules.columns = ['If Customer Likes...', 'They Also Like...', 'Support', 'Confidence', 'Lift']
    st.dataframe(display_rules, use_container_width=True, hide_index=True)

    st.divider()

    # Scatter: Confidence vs Lift
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confidence vs Lift")
        plot_rules = rules.head(100).copy()
        plot_rules['antecedents_str'] = plot_rules['antecedents'].apply(lambda x: ', '.join(list(x))[:40])
        plot_rules['consequents_str'] = plot_rules['consequents'].apply(lambda x: ', '.join(list(x))[:40])
        fig = px.scatter(plot_rules, x='confidence', y='lift', size='support',
                         color='lift', color_continuous_scale='Tealgrn',
                         hover_data=['antecedents_str', 'consequents_str'],
                         labels={'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'})
        fig.update_layout(margin=dict(t=10, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Support vs Confidence")
        fig = px.scatter(plot_rules, x='support', y='confidence', size='lift',
                         color='confidence', color_continuous_scale='Purples',
                         hover_data=['antecedents_str', 'consequents_str'],
                         labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
        fig.update_layout(margin=dict(t=10, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Network graph
    st.subheader("Item Co-occurrence Network")
    st.caption("Items connected by strong association rules (top 30 by lift)")

    top_net = rules.head(30)
    nodes = set()
    edges = []
    for _, row in top_net.iterrows():
        for a in row['antecedents']:
            nodes.add(a)
            for c in row['consequents']:
                nodes.add(c)
                edges.append((a, c, row['lift'], row['confidence']))

    node_list = list(nodes)
    np.random.seed(42)
    node_x = np.random.randn(len(node_list)) * 2
    node_y = np.random.randn(len(node_list)) * 2
    node_pos = {n: (node_x[i], node_y[i]) for i, n in enumerate(node_list)}

    edge_traces = []
    for a, c, lift, conf in edges:
        x0, y0 = node_pos[a]
        x1, y1 = node_pos[c]
        edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                                      line=dict(width=max(0.5, lift * 0.8), color='rgba(127,119,221,0.4)'),
                                      hoverinfo='none', showlegend=False))

    node_trace = go.Scatter(
        x=[node_pos[n][0] for n in node_list], y=[node_pos[n][1] for n in node_list],
        mode='markers+text', text=[n[:20] for n in node_list], textposition='top center',
        textfont=dict(size=10), marker=dict(size=14, color='#7F77DD', line=dict(width=1, color='white')),
        hovertext=node_list, showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Business Recommendations
    st.subheader("💡 Bundle Recommendations (Auto-generated)")
    top5 = rules.head(5)
    for idx, (_, row) in enumerate(top5.iterrows()):
        ante = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        st.markdown(
            f"**Bundle {idx+1}:** Customers who pick *{ante}* → **{row['confidence']:.0%}** "
            f"chance of also wanting *{cons}* (Lift: {row['lift']:.2f})"
        )
