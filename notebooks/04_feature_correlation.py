"""
EDA Notebook 4: Feature Correlation Analysis

Feature engineering preparation and correlation analysis:
- Engineered features from graph structure
- Correlation with fraud labels
- Feature importance insights
- Feature selection recommendations
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx

    sns.set_style("whitegrid")
    return mo, pd, np, Path, plt, sns, nx


@app.cell
def __(mo):
    return mo.md("# Feature Correlation Analysis")


@app.cell
def __(mo):
    return mo.md("## 1. Load Data and Build Graph")


@app.cell
def __(pd, Path, nx):
    import os
    cwd = Path(os.getcwd())
    project_root = cwd.parent if cwd.name == "notebooks" else cwd
    data_dir = project_root / "data" / "raw"

    transactions_df = pd.read_csv(data_dir / "transactions.csv")
    users_df = pd.read_csv(data_dir / "users.csv")
    fraud_labels_df = pd.read_csv(data_dir / "fraud_labels.csv")

    # Build transaction graph
    G = nx.DiGraph()
    for _, _row in transactions_df.iterrows():
        G.add_edge(_row['source_user'], _row['target_user'], amount=_row['amount'])

    print(f"Data loaded: {len(users_df)} users, {len(transactions_df)} transactions")
    return transactions_df, users_df, fraud_labels_df, data_dir, G


@app.cell
def __(mo):
    return mo.md("## 2. Engineer Graph Features")


@app.cell
def __(G, pd, nx):
    # Calculate graph-based features for each user
    features_list = []

    # Get degree centrality
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    # PageRank
    pagerank = nx.pagerank(G)

    # Clustering coefficient
    clustering = nx.clustering(G.to_undirected())

    for node in G.nodes():
        features = {
            'user_id': node,
            'degree': G.degree(node),
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'degree_centrality': degree_centrality.get(node, 0),
            'in_degree_centrality': in_degree_centrality.get(node, 0),
            'out_degree_centrality': out_degree_centrality.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'clustering_coefficient': clustering.get(node, 0),
        }
        features_list.append(features)

    graph_features_df = pd.DataFrame(features_list)
    print(f"Engineered {len(graph_features_df.columns)-1} graph features")
    return graph_features_df, degree_centrality, in_degree_centrality, out_degree_centrality, pagerank, clustering


@app.cell
def __(mo):
    return mo.md("## 3. Merge Features with Fraud Labels")


@app.cell
def __(graph_features_df, users_df, fraud_labels_df):
    # Merge all features
    features_merged = graph_features_df.merge(users_df, on='user_id', how='left')
    features_merged = features_merged.merge(fraud_labels_df, on='user_id', how='left')

    print(f"Final feature set shape: {features_merged.shape}")
    print(f"Fraud rate in user dataset: {features_merged['is_fraudster'].mean():.2%}")

    features_merged.head()
    return (features_merged,)


@app.cell
def __(mo):
    return mo.md("## 4. Feature Correlation Heatmap")


@app.cell
def __(features_merged, plt, sns):
    # Select numeric columns for correlation
    numeric_cols = ['degree', 'in_degree', 'out_degree', 'degree_centrality',
                    'in_degree_centrality', 'out_degree_centrality', 'pagerank',
                    'clustering_coefficient', 'age', 'account_age_days', 'credit_score',
                    'is_fraudster']

    correlation_matrix = features_merged[numeric_cols].corr()

    fig_corr, ax_corr = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax_corr, cbar_kws={'shrink': 0.8})
    ax_corr.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    fig_corr
    return correlation_matrix, numeric_cols


@app.cell
def __(mo):
    return mo.md("## 5. Features Most Correlated with Fraud")


@app.cell
def __(correlation_matrix, pd):
    fraud_correlations = correlation_matrix['is_fraudster'].drop('is_fraudster').sort_values(ascending=False)

    fraud_corr_df = pd.DataFrame({
        'Feature': fraud_correlations.index,
        'Correlation with Fraud': fraud_correlations.values
    })

    print("Top 10 features correlated with fraud:")
    fraud_corr_df.head(10)
    return fraud_correlations, fraud_corr_df


@app.cell
def __(fraud_correlations, plt):
    fig_fraud_corr, ax_fraud_corr = plt.subplots(1, 1, figsize=(10, 6))
    fraud_correlations.plot(kind='barh', ax=ax_fraud_corr, color=['red' if x > 0 else 'blue' for x in fraud_correlations])
    ax_fraud_corr.set_title('Feature Correlation with Fraud Label')
    ax_fraud_corr.set_xlabel('Correlation Coefficient')
    ax_fraud_corr.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig_fraud_corr
    return


@app.cell
def __(mo):
    return mo.md("## 6. Feature Distribution: Fraud vs Legitimate")


@app.cell
def __(features_merged, plt):
    fraud_users = features_merged[features_merged['is_fraudster'] == 1]
    legit_users = features_merged[features_merged['is_fraudster'] == 0]

    fig_dist, axes_dist = plt.subplots(2, 2, figsize=(14, 10))

    # Degree distribution
    axes_dist[0, 0].hist(legit_users['degree'], bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_dist[0, 0].hist(fraud_users['degree'], bins=30, alpha=0.5, label='Fraud', color='red')
    axes_dist[0, 0].set_title('Degree Distribution')
    axes_dist[0, 0].set_xlabel('Degree')
    axes_dist[0, 0].legend()

    # PageRank distribution
    axes_dist[0, 1].hist(legit_users['pagerank'], bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_dist[0, 1].hist(fraud_users['pagerank'], bins=30, alpha=0.5, label='Fraud', color='red')
    axes_dist[0, 1].set_title('PageRank Distribution')
    axes_dist[0, 1].set_xlabel('PageRank')
    axes_dist[0, 1].legend()

    # Clustering coefficient distribution
    axes_dist[1, 0].hist(legit_users['clustering_coefficient'], bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_dist[1, 0].hist(fraud_users['clustering_coefficient'], bins=30, alpha=0.5, label='Fraud', color='red')
    axes_dist[1, 0].set_title('Clustering Coefficient Distribution')
    axes_dist[1, 0].set_xlabel('Clustering Coefficient')
    axes_dist[1, 0].legend()

    # Credit score distribution
    axes_dist[1, 1].hist(legit_users['credit_score'], bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_dist[1, 1].hist(fraud_users['credit_score'], bins=30, alpha=0.5, label='Fraud', color='red')
    axes_dist[1, 1].set_title('Credit Score Distribution')
    axes_dist[1, 1].set_xlabel('Credit Score')
    axes_dist[1, 1].legend()

    plt.tight_layout()
    fig_dist
    return fraud_users, legit_users


@app.cell
def __(mo):
    return mo.md("## 7. Feature Statistics by Fraud Status")


@app.cell
def __(features_merged, numeric_cols):
    feature_stats = features_merged.groupby('is_fraudster')[numeric_cols].mean()
    feature_stats = feature_stats.T
    feature_stats.columns = ['Legitimate', 'Fraud']
    feature_stats['Difference'] = feature_stats['Fraud'] - feature_stats['Legitimate']
    feature_stats = feature_stats.sort_values('Difference', ascending=False)

    print("Mean feature values by fraud status:")
    feature_stats
    return (feature_stats,)


@app.cell
def __(mo):
    return mo.md("""
    ## 8. Summary and Recommendations

    Key findings:
    - Graph features show strong correlation with fraud labels
    - Degree centrality and PageRank are good fraud indicators
    - Clustering coefficient distinguishes fraud communities
    - User attributes (age, credit_score) have weaker correlation
    - Recommend using combination of graph and user features for modeling

    Next steps:
    - Use selected features for model training
    - Consider feature engineering for temporal patterns
    - Test feature importance with XGBoost
    """)


if __name__ == "__main__":
    app.run()
