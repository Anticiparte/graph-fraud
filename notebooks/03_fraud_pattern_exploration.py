"""
EDA Notebook 3: Fraud Pattern Exploration

Visualization and analysis of fraud patterns:
- Ring/circular patterns
- Velocity anomalies
- Community isolation
- Fraud vs legitimate transaction comparison
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
    return mo.md("# Fraud Pattern Exploration")


@app.cell
def __(mo):
    return mo.md("## 1. Load Data")


@app.cell
def __(pd, Path):
    import os
    cwd = Path(os.getcwd())
    project_root = cwd.parent if cwd.name == "notebooks" else cwd
    data_dir = project_root / "data" / "raw"

    transactions_df = pd.read_csv(data_dir / "transactions.csv")
    users_df = pd.read_csv(data_dir / "users.csv")
    fraud_labels_df = pd.read_csv(data_dir / "fraud_labels.csv")

    print(f"Total transactions: {len(transactions_df)}")
    print(f"Fraudulent transactions: {transactions_df['is_fraud'].sum()}")
    print(f"Fraud rate: {transactions_df['is_fraud'].mean():.2%}")
    return transactions_df, users_df, fraud_labels_df, data_dir


@app.cell
def __(mo):
    return mo.md("## 2. Fraud Type Distribution")


@app.cell
def __(transactions_df, plt):
    fraud_txns = transactions_df[transactions_df['is_fraud'] == True]

    fraud_type_counts = fraud_txns['fraud_type'].value_counts()

    fig_fraud_types, ax_fraud_types = plt.subplots(1, 1, figsize=(10, 6))
    fraud_type_counts.plot(kind='bar', ax=ax_fraud_types, color=['red', 'orange', 'darkred'])
    ax_fraud_types.set_title('Distribution of Fraud Types')
    ax_fraud_types.set_xlabel('Fraud Type')
    ax_fraud_types.set_ylabel('Count')
    ax_fraud_types.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig_fraud_types
    return fraud_txns, fraud_type_counts


@app.cell
def __(mo):
    return mo.md("## 3. Transaction Amount Comparison: Fraud vs Legitimate")


@app.cell
def __(transactions_df, plt):
    fraud_amounts = transactions_df[transactions_df['is_fraud'] == True]['amount']
    legit_amounts = transactions_df[transactions_df['is_fraud'] == False]['amount']

    fig_amounts, axes_amounts = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    axes_amounts[0].boxplot([legit_amounts, fraud_amounts], labels=['Legitimate', 'Fraud'])
    axes_amounts[0].set_title('Transaction Amount Distribution')
    axes_amounts[0].set_ylabel('Amount')

    # Histogram comparison
    axes_amounts[1].hist(legit_amounts, bins=50, alpha=0.5, label='Legitimate', color='blue')
    axes_amounts[1].hist(fraud_amounts, bins=50, alpha=0.5, label='Fraud', color='red')
    axes_amounts[1].set_title('Transaction Amount Histogram')
    axes_amounts[1].set_xlabel('Amount')
    axes_amounts[1].set_ylabel('Frequency')
    axes_amounts[1].legend()
    axes_amounts[1].set_yscale('log')

    plt.tight_layout()
    fig_amounts
    return fraud_amounts, legit_amounts


@app.cell
def __(mo):
    return mo.md("## 4. Fraud Pattern: Ring Detection")


@app.cell
def __(nx, transactions_df, np):
    # Build fraud subgraph
    fraud_txns_ring = transactions_df[transactions_df['fraud_type'] == 'ring']

    G_fraud = nx.DiGraph()
    for _, _row in fraud_txns_ring.iterrows():
        G_fraud.add_edge(_row['source_user'], _row['target_user'])

    # Find cycles (rings)
    try:
        cycles = list(nx.simple_cycles(G_fraud))
        print(f"Number of cycles detected: {len(cycles)}")
        print(f"Average cycle length: {np.mean([len(c) for c in cycles]) if cycles else 0:.2f}")
    except:
        cycles = []
        print("No cycles detected or graph too large")

    return fraud_txns_ring, G_fraud, cycles


@app.cell
def __(mo):
    return mo.md("## 5. Fraud Pattern: Velocity Anomalies")


@app.cell
def __(transactions_df, pd):
    # Analyze transaction velocity for fraud vs legitimate
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

    velocity_stats = transactions_df.groupby(['source_user', 'is_fraud']).agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'std']
    }).reset_index()

    velocity_stats.columns = ['user_id', 'is_fraud', 'txn_count', 'total_amount', 'avg_amount', 'std_amount']

    print("Velocity Statistics:")
    print(velocity_stats.groupby('is_fraud')[['txn_count', 'avg_amount']].mean())
    return (velocity_stats,)


@app.cell
def __(velocity_stats, plt):
    fig_velocity, axes_velocity = plt.subplots(1, 2, figsize=(14, 5))

    # Transaction count distribution
    fraud_velocity = velocity_stats[velocity_stats['is_fraud'] == True]
    legit_velocity = velocity_stats[velocity_stats['is_fraud'] == False]

    axes_velocity[0].hist(legit_velocity['txn_count'], bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_velocity[0].hist(fraud_velocity['txn_count'], bins=30, alpha=0.5, label='Fraud', color='red')
    axes_velocity[0].set_title('Transactions per User')
    axes_velocity[0].set_xlabel('Number of Transactions')
    axes_velocity[0].set_ylabel('Frequency')
    axes_velocity[0].legend()
    axes_velocity[0].set_yscale('log')

    # Average amount distribution
    axes_velocity[1].hist(legit_velocity['avg_amount'].dropna(), bins=30, alpha=0.5, label='Legitimate', color='blue')
    axes_velocity[1].hist(fraud_velocity['avg_amount'].dropna(), bins=30, alpha=0.5, label='Fraud', color='red')
    axes_velocity[1].set_title('Average Transaction Amount per User')
    axes_velocity[1].set_xlabel('Average Amount')
    axes_velocity[1].set_ylabel('Frequency')
    axes_velocity[1].legend()

    plt.tight_layout()
    fig_velocity
    return fraud_velocity, legit_velocity


@app.cell
def __(mo):
    return mo.md("## 6. Fraud Pattern: Community Isolation")


@app.cell
def __(nx, transactions_df, np):
    # Build separate graphs for fraud and legitimate transactions
    fraud_community_txns = transactions_df[transactions_df['fraud_type'] == 'community']
    legit_community_txns = transactions_df[transactions_df['is_fraud'] == False].sample(n=min(10000, len(transactions_df)))

    G_fraud_comm = nx.DiGraph()
    for _, _row_fraud in fraud_community_txns.iterrows():
        G_fraud_comm.add_edge(_row_fraud['source_user'], _row_fraud['target_user'])

    G_legit_comm = nx.DiGraph()
    for _, _row_legit in legit_community_txns.iterrows():
        G_legit_comm.add_edge(_row_legit['source_user'], _row_legit['target_user'])

    # Calculate clustering coefficients
    fraud_clustering = nx.clustering(G_fraud_comm.to_undirected())
    legit_clustering = nx.clustering(G_legit_comm.to_undirected())

    print(f"Fraud network average clustering: {np.mean(list(fraud_clustering.values())):.4f}")
    print(f"Legitimate network average clustering: {np.mean(list(legit_clustering.values())):.4f}")

    return fraud_community_txns, legit_community_txns, G_fraud_comm, G_legit_comm, fraud_clustering, legit_clustering


@app.cell
def __(mo):
    return mo.md("""
    ## 7. Summary

    Key fraud patterns identified:
    - **Ring patterns**: Circular money flows detected
    - **Velocity anomalies**: Fraudulent users show higher transaction velocity
    - **Community isolation**: Fraud networks have higher clustering coefficients
    - Ready for feature engineering phase
    """)


if __name__ == "__main__":
    app.run()
