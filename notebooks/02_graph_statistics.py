"""
EDA Notebook 2: Graph Statistics

Graph structure analysis including:
- Node and edge statistics
- Degree distributions
- Connected components
- Network density
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
    return mo.md("# Graph Statistics - Fraud Detection Network")


@app.cell
def __(mo):
    return mo.md("""
    ## 1. Load Transaction Data and Build Graph

    Constructing the transaction network from raw data.
    """)


@app.cell
def __(pd, Path):
    import os
    cwd = Path(os.getcwd())
    project_root = cwd.parent if cwd.name == "notebooks" else cwd
    data_dir = project_root / "data" / "raw"

    transactions_df = pd.read_csv(data_dir / "transactions.csv")
    users_df = pd.read_csv(data_dir / "users.csv")
    fraud_labels_df = pd.read_csv(data_dir / "fraud_labels.csv")

    print(f"Loaded {len(users_df)} users")
    print(f"Loaded {len(transactions_df)} transactions")
    return transactions_df, users_df, fraud_labels_df, data_dir


@app.cell
def __(mo):
    return mo.md("## 2. Build Transaction Graph")


@app.cell
def __(nx, transactions_df):
    G = nx.DiGraph()

    for _, row in transactions_df.iterrows():
        G.add_edge(
            row['source_user'],
            row['target_user'],
            amount=row['amount'],
            is_fraud=row['is_fraud']
        )

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return (G,)


@app.cell
def __(mo):
    return mo.md("## 3. Basic Graph Statistics")


@app.cell
def __(G, pd):
    graph_stats = pd.DataFrame({
        'Metric': [
            'Nodes',
            'Edges',
            'Density',
            'Is Directed',
            'Avg Degree',
            'Avg In-Degree',
            'Avg Out-Degree'
        ],
        'Value': [
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.density(G),
            G.is_directed(),
            sum(dict(G.degree()).values()) / G.number_of_nodes(),
            sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
            sum(dict(G.out_degree()).values()) / G.number_of_nodes()
        ]
    })
    graph_stats
    return (graph_stats,)


@app.cell
def __(mo):
    return mo.md("## 4. Degree Distribution")


@app.cell
def __(G, plt, np):
    degrees = [deg for node, deg in G.degree()]

    fig_deg, axes_deg = plt.subplots(1, 3, figsize=(18, 5))

    # Degree distribution
    axes_deg[0].hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    axes_deg[0].set_title('Degree Distribution')
    axes_deg[0].set_xlabel('Degree')
    axes_deg[0].set_ylabel('Frequency')

    # In-degree distribution
    in_degrees = [deg for node, deg in G.in_degree()]
    axes_deg[1].hist(in_degrees, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes_deg[1].set_title('In-Degree Distribution')
    axes_deg[1].set_xlabel('In-Degree')
    axes_deg[1].set_ylabel('Frequency')

    # Out-degree distribution
    out_degrees = [deg for node, deg in G.out_degree()]
    axes_deg[2].hist(out_degrees, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes_deg[2].set_title('Out-Degree Distribution')
    axes_deg[2].set_xlabel('Out-Degree')
    axes_deg[2].set_ylabel('Frequency')

    plt.tight_layout()
    fig_deg
    return (degrees, in_degrees, out_degrees)


@app.cell
def __(mo):
    return mo.md("## 5. Connected Components Analysis")


@app.cell
def __(G, nx):
    # Convert to undirected for component analysis
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))

    print(f"Number of connected components: {len(components)}")
    print(f"Largest component size: {len(max(components, key=len))}")
    print(f"Smallest component size: {len(min(components, key=len))}")
    return G_undirected, components


@app.cell
def __(components, plt):
    component_sizes = [len(c) for c in components]

    fig_comp, ax_comp = plt.subplots(1, 1, figsize=(10, 6))
    ax_comp.hist(component_sizes, bins=50, edgecolor='black', alpha=0.7)
    ax_comp.set_title('Connected Component Size Distribution')
    ax_comp.set_xlabel('Component Size')
    ax_comp.set_ylabel('Frequency')
    ax_comp.set_yscale('log')

    fig_comp
    return (component_sizes,)


@app.cell
def __(mo):
    return mo.md("## 6. Top Nodes by Degree")


@app.cell
def __(G, pd):
    top_degree = pd.DataFrame([
        {'user_id': node, 'degree': deg}
        for node, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
    ])

    top_in_degree = pd.DataFrame([
        {'user_id': node, 'in_degree': deg}
        for node, deg in sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:20]
    ])

    top_out_degree = pd.DataFrame([
        {'user_id': node, 'out_degree': deg}
        for node, deg in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:20]
    ])

    print("Top 20 nodes by total degree:")
    top_degree
    return top_degree, top_in_degree, top_out_degree


@app.cell
def __(mo):
    return mo.md("""
    ## 7. Graph Statistics Summary

    Key findings:
    - Network connectivity patterns identified
    - Degree distributions reveal hub nodes
    - Connected components show community structure
    - Ready for fraud pattern analysis
    """)


if __name__ == "__main__":
    app.run()
