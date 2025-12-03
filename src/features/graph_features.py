"""
Extract graph-based features from Neo4j for fraud detection.

Uses Neo4j Graph Data Science (GDS) library to compute centrality measures,
community detection, and other graph features.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


class GraphFeatureExtractor:
    """Extract graph features from Neo4j for fraud detection modeling."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def extract_user_features(self) -> pd.DataFrame:
        """
        Extract user node features.

        Returns:
            DataFrame with user features
        """
        print("Extracting user node features...")

        query = """
        MATCH (u:User)
        RETURN
            u.user_id AS user_id,
            u.age AS age,
            u.account_age_days AS account_age_days,
            u.credit_score AS credit_score,
            u.account_type AS account_type,
            u.is_fraudster AS is_fraudster
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted {len(df)} user records")
        return df

    def extract_degree_features(self) -> pd.DataFrame:
        """
        Extract degree centrality features.

        Returns:
            DataFrame with degree features
        """
        print("Extracting degree features...")

        query = """
        MATCH (u:User)
        WITH u,
             count {(u)-[:TRANSACTS_WITH]->()} AS out_degree,
             count {(u)<-[:TRANSACTS_WITH]-()} AS in_degree
        RETURN
            u.user_id AS user_id,
            out_degree,
            in_degree,
            out_degree + in_degree AS total_degree
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted degree features for {len(df)} users")
        return df

    def extract_transaction_features(self) -> pd.DataFrame:
        """
        Extract transaction-based features.

        Returns:
            DataFrame with transaction features
        """
        print("Extracting transaction features...")

        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[t_out:TRANSACTS_WITH]->()
        WITH u, collect(t_out) AS outgoing
        OPTIONAL MATCH (u)<-[t_in:TRANSACTS_WITH]-()
        WITH u, outgoing, collect(t_in) AS incoming
        RETURN
            u.user_id AS user_id,
            size(outgoing) AS txn_count_out,
            size(incoming) AS txn_count_in,
            size(outgoing) + size(incoming) AS txn_count_total,
            CASE WHEN size(outgoing) > 0
                THEN reduce(s = 0.0, t IN outgoing | s + t.amount) / size(outgoing)
                ELSE 0.0
            END AS avg_txn_amount_out,
            CASE WHEN size(incoming) > 0
                THEN reduce(s = 0.0, t IN incoming | s + t.amount) / size(incoming)
                ELSE 0.0
            END AS avg_txn_amount_in,
            CASE WHEN size(outgoing) > 0
                THEN reduce(s = 0.0, t IN outgoing | s + t.amount)
                ELSE 0.0
            END AS total_amount_out,
            CASE WHEN size(incoming) > 0
                THEN reduce(s = 0.0, t IN incoming | s + t.amount)
                ELSE 0.0
            END AS total_amount_in
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted transaction features for {len(df)} users")
        return df

    def extract_fraud_transaction_features(self) -> pd.DataFrame:
        """
        Extract fraud-specific transaction features.

        Returns:
            DataFrame with fraud transaction features
        """
        print("Extracting fraud transaction features...")

        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[t:TRANSACTS_WITH]->()
        WHERE t.is_fraud
        WITH u, count(t) AS fraud_txn_count_out, sum(t.amount) AS fraud_amount_out
        OPTIONAL MATCH (u)<-[t2:TRANSACTS_WITH]-()
        WHERE t2.is_fraud
        WITH u, fraud_txn_count_out, fraud_amount_out,
             count(t2) AS fraud_txn_count_in, sum(t2.amount) AS fraud_amount_in
        RETURN
            u.user_id AS user_id,
            coalesce(fraud_txn_count_out, 0) AS fraud_txn_count_out,
            coalesce(fraud_txn_count_in, 0) AS fraud_txn_count_in,
            coalesce(fraud_amount_out, 0.0) AS fraud_amount_out,
            coalesce(fraud_amount_in, 0.0) AS fraud_amount_in
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted fraud transaction features for {len(df)} users")
        return df

    def extract_neighbor_features(self) -> pd.DataFrame:
        """
        Extract neighbor-based features (fraud connections).

        Returns:
            DataFrame with neighbor features
        """
        print("Extracting neighbor features...")

        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:TRANSACTS_WITH]-(neighbor:User)
        WITH u, collect(DISTINCT neighbor) AS neighbors
        WITH u,
             size(neighbors) AS neighbor_count,
             CASE WHEN size(neighbors) > 0
                 THEN toFloat(size([n IN neighbors WHERE n.is_fraudster])) / size(neighbors)
                 ELSE 0.0
             END AS fraud_neighbor_ratio
        RETURN
            u.user_id AS user_id,
            neighbor_count,
            fraud_neighbor_ratio
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted neighbor features for {len(df)} users")
        return df

    def extract_device_features(self) -> pd.DataFrame:
        """
        Extract device sharing features.

        Returns:
            DataFrame with device features
        """
        print("Extracting device features...")

        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:USES_DEVICE]->(d:Device)
        WITH u, collect(DISTINCT d) AS devices
        WITH u, devices,
             [d IN devices | count {(d)<-[:USES_DEVICE]-()} ] AS device_user_counts
        RETURN
            u.user_id AS user_id,
            size(devices) AS device_count,
            CASE WHEN size(device_user_counts) > 0
                THEN reduce(s = 0, c IN device_user_counts | s + c) / toFloat(size(device_user_counts))
                ELSE 0.0
            END AS avg_users_per_device,
            CASE WHEN size(device_user_counts) > 0
                THEN reduce(s = 0, c IN device_user_counts |
                    CASE WHEN c > s THEN c ELSE s END)
                ELSE 0
            END AS max_users_per_device
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted device features for {len(df)} users")
        return df

    def extract_ip_features(self) -> pd.DataFrame:
        """
        Extract IP address sharing features.

        Returns:
            DataFrame with IP features
        """
        print("Extracting IP features...")

        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:USES_IP]->(ip:IPAddress)
        WITH u, collect(DISTINCT ip) AS ips
        WITH u, ips,
             [ip IN ips | count {(ip)<-[:USES_IP]-()} ] AS ip_user_counts
        RETURN
            u.user_id AS user_id,
            size(ips) AS ip_count,
            CASE WHEN size(ip_user_counts) > 0
                THEN reduce(s = 0, c IN ip_user_counts | s + c) / toFloat(size(ip_user_counts))
                ELSE 0.0
            END AS avg_users_per_ip,
            CASE WHEN size(ip_user_counts) > 0
                THEN reduce(s = 0, c IN ip_user_counts |
                    CASE WHEN c > s THEN c ELSE s END)
                ELSE 0
            END AS max_users_per_ip
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]

        df = pd.DataFrame(records)
        print(f"  Extracted IP features for {len(df)} users")
        return df

    def extract_all_features(self, output_path: Path = None) -> pd.DataFrame:
        """
        Extract all features and merge into single DataFrame.

        Args:
            output_path: Optional path to save features CSV

        Returns:
            DataFrame with all features
        """
        print("=" * 70)
        print("EXTRACTING GRAPH FEATURES FOR FRAUD DETECTION")
        print("=" * 70)
        print()

        # Extract all feature sets
        user_df = self.extract_user_features()
        degree_df = self.extract_degree_features()
        txn_df = self.extract_transaction_features()
        fraud_txn_df = self.extract_fraud_transaction_features()
        neighbor_df = self.extract_neighbor_features()
        device_df = self.extract_device_features()
        ip_df = self.extract_ip_features()

        # Merge all features
        print("\nMerging feature sets...")
        features = user_df
        for df in [degree_df, txn_df, fraud_txn_df, neighbor_df, device_df, ip_df]:
            features = features.merge(df, on="user_id", how="left")

        # Fill missing values
        features = features.fillna(0)

        # Convert account_type to numeric
        features["account_type_encoded"] = (
            features["account_type"].map({"personal": 0, "business": 1}).fillna(0)
        )

        print(f"\nFinal feature set: {features.shape}")
        print(f"  Total features: {len(features.columns)}")
        print(f"  Total users: {len(features)}")
        print(f"  Fraud rate: {features['is_fraudster'].mean():.2%}")

        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(output_path, index=False)
            print(f"\nFeatures saved to: {output_path}")

        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION COMPLETE")
        print("=" * 70)

        return features


def main():
    """Main entry point."""
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "frauddetection123")

    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "processed" / "graph_features.csv"

    extractor = GraphFeatureExtractor(uri, user, password)
    try:
        features = extractor.extract_all_features(output_path)
        print(f"\nFeature columns: {list(features.columns)}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
