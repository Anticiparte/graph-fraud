"""
Generate synthetic fraud transaction data with realistic patterns.

This module creates a simulated transaction network with three types of fraud:
1. Ring/Circular patterns: Money cycling through accounts
2. Velocity anomalies: Unusual transaction frequencies
3. Community isolation: Fraudster clusters
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


class FraudDataGenerator:
    """Generate synthetic fraud detection dataset."""

    def __init__(
        self,
        n_users: int = 10000,
        n_transactions: int = 100000,
        fraud_rate: float = 0.05,
        random_seed: int = 42,
    ):
        """
        Initialize fraud data generator.

        Args:
            n_users: Number of users to generate
            n_transactions: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions
            random_seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.graph = nx.DiGraph()
        self.users_df = None
        self.transactions_df = None
        self.fraud_labels = None

    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with realistic attributes."""
        print(f"Generating {self.n_users} users...")

        users = []
        for user_id in range(self.n_users):
            user = {
                "user_id": f"U{user_id:06d}",
                "age": np.random.randint(18, 80),
                "account_age_days": np.random.randint(1, 3650),
                "credit_score": np.random.randint(300, 850),
                "account_type": np.random.choice(["personal", "business"], p=[0.85, 0.15]),
                "registration_timestamp": datetime.now()
                - timedelta(days=np.random.randint(1, 3650)),
            }
            users.append(user)

        self.users_df = pd.DataFrame(users)
        return self.users_df

    def generate_fraud_rings(self, n_rings: int = 75) -> list:
        """
        Generate fraud ring patterns (circular money flows).

        Args:
            n_rings: Number of fraud rings to create

        Returns:
            List of fraud ring user IDs
        """
        print(f"Generating {n_rings} fraud rings...")
        fraudulent_users = []

        for ring_idx in range(n_rings):
            ring_size = np.random.randint(3, 9)
            ring_users = np.random.choice(
                self.users_df["user_id"].values, size=ring_size, replace=False
            )

            # Create circular transaction pattern
            for i in range(ring_size):
                source = ring_users[i]
                target = ring_users[(i + 1) % ring_size]

                # Add multiple transactions within the ring
                n_ring_txns = np.random.randint(5, 15)
                for _ in range(n_ring_txns):
                    amount = np.random.uniform(500, 5000)
                    timestamp = datetime.now() - timedelta(
                        days=np.random.randint(1, 180)
                    )

                    self.graph.add_edge(
                        source,
                        target,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="transfer",
                        is_fraud=True,
                        fraud_type="ring",
                    )

            fraudulent_users.extend(ring_users)

        return list(set(fraudulent_users))

    def generate_velocity_anomalies(self, n_compromised: int = 250) -> list:
        """
        Generate velocity anomaly patterns (unusual transaction bursts).

        Args:
            n_compromised: Number of compromised accounts

        Returns:
            List of compromised user IDs
        """
        print(f"Generating {n_compromised} velocity anomalies...")
        compromised_users = np.random.choice(
            self.users_df["user_id"].values, size=n_compromised, replace=False
        )

        for user_id in compromised_users:
            # Sudden burst of transactions
            burst_size = np.random.randint(15, 50)
            burst_start = datetime.now() - timedelta(days=np.random.randint(1, 90))

            for txn_idx in range(burst_size):
                target = np.random.choice(self.users_df["user_id"].values)
                if target == user_id:
                    continue

                amount = np.random.uniform(100, 3000)
                timestamp = burst_start + timedelta(minutes=np.random.randint(1, 120))

                self.graph.add_edge(
                    user_id,
                    target,
                    amount=amount,
                    timestamp=timestamp,
                    transaction_type="transfer",
                    is_fraud=True,
                    fraud_type="velocity",
                )

        return list(compromised_users)

    def generate_fraud_communities(self, n_communities: int = 8) -> list:
        """
        Generate isolated fraudster communities.

        Args:
            n_communities: Number of fraud communities

        Returns:
            List of fraudster user IDs
        """
        print(f"Generating {n_communities} fraud communities...")
        fraudulent_users = []

        for comm_idx in range(n_communities):
            comm_size = np.random.randint(5, 15)
            community_users = np.random.choice(
                self.users_df["user_id"].values, size=comm_size, replace=False
            )

            # High internal connectivity
            for source in community_users:
                n_internal_txns = np.random.randint(3, 8)
                targets = np.random.choice(
                    community_users, size=n_internal_txns, replace=True
                )

                for target in targets:
                    if source == target:
                        continue

                    amount = np.random.uniform(200, 2000)
                    timestamp = datetime.now() - timedelta(
                        days=np.random.randint(1, 180)
                    )

                    self.graph.add_edge(
                        source,
                        target,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="transfer",
                        is_fraud=True,
                        fraud_type="community",
                    )

            fraudulent_users.extend(community_users)

        return list(set(fraudulent_users))

    def generate_legitimate_transactions(self):
        """Generate realistic legitimate transaction patterns."""
        print("Generating legitimate transactions...")

        # Calculate remaining transactions needed
        current_edges = len(self.graph.edges())
        remaining_txns = self.n_transactions - current_edges

        for _ in range(remaining_txns):
            source = np.random.choice(self.users_df["user_id"].values)
            target = np.random.choice(self.users_df["user_id"].values)

            if source == target:
                continue

            # Legitimate transactions have different characteristics
            amount = np.random.lognormal(mean=4.5, sigma=1.2)
            amount = max(10, min(amount, 10000))

            timestamp = datetime.now() - timedelta(days=np.random.randint(1, 365))

            self.graph.add_edge(
                source,
                target,
                amount=amount,
                timestamp=timestamp,
                transaction_type=np.random.choice(
                    ["transfer", "payment", "purchase"], p=[0.3, 0.4, 0.3]
                ),
                is_fraud=False,
                fraud_type=None,
            )

    def create_shared_entities(self):
        """Add shared device/IP patterns for fraud detection."""
        print("Adding shared device and IP patterns...")

        devices = [f"DEV{i:05d}" for i in range(self.n_users // 2)]
        ips = [f"192.168.{i//256}.{i%256}" for i in range(self.n_users // 3)]

        device_assignments = []
        ip_assignments = []

        for user_id in self.users_df["user_id"].values:
            device_assignments.append(
                {"user_id": user_id, "device_id": np.random.choice(devices)}
            )

            ip_assignments.append(
                {"user_id": user_id, "ip_address": np.random.choice(ips)}
            )

        self.device_df = pd.DataFrame(device_assignments)
        self.ip_df = pd.DataFrame(ip_assignments)

    def build_transaction_dataframe(self) -> pd.DataFrame:
        """Convert graph edges to transaction DataFrame."""
        print("Building transaction DataFrame...")

        transactions = []
        for idx, (source, target, data) in enumerate(self.graph.edges(data=True)):
            txn = {
                "transaction_id": f"TXN{idx:08d}",
                "source_user": source,
                "target_user": target,
                "amount": data["amount"],
                "timestamp": data["timestamp"],
                "transaction_type": data["transaction_type"],
                "is_fraud": data.get("is_fraud", False),
                "fraud_type": data.get("fraud_type", None),
            }
            transactions.append(txn)

        self.transactions_df = pd.DataFrame(transactions)

        # Add device and IP information
        self.transactions_df = self.transactions_df.merge(
            self.device_df, left_on="source_user", right_on="user_id", how="left"
        )
        self.transactions_df = self.transactions_df.merge(
            self.ip_df, left_on="source_user", right_on="user_id", how="left"
        )
        self.transactions_df = self.transactions_df.drop(columns=["user_id_x", "user_id_y"])

        return self.transactions_df

    def generate_fraud_labels(self) -> pd.DataFrame:
        """Create fraud label dataset at user level."""
        print("Generating fraud labels...")

        fraud_users = set()
        for _, _, data in self.graph.edges(data=True):
            if data.get("is_fraud", False):
                fraud_users.add(_)
                fraud_users.add(__)

        labels = []
        for user_id in self.users_df["user_id"].values:
            labels.append(
                {
                    "user_id": user_id,
                    "is_fraudster": 1 if user_id in fraud_users else 0,
                }
            )

        self.fraud_labels = pd.DataFrame(labels)
        return self.fraud_labels

    def save_data(self, output_dir: Path):
        """Save generated data to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving data to {output_dir}...")

        self.users_df.to_csv(output_dir / "users.csv", index=False)
        self.transactions_df.to_csv(output_dir / "transactions.csv", index=False)
        self.fraud_labels.to_csv(output_dir / "fraud_labels.csv", index=False)
        self.device_df.to_csv(output_dir / "devices.csv", index=False)
        self.ip_df.to_csv(output_dir / "ip_addresses.csv", index=False)

        print("Data generation complete!")
        print(f"  Users: {len(self.users_df)}")
        print(f"  Transactions: {len(self.transactions_df)}")
        print(f"  Fraud rate: {self.transactions_df['is_fraud'].mean():.2%}")
        print(f"  Fraudulent users: {self.fraud_labels['is_fraudster'].sum()}")

    def generate_all(self, output_dir: Path):
        """Run complete data generation pipeline."""
        self.generate_users()
        fraud_ring_users = self.generate_fraud_rings()
        velocity_users = self.generate_velocity_anomalies()
        community_users = self.generate_fraud_communities()
        self.generate_legitimate_transactions()
        self.create_shared_entities()
        self.build_transaction_dataframe()
        self.generate_fraud_labels()
        self.save_data(output_dir)


def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud detection dataset"
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=10000,
        help="Number of users to generate",
    )
    parser.add_argument(
        "--n-transactions",
        type=int,
        default=100000,
        help="Number of transactions to generate",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.05,
        help="Target fraud rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    generator = FraudDataGenerator(
        n_users=args.n_users,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        random_seed=args.seed,
    )

    generator.generate_all(Path(args.output_dir))


if __name__ == "__main__":
    main()
