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
        # Noise parameters for realistic data (tuned for F1 0.82-0.86)
        fraud_legitimate_mix: float = 0.30,  # Reduced from 0.60
        n_power_users: int = 30,  # Reduced from 80
        n_business_networks: int = 5,  # Reduced from 15
        n_money_pooling_groups: int = 5,  # Reduced from 10
        timestamp_jitter_minutes: int = 30,  # Reduced from 60
        amount_noise_pct: float = 0.15,  # Reduced from 0.30
        incomplete_pattern_rate: float = 0.10,  # Reduced from 0.30
    ):
        """
        Initialize fraud data generator.

        Args:
            n_users: Number of users to generate
            n_transactions: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions
            random_seed: Random seed for reproducibility
            fraud_legitimate_mix: Proportion of fraudster transactions that are legitimate
            n_power_users: Number of legitimate high-velocity users
            n_business_networks: Number of legitimate business payment networks
            n_money_pooling_groups: Number of legitimate money pooling groups
            timestamp_jitter_minutes: Maximum timestamp noise in minutes
            amount_noise_pct: Percentage noise to add to transaction amounts
            incomplete_pattern_rate: Proportion of fraud patterns to make incomplete
        """
        self.n_users = n_users
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed
        
        # Noise parameters
        self.fraud_legitimate_mix = fraud_legitimate_mix
        self.n_power_users = n_power_users
        self.n_business_networks = n_business_networks
        self.n_money_pooling_groups = n_money_pooling_groups
        self.timestamp_jitter_minutes = timestamp_jitter_minutes
        self.amount_noise_pct = amount_noise_pct
        self.incomplete_pattern_rate = incomplete_pattern_rate

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.graph = nx.DiGraph()
        self.users_df = None
        self.transactions_df = None
        self.fraud_labels = None
        self.fraudulent_users = set()  # Track all fraudulent users

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

    def generate_fraud_rings(self, n_rings: int = 8) -> list:
        """
        Generate fraud ring patterns (circular money flows) with noise.

        Args:
            n_rings: Number of fraud rings to create

        Returns:
            List of fraud ring user IDs
        """
        print(f"Generating {n_rings} fraud rings with noise...")
        fraudulent_users = []

        for ring_idx in range(n_rings):
            ring_size = np.random.randint(3, 9)
            ring_users = np.random.choice(
                self.users_df["user_id"].values, size=ring_size, replace=False
            )

            # Create circular transaction pattern
            for i in range(ring_size):
                # Make some rings incomplete
                if np.random.random() < self.incomplete_pattern_rate:
                    continue
                    
                source = ring_users[i]
                target = ring_users[(i + 1) % ring_size]

                # Add multiple transactions within the ring with variation
                n_ring_txns = np.random.randint(5, 15)
                for _ in range(n_ring_txns):
                    # Add amount noise (±20%)
                    base_amount = np.random.uniform(500, 5000)
                    amount = base_amount * (1 + np.random.uniform(-self.amount_noise_pct, self.amount_noise_pct))
                    
                    # Add temporal variation
                    base_timestamp = datetime.now() - timedelta(
                        days=np.random.randint(1, 180)
                    )
                    timestamp = base_timestamp + timedelta(
                        minutes=np.random.randint(-self.timestamp_jitter_minutes, self.timestamp_jitter_minutes)
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

    def generate_velocity_anomalies(self, n_compromised: int = 50) -> list:
        """
        Generate velocity anomaly patterns (unusual transaction bursts) with noise.

        Args:
            n_compromised: Number of compromised accounts

        Returns:
            List of compromised user IDs
        """
        print(f"Generating {n_compromised} velocity anomalies with noise...")
        compromised_users = np.random.choice(
            self.users_df["user_id"].values, size=n_compromised, replace=False
        )

        for user_id in compromised_users:
            # Sudden burst of transactions with more variation
            burst_size = np.random.randint(15, 50)
            burst_start = datetime.now() - timedelta(days=np.random.randint(1, 90))

            for txn_idx in range(burst_size):
                target = np.random.choice(self.users_df["user_id"].values)
                if target == user_id:
                    continue

                # Add amount variation
                base_amount = np.random.uniform(100, 3000)
                amount = base_amount * (1 + np.random.uniform(-self.amount_noise_pct, self.amount_noise_pct))
                
                # Add more temporal spread - some bursts over hours, some over days
                time_spread = np.random.choice([120, 1440, 4320])  # 2 hours, 1 day, 3 days
                timestamp = burst_start + timedelta(minutes=np.random.randint(1, time_spread))

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

    def generate_fraud_communities(self, n_communities: int = 2) -> list:
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

    def generate_power_users(self) -> list:
        """
        Generate legitimate power users with high transaction velocity.
        
        These users have similar transaction counts to velocity fraudsters
        but spread over longer time periods with legitimate patterns.
        
        Returns:
            List of power user IDs
        """
        print(f"Generating {self.n_power_users} legitimate power users...")
        power_users = np.random.choice(
            self.users_df["user_id"].values, size=self.n_power_users, replace=False
        )
        
        for user_id in power_users:
            # High transaction count but spread over weeks/months
            n_txns = np.random.randint(20, 40)
            
            for _ in range(n_txns):
                target = np.random.choice(self.users_df["user_id"].values)
                if target == user_id:
                    continue
                
                # Legitimate amounts and types
                amount = np.random.lognormal(mean=4.5, sigma=1.2)
                amount = max(10, min(amount, 10000))
                
                # Spread over weeks, not hours
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(1, 90),
                    hours=np.random.randint(0, 24)
                )
                
                self.graph.add_edge(
                    user_id,
                    target,
                    amount=amount,
                    timestamp=timestamp,
                    transaction_type=np.random.choice(
                        ["transfer", "payment", "purchase"], p=[0.2, 0.5, 0.3]
                    ),
                    is_fraud=False,
                    fraud_type=None,
                )
        
        return list(power_users)

    def generate_business_networks(self) -> list:
        """
        Generate legitimate business networks with circular payment patterns.
        
        These simulate payroll, vendor payments, and business transactions
        that may look like fraud rings but are legitimate.
        
        Returns:
            List of business network user IDs
        """
        print(f"Generating {self.n_business_networks} legitimate business networks...")
        business_users = []
        
        for network_idx in range(self.n_business_networks):
            network_size = np.random.randint(3, 6)
            network_members = np.random.choice(
                self.users_df["user_id"].values, size=network_size, replace=False
            )
            
            # Create circular payment pattern (like payroll cycle)
            for i in range(network_size):
                source = network_members[i]
                target = network_members[(i + 1) % network_size]
                
                # Regular monthly payments, smaller amounts
                n_payments = np.random.randint(2, 6)
                for month in range(n_payments):
                    amount = np.random.uniform(50, 500)  # Smaller than fraud rings
                    
                    # Regular monthly pattern
                    timestamp = datetime.now() - timedelta(
                        days=30 * month + np.random.randint(-3, 3)
                    )
                    
                    self.graph.add_edge(
                        source,
                        target,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="payment",
                        is_fraud=False,
                        fraud_type=None,
                    )
            
            business_users.extend(network_members)
        
        return list(set(business_users))

    def generate_money_pooling_groups(self) -> list:
        """
        Generate legitimate money pooling groups (friends/family).
        
        These have high internal connectivity like fraud communities
        but smaller groups and lower transaction frequencies.
        
        Returns:
            List of group member user IDs
        """
        print(f"Generating {self.n_money_pooling_groups} legitimate money pooling groups...")
        pooling_users = []
        
        for group_idx in range(self.n_money_pooling_groups):
            group_size = np.random.randint(3, 5)  # Smaller than fraud communities
            group_members = np.random.choice(
                self.users_df["user_id"].values, size=group_size, replace=False
            )
            
            # Moderate internal connectivity
            for source in group_members:
                n_internal_txns = np.random.randint(2, 5)  # Lower than fraud
                targets = np.random.choice(
                    group_members, size=n_internal_txns, replace=True
                )
                
                for target in targets:
                    if source == target:
                        continue
                    
                    # Small amounts (splitting bills, etc.)
                    amount = np.random.uniform(20, 200)
                    timestamp = datetime.now() - timedelta(
                        days=np.random.randint(1, 180)
                    )
                    
                    self.graph.add_edge(
                        source,
                        target,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type=np.random.choice(["transfer", "payment"]),
                        is_fraud=False,
                        fraud_type=None,
                    )
            
            pooling_users.extend(group_members)
        
        return list(set(pooling_users))

    def add_legitimate_transactions_to_fraudsters(self, fraud_users: list):
        """
        Add legitimate transactions to fraudster profiles to create ambiguity.
        
        Args:
            fraud_users: List of fraudulent user IDs
        """
        print(f"Adding legitimate transactions to {len(fraud_users)} fraudsters...")
        
        for user_id in fraud_users:
            # Each fraudster does some legitimate transactions
            n_legit_txns = int(np.random.randint(10, 30) * self.fraud_legitimate_mix)
            
            for _ in range(n_legit_txns):
                target = np.random.choice(self.users_df["user_id"].values)
                if target == user_id:
                    continue
                
                # Normal legitimate transaction
                amount = np.random.lognormal(mean=4.5, sigma=1.2)
                amount = max(10, min(amount, 10000))
                
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(1, 365)
                )
                
                self.graph.add_edge(
                    user_id,
                    target,
                    amount=amount,
                    timestamp=timestamp,
                    transaction_type=np.random.choice(
                        ["transfer", "payment", "purchase"], p=[0.3, 0.4, 0.3]
                    ),
                    is_fraud=False,
                    fraud_type=None,
                )

    def add_fraud_transactions_to_legitimate_users(self, n_users: int = 100):
        """
        Add some fraud-labeled transactions to legitimate users.
        
        This creates false positives and makes the task more challenging.
        
        Args:
            n_users: Number of legitimate users to add fraud transactions to
        """
        print(f"Adding fraud transactions to {n_users} legitimate users (label noise)...")
        
        # Get legitimate users only
        legitimate_users = [
            uid for uid in self.users_df["user_id"].values 
            if uid not in self.fraudulent_users
        ]
        
        if len(legitimate_users) < n_users:
            n_users = len(legitimate_users)
        
        selected_users = np.random.choice(legitimate_users, size=n_users, replace=False)
        
        for user_id in selected_users:
            # Add 1-3 fraud-labeled transactions
            n_fraud_txns = np.random.randint(1, 4)
            
            for _ in range(n_fraud_txns):
                target = np.random.choice(self.users_df["user_id"].values)
                if target == user_id:
                    continue
                
                # Fraud-like amounts
                amount = np.random.uniform(500, 3000)
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(1, 180)
                )
                
                self.graph.add_edge(
                    user_id,
                    target,
                    amount=amount,
                    timestamp=timestamp,
                    transaction_type="transfer",
                    is_fraud=True,  # Label as fraud even though user is legitimate
                    fraud_type="noise",
                )

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
        """Create fraud label dataset at user level using tracked fraudulent users."""
        print("Generating fraud labels...")

        # Use the tracked fraudulent users set instead of deriving from transactions
        # This prevents label noise from fraud transactions added to legitimate users
        labels = []
        for user_id in self.users_df["user_id"].values:
            labels.append(
                {
                    "user_id": user_id,
                    "is_fraudster": 1 if user_id in self.fraudulent_users else 0,
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
        """Run complete data generation pipeline with noise injection."""
        # Generate base users
        self.generate_users()
        
        # Generate fraud patterns (tuned for F1 0.82-0.86)
        fraud_ring_users = self.generate_fraud_rings(n_rings=10)  # Increased from 5
        velocity_users = self.generate_velocity_anomalies(n_compromised=60)  # Increased from 30
        community_users = self.generate_fraud_communities()
        
        # Track all fraudulent users
        all_fraud_users = list(set(fraud_ring_users + velocity_users + community_users))
        self.fraudulent_users.update(all_fraud_users)
        
        # Generate legitimate users with fraud-like patterns (noise injection)
        power_users = self.generate_power_users()
        business_users = self.generate_business_networks()
        pooling_users = self.generate_money_pooling_groups()
        
        # Add legitimate transactions to fraudsters (creates ambiguity)
        self.add_legitimate_transactions_to_fraudsters(all_fraud_users)
        
        # Add fraud transactions to some legitimate users (creates false positives)
        self.add_fraud_transactions_to_legitimate_users(n_users=50)  # Reduced from 150
        
        # Fill remaining with normal legitimate transactions
        self.generate_legitimate_transactions()
        
        # Add shared entities
        self.create_shared_entities()
        
        # Build final datasets
        self.build_transaction_dataframe()
        self.generate_fraud_labels()
        
        # Save to disk
        self.save_data(output_dir)
        
        # Print noise injection summary
        print("\n" + "=" * 70)
        print("NOISE INJECTION SUMMARY")
        print("=" * 70)
        print(f"Fraudulent users: {len(all_fraud_users)}")
        print(f"  - With mixed legitimate behavior: {len(all_fraud_users)}")
        print(f"Legitimate users with fraud-like patterns:")
        print(f"  - Power users (high velocity): {len(power_users)}")
        print(f"  - Business networks (circular): {len(business_users)}")
        print(f"  - Money pooling groups (community): {len(pooling_users)}")
        print(f"Noise parameters:")
        print(f"  - Fraud-legitimate mix: {self.fraud_legitimate_mix:.0%}")
        print(f"  - Amount noise: ±{self.amount_noise_pct:.0%}")
        print(f"  - Timestamp jitter: ±{self.timestamp_jitter_minutes} minutes")
        print(f"  - Incomplete patterns: {self.incomplete_pattern_rate:.0%}")
        print("=" * 70)


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
