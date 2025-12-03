"""
Load fraud detection data into Neo4j graph database.

This script loads users, transactions, devices, and IP addresses into Neo4j,
creating the graph structure needed for fraud detection analysis.
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


class Neo4jLoader:
    """Load fraud detection data into Neo4j."""

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

    def execute_query(self, query: str, description: str = ""):
        """
        Execute a Cypher query.

        Args:
            query: Cypher query to execute
            description: Description of the query for logging
        """
        with self.driver.session() as session:
            start_time = time.time()
            if description:
                print(f"\n{description}...")

            try:
                result = session.run(query)
                summary = result.consume()
                elapsed = time.time() - start_time

                print(f"  ✓ Completed in {elapsed:.2f}s")
                if summary.counters:
                    counters = summary.counters
                    if counters.nodes_created:
                        print(f"    Nodes created: {counters.nodes_created}")
                    if counters.relationships_created:
                        print(f"    Relationships created: {counters.relationships_created}")
                    if counters.properties_set:
                        print(f"    Properties set: {counters.properties_set}")

                return True
            except Exception as e:
                print(f"  ✗ Error: {e}")
                return False

    def create_constraints(self):
        """Create constraints and indices."""
        print("="*70)
        print("STEP 1: Creating Constraints and Indices")
        print("="*70)

        queries = [
            ("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
             "Creating unique constraint on User.user_id"),
            ("CREATE INDEX user_age_idx IF NOT EXISTS FOR (u:User) ON (u.age)",
             "Creating index on User.age"),
            ("CREATE INDEX user_fraud_idx IF NOT EXISTS FOR (u:User) ON (u.is_fraudster)",
             "Creating index on User.is_fraudster"),
        ]

        for query, desc in queries:
            self.execute_query(query, desc)

    def load_users(self):
        """Load user nodes."""
        print("\n" + "="*70)
        print("STEP 2: Loading Users")
        print("="*70)

        query = """
        LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
        CREATE (u:User {
            user_id: row.user_id,
            age: toInteger(row.age),
            account_age_days: toInteger(row.account_age_days),
            credit_score: toInteger(row.credit_score),
            account_type: row.account_type,
            registration_timestamp: datetime(replace(row.registration_timestamp, ' ', 'T'))
        })
        """
        self.execute_query(query, "Loading user nodes")

    def load_fraud_labels(self):
        """Add fraud labels to users."""
        print("\n" + "="*70)
        print("STEP 3: Adding Fraud Labels")
        print("="*70)

        query = """
        LOAD CSV WITH HEADERS FROM 'file:///fraud_labels.csv' AS row
        MATCH (u:User {user_id: row.user_id})
        SET u.is_fraudster = (toInteger(row.is_fraudster) = 1)
        """
        self.execute_query(query, "Adding fraud labels to users")

    def load_transactions(self):
        """Load transaction relationships."""
        print("\n" + "="*70)
        print("STEP 4: Loading Transactions (this may take a while...)")
        print("="*70)

        query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row RETURN row",
            "MATCH (source:User {user_id: row.source_user})
             MATCH (target:User {user_id: row.target_user})
             CREATE (source)-[t:TRANSACTS_WITH {
                 transaction_id: row.transaction_id,
                 amount: toFloat(row.amount),
                 timestamp: datetime(replace(row.timestamp, ' ', 'T')),
                 transaction_type: row.transaction_type,
                 is_fraud: (row.is_fraud = 'True'),
                 fraud_type: CASE WHEN row.fraud_type = '' THEN null ELSE row.fraud_type END,
                 device_id: row.device_id,
                 ip_address: row.ip_address
             }]->(target)",
            {batchSize: 1000, parallel: false}
        )
        """
        self.execute_query(query, "Loading transaction relationships")

    def load_devices(self):
        """Load device sharing relationships."""
        print("\n" + "="*70)
        print("STEP 5: Loading Device Relationships")
        print("="*70)

        query = """
        LOAD CSV WITH HEADERS FROM 'file:///devices.csv' AS row
        MERGE (d:Device {device_id: row.device_id})
        WITH d, row
        MATCH (u:User {user_id: row.user_id})
        MERGE (u)-[:USES_DEVICE]->(d)
        """
        self.execute_query(query, "Loading device sharing relationships")

    def load_ip_addresses(self):
        """Load IP address sharing relationships."""
        print("\n" + "="*70)
        print("STEP 6: Loading IP Address Relationships")
        print("="*70)

        query = """
        LOAD CSV WITH HEADERS FROM 'file:///ip_addresses.csv' AS row
        MERGE (ip:IPAddress {ip_address: row.ip_address})
        WITH ip, row
        MATCH (u:User {user_id: row.user_id})
        MERGE (u)-[:USES_IP]->(ip)
        """
        self.execute_query(query, "Loading IP address sharing relationships")

    def verify_data(self):
        """Verify loaded data."""
        print("\n" + "="*70)
        print("STEP 7: Data Verification")
        print("="*70)

        queries = [
            ("MATCH (u:User) RETURN count(u) AS count", "Total Users"),
            ("MATCH (d:Device) RETURN count(d) AS count", "Total Devices"),
            ("MATCH (ip:IPAddress) RETURN count(ip) AS count", "Total IP Addresses"),
            ("MATCH ()-[t:TRANSACTS_WITH]->() RETURN count(t) AS count", "Total Transactions"),
            ("MATCH ()-[u:USES_DEVICE]->() RETURN count(u) AS count", "Device Relationships"),
            ("MATCH ()-[u:USES_IP]->() RETURN count(u) AS count", "IP Relationships"),
        ]

        with self.driver.session() as session:
            print()
            for query, label in queries:
                result = session.run(query)
                record = result.single()
                count = record["count"] if record else 0
                print(f"  {label}: {count:,}")

        # Fraud statistics
        print("\nFraud Statistics:")
        with self.driver.session() as session:
            query = """
            MATCH (u:User)
            RETURN
                count(u) AS total_users,
                sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END) AS fraudulent_users
            """
            result = session.run(query)
            record = result.single()
            total = record["total_users"]
            fraud = record["fraudulent_users"]
            fraud_rate = (fraud / total * 100) if total > 0 else 0
            print(f"  Fraudulent Users: {fraud:,} ({fraud_rate:.2f}%)")

            query = """
            MATCH ()-[t:TRANSACTS_WITH]->()
            RETURN
                count(t) AS total_transactions,
                sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraudulent_transactions
            """
            result = session.run(query)
            record = result.single()
            total = record["total_transactions"]
            fraud = record["fraudulent_transactions"]
            fraud_rate = (fraud / total * 100) if total > 0 else 0
            print(f"  Fraudulent Transactions: {fraud:,} ({fraud_rate:.2f}%)")

    def load_all(self):
        """Execute complete data loading pipeline."""
        print("\n" + "="*70)
        print("Neo4j Fraud Detection Data Loader")
        print("="*70)

        try:
            self.create_constraints()
            self.load_users()
            self.load_fraud_labels()
            self.load_transactions()
            self.load_devices()
            self.load_ip_addresses()
            self.verify_data()

            print("\n" + "="*70)
            print("✓ Data Loading Complete!")
            print("="*70)
            print("\nAccess Neo4j Browser at: http://localhost:7474")
            print("  Username: neo4j")
            print("  Password: frauddetection123")
            print()

        except Exception as e:
            print(f"\n✗ Error during data loading: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load fraud detection data into Neo4j")
    parser.add_argument(
        "--uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD", "frauddetection123"),
        help="Neo4j password",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    loader = Neo4jLoader(args.uri, args.user, args.password)
    try:
        loader.load_all()
    finally:
        loader.close()


if __name__ == "__main__":
    main()
