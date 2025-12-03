"""
Verify Neo4j graph data and fraud patterns.

Run key queries to validate the loaded data and demonstrate fraud patterns.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


def run_verification():
    """Run verification queries and display results."""
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "frauddetection123")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    queries = {
        "Node Counts": """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """,
        "Fraud Statistics": """
            MATCH (u:User)
            RETURN
                count(u) AS total_users,
                sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END) AS fraudulent_users,
                toFloat(sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END)) / count(u) * 100 AS fraud_rate_pct
        """,
        "Transaction Fraud Rate": """
            MATCH ()-[t:TRANSACTS_WITH]->()
            RETURN
                count(t) AS total_transactions,
                sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraudulent_transactions,
                toFloat(sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)) / count(t) * 100 AS fraud_rate_pct
        """,
        "Ring Patterns (3-node cycles)": """
            MATCH (a:User)-[t1:TRANSACTS_WITH]->(b:User)-[t2:TRANSACTS_WITH]->(c:User)-[t3:TRANSACTS_WITH]->(a)
            WHERE t1.is_fraud OR t2.is_fraud OR t3.is_fraud
            RETURN count(*) AS ring_count
        """,
        "High Velocity Users": """
            MATCH (u:User)-[t:TRANSACTS_WITH]->()
            WITH u, count(t) AS txn_count
            WHERE txn_count > 15
            RETURN
                count(*) AS high_velocity_users,
                avg(txn_count) AS avg_transactions
        """,
        "Device Sharing": """
            MATCH (u:User)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:User)
            WHERE u <> other
            WITH d, count(DISTINCT u) AS user_count
            WHERE user_count > 1
            RETURN
                count(d) AS shared_devices,
                avg(user_count) AS avg_users_per_device,
                max(user_count) AS max_users_per_device
        """,
        "IP Sharing": """
            MATCH (u:User)-[:USES_IP]->(ip:IPAddress)<-[:USES_IP]-(other:User)
            WHERE u <> other
            WITH ip, count(DISTINCT u) AS user_count
            WHERE user_count > 1
            RETURN
                count(ip) AS shared_ips,
                avg(user_count) AS avg_users_per_ip,
                max(user_count) AS max_users_per_ip
        """,
        "Fraud Community Isolation": """
            MATCH (fraudster:User {is_fraudster: true})-[:TRANSACTS_WITH]-(other:User)
            WITH fraudster,
                 count(other) AS total_connections,
                 sum(CASE WHEN other.is_fraudster THEN 1 ELSE 0 END) AS fraud_connections
            WHERE total_connections > 0
            WITH fraudster,
                 toFloat(fraud_connections) / total_connections AS fraud_ratio
            WHERE fraud_ratio >= 0.8
            RETURN count(*) AS isolated_fraudsters
        """,
        "Transaction Amount Comparison": """
            MATCH ()-[t:TRANSACTS_WITH]->()
            RETURN
                t.is_fraud,
                avg(t.amount) AS avg_amount,
                min(t.amount) AS min_amount,
                max(t.amount) AS max_amount
            ORDER BY t.is_fraud
        """,
        "Hub Users (High Degree)": """
            MATCH (u:User)
            WITH u, count {(u)-[:TRANSACTS_WITH]-()} AS degree
            WHERE degree > 15
            RETURN
                count(*) AS hub_users,
                avg(degree) AS avg_degree,
                max(degree) AS max_degree
        """,
    }

    print("=" * 80)
    print("NEO4J GRAPH VERIFICATION")
    print("=" * 80)

    with driver.session() as session:
        for title, query in queries.items():
            print(f"\n{title}:")
            print("-" * 80)
            try:
                result = session.run(query)
                records = list(result)

                if records:
                    for record in records:
                        # Format output based on record content
                        items = []
                        for key in record.keys():
                            value = record[key]
                            if isinstance(value, float):
                                items.append(f"{key}: {value:.2f}")
                            else:
                                items.append(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
                        print("  " + ", ".join(items))
                else:
                    print("  No results")
            except Exception as e:
                print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nNeo4j Browser: http://localhost:7474")
    print(f"Username: neo4j")
    print(f"Password: frauddetection123")
    print()

    driver.close()


if __name__ == "__main__":
    run_verification()
