// Neo4j Data Loading Script for Fraud Detection
// This script loads users, transactions, and fraud labels into Neo4j

// ============================================================================
// 1. Create Constraints and Indices
// ============================================================================

// Create constraint on User ID
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

// Create indices for better query performance
CREATE INDEX user_age_idx IF NOT EXISTS FOR (u:User) ON (u.age);
CREATE INDEX user_fraud_idx IF NOT EXISTS FOR (u:User) ON (u.is_fraudster);
CREATE INDEX txn_fraud_idx IF NOT EXISTS FOR ()-[t:TRANSACTS_WITH]-() ON (t.is_fraud);
CREATE INDEX txn_timestamp_idx IF NOT EXISTS FOR ()-[t:TRANSACTS_WITH]-() ON (t.timestamp);

// ============================================================================
// 2. Load Users
// ============================================================================

LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
CREATE (u:User {
    user_id: row.user_id,
    age: toInteger(row.age),
    account_age_days: toInteger(row.account_age_days),
    credit_score: toInteger(row.credit_score),
    account_type: row.account_type,
    registration_timestamp: datetime(row.registration_timestamp)
});

// ============================================================================
// 3. Load Fraud Labels
// ============================================================================

LOAD CSV WITH HEADERS FROM 'file:///fraud_labels.csv' AS row
MATCH (u:User {user_id: row.user_id})
SET u.is_fraudster = toInteger(row.is_fraudster) = 1;

// ============================================================================
// 4. Load Transactions as Relationships
// ============================================================================

// Load in batches for better performance
:auto USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
MATCH (source:User {user_id: row.source_user})
MATCH (target:User {user_id: row.target_user})
CREATE (source)-[t:TRANSACTS_WITH {
    transaction_id: row.transaction_id,
    amount: toFloat(row.amount),
    timestamp: datetime(row.timestamp),
    transaction_type: row.transaction_type,
    is_fraud: row.is_fraud = 'True',
    fraud_type: row.fraud_type,
    device_id: row.device_id,
    ip_address: row.ip_address
}]->(target);

// ============================================================================
// 5. Load Device Sharing Relationships
// ============================================================================

LOAD CSV WITH HEADERS FROM 'file:///devices.csv' AS row
MERGE (d:Device {device_id: row.device_id})
WITH d, row
MATCH (u:User {user_id: row.user_id})
MERGE (u)-[:USES_DEVICE]->(d);

// ============================================================================
// 6. Load IP Sharing Relationships
// ============================================================================

LOAD CSV WITH HEADERS FROM 'file:///ip_addresses.csv' AS row
MERGE (ip:IPAddress {ip_address: row.ip_address})
WITH ip, row
MATCH (u:User {user_id: row.user_id})
MERGE (u)-[:USES_IP]->(ip);

// ============================================================================
// 7. Verification Queries
// ============================================================================

// Count nodes
MATCH (u:User) RETURN 'Users' AS type, count(u) AS count
UNION
MATCH (d:Device) RETURN 'Devices' AS type, count(d) AS count
UNION
MATCH (ip:IPAddress) RETURN 'IP Addresses' AS type, count(ip) AS count;

// Count relationships
MATCH ()-[t:TRANSACTS_WITH]->() RETURN 'Transactions' AS type, count(t) AS count
UNION
MATCH ()-[u:USES_DEVICE]->() RETURN 'Device Usage' AS type, count(u) AS count
UNION
MATCH ()-[u:USES_IP]->() RETURN 'IP Usage' AS type, count(u) AS count;

// Fraud statistics
MATCH (u:User)
RETURN
    count(u) AS total_users,
    sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END) AS fraudulent_users,
    toFloat(sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END)) / count(u) * 100 AS fraud_rate_pct;

MATCH ()-[t:TRANSACTS_WITH]->()
RETURN
    count(t) AS total_transactions,
    sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraudulent_transactions,
    toFloat(sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)) / count(t) * 100 AS fraud_rate_pct;
