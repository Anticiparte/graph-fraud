// ============================================================================
// Fraud Detection Graph Exploration Queries
// ============================================================================
// This file contains Cypher queries for exploring fraud patterns in the graph

// ============================================================================
// 1. Basic Statistics
// ============================================================================

// Count all nodes and relationships
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY count DESC;

// Count all relationship types
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(r) AS count
ORDER BY count DESC;

// Fraud statistics overview
MATCH (u:User)
RETURN
    count(u) AS total_users,
    sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END) AS fraudulent_users,
    toFloat(sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END)) / count(u) * 100 AS fraud_rate_pct;

// Transaction fraud statistics
MATCH ()-[t:TRANSACTS_WITH]->()
RETURN
    count(t) AS total_transactions,
    sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraudulent_transactions,
    toFloat(sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)) / count(t) * 100 AS fraud_rate_pct;

// ============================================================================
// 2. Fraud Pattern: Ring Detection
// ============================================================================

// Find circular transaction patterns (rings of length 3)
MATCH (a:User)-[t1:TRANSACTS_WITH]->(b:User)-[t2:TRANSACTS_WITH]->(c:User)-[t3:TRANSACTS_WITH]->(a)
WHERE t1.is_fraud OR t2.is_fraud OR t3.is_fraud
RETURN
    a.user_id AS user1,
    b.user_id AS user2,
    c.user_id AS user3,
    t1.amount AS amount1,
    t2.amount AS amount2,
    t3.amount AS amount3,
    t1.is_fraud AS fraud1,
    t2.is_fraud AS fraud2,
    t3.is_fraud AS fraud3
LIMIT 20;

// Find rings of length 4
MATCH (a:User)-[:TRANSACTS_WITH]->(b:User)-[:TRANSACTS_WITH]->(c:User)-[:TRANSACTS_WITH]->(d:User)-[:TRANSACTS_WITH]->(a)
WHERE a.is_fraudster
RETURN a.user_id, b.user_id, c.user_id, d.user_id
LIMIT 10;

// Count ring structures by length
MATCH path = (a:User)-[:TRANSACTS_WITH*3..5]->(a)
WHERE a.is_fraudster
RETURN length(path) AS ring_length, count(*) AS count
ORDER BY ring_length;

// ============================================================================
// 3. Fraud Pattern: Velocity Anomalies
// ============================================================================

// Find users with high transaction velocity
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WITH u, count(t) AS txn_count, sum(t.amount) AS total_amount
WHERE txn_count > 10
RETURN
    u.user_id,
    u.is_fraudster,
    txn_count,
    total_amount,
    total_amount / txn_count AS avg_amount
ORDER BY txn_count DESC
LIMIT 20;

// Compare transaction velocity: fraud vs legitimate
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WITH u.is_fraudster AS is_fraud, count(t) AS txn_count
RETURN
    is_fraud,
    avg(txn_count) AS avg_transactions,
    min(txn_count) AS min_transactions,
    max(txn_count) AS max_transactions
ORDER BY is_fraud;

// Find burst activity (multiple transactions in short time)
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WITH u, t
ORDER BY u.user_id, t.timestamp
WITH u, collect(t) AS transactions
WHERE size(transactions) >= 3
UNWIND range(0, size(transactions)-3) AS i
WITH u, transactions[i] AS t1, transactions[i+1] AS t2, transactions[i+2] AS t3
WHERE duration.between(t1.timestamp, t3.timestamp).seconds < 3600
RETURN
    u.user_id,
    u.is_fraudster,
    t1.timestamp AS first_txn,
    t3.timestamp AS last_txn,
    duration.between(t1.timestamp, t3.timestamp).seconds AS time_window_sec
ORDER BY time_window_sec
LIMIT 20;

// ============================================================================
// 4. Fraud Pattern: Community Isolation
// ============================================================================

// Find isolated fraud communities (users only transacting with other fraudsters)
MATCH (fraudster:User {is_fraudster: true})-[:TRANSACTS_WITH]-(other:User)
WITH fraudster,
     count(other) AS total_connections,
     sum(CASE WHEN other.is_fraudster THEN 1 ELSE 0 END) AS fraud_connections
WHERE total_connections > 0
WITH fraudster,
     toFloat(fraud_connections) / total_connections AS fraud_ratio
WHERE fraud_ratio >= 0.8
RETURN
    fraudster.user_id,
    fraud_ratio,
    total_connections
ORDER BY fraud_ratio DESC, total_connections DESC
LIMIT 20;

// Find dense fraud subgraphs
MATCH (u1:User {is_fraudster: true})-[:TRANSACTS_WITH]-(u2:User {is_fraudster: true})
WITH u1, count(DISTINCT u2) AS fraud_neighbors
WHERE fraud_neighbors >= 5
RETURN u1.user_id, fraud_neighbors
ORDER BY fraud_neighbors DESC
LIMIT 20;

// ============================================================================
// 5. Device and IP Sharing Analysis
// ============================================================================

// Find devices shared between multiple users
MATCH (u:User)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:User)
WITH d, count(DISTINCT u) AS user_count,
     collect(DISTINCT u.user_id) AS users,
     sum(CASE WHEN u.is_fraudster THEN 1 ELSE 0 END) AS fraud_count
WHERE user_count > 1
RETURN
    d.device_id,
    user_count,
    fraud_count,
    users[0..5] AS sample_users
ORDER BY user_count DESC
LIMIT 20;

// Find IP addresses shared between fraudsters and legitimate users
MATCH (fraudster:User {is_fraudster: true})-[:USES_IP]->(ip:IPAddress)<-[:USES_IP]-(legit:User {is_fraudster: false})
WITH ip, count(DISTINCT fraudster) AS fraud_count, count(DISTINCT legit) AS legit_count
RETURN
    ip.ip_address,
    fraud_count,
    legit_count
ORDER BY fraud_count DESC
LIMIT 20;

// Users with shared devices and fraudulent behavior
MATCH (u:User)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:User)
WHERE u.is_fraudster AND other.is_fraudster AND u <> other
RETURN
    d.device_id,
    collect(DISTINCT u.user_id) AS fraudulent_users
LIMIT 20;

// ============================================================================
// 6. High-Risk User Identification
// ============================================================================

// Find users with most fraudulent transactions
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WHERE t.is_fraud
WITH u, count(t) AS fraud_txn_count, sum(t.amount) AS fraud_amount
RETURN
    u.user_id,
    u.is_fraudster,
    fraud_txn_count,
    fraud_amount
ORDER BY fraud_txn_count DESC
LIMIT 20;

// Find users receiving many fraudulent transactions
MATCH (u:User)<-[t:TRANSACTS_WITH]-()
WHERE t.is_fraud
WITH u, count(t) AS incoming_fraud_count, sum(t.amount) AS incoming_fraud_amount
RETURN
    u.user_id,
    u.is_fraudster,
    incoming_fraud_count,
    incoming_fraud_amount
ORDER BY incoming_fraud_count DESC
LIMIT 20;

// Find users with fraud neighbors (connected to many fraudsters)
MATCH (u:User)-[:TRANSACTS_WITH]-(fraudster:User {is_fraudster: true})
WITH u, count(DISTINCT fraudster) AS fraud_neighbor_count
WHERE fraud_neighbor_count >= 3
RETURN
    u.user_id,
    u.is_fraudster,
    fraud_neighbor_count
ORDER BY fraud_neighbor_count DESC
LIMIT 20;

// ============================================================================
// 7. Transaction Amount Analysis
// ============================================================================

// Compare transaction amounts: fraud vs legitimate
MATCH ()-[t:TRANSACTS_WITH]->()
RETURN
    t.is_fraud,
    avg(t.amount) AS avg_amount,
    min(t.amount) AS min_amount,
    max(t.amount) AS max_amount,
    percentileCont(t.amount, 0.5) AS median_amount,
    percentileCont(t.amount, 0.95) AS p95_amount
ORDER BY t.is_fraud;

// Large fraudulent transactions
MATCH (source:User)-[t:TRANSACTS_WITH]->(target:User)
WHERE t.is_fraud
RETURN
    source.user_id,
    target.user_id,
    t.amount,
    t.timestamp,
    t.fraud_type
ORDER BY t.amount DESC
LIMIT 20;

// ============================================================================
// 8. Temporal Pattern Analysis
// ============================================================================

// Transaction distribution by hour of day
MATCH ()-[t:TRANSACTS_WITH]->()
RETURN
    t.timestamp.hour AS hour,
    count(t) AS total_txns,
    sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_txns,
    toFloat(sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)) / count(t) * 100 AS fraud_rate_pct
ORDER BY hour;

// Transaction distribution by day of week
MATCH ()-[t:TRANSACTS_WITH]->()
RETURN
    t.timestamp.dayOfWeek AS day_of_week,
    count(t) AS total_txns,
    sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_txns
ORDER BY day_of_week;

// ============================================================================
// 9. User Profile Analysis
// ============================================================================

// Compare user attributes: fraudster vs legitimate
MATCH (u:User)
RETURN
    u.is_fraudster,
    avg(u.age) AS avg_age,
    avg(u.account_age_days) AS avg_account_age_days,
    avg(u.credit_score) AS avg_credit_score,
    count(u) AS user_count
ORDER BY u.is_fraudster;

// Young accounts with high fraud activity
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WHERE u.account_age_days < 180 AND t.is_fraud
WITH u, count(t) AS fraud_count
RETURN
    u.user_id,
    u.account_age_days,
    u.credit_score,
    fraud_count
ORDER BY fraud_count DESC
LIMIT 20;

// Low credit score users involved in fraud
MATCH (u:User)
WHERE u.credit_score < 600 AND u.is_fraudster
RETURN
    u.user_id,
    u.credit_score,
    u.age,
    u.account_age_days
ORDER BY u.credit_score
LIMIT 20;

// ============================================================================
// 10. Graph Degree Analysis
// ============================================================================

// Find hub users (high degree centrality)
MATCH (u:User)
WITH u, count {(u)-[:TRANSACTS_WITH]-()} AS degree
WHERE degree > 0
RETURN
    u.user_id,
    u.is_fraudster,
    degree
ORDER BY degree DESC
LIMIT 20;

// In-degree vs out-degree for fraudsters
MATCH (u:User)
WHERE u.is_fraudster
WITH u,
     count {(u)-[:TRANSACTS_WITH]->()} AS out_degree,
     count {(u)<-[:TRANSACTS_WITH]-()} AS in_degree
RETURN
    u.user_id,
    in_degree,
    out_degree,
    in_degree + out_degree AS total_degree
ORDER BY total_degree DESC
LIMIT 20;
