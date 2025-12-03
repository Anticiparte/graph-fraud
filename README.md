# Graph-Based Fraud Detection MLOps Pipeline

Production-ready fraud detection system using graph-based machine learning with Neo4j, XGBoost, and MLflow.

## Overview

This project demonstrates how graph analysis and machine learning can detect fraudulent transactions in financial networks. By modeling users as nodes and transactions as edges, the system identifies complex fraud patterns that traditional methods miss.

### Detected Fraud Patterns

1. **Ring/Circular Patterns** (618 detected): Money laundering schemes where funds cycle through multiple accounts
2. **Velocity Anomalies** (501 high-velocity users): Unusual transaction frequencies indicating account compromise
3. **Community Isolation** (466 isolated fraudsters): Fraudulent users forming tight clusters with 80%+ fraud connections

### Key Results

- **Model Performance**: 82.5% ROC-AUC, 48.9% F1-score (realistic performance without data leakage)
- **Inference Speed**: 10,000 predictions in 0.01 seconds
- **Feature Importance**: Balanced features - top feature (total_amount_in) at 16% importance
- **No Data Leakage**: All features derived from observable behavioral patterns
- **Realistic Data**: Includes noise, legitimate users with fraud-like patterns, and mixed fraudster behavior

## Architecture

```
graph-fraud/
├── data/
│   ├── raw/              # Generated transaction data (10K users, 100K txns)
│   ├── processed/        # Engineered features (29 features)
│   └── inference/        # Batch predictions with fraud scores
├── notebooks/            # Mandatory EDA (marimo)
│   ├── 01_data_quality_check.py
│   ├── 02_graph_statistics.py
│   ├── 03_fraud_pattern_exploration.py
│   └── 04_feature_correlation.py
├── src/
│   ├── data/
│   │   ├── generate_fraud_data.py      # Synthetic data generation
│   │   ├── load_to_neo4j.py            # Automated Neo4j loader
│   │   └── load_neo4j.cypher           # Manual Cypher queries
│   ├── features/
│   │   └── graph_features.py           # Extract 29 graph features
│   ├── models/
│   │   ├── train.py                    # XGBoost training w/ MLflow
│   │   └── inference.py                # Batch prediction service
│   └── queries/
│       ├── fraud_exploration.cypher    # 10 categories of queries
│       └── verify_graph.py             # Data verification
├── config/
│   ├── training_config.yaml            # Model hyperparameters
│   ├── inference_config.yaml           # Batch inference settings
│   ├── feature_config.yaml             # Feature specifications
│   ├── prometheus/prometheus.yml       # Metrics scraping
│   └── grafana/fraud_detection_dashboard.json  # Monitoring dashboard
├── models/
│   └── fraud_detector.json             # Trained XGBoost model
├── docker-compose.yml                   # Neo4j, Prometheus, Grafana
└── pyproject.toml                       # Dependencies (uv managed)
```

## Requirements

- **Python**: 3.12
- **Memory**: 16GB RAM minimum
- **Docker**: Docker Engine + Docker Compose
- **Package Manager**: uv
- **Neo4j**: 5.15+ (via Docker)

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/Anticiparte/graph-fraud.git
cd graph-fraud

# Install dependencies with uv
uv sync --locked

# Activate virtual environment
source .venv/bin/activate

# Create environment file
cp .env.example .env
```

### 2. Start Infrastructure

```bash
# Start Neo4j, Prometheus, Grafana
docker-compose up -d

# Verify all services healthy
docker-compose ps

# Access points:
# - Neo4j Browser: http://localhost:7474 (neo4j / frauddetection123)
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin / frauddetection123)
```

### 3. Generate Synthetic Data

```bash
# Generate fraud transaction network
python src/data/generate_fraud_data.py \
    --n-users 10000 \
    --n-transactions 100000 \
    --fraud-rate 0.05 \
    --output-dir data/raw

# Output: 5 CSV files (users, transactions, fraud_labels, devices, ip_addresses)
```

**Realistic Data Features:**
- **Fraud patterns**: 10 fraud rings, 60 velocity anomalies, 2 fraud communities
- **Legitimate fraud-like users**: 30 power users, 5 business networks, 5 money pooling groups
- **Noise injection**: 30% fraud-legitimate mix, ±15% amount noise, ±30 min timestamp jitter
- **Pattern variations**: 10% incomplete fraud patterns, mixed transaction types

### 4. Explore Data (Mandatory EDA)

```bash
# Run all 4 marimo notebooks sequentially
marimo edit notebooks/01_data_quality_check.py
marimo edit notebooks/02_graph_statistics.py
marimo edit notebooks/03_fraud_pattern_exploration.py
marimo edit notebooks/04_feature_correlation.py

# Or run programmatically (non-interactive)
python notebooks/01_data_quality_check.py
```

### 5. Load Data into Neo4j

```bash
# Automated loading (recommended)
python src/data/load_to_neo4j.py

# Manual option: Use Cypher queries
# Copy CSVs: cp data/raw/*.csv neo4j_data/import/
# Execute: src/data/load_neo4j.cypher in Neo4j Browser
```

### 6. Verify Graph Data

```bash
# Run verification queries
python src/queries/verify_graph.py

# Expected output:
# - 10,000 users, 99,945 transactions
# - 618 ring patterns, 466 isolated fraudsters
# - Fraud amounts 8.6x higher ($1,589 vs $184)
```

### 7. Extract Graph Features

```bash
# Extract 29 features from Neo4j
python src/features/graph_features.py

# Features saved to: data/processed/graph_features.csv
# Includes: degree centrality, transaction metrics, neighbor analysis,
#           device/IP sharing features
```

### 8. Train Fraud Detection Model

```bash
# Train XGBoost with MLflow tracking
python src/models/train.py

# View MLflow experiments
mlflow ui

# Model registered as: fraud-detector v1
# Performance: 100% precision, recall, F1, ROC-AUC
```

### 9. Run Batch Inference

```bash
# Make predictions for all users
python src/models/inference.py --metrics-port 8000

# Output:
# - Predictions written to Neo4j (fraud_score, fraud_prediction)
# - CSV saved to: data/inference/predictions.csv
# - Prometheus metrics: http://localhost:8000/metrics
```

## Complete Pipeline Example

```bash
# End-to-end execution
docker-compose up -d && \
python src/data/generate_fraud_data.py && \
python src/data/load_to_neo4j.py && \
python src/features/graph_features.py && \
python src/models/train.py && \
python src/models/inference.py
```

## Neo4j Fraud Queries

Example queries in `src/queries/fraud_exploration.cypher`:

```cypher
// Find fraud rings (3-node cycles)
MATCH (a:User)-[t1:TRANSACTS_WITH]->(b:User)-[t2:TRANSACTS_WITH]->(c:User)-[t3:TRANSACTS_WITH]->(a)
WHERE t1.is_fraud OR t2.is_fraud OR t3.is_fraud
RETURN a.user_id, b.user_id, c.user_id, t1.amount, t2.amount, t3.amount
LIMIT 10;

// High-velocity fraudsters
MATCH (u:User)-[t:TRANSACTS_WITH]->()
WHERE u.is_fraudster
WITH u, count(t) AS txn_count
WHERE txn_count > 20
RETURN u.user_id, txn_count
ORDER BY txn_count DESC;

// Device sharing between fraudsters
MATCH (u:User)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:User)
WHERE u.is_fraudster AND other.is_fraudster AND u <> other
RETURN d.device_id, collect(u.user_id) AS fraudulent_users;
```

## Feature Engineering

27 graph-based features extracted (no data leakage):

**User Attributes** (4): age, account_age_days, credit_score, account_type_encoded

**Degree Features** (3): in_degree, out_degree, total_degree

**Transaction Features** (6): txn_count_out/in/total, avg_txn_amount_out/in, total_amount_out/in

**Behavioral Features** (2): max_to_avg_amount_ratio, bidirectional_partners, circular_paths

**Neighbor Features** (1): neighbor_count

**Device Features** (3): device_count, avg_users_per_device, max_users_per_device

**IP Features** (3): ip_count, avg_users_per_ip, max_users_per_ip

**Top Features by Importance:**
1. total_amount_in: 16.32%
2. avg_txn_amount_in: 8.49%
3. out_degree: 7.00%
4. bidirectional_partners: 4.68%

**Note**: Fraud-specific features (fraud_txn_count, fraud_amount, fraud_neighbor_ratio) were removed to eliminate data leakage. All features are now derived from observable behavioral patterns.

## Model Training

XGBoost configuration optimized for fraud detection:

```yaml
# config/training_config.yaml
model:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 200
  scale_pos_weight: 0.733  # Handle class imbalance
  subsample: 0.8
  colsample_bytree: 0.8

metrics:
  primary: roc_auc
  secondary: f1_score
  acceptance:
    roc_auc_min: 0.80  # Realistic target without data leakage
    f1_min: 0.75
```

**Current Performance (No Data Leakage):**
- ROC-AUC: 0.825
- F1-Score: 0.489
- Precision: 0.428
- Recall: 0.570

**Note**: Performance is realistic for behavioral pattern detection without ground truth leakage. F1 score can be improved through feature engineering and parameter tuning while maintaining data integrity.

## Monitoring

### Grafana Dashboard

8 panels tracking fraud detection metrics:
- Total predictions and fraud vs legitimate pie chart
- Prediction rate over time (per minute)
- Fraud score distribution heatmap
- Inference duration percentiles (p50, p95, p99)
- Fraud detection rate percentage
- Average fraud score gauge with thresholds
- High-risk predictions count (score > 0.9)

Import dashboard: `config/grafana/fraud_detection_dashboard.json`

### Prometheus Metrics

```python
# Exported metrics from inference service
fraud_predictions_total{prediction="fraud|legitimate"}  # Counter
fraud_prediction_score  # Histogram (0.0-1.0)
fraud_inference_duration_seconds  # Histogram
```

## Development

### Code Quality

```bash
# Format code (4 spaces, line length 100)
black src/ tests/ --line-length 100

# Lint and fix
ruff check src/ tests/ --fix

# Type check
mypy src/

# Run tests
pytest tests/ --cov=src --cov-report=html
```

### Git Workflow

Conventional commits format:

```bash
feat: add fraud ring detection query
fix: correct datetime parsing in Neo4j loader
docs: update README with inference instructions
chore: update dependencies in uv.lock
```

## Roadmap & TODO

### ✅ Phase 1: Local Development (Complete)
- [x] Synthetic data generation with realistic noise
- [x] EDA with marimo notebooks
- [x] Neo4j deployment and data loading
- [x] Feature engineering from graph (no leakage)
- [x] Model training with MLflow
- [x] Batch inference pipeline
- [x] Prometheus/Grafana monitoring

### 🚧 Phase 2: Cloud Infrastructure (Pending)
- [ ] **Infrastructure as Code**: Implement Terraform for GCP resource provisioning
- [ ] **Data Storage**: Setup GCS buckets for raw data and model artifacts
- [ ] **Graph Database**: Deploy Neo4j AuraDB or self-hosted Neo4j on GCE
- [ ] **Container Registry**: Push Docker images to Artifact Registry

### 🚀 Phase 3: Production Serving (Pending)
- [ ] **Model Serving**: Deploy model to Vertex AI or Cloud Run
- [ ] **Batch Processing**: Orchestrate data pipeline with Cloud Workflows/Airflow
- [ ] **Automation**: Configure Cloud Scheduler for periodic retraining
- [ ] **CI/CD**: Setup GitHub Actions for automated testing and deployment

## Configuration Files

### training_config.yaml
XGBoost hyperparameters, metrics, acceptance criteria

### inference_config.yaml
Batch inference settings, Neo4j connection, output paths

### feature_config.yaml
Graph algorithms (degree, pagerank, betweenness), temporal features, network features

### prometheus.yml
Scrape targets: fraud-inference (8000), neo4j (2004), prometheus (9090)

## Performance

### Data Pipeline
- **Data Generation**: 100K transactions in ~2s
- **Neo4j Loading**: 10K users + 100K txns in ~16s
- **Feature Extraction**: 29 features for 10K users in ~8s

### Model Training
- **Training Time**: ~1s for 200 estimators
- **Test Set**: 2,000 users (20% split, stratified)
- **Metrics**: 100% precision, recall, F1, ROC-AUC

### Inference
- **Throughput**: 10,000 predictions in 0.01s (1M predictions/sec)
- **Latency**: <1ms per prediction
- **Memory**: <100MB for model

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j status
docker-compose ps neo4j

# View Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### Import Directory Not Found

```bash
# Create import directory
mkdir -p neo4j_data/import

# Copy CSV files
cp data/raw/*.csv neo4j_data/import/
```

### MLflow Database Locked

```bash
# Use filesystem backend
mlflow ui --backend-store-uri ./mlruns

# Or switch to SQLite
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Neo4j Browser | http://localhost:7474 | neo4j / frauddetection123 |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / frauddetection123 |
| Inference Metrics | http://localhost:8000/metrics | None |
| MLflow UI | `mlflow ui` then http://localhost:5000 | None |

## Repository

https://github.com/Anticiparte/graph-fraud

## License

MIT License

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Follow code quality standards (black, ruff, mypy)
3. Add tests for new features
4. Use conventional commits
5. Push and create pull request

For questions or issues, open a GitHub issue.

## Data Leakage Prevention

This project implements strict measures to prevent data leakage:

### Removed Features
The following features were **removed** as they used ground truth labels:
- `fraud_txn_count_in/out`: Counted transactions where `is_fraud=True`
- `fraud_amount_in/out`: Summed amounts where `is_fraud=True`
- `fraud_neighbor_ratio`: Ratio of neighbors where `is_fraudster=True`

### Behavioral Features Only
All features are derived from **observable patterns**:
- Transaction amounts and frequencies
- Network connectivity (degree, bidirectional partners)
- Temporal patterns (max-to-avg ratios)
- Device and IP sharing patterns
- Circular transaction paths

### Realistic Data Generation
- **Mixed behavior**: Fraudsters perform 30% legitimate transactions
- **Fraud-like legitimate users**: Power users, business networks, money pooling groups
- **Noise injection**: Amount variations, timestamp jitter, incomplete patterns
- **Label integrity**: Fraud labels based on user identity, not transaction participation
