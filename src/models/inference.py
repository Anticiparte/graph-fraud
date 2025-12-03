"""
Batch inference service for fraud detection.

Loads trained model, extracts features from Neo4j, makes predictions,
and writes results back to Neo4j.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from neo4j import GraphDatabase
from prometheus_client import Counter, Histogram, start_http_server

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.graph_features import GraphFeatureExtractor

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions made",
    ["prediction"],
)
PREDICTION_SCORE = Histogram(
    "fraud_prediction_score",
    "Distribution of fraud prediction scores",
)
INFERENCE_DURATION = Histogram(
    "fraud_inference_duration_seconds",
    "Time spent on inference",
)


class FraudInferenceService:
    """Batch inference service for fraud detection."""

    def __init__(
        self,
        model_path: Path,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        """
        Initialize inference service.

        Args:
            model_path: Path to trained XGBoost model
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.model_path = model_path
        self.model = None
        self.feature_cols = None

        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None

    def load_model(self):
        """Load trained model."""
        print("Loading trained model...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        print(f"  Model loaded from: {self.model_path}")

    def connect_neo4j(self):
        """Connect to Neo4j."""
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )
        print("Connected to Neo4j")

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def extract_features(self) -> pd.DataFrame:
        """Extract features from Neo4j."""
        print("\nExtracting features from Neo4j...")
        extractor = GraphFeatureExtractor(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password,
        )
        try:
            features = extractor.extract_all_features()
            return features
        finally:
            extractor.close()

    def make_predictions(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make fraud predictions.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        print("\nMaking predictions...")
        start_time = time.time()

        # Prepare features
        exclude_cols = ["user_id", "is_fraudster", "account_type"]
        self.feature_cols = [
            col for col in features.columns if col not in exclude_cols
        ]
        X = features[self.feature_cols]

        # Make predictions
        predictions = self.model.predict(X)
        prediction_proba = self.model.predict_proba(X)[:, 1]

        # Create results DataFrame
        results = pd.DataFrame({
            "user_id": features["user_id"],
            "fraud_prediction": predictions,
            "fraud_score": prediction_proba,
            "prediction_timestamp": pd.Timestamp.now(),
        })

        # Update Prometheus metrics
        duration = time.time() - start_time
        INFERENCE_DURATION.observe(duration)

        for pred in predictions:
            PREDICTIONS_TOTAL.labels(
                prediction="fraud" if pred == 1 else "legitimate"
            ).inc()

        for score in prediction_proba:
            PREDICTION_SCORE.observe(score)

        print(f"  Predictions made: {len(results):,}")
        print(f"  Fraudulent: {(predictions == 1).sum():,} ({(predictions == 1).mean():.2%})")
        print(f"  Legitimate: {(predictions == 0).sum():,} ({(predictions == 0).mean():.2%})")
        print(f"  Inference duration: {duration:.2f}s")

        return results

    def write_predictions_to_neo4j(self, predictions: pd.DataFrame):
        """
        Write predictions back to Neo4j.

        Args:
            predictions: DataFrame with predictions
        """
        print("\nWriting predictions to Neo4j...")

        query = """
        UNWIND $predictions AS pred
        MATCH (u:User {user_id: pred.user_id})
        SET u.fraud_prediction = pred.fraud_prediction,
            u.fraud_score = pred.fraud_score,
            u.prediction_timestamp = datetime(pred.prediction_timestamp)
        """

        # Prepare data for Neo4j
        predictions_list = predictions.to_dict("records")

        with self.driver.session() as session:
            session.run(query, predictions=predictions_list)

        print(f"  Updated {len(predictions):,} user nodes with predictions")

    def run_batch_inference(self, output_path: Path = None):
        """
        Execute batch inference pipeline.

        Args:
            output_path: Optional path to save predictions CSV
        """
        print("=" * 70)
        print("FRAUD DETECTION BATCH INFERENCE")
        print("=" * 70)

        # Load model
        self.load_model()

        # Connect to Neo4j
        self.connect_neo4j()

        try:
            # Extract features
            features = self.extract_features()

            # Make predictions
            predictions = self.make_predictions(features)

            # Write predictions to Neo4j
            self.write_predictions_to_neo4j(predictions)

            # Save predictions to CSV if requested
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(output_path, index=False)
                print(f"\nPredictions saved to: {output_path}")

            print("\n" + "=" * 70)
            print("BATCH INFERENCE COMPLETE")
            print("=" * 70)

            return predictions

        finally:
            self.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run fraud detection batch inference")
    parser.add_argument(
        "--model",
        type=str,
        default="models/fraud_detector.json",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/inference/predictions.csv",
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port for Prometheus metrics",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Start Prometheus metrics server
    start_http_server(args.metrics_port)
    print(f"Prometheus metrics available at: http://localhost:{args.metrics_port}")

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / args.model
    output_path = project_root / args.output if args.output else None

    # Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "frauddetection123")

    # Run inference
    service = FraudInferenceService(
        model_path=model_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    predictions = service.run_batch_inference(output_path)

    # Print sample predictions
    print("\nSample Predictions:")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
