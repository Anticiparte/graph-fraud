"""
Train fraud detection model using XGBoost with MLflow tracking.

Optimizes for Precision (primary) and F1-score (secondary) metrics.
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


class FraudDetectionTrainer:
    """Train and evaluate fraud detection model."""

    def __init__(
        self,
        features_path: Path,
        model_output_path: Path,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            features_path: Path to feature CSV
            model_output_path: Path to save trained model
            test_size: Test set proportion
            random_state: Random seed for reproducibility
        """
        self.features_path = features_path
        self.model_output_path = model_output_path
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        """Load features and prepare train/test split."""
        print("=" * 70)
        print("LOADING AND PREPARING DATA")
        print("=" * 70)

        # Load features
        df = pd.read_csv(self.features_path)
        print(f"\nLoaded {len(df)} records with {len(df.columns)} columns")

        # Separate features and target
        target_col = "is_fraudster"
        exclude_cols = ["user_id", "is_fraudster", "account_type"]
        self.feature_cols = [
            col for col in df.columns if col not in exclude_cols
        ]

        X = df[self.feature_cols]
        y = df[target_col].astype(int)

        print(f"\nFeature set: {len(self.feature_cols)} features")
        print(f"Target distribution:")
        print(f"  Legitimate: {(y == 0).sum():,} ({(y == 0).mean():.2%})")
        print(f"  Fraudulent: {(y == 1).sum():,} ({(y == 1).mean():.2%})")

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTrain set: {len(self.X_train):,} samples")
        print(f"Test set: {len(self.X_test):,} samples")

    def train_model(self, params: dict = None):
        """
        Train XGBoost model.

        Args:
            params: Optional XGBoost parameters
        """
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST MODEL")
        print("=" * 70)

        # Default parameters optimized for fraud detection
        if params is None:
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

            params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": scale_pos_weight,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
            }

        print("\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False,
        )

        print("\nTraining complete!")

    def evaluate_model(self):
        """Evaluate model and return metrics."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        # Print metrics
        print("\nTest Set Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:,}  |  FP: {cm[0, 1]:,}")
        print(f"  FN: {cm[1, 0]:,}  |  TP: {cm[1, 1]:,}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return metrics, feature_importance

    def save_model(self):
        """Save trained model."""
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_output_path))
        print(f"\nModel saved to: {self.model_output_path}")

    def train_and_evaluate(self):
        """Execute full training pipeline with MLflow tracking."""
        # Start MLflow run
        mlflow.set_experiment("fraud-detection")

        with mlflow.start_run():
            # Load data
            self.load_and_prepare_data()

            # Log data parameters
            mlflow.log_param("n_features", len(self.feature_cols))
            mlflow.log_param("n_train", len(self.X_train))
            mlflow.log_param("n_test", len(self.X_test))
            mlflow.log_param("fraud_rate", self.y_train.mean())

            # Train model
            self.train_model()

            # Log model parameters
            for key, value in self.model.get_params().items():
                mlflow.log_param(key, value)

            # Evaluate model
            metrics, feature_importance = self.evaluate_model()

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Save model
            self.save_model()

            # Log model to MLflow
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name="fraud-detector",
            )

            # Log feature importance
            feature_importance_path = self.model_output_path.parent / "feature_importance.csv"
            feature_importance.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(str(feature_importance_path))

            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"Model registered as: fraud-detector")

            return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/graph_features.csv",
        help="Path to feature CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/fraud_detector.json",
        help="Path to save model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    features_path = project_root / args.features
    model_output_path = project_root / args.output

    # Train model
    trainer = FraudDetectionTrainer(
        features_path=features_path,
        model_output_path=model_output_path,
        test_size=args.test_size,
    )

    metrics = trainer.train_and_evaluate()

    # Check acceptance criteria
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA CHECK")
    print("=" * 70)
    print(f"  Precision: {metrics['precision']:.4f} (target: > 0.85)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} (target: > 0.75)")

    if metrics["precision"] > 0.85 and metrics["f1_score"] > 0.75:
        print("\n✓ Model meets acceptance criteria!")
    else:
        print("\n⚠ Model does not meet acceptance criteria")


if __name__ == "__main__":
    main()
