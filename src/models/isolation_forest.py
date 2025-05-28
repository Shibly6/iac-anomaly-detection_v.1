#!/usr/bin/env python3
"""
Isolation Forest model for unsupervised anomaly detection in S3 bucket configurations.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger("anomaly_detector.models.isolation_forest")

def train_isolation_forest_model(X_train, contamination=0.1, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection.

    Args:
        X_train: Feature matrix for training
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility

    Returns:
        Trained Isolation Forest model
    """
    # Increase number of trees and use bootstrap samples for better separation
    n_estimators = 200
    max_samples = 256 if X_train.shape[0] > 256 else "auto"

    logger.info(f"Training Isolation Forest with n_estimators={n_estimators}, max_samples={max_samples}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        bootstrap=True,  # Use bootstrap samples
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X_train)
    return model

def predict_anomalies(model, X):
    """
    Predict anomalies using a trained Isolation Forest model.

    Args:
        model: Trained Isolation Forest model
        X: Feature matrix for prediction

    Returns:
        tuple: (anomaly_scores, anomaly_predictions)
    """
    # Get anomaly scores (higher means more anomalous)
    anomaly_scores = -model.decision_function(X)

    # Get binary predictions (1 for anomalies, 0 for normal)
    # Note: predict() returns 1 for normal and -1 for anomalies, so we convert to standard format
    raw_predictions = model.predict(X)
    anomaly_predictions = np.where(raw_predictions == -1, 1, 0)

    return anomaly_scores, anomaly_predictions