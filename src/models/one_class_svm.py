#!/usr/bin/env python3
"""
One-Class SVM model for unsupervised anomaly detection in S3 bucket configurations.
"""

import numpy as np
from sklearn.svm import OneClassSVM
import logging

logger = logging.getLogger("anomaly_detector.models.one_class_svm")

def train_one_class_svm_model(X_train, contamination=0.1, kernel='rbf'):
    """
    Train a One-Class SVM model for anomaly detection.

    Args:
        X_train: Feature matrix for training
        contamination: Expected proportion of anomalies
        kernel: Kernel type (rbf, linear, poly)

    Returns:
        Trained One-Class SVM model
    """
    # Set nu parameter based on contamination (nu is the upper bound on fraction of outliers)
    nu = contamination

    logger.info(f"Training One-Class SVM with nu={nu}, kernel={kernel}")

    # Gamma parameter often needs tuning
    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma='auto',  # Changed from 'scale' to 'auto'
        cache_size=500  # Increase cache size for faster training
    )

    model.fit(X_train)
    return model

def predict_anomalies(model, X):
    """
    Predict anomalies using a trained One-Class SVM model.

    Args:
        model: Trained One-Class SVM model
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