#!/usr/bin/env python3
"""
Autoencoder model for unsupervised anomaly detection in S3 bucket configurations.
This implementation uses scikit-learn components instead of TensorFlow for wider compatibility.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
import logging

logger = logging.getLogger("anomaly_detector.models.autoencoder")


class SklearnAutoencoder:
    """
    Autoencoder implementation using scikit-learn's MLPRegressor.
    This serves as a TensorFlow-free alternative for anomaly detection.
    """

    def __init__(self, hidden_layer_sizes=None, activation='relu', random_state=42):
        """
        Initialize the autoencoder model.

        Args:
            hidden_layer_sizes: Sizes of hidden layers (if None, will be determined based on input size)
            activation: Activation function ('relu', 'tanh', 'logistic')
            random_state: Random seed for reproducibility
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.random_state = random_state
        self.model = None
        self.n_features = None

    def fit(self, X_train, X_val=None, max_iter=500):
        """
        Train the autoencoder model.

        Args:
            X_train: Training data
            X_val: Validation data (unused in this implementation)
            max_iter: Maximum number of training iterations

        Returns:
            Trained model
        """
        self.n_features = X_train.shape[1]

        # Determine hidden layer sizes if not provided
        if self.hidden_layer_sizes is None:
            encoding_dim = max(int(self.n_features * 0.75), 2)
            hidden_dim = max(int(self.n_features * 0.5), 1)
            self.hidden_layer_sizes = (encoding_dim, hidden_dim, encoding_dim)

        logger.info(f"Training Autoencoder with hidden layers: {self.hidden_layer_sizes}")

        # Create and train the model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver='adam',
            alpha=0.0001,
            max_iter=max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1 if X_val is None else 0.0
        )

        # Train on training data
        self.model.fit(X_train, X_train)

        return self

    def predict(self, X):
        """
        Generate reconstructions of input data.

        Args:
            X: Input data

        Returns:
            Reconstructed data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def decision_function(self, X):
        """
        Calculate reconstruction error as anomaly score.

        Args:
            X: Input data

        Returns:
            Anomaly scores (negative reconstruction error)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get reconstructions
        X_pred = self.predict(X)

        # Calculate mean squared reconstruction error
        mse = np.mean(np.square(X - X_pred), axis=1)

        # Return negative error so that higher values indicate anomalies
        # (for consistency with other models)
        return -mse


def train_autoencoder_model(X_train, X_val=None, contamination=0.1):
    """
    Train an Autoencoder model for anomaly detection.

    Args:
        X_train: Feature matrix for training
        X_val: Validation data (optional)
        contamination: Expected proportion of anomalies

    Returns:
        Trained Autoencoder model
    """
    # Create and train autoencoder
    autoencoder = SklearnAutoencoder(random_state=42)
    autoencoder.fit(X_train, X_val)

    return autoencoder


def predict_anomalies(model, X, contamination=0.1):
    """
    Predict anomalies using a trained Autoencoder model.

    Args:
        model: Trained Autoencoder model
        X: Feature matrix for prediction
        contamination: Expected proportion of anomalies

    Returns:
        tuple: (anomaly_scores, anomaly_predictions)
    """
    # Get anomaly scores (higher means more anomalous)
    anomaly_scores = -model.decision_function(X)

    # Calculate threshold based on contamination
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))

    # Get binary predictions (1 for anomalies, 0 for normal)
    anomaly_predictions = (anomaly_scores > threshold).astype(int)

    return anomaly_scores, anomaly_predictions