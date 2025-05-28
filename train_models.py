#!/usr/bin/env python3
"""
Train unsupervised anomaly detection models on extracted features.
This script handles model training, hyperparameter optimization, and model saving.
"""

import os
import logging
import numpy as np
import pandas as pd
import glob
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

# Configure logging
logger = logging.getLogger("anomaly_detector.models")
console = Console()


class ImprovedAutoencoder:
    """Improved Autoencoder for anomaly detection with better architecture"""

    def __init__(self, hidden_layer_sizes=None, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, X_val=None):
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_train)

        n_features = X_scaled.shape[1]
        if self.hidden_layer_sizes is None:
            # Create a more sophisticated architecture
            encoding_dim = max(int(n_features * 0.3), 2)  # Smaller bottleneck
            hidden1 = max(int(n_features * 0.7), 3)
            hidden2 = max(int(n_features * 0.5), 2)
            self.hidden_layer_sizes = (hidden1, hidden2, encoding_dim, hidden2, hidden1)

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='tanh',  # tanh often works better than relu for autoencoders
            solver='adam',
            alpha=0.001,  # Increased regularization
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.random_state
        )

        self.model.fit(X_scaled, X_scaled)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        X_pred = self.model.predict(X_scaled)
        # Calculate reconstruction error
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)
        return mse  # Higher values indicate anomalies


def train_isolation_forest(X_train, contamination=0.1):
    """Train Isolation Forest with improved parameters"""
    console.print("[cyan]Training Isolation Forest model...[/cyan]")

    # Adaptive parameters based on dataset size
    n_samples = X_train.shape[0]

    if n_samples < 100:
        n_estimators = 150
        max_samples = min(64, n_samples)
    elif n_samples < 500:
        n_estimators = 200
        max_samples = min(128, n_samples)
    else:
        n_estimators = 300
        max_samples = 256

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=0.8,  # Use 80% of features per tree
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train)
    return model


def train_one_class_svm(X_train, contamination=0.1):
    """Train One-Class SVM with improved parameters"""
    console.print("[cyan]Training One-Class SVM model...[/cyan]")

    # Use RBF kernel with auto gamma for better performance
    model = OneClassSVM(
        nu=contamination,
        kernel='rbf',  # RBF kernel generally works better
        gamma='scale',  # Use scale for better performance
        cache_size=500
    )

    model.fit(X_train)
    return model


def train_autoencoder(X_train, X_val=None):
    """Train improved autoencoder model"""
    console.print("[cyan]Training Autoencoder model...[/cyan]")

    autoencoder = ImprovedAutoencoder(random_state=42)
    autoencoder.fit(X_train, X_val)
    return autoencoder


def train_local_outlier_factor(X_train, contamination=0.1):
    """Train LOF with improved parameters"""
    console.print("[cyan]Training Local Outlier Factor model...[/cyan]")

    n_samples = X_train.shape[0]

    # Adaptive number of neighbors
    if n_samples < 50:
        n_neighbors = max(5, n_samples // 10)
    elif n_samples < 200:
        n_neighbors = 20
    else:
        n_neighbors = 30

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        algorithm='auto',
        leaf_size=30,
        n_jobs=-1
    )

    model.fit(X_train)
    return model


def optimize_threshold(y_true, scores):
    """Find optimal threshold that maximizes F1 score"""
    if y_true is None or len(np.unique(y_true)) < 2:
        # Default to 90th percentile if no labels
        return np.percentile(scores, 90)

    best_f1 = 0
    best_threshold = np.percentile(scores, 90)

    # Try different percentiles
    for percentile in range(70, 99):
        threshold = np.percentile(scores, percentile)
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def evaluate_model(model, X_test, y_test=None, model_type='isolation_forest'):
    """Improved model evaluation with better scoring"""
    results = {}

    # Get raw anomaly scores based on model type
    if model_type == 'isolation_forest':
        raw_scores = model.decision_function(X_test)
        # For isolation forest, negative scores indicate anomalies
        # Convert to positive anomaly scores (higher = more anomalous)
        scores = -raw_scores

    elif model_type == 'one_class_svm':
        raw_scores = model.decision_function(X_test)
        # For One-Class SVM, negative scores indicate anomalies
        scores = -raw_scores

    elif model_type == 'autoencoder':
        # For autoencoder, higher reconstruction error = more anomalous
        scores = model.decision_function(X_test)

    elif model_type == 'local_outlier_factor':
        raw_scores = model.decision_function(X_test)
        # For LOF, negative scores indicate outliers
        scores = -raw_scores
    else:
        scores = np.random.random(X_test.shape[0])  # Fallback
        logger.warning(f"Unknown model type: {model_type}")

    # Ensure scores are positive and well-scaled
    scores = np.maximum(scores, 0)  # Ensure non-negative
    if scores.max() > scores.min():
        # Normalize to 0-1 range for consistency
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    results['anomaly_scores'] = scores

    # Find optimal threshold if labels are available
    if y_test is not None:
        optimal_threshold = optimize_threshold(y_test, scores)
        predictions = (scores >= optimal_threshold).astype(int)
        results['predictions'] = predictions
        results['threshold'] = optimal_threshold

        try:
            # Calculate metrics
            results['roc_auc'] = roc_auc_score(y_test, scores)
            precision, recall, _ = precision_recall_curve(y_test, scores)
            results['pr_auc'] = auc(recall, precision)
            results['f1'] = f1_score(y_test, predictions, zero_division=0)

            # Calculate detection statistics
            results['true_positives'] = np.sum((y_test == 1) & (predictions == 1))
            results['false_positives'] = np.sum((y_test == 0) & (predictions == 1))
            results['true_negatives'] = np.sum((y_test == 0) & (predictions == 0))
            results['false_negatives'] = np.sum((y_test == 1) & (predictions == 0))

        except Exception as e:
            logger.warning(f"Could not calculate evaluation metrics: {str(e)}")
            results['roc_auc'] = None
            results['pr_auc'] = None
            results['f1'] = None
    else:
        # Use default threshold (95th percentile for conservative detection)
        threshold = np.percentile(scores, 95)
        predictions = (scores >= threshold).astype(int)
        results['predictions'] = predictions
        results['threshold'] = threshold

    return results


def create_improved_ensemble(models_results, X_test, y_test=None):
    """Create an improved ensemble with adaptive weighting"""
    if len(models_results) < 2:
        return None

    # Collect all normalized scores
    all_scores = []
    model_names = []
    model_performance = {}

    for model_name, results in models_results.items():
        scores = results['anomaly_scores']
        all_scores.append(scores)
        model_names.append(model_name)

        # Calculate model performance weight
        if y_test is not None and 'f1' in results and results['f1'] is not None:
            model_performance[model_name] = results['f1']
        else:
            # Use diversity-based weighting when no labels available
            model_performance[model_name] = 1.0

    all_scores = np.array(all_scores)

    # Calculate model weights based on performance and diversity
    if y_test is not None and any(perf > 0 for perf in model_performance.values()):
        # Performance-based weighting
        total_performance = sum(model_performance.values())
        weights = [model_performance[name] / total_performance for name in model_names]
    else:
        # Equal weighting with slight preference for ensemble diversity
        base_weights = {
            'isolation_forest': 0.3,
            'one_class_svm': 0.25,
            'autoencoder': 0.25,
            'local_outlier_factor': 0.2
        }
        weights = [base_weights.get(name, 1.0 / len(model_names)) for name in model_names]
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

    # Create weighted ensemble
    ensemble_scores = np.zeros(len(X_test))
    for i, (scores, weight) in enumerate(zip(all_scores, weights)):
        ensemble_scores += scores * weight

    # Apply final normalization
    if ensemble_scores.max() > ensemble_scores.min():
        ensemble_scores = (ensemble_scores - ensemble_scores.min()) / \
                          (ensemble_scores.max() - ensemble_scores.min())

    return ensemble_scores, dict(zip(model_names, weights))


def train_models_main(output_dir, models_list=None):
    """Main function for improved model training pipeline"""
    if models_list is None:
        models_list = ['isolation_forest', 'one_class_svm', 'autoencoder', 'local_outlier_factor']

    # Create necessary directories
    os.makedirs(f"{output_dir}/models", exist_ok=True)

    # Load preprocessed data
    console.print("[cyan]Loading preprocessed data...[/cyan]")
    try:
        X_train = np.load(f"{output_dir}/features/X_train.npy")
        X_test = np.load(f"{output_dir}/features/X_test.npy")
        X_val = np.load(f"{output_dir}/features/X_val.npy")

        # Load labels if available
        try:
            y_test = np.load(f"{output_dir}/features/y_test.npy")
            y_val = np.load(f"{output_dir}/features/y_val.npy")
        except:
            y_test = None
            y_val = None
            logger.info("Test labels not found. Using unsupervised evaluation.")

    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        console.print(f"[red]Error loading preprocessed data: {str(e)}[/red]")
        return

    # Calculate contamination rate from actual data distribution
    if y_test is not None:
        contamination = max(0.05, min(0.3, np.mean(y_test)))  # Between 5% and 30%
    else:
        # Try to estimate from file counts
        try:
            misconfig_count = len([f for f in glob.glob('data/terraform/misconfig/*.tf')])
            correct_count = len([f for f in glob.glob('data/terraform/correct/*.tf')])
            total_count = misconfig_count + correct_count
            if total_count > 0:
                contamination = min(0.3, max(0.05, misconfig_count / total_count))
            else:
                contamination = 0.1
        except:
            contamination = 0.1

    logger.info(f"Using contamination rate: {contamination:.3f}")

    # Train models
    models_results = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        training_task = progress.add_task("[cyan]Training models...", total=len(models_list))

        for model_type in models_list:
            try:
                # Train the model
                if model_type == 'isolation_forest':
                    model = train_isolation_forest(X_train, contamination)
                elif model_type == 'one_class_svm':
                    model = train_one_class_svm(X_train, contamination)
                elif model_type == 'autoencoder':
                    model = train_autoencoder(X_train, X_val)
                elif model_type == 'local_outlier_factor':
                    model = train_local_outlier_factor(X_train, contamination)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue

                # Evaluate model
                evaluation = evaluate_model(model, X_test, y_test, model_type)

                # Save model
                if model_type == 'autoencoder':
                    # Save autoencoder components
                    import pickle
                    with open(f"{output_dir}/models/{model_type}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                else:
                    pd.to_pickle(model, f"{output_dir}/models/{model_type}.pkl")

                # Save results
                np.save(f"{output_dir}/models/{model_type}_scores.npy", evaluation['anomaly_scores'])
                if 'predictions' in evaluation:
                    np.save(f"{output_dir}/models/{model_type}_predictions.npy", evaluation['predictions'])

                # Store results
                models_results[model_type] = evaluation

                # Log performance
                if y_test is not None and 'f1' in evaluation and evaluation['f1'] is not None:
                    logger.info(f"{model_type} - F1: {evaluation['f1']:.4f}, "
                                f"ROC AUC: {evaluation.get('roc_auc', 0):.4f}, "
                                f"Threshold: {evaluation.get('threshold', 0):.4f}")

                    # Log detection statistics
                    tp = evaluation.get('true_positives', 0)
                    fp = evaluation.get('false_positives', 0)
                    tn = evaluation.get('true_negatives', 0)
                    fn = evaluation.get('false_negatives', 0)

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    logger.info(f"{model_type} - Precision: {precision:.4f}, Recall: {recall:.4f}")
                    logger.info(f"{model_type} - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

                progress.update(training_task, advance=1)

            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                progress.update(training_task, advance=1)
                continue

    # Create improved ensemble
    if len(models_results) > 1:
        console.print("[cyan]Creating improved ensemble...[/cyan]")
        try:
            ensemble_result = create_improved_ensemble(models_results, X_test, y_test)
            if ensemble_result is not None:
                ensemble_scores, ensemble_weights = ensemble_result

                # Evaluate ensemble
                ensemble_evaluation = {'anomaly_scores': ensemble_scores}

                if y_test is not None:
                    optimal_threshold = optimize_threshold(y_test, ensemble_scores)
                    ensemble_predictions = (ensemble_scores >= optimal_threshold).astype(int)

                    ensemble_evaluation.update({
                        'predictions': ensemble_predictions,
                        'threshold': optimal_threshold,
                        'roc_auc': roc_auc_score(y_test, ensemble_scores),
                        'f1': f1_score(y_test, ensemble_predictions, zero_division=0)
                    })

                    precision, recall, _ = precision_recall_curve(y_test, ensemble_scores)
                    ensemble_evaluation['pr_auc'] = auc(recall, precision)

                # Save ensemble results
                np.save(f"{output_dir}/models/ensemble_scores.npy", ensemble_scores)
                if 'predictions' in ensemble_evaluation:
                    np.save(f"{output_dir}/models/ensemble_predictions.npy", ensemble_evaluation['predictions'])

                # Save ensemble weights
                pd.DataFrame([ensemble_weights]).to_csv(f"{output_dir}/models/ensemble_weights.csv", index=False)

                # Log ensemble performance
                if y_test is not None and 'f1' in ensemble_evaluation:
                    logger.info(f"Ensemble - F1: {ensemble_evaluation['f1']:.4f}, "
                                f"ROC AUC: {ensemble_evaluation.get('roc_auc', 0):.4f}")
                    logger.info(f"Ensemble weights: {ensemble_weights}")

                models_results['ensemble'] = ensemble_evaluation

        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")

    console.print(f"[green]Model training completed. Trained {len(models_results)} models.[/green]")

    # Print summary
    if y_test is not None:
        console.print("\n[bold]Model Performance Summary:[/bold]")
        for model_name, results in models_results.items():
            if 'f1' in results and results['f1'] is not None:
                f1_score = results['f1']
                roc_auc = results.get('roc_auc', 0)
                console.print(f"  {model_name}: F1={f1_score:.4f}, ROC AUC={roc_auc:.4f}")


if __name__ == "__main__":
    # If run directly, use default path
    output_dir = "data/output"
    train_models_main(output_dir)