#!/usr/bin/env python3
"""
Anomaly visualization utilities for anomaly detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import logging

logger = logging.getLogger("anomaly_detector.visualization.anomaly_visualizer")


def plot_anomaly_score_distribution(anomaly_scores, labels, model_name, output_dir):
    """
    Create distribution plot for anomaly scores.

    Args:
        anomaly_scores: Anomaly scores from model
        labels: True labels (if available)
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # If labels are available, color by label
    if labels is not None:
        sns.histplot(
            data=pd.DataFrame({
                'anomaly_score': anomaly_scores,
                'label': labels
            }),
            x='anomaly_score', hue='label',
            kde=True, element='step', common_norm=False
        )
    else:
        sns.histplot(anomaly_scores, kde=True)

    # Add threshold line (90th percentile)
    threshold = np.percentile(anomaly_scores, 90)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (90th percentile): {threshold:.3f}')

    plt.title(f'Anomaly Score Distribution - {model_name}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_score_distribution.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, anomaly_scores, model_name, output_dir):
    """
    Create ROC curve for anomaly detection model.

    Args:
        y_true: True labels
        anomaly_scores: Anomaly scores from model
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Skip if labels are not available or all same class
    if y_true is None or len(np.unique(y_true)) < 2:
        logger.warning("Cannot create ROC curve: insufficient ground truth labels")
        return

    try:
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        roc_auc = auc(fpr, tpr)

        # Create ROC curve plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create ROC curve: {str(e)}")


def plot_precision_recall_curve(y_true, anomaly_scores, model_name, output_dir):
    """
    Create precision-recall curve for anomaly detection model.

    Args:
        y_true: True labels
        anomaly_scores: Anomaly scores from model
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Skip if labels are not available or all same class
    if y_true is None or len(np.unique(y_true)) < 2:
        logger.warning("Cannot create precision-recall curve: insufficient ground truth labels")
        return

    try:
        # Calculate precision-recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = auc(recall, precision)

        # Create precision-recall curve plot
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.axhline(y=sum(y_true) / len(y_true), color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_pr_curve.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create precision-recall curve: {str(e)}")


def plot_anomaly_detection_2d(X, anomaly_scores, anomaly_predictions, model_name, output_dir, y_true=None):
    """
    Create 2D visualization of detected anomalies.

    Args:
        X: Feature matrix
        anomaly_scores: Anomaly scores from model
        anomaly_predictions: Binary anomaly predictions
        model_name: Name of the model
        output_dir: Directory to save plots
        y_true: True labels (if available, for comparison)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create plot with anomaly scores
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=anomaly_scores, cmap='YlOrRd', alpha=0.7, s=60
    )
    plt.colorbar(scatter, label='Anomaly Score')

    # Highlight predicted anomalies
    anomalies_mask = anomaly_predictions == 1
    plt.scatter(
        X_2d[anomalies_mask, 0], X_2d[anomalies_mask, 1],
        edgecolors='black', facecolors='none', s=120, linewidth=2,
        label='Predicted Anomalies'
    )

    plt.title(f'Anomaly Detection Visualization - {model_name}')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_detection.png"), dpi=300)
    plt.close()

    # If true labels are available, create comparison plot
    if y_true is not None:
        plt.figure(figsize=(15, 7))

        # Plot 1: Predicted anomalies
        plt.subplot(1, 2, 1)
        normal_mask = anomaly_predictions == 0
        anomalies_mask = anomaly_predictions == 1

        plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], c='blue', alpha=0.5, label='Normal')
        plt.scatter(X_2d[anomalies_mask, 0], X_2d[anomalies_mask, 1], c='red', alpha=0.7, label='Anomaly')

        plt.title(f'Predicted Anomalies - {model_name}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: True labels
        plt.subplot(1, 2, 2)
        normal_mask = y_true == 0
        anomalies_mask = y_true == 1

        plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], c='blue', alpha=0.5, label='Normal')
        plt.scatter(X_2d[anomalies_mask, 0], X_2d[anomalies_mask, 1], c='red', alpha=0.7, label='Anomaly')

        plt.title('True Labels (for comparison)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_comparison.png"), dpi=300)
        plt.close()


def plot_model_comparison(models_results, y_true, output_dir):
    """
    Create comparison plots for multiple models.

    Args:
        models_results: Dictionary mapping model names to their results
        y_true: True labels
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Skip if labels are not available or all same class
    if y_true is None or len(np.unique(y_true)) < 2:
        logger.warning("Cannot create model comparison: insufficient ground truth labels")
        return

    try:
        # Create ROC curve comparison
        plt.figure(figsize=(10, 8))

        for model_name, results in models_results.items():
            anomaly_scores = results.get('anomaly_scores')
            if anomaly_scores is not None:
                fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_comparison.png"), dpi=300)
        plt.close()

        # Create precision-recall curve comparison
        plt.figure(figsize=(10, 8))

        for model_name, results in models_results.items():
            anomaly_scores = results.get('anomaly_scores')
            if anomaly_scores is not None:
                precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')

        plt.axhline(y=sum(y_true) / len(y_true), color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pr_comparison.png"), dpi=300)
        plt.close()

        # Create F1 score bar chart
        plt.figure(figsize=(10, 6))

        model_names = []
        f1_scores = []

        for model_name, results in models_results.items():
            if 'f1' in results:
                model_names.append(model_name)
                f1_scores.append(results['f1'])

        if model_names:
            sns.barplot(x=model_names, y=f1_scores)
            plt.title('F1 Score Comparison')
            plt.xlabel('Model')
            plt.ylabel('F1 Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=300)
            plt.close()
    except Exception as e:
        logger.warning(f"Could not create model comparison: {str(e)}")