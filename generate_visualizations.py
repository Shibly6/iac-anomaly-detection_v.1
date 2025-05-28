#!/usr/bin/env python3
"""
Generate visualizations for anomaly detection results.
This script creates professional-quality diagrams for analysis with proper error handling.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger("anomaly_detector.visualization")
console = Console()

# Set plotting style
plt.style.use('default')
sns.set_palette("viridis")
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def prepare_marker_sizes(values, min_size=5, max_size=20):
    """Convert values to positive marker sizes suitable for visualization."""
    values = np.array(values)

    # Ensure all values are positive
    positive_values = np.abs(values)

    # If all values are very close to 0, use a default size
    if np.all(positive_values < 0.001):
        return [min_size] * len(values)

    # Scale values to desired range
    min_val = np.min(positive_values)
    max_val = np.max(positive_values)
    range_val = max_val - min_val

    if range_val < 0.001:  # Handle case where all values are identical
        return [min_size] * len(values)

    # Scale to desired size range
    sizes = (positive_values - min_val) / range_val * (max_size - min_size) + min_size

    return sizes.tolist()


def safe_load_data(file_path, default_value=None):
    """Safely load numpy data with error handling."""
    try:
        data = np.load(file_path)
        # Check for NaN values
        if np.isnan(data).any():
            logger.warning(f"NaN values found in {file_path}, filling with zeros")
            data = np.nan_to_num(data, nan=0.0)
        return data
    except Exception as e:
        logger.warning(f"Could not load {file_path}: {str(e)}")
        return default_value


def clean_data_for_visualization(X, y=None):
    """Clean data for visualization by handling NaN and infinite values."""
    # Handle NaN values
    if np.isnan(X).any():
        logger.warning("NaN values found in visualization data, filling with zeros")
        X = np.nan_to_num(X, nan=0.0)

    # Handle infinite values
    if np.isinf(X).any():
        logger.warning("Infinite values found in visualization data, clipping")
        X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

    if y is not None:
        if np.isnan(y).any():
            logger.warning("NaN values found in labels, filling with zeros")
            y = np.nan_to_num(y, nan=0.0).astype(int)

    return X, y


def create_feature_distribution_plots(features_df, feature_names, output_dir):
    """Create histograms for feature distributions with error handling."""
    console.print("[cyan]Creating feature distribution plots...[/cyan]")

    try:
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots/features", exist_ok=True)

        # Select features to plot (max 6)
        if len(feature_names) > 6:
            plot_features = feature_names[:6]
        else:
            plot_features = feature_names

        # Create grid of histograms
        if plot_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, feature in enumerate(plot_features):
                if i >= len(axes):
                    break

                if feature in features_df.columns:
                    try:
                        # Clean the data
                        feature_data = features_df[feature].fillna(0)

                        if 'assumed_anomalous' in features_df.columns:
                            # Create histogram with hue
                            normal_data = feature_data[features_df['assumed_anomalous'] == 0]
                            anomaly_data = feature_data[features_df['assumed_anomalous'] == 1]

                            axes[i].hist(normal_data, bins=10, alpha=0.7, label='Normal', color='blue')
                            axes[i].hist(anomaly_data, bins=10, alpha=0.7, label='Anomalous', color='red')
                            axes[i].legend()
                        else:
                            axes[i].hist(feature_data, bins=10, alpha=0.7, color='blue')

                        axes[i].set_title(f'Distribution of {feature}')
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True, alpha=0.3)

                    except Exception as e:
                        logger.warning(f"Could not plot feature {feature}: {str(e)}")
                        axes[i].text(0.5, 0.5, f'Error plotting\n{feature}',
                                     ha='center', va='center', transform=axes[i].transAxes)

            # Hide unused subplots
            for i in range(len(plot_features), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/features/feature_distributions_grid.png", dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"Error creating feature distribution plots: {str(e)}")


def create_dimension_reduction_plots(X, y, feature_names, output_dir):
    """Create PCA and t-SNE plots with robust error handling."""
    console.print("[cyan]Creating dimensionality reduction plots...[/cyan]")

    try:
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots/dimension_reduction", exist_ok=True)

        # Clean data
        X_clean, y_clean = clean_data_for_visualization(X, y)

        if X_clean.shape[1] < 2:
            logger.warning("Not enough features for dimensionality reduction")
            return

        # Create PCA plot
        try:
            n_components = min(2, X_clean.shape[1], X_clean.shape[0] - 1)
            if n_components >= 2:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_clean)

                plt.figure(figsize=(10, 8))
                if y_clean is not None:
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_clean, cmap='viridis', alpha=0.7, s=50)
                    plt.colorbar(scatter, label='Anomaly (from folder)')
                else:
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)

                plt.title('PCA Visualization of Features')
                plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/plots/dimension_reduction/pca_visualization.png", dpi=300,
                            bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.warning(f"Could not create PCA plot: {str(e)}")

        # Create t-SNE plot (only for smaller datasets)
        if X_clean.shape[0] <= 1000 and X_clean.shape[0] >= 5:
            try:
                perplexity = min(30, X_clean.shape[0] - 1, 50)
                if perplexity >= 5:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    X_tsne = tsne.fit_transform(X_clean)

                    plt.figure(figsize=(10, 8))
                    if y_clean is not None:
                        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_clean, cmap='viridis', alpha=0.7, s=50)
                        plt.colorbar(scatter, label='Anomaly (from folder)')
                    else:
                        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=50)

                    plt.title('t-SNE Visualization of Features')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/dimension_reduction/tsne_visualization.png", dpi=300,
                                bbox_inches='tight')
                    plt.close()

            except Exception as e:
                logger.warning(f"Could not create t-SNE plot: {str(e)}")

    except Exception as e:
        logger.error(f"Error creating dimension reduction plots: {str(e)}")


def create_anomaly_score_plots(anomaly_scores, y, model_name, output_dir):
    """Create plots for anomaly score distributions and evaluation curves."""
    console.print(f"[cyan]Creating anomaly score plots for {model_name}...[/cyan]")

    try:
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots/anomaly_scores", exist_ok=True)

        # Clean anomaly scores
        scores_clean = np.nan_to_num(anomaly_scores, nan=0.0)

        # Create anomaly score distribution
        plt.figure(figsize=(10, 6))

        if y is not None and len(np.unique(y)) > 1:
            y_clean = np.nan_to_num(y, nan=0.0).astype(int)

            # Separate scores by class
            normal_scores = scores_clean[y_clean == 0]
            anomaly_scores_pos = scores_clean[y_clean == 1]

            if len(normal_scores) > 0:
                plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue', density=True)
            if len(anomaly_scores_pos) > 0:
                plt.hist(anomaly_scores_pos, bins=20, alpha=0.7, label='Anomalous', color='red', density=True)

            plt.legend()
        else:
            plt.hist(scores_clean, bins=20, alpha=0.7, color='blue', density=True)

        # Add threshold line
        if len(scores_clean) > 0:
            threshold = np.percentile(scores_clean, 90)
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold (90th percentile): {threshold:.3f}')

        plt.title(f'Anomaly Score Distribution - {model_name}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/anomaly_scores/{model_name}_score_distribution.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Create ROC curve if labels are available
        if y is not None and len(np.unique(y)) > 1:
            try:
                y_clean = np.nan_to_num(y, nan=0.0).astype(int)
                fpr, tpr, thresholds = roc_curve(y_clean, scores_clean)
                roc_auc = auc(fpr, tpr)

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
                plt.savefig(f"{output_dir}/plots/anomaly_scores/{model_name}_roc_curve.png", dpi=300,
                            bbox_inches='tight')
                plt.close()

                # Create precision-recall curve
                precision, recall, _ = precision_recall_curve(y_clean, scores_clean)
                pr_auc = auc(recall, precision)

                plt.figure(figsize=(8, 8))
                plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
                plt.axhline(y=np.mean(y_clean), color='navy', lw=2, linestyle='--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {model_name}')
                plt.legend(loc="lower left")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/plots/anomaly_scores/{model_name}_pr_curve.png", dpi=300,
                            bbox_inches='tight')
                plt.close()

            except Exception as e:
                logger.warning(f"Could not create evaluation curves for {model_name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error creating anomaly score plots for {model_name}: {str(e)}")


def create_anomaly_detection_visualization(X, anomaly_scores, y, model_name, output_dir):
    """Create visualizations of detected anomalies in 2D space."""
    console.print(f"[cyan]Creating anomaly detection visualization for {model_name}...[/cyan]")

    try:
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots/anomaly_detection", exist_ok=True)

        # Clean data
        X_clean, y_clean = clean_data_for_visualization(X, y)
        scores_clean = np.nan_to_num(anomaly_scores, nan=0.0)

        if X_clean.shape[1] < 2:
            logger.warning(f"Not enough features for anomaly detection visualization for {model_name}")
            return

        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_clean)

        # Determine threshold
        threshold = np.percentile(scores_clean, 90)
        y_pred = (scores_clean >= threshold).astype(int)

        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=scores_clean, cmap='YlOrRd', alpha=0.7, s=60)
        plt.colorbar(scatter, label='Anomaly Score')

        # Highlight predicted anomalies
        anomaly_mask = y_pred == 1
        if np.any(anomaly_mask):
            plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1],
                        edgecolors='black', facecolors='none', s=120, linewidth=2,
                        label='Predicted Anomalies')

        plt.title(f'Anomaly Detection Visualization - {model_name}')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/anomaly_detection/{model_name}_detection.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create comparison plot if ground truth is available
        if y_clean is not None:
            plt.figure(figsize=(15, 7))

            # Plot 1: Predicted anomalies
            plt.subplot(1, 2, 1)
            normal_mask = y_pred == 0
            anomaly_mask = y_pred == 1

            if np.any(normal_mask):
                plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], c='blue', alpha=0.5, label='Normal')
            if np.any(anomaly_mask):
                plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], c='red', alpha=0.7, label='Anomaly')

            plt.title(f'Predicted Anomalies - {model_name}')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Plot 2: Ground truth
            plt.subplot(1, 2, 2)
            normal_mask = y_clean == 0
            anomaly_mask = y_clean == 1

            if np.any(normal_mask):
                plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], c='blue', alpha=0.5, label='Normal')
            if np.any(anomaly_mask):
                plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], c='red', alpha=0.7, label='Anomaly')

            plt.title('Ground Truth Labels')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/anomaly_detection/{model_name}_comparison.png", dpi=300,
                        bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"Error creating anomaly detection visualization for {model_name}: {str(e)}")


def create_model_comparison_plots(model_scores, y, output_dir):
    """Create plots comparing different models."""
    console.print("[cyan]Creating model comparison plots...[/cyan]")

    try:
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots/model_comparison", exist_ok=True)

        if not model_scores or len(model_scores) < 2:
            logger.warning("Not enough models for comparison")
            return

        y_clean = None
        if y is not None:
            y_clean = np.nan_to_num(y, nan=0.0).astype(int)

        # Create ROC curve comparison
        if y_clean is not None and len(np.unique(y_clean)) > 1:
            plt.figure(figsize=(10, 8))

            for model_name, scores in model_scores.items():
                try:
                    scores_clean = np.nan_to_num(scores, nan=0.0)
                    fpr, tpr, _ = roc_curve(y_clean, scores_clean)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not create ROC curve for {model_name}: {str(e)}")

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/model_comparison/roc_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Create precision-recall curve comparison
            plt.figure(figsize=(10, 8))

            for model_name, scores in model_scores.items():
                try:
                    scores_clean = np.nan_to_num(scores, nan=0.0)
                    precision, recall, _ = precision_recall_curve(y_clean, scores_clean)
                    pr_auc = auc(recall, precision)
                    plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not create PR curve for {model_name}: {str(e)}")

            plt.axhline(y=np.mean(y_clean), color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve Comparison')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/model_comparison/pr_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Create anomaly score distribution comparison
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for i, (model_name, scores) in enumerate(model_scores.items()):
            try:
                scores_clean = np.nan_to_num(scores, nan=0.0)

                # Normalize scores to 0-1 range for comparison
                if scores_clean.max() > scores_clean.min():
                    normalized_scores = (scores_clean - scores_clean.min()) / (scores_clean.max() - scores_clean.min())
                else:
                    normalized_scores = scores_clean

                color = colors[i % len(colors)]
                plt.hist(normalized_scores, bins=20, alpha=0.5, label=model_name,
                         color=color, density=True, histtype='step', linewidth=2)

            except Exception as e:
                logger.warning(f"Could not plot distribution for {model_name}: {str(e)}")

        plt.xlim([0.0, 1.0])
        plt.xlabel('Normalized Anomaly Score')
        plt.ylabel('Density')
        plt.title('Anomaly Score Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/model_comparison/score_distribution_comparison.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error creating model comparison plots: {str(e)}")


def generate_visualizations_main(output_dir, models_list=None):
    """Main function for visualization generation with robust error handling."""
    if models_list is None or not models_list:
        models_list = ['isolation_forest', 'one_class_svm', 'autoencoder', 'local_outlier_factor']

    # Ensure plots directory exists
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # Load data with error handling
    console.print("[cyan]Loading data for visualization...[/cyan]")
    try:
        # Load feature data
        features_df = None
        feature_names = []

        try:
            features_df = pd.read_csv(f"{output_dir}/features/raw_features.csv")
            feature_names_df = pd.read_csv(f"{output_dir}/features/feature_names.csv")
            feature_names = feature_names_df['feature_name'].tolist()
        except Exception as e:
            logger.warning(f"Could not load feature data: {str(e)}")

        # Load test data
        X_test = safe_load_data(f"{output_dir}/features/X_test.npy")
        y_test = safe_load_data(f"{output_dir}/features/y_test.npy")

        if X_test is None:
            logger.error("Could not load test data. Skipping visualizations.")
            return

        # Clean the loaded data
        X_test, y_test = clean_data_for_visualization(X_test, y_test)

    except Exception as e:
        logger.error(f"Error loading data for visualization: {str(e)}")
        return

    # Create feature distribution plots
    if features_df is not None and feature_names:
        create_feature_distribution_plots(features_df, feature_names, output_dir)

    # Create dimension reduction plots
    if X_test is not None:
        create_dimension_reduction_plots(X_test, y_test, feature_names, output_dir)

    # Load and visualize model results
    model_scores = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        visualization_task = progress.add_task("[cyan]Creating model visualizations...", total=len(models_list))

        for model_name in models_list:
            try:
                scores = safe_load_data(f"{output_dir}/models/{model_name}_scores.npy")

                if scores is not None:
                    model_scores[model_name] = scores

                    # Create anomaly score plots
                    create_anomaly_score_plots(scores, y_test, model_name, output_dir)

                    # Create anomaly detection visualization
                    if X_test is not None:
                        create_anomaly_detection_visualization(X_test, scores, y_test, model_name, output_dir)
                else:
                    logger.warning(f"Could not load scores for {model_name}")

                progress.update(visualization_task, advance=1)

            except Exception as e:
                logger.warning(f"Could not create visualizations for {model_name}: {str(e)}")
                progress.update(visualization_task, advance=1)

        # Load ensemble scores if available
        try:
            ensemble_scores = safe_load_data(f"{output_dir}/models/ensemble_scores.npy")
            if ensemble_scores is not None:
                model_scores['ensemble'] = ensemble_scores
                create_anomaly_score_plots(ensemble_scores, y_test, 'ensemble', output_dir)
                if X_test is not None:
                    create_anomaly_detection_visualization(X_test, ensemble_scores, y_test, 'ensemble', output_dir)
        except Exception as e:
            logger.info(f"Ensemble scores not available: {str(e)}")

    # Create model comparison plots
    if len(model_scores) > 1:
        create_model_comparison_plots(model_scores, y_test, output_dir)

    console.print(f"[green]Visualization generation completed. Created plots for {len(model_scores)} models.[/green]")


if __name__ == "__main__":
    # If run directly, use default path
    output_dir = "data/output"
    generate_visualizations_main(output_dir)