#!/usr/bin/env python3
"""
Feature visualization utilities for anomaly detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger("anomaly_detector.visualization.feature_visualizer")


def plot_feature_distributions(df, feature_names, output_dir, hue_col='assumed_anomalous'):
    """
    Create distribution plots for selected features.

    Args:
        df: DataFrame with features
        feature_names: List of feature names to plot
        output_dir: Directory to save plots
        hue_col: Column to use for coloring data points
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select top features to plot (max 6)
    if len(feature_names) > 6:
        plot_features = feature_names[:6]
    else:
        plot_features = feature_names

    # Create individual histograms
    for feature in plot_features:
        if feature not in df.columns:
            continue

        plt.figure(figsize=(8, 6))

        # Check feature type
        if df[feature].dtype in [np.int64, np.float64]:
            # For numeric features, create histplot
            sns.histplot(
                data=df, x=feature, hue=hue_col,
                kde=True, element='step', common_norm=False
            )
        else:
            # For categorical features, create countplot
            sns.countplot(
                data=df, x=feature, hue=hue_col
            )

        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"), dpi=300)
        plt.close()

    # Create correlation heatmap
    try:
        numeric_df = df.select_dtypes(include=[np.int64, np.float64])

        plt.figure(figsize=(12, 10))
        correlation = numeric_df.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))

        sns.heatmap(
            correlation, mask=mask, cmap='coolwarm', annot=False,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}
        )

        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create correlation heatmap: {str(e)}")


def plot_pca_visualization(X, labels, feature_names, output_dir, n_components=2):
    """
    Create PCA visualization of features.

    Args:
        X: Feature matrix
        labels: Labels for coloring points
        feature_names: List of feature names
        output_dir: Directory to save plots
        n_components: Number of PCA components
    """
    os.makedirs(output_dir, exist_ok=True)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Create PCA scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Label')
    plt.title('PCA Visualization of Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_visualization.png"), dpi=300)
    plt.close()

    # Create PCA with feature loadings
    plt.figure(figsize=(12, 10))

    # Plot points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=70)

    # Plot feature loadings (arrows)
    plot_features = feature_names[:min(len(feature_names), 10)]  # Limit to top 10 features for clarity

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(plot_features):
        if i < len(loadings):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.5)
            plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color='red', ha='center', va='center')

    plt.title('PCA Visualization with Feature Loadings')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_with_loadings.png"), dpi=300)
    plt.close()

    # Create feature importance based on PCA loadings
    feature_importance = np.abs(pca.components_).sum(axis=0)

    # Plot feature importance
    plt.figure(figsize=(12, 8))

    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    features = [feature_names[i] for i in indices[:15]]  # Top 15 features
    importance = feature_importance[indices[:15]]

    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance from PCA')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()


def plot_tsne_visualization(X, labels, output_dir):
    """
    Create t-SNE visualization of features.

    Args:
        X: Feature matrix
        labels: Labels for coloring points
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only attempt t-SNE if dataset is not too large
    if X.shape[0] > 5000:
        logger.warning("Dataset too large for t-SNE visualization. Skipping.")
        return

    try:
        # Apply t-SNE
        perplexity = min(30, X.shape[0] - 1)  # Perplexity must be < n_samples - 1
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)

        # Create t-SNE scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Label')
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create t-SNE visualization: {str(e)}")
        logger.warning(f"Could not create t-SNE visualization: {str(e)}")