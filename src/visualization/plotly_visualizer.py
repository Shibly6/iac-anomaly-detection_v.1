#!/usr/bin/env python3
"""
Plotly visualization utilities for anomaly detection.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from .marker_utils import prepare_marker_sizes

logger = logging.getLogger("anomaly_detector.visualization.plotly_visualizer")


def create_scatter_plot(x, y, color=None, size=None, title='Scatter Plot',
                        x_label='X', y_label='Y', color_label='Color',
                        output_path=None):
    """
    Create a basic scatter plot with Plotly.

    Args:
        x: X-axis values
        y: Y-axis values
        color: Values for coloring points
        size: Values for sizing points (will be processed for positive values)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        color_label: Color scale label
        output_path: Path to save the plot

    Returns:
        Plotly figure object
    """
    # Process marker sizes if provided
    marker_size = 8  # Default size
    if size is not None:
        marker_size = prepare_marker_sizes(size, min_size=5, max_size=20)

    # Create figure
    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color,
            colorscale='Viridis',
            showscale=color is not None,
            colorbar=dict(title=color_label) if color is not None else None
        )
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white'
    )

    # Save if output path is provided
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
        except Exception as e:
            logger.warning(f"Could not save Plotly plot to {output_path}: {str(e)}")

    return fig


def create_tsne_plot(X, anomaly_scores, predictions=None, model_name='Model', output_dir=None):
    """
    Create t-SNE visualization with Plotly.

    Args:
        X: Feature matrix
        anomaly_scores: Anomaly scores for sizing/coloring points
        predictions: Binary predictions for highlighting anomalies
        model_name: Name of the model
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    # Apply t-SNE
    try:
        perplexity = min(30, X.shape[0] - 1)  # Perplexity must be < n_samples - 1
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(X)

        # Process marker sizes (ensure positive values)
        marker_size = prepare_marker_sizes(anomaly_scores, min_size=5, max_size=20)

        # Create color array
        if predictions is not None:
            colors = ['normal' if p == 0 else 'anomaly' for p in predictions]
        else:
            colors = anomaly_scores

        # Create figure
        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=colors,
                colorscale='Viridis' if predictions is None else None,
                showscale=predictions is None
            ),
            text=[f"Score: {score:.3f}" for score in anomaly_scores],
            hoverinfo='text'
        ))

        # Update layout
        fig.update_layout(
            title=f't-SNE Visualization - {model_name}',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            template='plotly_white'
        )

        # Save if output directory is provided
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                fig.write_html(os.path.join(output_dir, f"{model_name}_tsne.html"))
            except Exception as e:
                logger.warning(f"Could not save t-SNE plot: {str(e)}")

        return fig

    except Exception as e:
        logger.warning(f"Could not create t-SNE visualization: {str(e)}")
        return None


def create_pca_plot(X, anomaly_scores, predictions=None, model_name='Model', output_dir=None):
    """
    Create PCA visualization with Plotly.

    Args:
        X: Feature matrix
        anomaly_scores: Anomaly scores for sizing/coloring points
        predictions: Binary predictions for highlighting anomalies
        model_name: Name of the model
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    # Apply PCA
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        # Process marker sizes (ensure positive values)
        marker_size = prepare_marker_sizes(anomaly_scores, min_size=5, max_size=20)

        # Create color array
        if predictions is not None:
            colors = ['normal' if p == 0 else 'anomaly' for p in predictions]
        else:
            colors = anomaly_scores

        # Create figure
        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=colors,
                colorscale='Viridis' if predictions is None else None,
                showscale=predictions is None
            ),
            text=[f"Score: {score:.3f}" for score in anomaly_scores],
            hoverinfo='text'
        ))

        # Update layout
        fig.update_layout(
            title=f'PCA Visualization - {model_name}',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
            template='plotly_white'
        )

        # Save if output directory is provided
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                fig.write_html(os.path.join(output_dir, f"{model_name}_pca.html"))
            except Exception as e:
                logger.warning(f"Could not save PCA plot: {str(e)}")

        return fig

    except Exception as e:
        logger.warning(f"Could not create PCA visualization: {str(e)}")
        return None


def create_model_comparison_plot(models_results, y_true=None, output_dir=None):
    """
    Create model comparison visualizations with Plotly.

    Args:
        models_results: Dictionary of model results
        y_true: True labels (if available)
        output_dir: Directory to save plots

    Returns:
        Dictionary of Plotly figure objects
    """
    figures = {}

    # Create comparison bar chart
    try:
        model_names = list(models_results.keys())
        scores = [models_results[name].get('score', 0) for name in model_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=scores,
            marker_color='lightblue'
        ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Performance Score',
            template='plotly_white'
        )

        figures['performance'] = fig

        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                fig.write_html(os.path.join(output_dir, "model_comparison.html"))
            except Exception as e:
                logger.warning(f"Could not save model comparison plot: {str(e)}")

    except Exception as e:
        logger.warning(f"Could not create model comparison: {str(e)}")

    return figures