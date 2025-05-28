#!/usr/bin/env python3
"""
Enhanced evaluation utilities for anomaly detection models.
Provides comprehensive evaluation metrics and threshold optimization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from scipy import stats
import logging

logger = logging.getLogger("anomaly_detector.utils.evaluation")


def evaluate_anomaly_detector(y_true, anomaly_scores, anomaly_predictions=None,
                              threshold=None, model_name="Model"):
    """
    Comprehensive evaluation of an anomaly detection model.
    """
    if y_true is None or len(np.unique(y_true)) < 2:
        logger.warning(f"Cannot evaluate {model_name}: insufficient ground truth labels")
        return create_unsupervised_evaluation(anomaly_scores, anomaly_predictions, threshold)

    results = {'model_name': model_name}

    # Ensure anomaly_scores are properly formatted
    anomaly_scores = np.array(anomaly_scores)

    # Optimize threshold if not provided
    if threshold is None:
        threshold, _ = optimize_threshold(anomaly_scores, y_true, metric='f1')
        results['optimized_threshold'] = True
    else:
        results['optimized_threshold'] = False

    results['threshold'] = threshold

    # Generate predictions if not provided
    if anomaly_predictions is None:
        anomaly_predictions = (anomaly_scores >= threshold).astype(int)

    # Core metrics
    try:
        results['roc_auc'] = roc_auc_score(y_true, anomaly_scores)
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC for {model_name}: {str(e)}")
        results['roc_auc'] = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        results['pr_auc'] = auc(recall, precision)
    except Exception as e:
        logger.warning(f"Could not compute PR AUC for {model_name}: {str(e)}")
        results['pr_auc'] = np.nan

    # Classification metrics
    results['accuracy'] = accuracy_score(y_true, anomaly_predictions)
    results['precision'] = precision_score(y_true, anomaly_predictions, zero_division=0)
    results['recall'] = recall_score(y_true, anomaly_predictions, zero_division=0)
    results['f1'] = f1_score(y_true, anomaly_predictions, zero_division=0)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, anomaly_predictions).ravel()
    results['true_positives'] = tp
    results['false_positives'] = fp
    results['true_negatives'] = tn
    results['false_negatives'] = fn

    # Additional metrics
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    results['balanced_accuracy'] = (results['sensitivity'] + results['specificity']) / 2

    # Detection statistics
    results['detection_rate'] = np.mean(anomaly_predictions)
    results['anomaly_rate'] = np.mean(y_true)
    results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    results['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Score distribution analysis
    results.update(analyze_score_distribution(anomaly_scores, y_true))

    return results


def create_unsupervised_evaluation(anomaly_scores, anomaly_predictions=None, threshold=None):
    """
    Create evaluation metrics for unsupervised case (no ground truth).
    """
    results = {'supervised': False}

    if threshold is None:
        threshold = np.percentile(anomaly_scores, 90)  # Conservative threshold

    results['threshold'] = threshold

    if anomaly_predictions is None:
        anomaly_predictions = (anomaly_scores >= threshold).astype(int)

    # Basic statistics
    results['detection_rate'] = np.mean(anomaly_predictions)
    results['mean_anomaly_score'] = np.mean(anomaly_scores)
    results['std_anomaly_score'] = np.std(anomaly_scores)
    results['score_range'] = np.max(anomaly_scores) - np.min(anomaly_scores)

    # Score distribution analysis
    results['score_percentiles'] = {
        'p25': np.percentile(anomaly_scores, 25),
        'p50': np.percentile(anomaly_scores, 50),
        'p75': np.percentile(anomaly_scores, 75),
        'p90': np.percentile(anomaly_scores, 90),
        'p95': np.percentile(anomaly_scores, 95),
        'p99': np.percentile(anomaly_scores, 99)
    }

    return results


def analyze_score_distribution(anomaly_scores, y_true):
    """
    Analyze the distribution of anomaly scores for normal vs anomalous samples.
    """
    analysis = {}

    normal_scores = anomaly_scores[y_true == 0]
    anomaly_scores_positive = anomaly_scores[y_true == 1]

    if len(normal_scores) > 0:
        analysis['normal_score_mean'] = np.mean(normal_scores)
        analysis['normal_score_std'] = np.std(normal_scores)
        analysis['normal_score_median'] = np.median(normal_scores)

    if len(anomaly_scores_positive) > 0:
        analysis['anomaly_score_mean'] = np.mean(anomaly_scores_positive)
        analysis['anomaly_score_std'] = np.std(anomaly_scores_positive)
        analysis['anomaly_score_median'] = np.median(anomaly_scores_positive)

    # Separation analysis
    if len(normal_scores) > 0 and len(anomaly_scores_positive) > 0:
        # Calculate separation between distributions
        analysis['score_separation'] = (
                analysis['anomaly_score_mean'] - analysis['normal_score_mean']
        )

        # Statistical test for distribution difference
        try:
            stat, p_value = stats.mannwhitneyu(anomaly_scores_positive, normal_scores,
                                               alternative='greater')
            analysis['distribution_test_pvalue'] = p_value
            analysis['distribution_significantly_different'] = p_value < 0.05
        except:
            analysis['distribution_test_pvalue'] = np.nan
            analysis['distribution_significantly_different'] = False

    return analysis


def optimize_threshold(anomaly_scores, y_true, metric='f1'):
    """
    Find the optimal threshold that maximizes the specified metric.
    """
    if y_true is None or len(np.unique(y_true)) < 2:
        # Default threshold for unsupervised case
        return np.percentile(anomaly_scores, 90), 0

    best_score = -1
    best_threshold = np.percentile(anomaly_scores, 90)

    # Try different percentiles as thresholds
    percentiles = np.linspace(50, 99, 50)  # More granular search

    for percentile in percentiles:
        threshold = np.percentile(anomaly_scores, percentile)
        y_pred = (anomaly_scores >= threshold).astype(int)

        # Calculate metric
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            score = f1_score(y_true, y_pred, zero_division=0)  # Default to F1

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def compare_models(models_results, ranking_criteria=None):
    """
    Compare multiple anomaly detection models with enhanced ranking.
    """
    if not models_results:
        return None, {}

    if ranking_criteria is None:
        ranking_criteria = [
            ('f1', 3.0),  # F1 score (highest weight)
            ('roc_auc', 2.0),  # ROC AUC
            ('pr_auc', 2.0),  # PR AUC
            ('balanced_accuracy', 1.5),  # Balanced accuracy
            ('precision', 1.0),  # Precision
            ('recall', 1.0)  # Recall
        ]

    # Create comparison dictionary
    comparison = {}
    model_scores = {}

    # Extract metrics for all models
    all_metrics = set()
    for model_results in models_results.values():
        all_metrics.update(model_results.keys())

    for metric in all_metrics:
        comparison[metric] = {}
        for model_name, results in models_results.items():
            comparison[metric][model_name] = results.get(metric, np.nan)

    # Calculate composite scores for ranking
    for model_name in models_results:
        composite_score = 0
        total_weight = 0

        for metric, weight in ranking_criteria:
            if metric in models_results[model_name]:
                value = models_results[model_name][metric]
                if not np.isnan(value):
                    composite_score += value * weight
                    total_weight += weight

        if total_weight > 0:
            model_scores[model_name] = composite_score / total_weight
        else:
            model_scores[model_name] = 0

    # Find best model
    best_model = max(model_scores, key=model_scores.get) if model_scores else None

    # Add ranking information to comparison
    comparison['composite_scores'] = model_scores
    comparison['ranking'] = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    return best_model, comparison


def create_evaluation_report(models_results, output_path=None):
    """
    Create a comprehensive evaluation report.
    """
    if not models_results:
        return "No model results to report."

    report_lines = []
    report_lines.append("ANOMALY DETECTION MODEL EVALUATION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Model comparison
    best_model, comparison = compare_models(models_results)

    if best_model:
        report_lines.append(f"BEST PERFORMING MODEL: {best_model}")
        report_lines.append("")

    # Individual model results
    for model_name, results in models_results.items():
        report_lines.append(f"MODEL: {model_name.upper()}")
        report_lines.append("-" * 30)

        # Core metrics
        if 'f1' in results and not np.isnan(results['f1']):
            report_lines.append(f"F1 Score:           {results['f1']:.4f}")
        if 'roc_auc' in results and not np.isnan(results['roc_auc']):
            report_lines.append(f"ROC AUC:            {results['roc_auc']:.4f}")
        if 'pr_auc' in results and not np.isnan(results['pr_auc']):
            report_lines.append(f"PR AUC:             {results['pr_auc']:.4f}")
        if 'balanced_accuracy' in results:
            report_lines.append(f"Balanced Accuracy:  {results['balanced_accuracy']:.4f}")

        # Classification metrics
        if 'precision' in results:
            report_lines.append(f"Precision:          {results['precision']:.4f}")
        if 'recall' in results:
            report_lines.append(f"Recall:             {results['recall']:.4f}")
        if 'specificity' in results:
            report_lines.append(f"Specificity:        {results['specificity']:.4f}")

        # Detection statistics
        if 'detection_rate' in results:
            report_lines.append(f"Detection Rate:     {results['detection_rate']:.4f}")
        if 'false_positive_rate' in results:
            report_lines.append(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        if 'threshold' in results:
            report_lines.append(f"Threshold:          {results['threshold']:.4f}")

        # Confusion matrix
        if all(k in results for k in ['true_positives', 'false_positives',
                                      'true_negatives', 'false_negatives']):
            report_lines.append("")
            report_lines.append("Confusion Matrix:")
            report_lines.append(f"  TP: {results['true_positives']:3d}  FP: {results['false_positives']:3d}")
            report_lines.append(f"  FN: {results['false_negatives']:3d}  TN: {results['true_negatives']:3d}")

        report_lines.append("")

    # Model ranking
    if 'ranking' in comparison:
        report_lines.append("MODEL RANKING (by composite score):")
        report_lines.append("-" * 30)
        for i, (model_name, score) in enumerate(comparison['ranking'], 1):
            report_lines.append(f"{i}. {model_name}: {score:.4f}")
        report_lines.append("")

    # Summary statistics
    report_lines.append("SUMMARY STATISTICS:")
    report_lines.append("-" * 20)

    # Average performance across models
    avg_metrics = {}
    for metric in ['f1', 'roc_auc', 'pr_auc', 'precision', 'recall']:
        if metric in comparison:
            values = [v for v in comparison[metric].values() if not np.isnan(v)]
            if values:
                avg_metrics[metric] = np.mean(values)

    for metric, avg_value in avg_metrics.items():
        report_lines.append(f"Average {metric.upper()}: {avg_value:.4f}")

    report = "\n".join(report_lines)

    # Save to file if path provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Could not save report to {output_path}: {str(e)}")

    return report


def calculate_detection_performance(y_true, anomaly_scores, thresholds=None):
    """
    Calculate detection performance at multiple thresholds.
    """
    if thresholds is None:
        thresholds = np.percentile(anomaly_scores, np.linspace(50, 99, 20))

    performance_data = []

    for threshold in thresholds:
        y_pred = (anomaly_scores >= threshold).astype(int)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        performance_data.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'detection_rate': np.mean(y_pred)
        })

    return pd.DataFrame(performance_data)


def analyze_model_reliability(models_results, consistency_threshold=0.1):
    """
    Analyze the reliability and consistency of model predictions.
    """
    reliability_analysis = {}

    if len(models_results) < 2:
        return reliability_analysis

    # Extract anomaly scores from all models
    model_names = list(models_results.keys())
    all_scores = []

    for model_name in model_names:
        if 'anomaly_scores' in models_results[model_name]:
            all_scores.append(models_results[model_name]['anomaly_scores'])

    if len(all_scores) < 2:
        return reliability_analysis

    all_scores = np.array(all_scores)

    # Calculate pairwise correlations
    correlations = []
    for i in range(len(all_scores)):
        for j in range(i + 1, len(all_scores)):
            try:
                corr = np.corrcoef(all_scores[i], all_scores[j])[0, 1]
                correlations.append(corr)
            except:
                continue

    if correlations:
        reliability_analysis['score_correlations'] = {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'min': np.min(correlations),
            'max': np.max(correlations)
        }

        # High correlation suggests consistent scoring
        reliability_analysis['models_consistent'] = np.mean(correlations) > (1 - consistency_threshold)

    return reliability_analysis