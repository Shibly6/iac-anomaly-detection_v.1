#!/usr/bin/env python3
"""
Data preprocessing utilities for anomaly detection in IaC scripts.
Enhanced with better feature selection and preprocessing for improved anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger("anomaly_detector.preprocessing")


def preprocess_features(df, exclude_cols=None):
    """
    Enhanced feature preprocessing with better handling of anomaly detection requirements.
    """
    if exclude_cols is None:
        exclude_cols = ['bucket_name', 'acl', 'source_file', 'assumed_anomalous']

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Create feature matrix
    X = df[feature_cols].copy()

    # Handle missing values with more sophisticated approach
    X = handle_missing_values(X)

    # Remove features with zero or near-zero variance
    X, feature_cols = remove_low_variance_features(X, feature_cols)

    # Handle infinite values
    X = handle_infinite_values(X)

    # Convert to numpy array
    X_array = X.values

    logger.info(f"Preprocessing completed: {X_array.shape[1]} features, {X_array.shape[0]} samples")

    return X_array, feature_cols


def handle_missing_values(X):
    """
    Handle missing values with context-aware imputation.
    """
    for col in X.columns:
        if X[col].isnull().any():
            if col.endswith('_enabled') or col in ['is_public_acl', 'has_public_policy']:
                # Binary features: fill with 0 (disabled/private)
                X[col] = X[col].fillna(0)
            elif col.endswith('_count') or col.endswith('_rules'):
                # Count features: fill with 0
                X[col] = X[col].fillna(0)
            elif col.endswith('_score'):
                # Score features: fill with median
                X[col] = X[col].fillna(X[col].median())
            elif col.endswith('_ratio'):
                # Ratio features: fill with 1 (neutral ratio)
                X[col] = X[col].fillna(1.0)
            else:
                # Other numeric features: fill with median
                X[col] = X[col].fillna(X[col].median())

    return X


def remove_low_variance_features(X, feature_cols, threshold=0.01):
    """
    Remove features with very low variance that won't help with anomaly detection.
    """
    # Calculate variance for each feature
    variances = X.var()

    # Keep features with variance above threshold
    high_variance_mask = variances > threshold

    if high_variance_mask.sum() == 0:
        logger.warning("All features have low variance, keeping all features")
        return X, feature_cols

    # Filter features
    X_filtered = X.loc[:, high_variance_mask]
    feature_cols_filtered = [col for col, keep in zip(feature_cols, high_variance_mask) if keep]

    removed_count = len(feature_cols) - len(feature_cols_filtered)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} low-variance features")

    return X_filtered, feature_cols_filtered


def handle_infinite_values(X):
    """
    Handle infinite values in the dataset.
    """
    # Replace infinite values with large but finite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN values (which were inf) with appropriate values
    for col in X.columns:
        if X[col].isnull().any():
            if col.endswith('_ratio'):
                # For ratios, use a reasonable maximum value
                X[col] = X[col].fillna(10.0)
            else:
                # For other features, use the 95th percentile
                percentile_95 = X[col].quantile(0.95)
                X[col] = X[col].fillna(percentile_95)

    return X


def split_dataset(X, test_size=0.2, val_size=0.1, random_state=42, y=None):
    """
    Enhanced dataset splitting with stratification support.
    """
    if y is not None and len(np.unique(y)) > 1:
        # Stratified split to maintain class balance
        if val_size > 0:
            # First split: train+val vs test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio,
                stratify=y_train_val, random_state=random_state
            )

            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            # Simple train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            return X_train, X_test, None, y_train, y_test, None

    else:
        # Random split without stratification
        if val_size > 0:
            # First split
            X_train_val, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )

            # Second split
            val_ratio = val_size / (1 - test_size)
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_ratio, random_state=random_state
            )

            return X_train, X_test, X_val
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, None


def select_best_features(X, feature_names, k=15, y=None, method='variance'):
    """
    Enhanced feature selection with multiple methods.
    """
    if X.shape[1] <= k:
        return X, feature_names

    logger.info(f"Selecting {k} best features from {X.shape[1]} using {method} method")

    if method == 'variance' or y is None:
        # Variance-based selection
        variances = np.var(X, axis=0)
        top_k_indices = np.argsort(variances)[-k:]

    elif method == 'statistical' and y is not None:
        # Statistical test-based selection
        try:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            top_k_indices = selector.get_support(indices=True)
        except:
            # Fallback to variance if statistical method fails
            logger.warning("Statistical feature selection failed, using variance method")
            variances = np.var(X, axis=0)
            top_k_indices = np.argsort(variances)[-k:]

    elif method == 'combined' and y is not None:
        # Combined approach: variance + statistical importance
        try:
            # Get variance scores (normalized)
            variances = np.var(X, axis=0)
            variance_scores = (variances - variances.min()) / (variances.max() - variances.min())

            # Get statistical scores (normalized)
            selector = SelectKBest(score_func=f_classif, k=X.shape[1])
            selector.fit(X, y)
            stat_scores = selector.scores_
            stat_scores = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min())

            # Combine scores (equal weight)
            combined_scores = variance_scores + stat_scores
            top_k_indices = np.argsort(combined_scores)[-k:]

        except:
            # Fallback to variance
            logger.warning("Combined feature selection failed, using variance method")
            variances = np.var(X, axis=0)
            top_k_indices = np.argsort(variances)[-k:]

    else:
        # Default to variance
        variances = np.var(X, axis=0)
        top_k_indices = np.argsort(variances)[-k:]

    # Select features
    X_selected = X[:, top_k_indices]
    feature_names_selected = [feature_names[i] for i in top_k_indices]

    logger.info(f"Selected features: {feature_names_selected}")

    return X_selected, feature_names_selected


def apply_advanced_scaling(X, method='robust', feature_types=None):
    """
    Apply advanced scaling methods suitable for anomaly detection.
    """
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'mixed':
        # Use different scalers for different feature types
        return apply_mixed_scaling(X, feature_types)
    else:
        scaler = RobustScaler()  # Default

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def apply_mixed_scaling(X, feature_types=None):
    """
    Apply different scaling methods to different types of features.
    """
    if feature_types is None:
        # Default: use RobustScaler for all
        scaler = RobustScaler()
        return scaler.fit_transform(X), scaler

    X_scaled = X.copy()
    scalers = {}

    # Define feature type groups
    type_groups = {
        'binary': [],
        'count': [],
        'score': [],
        'ratio': [],
        'other': []
    }

    # Classify features
    for i, feature_type in enumerate(feature_types):
        if 'binary' in feature_type or 'enabled' in feature_type:
            type_groups['binary'].append(i)
        elif 'count' in feature_type or 'rules' in feature_type:
            type_groups['count'].append(i)
        elif 'score' in feature_type:
            type_groups['score'].append(i)
        elif 'ratio' in feature_type:
            type_groups['ratio'].append(i)
        else:
            type_groups['other'].append(i)

    # Apply appropriate scaling to each group
    for group_name, indices in type_groups.items():
        if not indices:
            continue

        if group_name == 'binary':
            # Don't scale binary features
            scalers[group_name] = None
        elif group_name == 'count':
            # Use StandardScaler for count features
            scaler = StandardScaler()
            X_scaled[:, indices] = scaler.fit_transform(X[:, indices])
            scalers[group_name] = scaler
        elif group_name == 'score':
            # Use MinMaxScaler for score features
            scaler = MinMaxScaler()
            X_scaled[:, indices] = scaler.fit_transform(X[:, indices])
            scalers[group_name] = scaler
        elif group_name == 'ratio':
            # Use RobustScaler for ratio features (may have outliers)
            scaler = RobustScaler()
            X_scaled[:, indices] = scaler.fit_transform(X[:, indices])
            scalers[group_name] = scaler
        else:
            # Use RobustScaler for other features
            scaler = RobustScaler()
            X_scaled[:, indices] = scaler.fit_transform(X[:, indices])
            scalers[group_name] = scaler

    return X_scaled, scalers


def detect_and_handle_outliers(X, method='iqr', contamination=0.1):
    """
    Detect and optionally handle outliers in the feature space.
    """
    outlier_mask = np.zeros(X.shape[0], dtype=bool)

    if method == 'iqr':
        # Use IQR method for each feature
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            Q1 = np.percentile(feature_values, 25)
            Q3 = np.percentile(feature_values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            feature_outliers = (feature_values < lower_bound) | (feature_values > upper_bound)
            outlier_mask |= feature_outliers

    elif method == 'zscore':
        # Use Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(X, axis=0))
        outlier_mask = np.any(z_scores > 3, axis=1)

    elif method == 'isolation_forest':
        # Use Isolation Forest for multivariate outlier detection
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        outlier_mask = outlier_labels == -1

    outlier_count = outlier_mask.sum()
    logger.info(f"Detected {outlier_count} outliers using {method} method")

    return outlier_mask


def create_feature_summary(X, feature_names):
    """
    Create a summary of feature statistics.
    """
    summary = pd.DataFrame({
        'feature': feature_names,
        'mean': X.mean(axis=0),
        'std': X.std(axis=0),
        'min': X.min(axis=0),
        'max': X.max(axis=0),
        'variance': X.var(axis=0)
    })

    # Add additional statistics
    summary['range'] = summary['max'] - summary['min']
    summary['cv'] = summary['std'] / np.maximum(summary['mean'], 1e-8)  # Coefficient of variation

    return summary


def prepare_data_for_anomaly_detection(df, target_col='assumed_anomalous',
                                       test_size=0.2, val_size=0.1,
                                       n_features=15, scaling_method='robust'):
    """
    Complete data preparation pipeline for anomaly detection.
    """
    logger.info("Starting complete data preparation for anomaly detection...")

    # Step 1: Basic preprocessing
    exclude_cols = ['bucket_name', 'acl', 'source_file', target_col]
    X, feature_names = preprocess_features(df, exclude_cols)

    # Extract target if available
    y = df[target_col].values if target_col in df.columns else None

    # Step 2: Feature selection
    X_selected, selected_features = select_best_features(
        X, feature_names, k=n_features, y=y, method='combined'
    )

    # Step 3: Train-test-val split
    if y is not None:
        split_results = split_dataset(X_selected, test_size, val_size, y=y)
        if len(split_results) == 6:
            X_train, X_test, X_val, y_train, y_test, y_val = split_results
        else:
            X_train, X_test, X_val = split_results
            y_train, y_test, y_val = None, None, None
    else:
        X_train, X_test, X_val = split_dataset(X_selected, test_size, val_size)
        y_train, y_test, y_val = None, None, None

    # Step 4: Feature scaling
    X_train_scaled, scaler = apply_advanced_scaling(X_train, method=scaling_method)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    # Step 5: Create feature summary
    feature_summary = create_feature_summary(X_train_scaled, selected_features)

    logger.info("Data preparation completed successfully")

    # Return comprehensive results
    results = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_val': X_val_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'feature_names': selected_features,
        'scaler': scaler,
        'feature_summary': feature_summary,
        'data_info': {
            'original_features': len(feature_names),
            'selected_features': len(selected_features),
            'train_samples': len(X_train_scaled),
            'test_samples': len(X_test_scaled),
            'val_samples': len(X_val_scaled) if X_val_scaled is not None else 0,
            'anomaly_ratio': np.mean(y_train) if y_train is not None else None
        }
    }

    return results