#!/usr/bin/env python3
"""
feature_engineering.py - Fixed feature engineering for S3 bucket security analysis

This module provides robust functions for creating discriminative features from basic S3 bucket
configurations with proper NaN handling and improved feature discrimination.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import logging
from scipy import stats

# Setup logging
log = logging.getLogger("rich")


def create_numerical_features(df):
    """
    Create enhanced numerical features with robust NaN handling.
    """
    df_enhanced = df.copy()

    # Ensure all numeric columns are properly typed
    numeric_cols = ['is_public_acl', 'has_public_policy', 'versioning_enabled', 'logging_enabled',
                    'encryption_enabled', 'lifecycle_rules', 'secure_transport', 'bucket_policy_complexity',
                    'public_access_blocks', 'cors_enabled', 'website_enabled', 'replication_enabled',
                    'object_lock_enabled', 'intelligent_tiering_enabled', 'analytics_enabled',
                    'inventory_enabled', 'accelerate_enabled', 'security_score', 'accessibility_score', 'risk_ratio']

    for col in numeric_cols:
        if col in df_enhanced.columns:
            df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce').fillna(0)

    # Core security features aggregation
    security_cols = ['versioning_enabled', 'logging_enabled', 'encryption_enabled', 'secure_transport']
    available_security_cols = [col for col in security_cols if col in df_enhanced.columns]

    if available_security_cols:
        df_enhanced['security_features_count'] = df_enhanced[available_security_cols].sum(axis=1)
        df_enhanced['security_coverage'] = df_enhanced['security_features_count'] / len(available_security_cols)
    else:
        df_enhanced['security_features_count'] = 0
        df_enhanced['security_coverage'] = 0

    # PUBLIC ACCESS INDICATORS (MOST IMPORTANT FOR DISCRIMINATION)
    if 'is_public_acl' in df_enhanced.columns and 'has_public_policy' in df_enhanced.columns:
        # Strong binary indicator for ANY public access
        df_enhanced['has_any_public_access'] = (
                (df_enhanced['is_public_acl'] > 0) | (df_enhanced['has_public_policy'] > 0)
        ).astype(int)

        # Weighted public access score
        df_enhanced['public_access_score'] = (
                df_enhanced['is_public_acl'] * 5 + df_enhanced['has_public_policy'] * 4
        )

        # Public access type classification
        df_enhanced['public_access_type'] = np.select([
            (df_enhanced['is_public_acl'] == 0) & (df_enhanced['has_public_policy'] == 0),
            (df_enhanced['is_public_acl'] == 1) & (df_enhanced['has_public_policy'] == 0),
            (df_enhanced['is_public_acl'] == 0) & (df_enhanced['has_public_policy'] == 1),
            (df_enhanced['is_public_acl'] == 1) & (df_enhanced['has_public_policy'] == 1)
        ], [0, 1, 2, 3], default=0)

    else:
        df_enhanced['has_any_public_access'] = 0
        df_enhanced['public_access_score'] = 0
        df_enhanced['public_access_type'] = 0

    # Configuration complexity
    feature_cols = ['cors_enabled', 'website_enabled', 'intelligent_tiering_enabled',
                    'analytics_enabled', 'inventory_enabled', 'accelerate_enabled']
    available_feature_cols = [col for col in feature_cols if col in df_enhanced.columns]

    if available_feature_cols:
        df_enhanced['advanced_features_count'] = df_enhanced[available_feature_cols].sum(axis=1)
        df_enhanced['feature_complexity'] = df_enhanced['advanced_features_count'] / len(available_feature_cols)
    else:
        df_enhanced['advanced_features_count'] = 0
        df_enhanced['feature_complexity'] = 0

    # Security deficiency indicators
    if 'public_access_blocks' in df_enhanced.columns:
        df_enhanced['public_access_blocks'] = df_enhanced['public_access_blocks'].fillna(0)
        df_enhanced['access_blocks_ratio'] = df_enhanced['public_access_blocks'] / 4.0
        df_enhanced['insufficient_blocks'] = (df_enhanced['public_access_blocks'] < 4).astype(int)
    else:
        df_enhanced['public_access_blocks'] = 0
        df_enhanced['access_blocks_ratio'] = 0
        df_enhanced['insufficient_blocks'] = 0

    # Critical security gaps
    critical_features = ['versioning_enabled', 'logging_enabled', 'encryption_enabled']
    available_critical = [col for col in critical_features if col in df_enhanced.columns]

    if available_critical:
        df_enhanced['critical_security_gaps'] = len(available_critical) - df_enhanced[available_critical].sum(axis=1)
        df_enhanced['has_critical_gaps'] = (df_enhanced['critical_security_gaps'] > 1).astype(int)
    else:
        df_enhanced['critical_security_gaps'] = 0
        df_enhanced['has_critical_gaps'] = 0

    return df_enhanced


def create_security_score_features(df):
    """
    Create comprehensive security scoring with improved discrimination.
    """
    df_enhanced = df.copy()

    # Recalculate security score with proper weights
    security_components = {
        'versioning_enabled': 2.0,
        'logging_enabled': 2.0,
        'encryption_enabled': 3.0,
        'secure_transport': 2.0,
        'public_access_blocks': 2.5,  # Will be normalized
        'replication_enabled': 1.0,
        'object_lock_enabled': 1.0
    }

    total_security_score = 0
    max_possible_score = 0

    for feature, weight in security_components.items():
        if feature in df_enhanced.columns:
            feature_values = df_enhanced[feature].fillna(0)

            if feature == 'public_access_blocks':
                # Normalize to 0-1 range
                feature_values = feature_values / 4.0

            total_security_score += feature_values * weight
            max_possible_score += weight

    if max_possible_score > 0:
        df_enhanced['recalc_security_score'] = (total_security_score / max_possible_score) * 10
    else:
        df_enhanced['recalc_security_score'] = 0

    # Security penalty for public access
    if 'has_any_public_access' in df_enhanced.columns:
        df_enhanced['security_penalty'] = df_enhanced['has_any_public_access'] * 6  # Heavy penalty
        df_enhanced['adjusted_security_score'] = np.maximum(
            0, df_enhanced['recalc_security_score'] - df_enhanced['security_penalty']
        )
    else:
        df_enhanced['security_penalty'] = 0
        df_enhanced['adjusted_security_score'] = df_enhanced['recalc_security_score']

    # Risk scoring
    df_enhanced['risk_score'] = 10 - df_enhanced['adjusted_security_score']

    # Vulnerability indicators
    vulnerability_factors = []

    # Public without protection
    if all(col in df_enhanced.columns for col in ['has_any_public_access', 'insufficient_blocks']):
        vuln_1 = (df_enhanced['has_any_public_access'] == 1) & (df_enhanced['insufficient_blocks'] == 1)
        vulnerability_factors.append(vuln_1.astype(int) * 4)

    # No encryption on public bucket
    if all(col in df_enhanced.columns for col in ['has_any_public_access', 'encryption_enabled']):
        vuln_2 = (df_enhanced['has_any_public_access'] == 1) & (df_enhanced['encryption_enabled'] == 0)
        vulnerability_factors.append(vuln_2.astype(int) * 3)

    # No logging on public bucket
    if all(col in df_enhanced.columns for col in ['has_any_public_access', 'logging_enabled']):
        vuln_3 = (df_enhanced['has_any_public_access'] == 1) & (df_enhanced['logging_enabled'] == 0)
        vulnerability_factors.append(vuln_3.astype(int) * 2)

    if vulnerability_factors:
        df_enhanced['vulnerability_score'] = sum(vulnerability_factors)
    else:
        df_enhanced['vulnerability_score'] = 0

    return df_enhanced


def create_interaction_features(df):
    """
    Create interaction features that capture anomalous patterns.
    """
    df_enhanced = df.copy()

    # High-risk combinations
    risk_combinations = []

    # Public access without any protection
    if all(col in df_enhanced.columns for col in ['has_any_public_access', 'security_features_count']):
        combo_1 = (df_enhanced['has_any_public_access'] == 1) & (df_enhanced['security_features_count'] == 0)
        risk_combinations.append(combo_1.astype(int) * 5)  # Highest risk

    # Website hosting without proper security
    if all(col in df_enhanced.columns for col in ['website_enabled', 'has_any_public_access', 'encryption_enabled']):
        combo_2 = (df_enhanced['website_enabled'] == 1) & (df_enhanced['has_any_public_access'] == 1) & (
                    df_enhanced['encryption_enabled'] == 0)
        risk_combinations.append(combo_2.astype(int) * 3)

    # Public with full access (read-write)
    if 'is_public_acl' in df_enhanced.columns:
        # Assume public-read-write if is_public_acl is 1 and has high accessibility
        combo_3 = (df_enhanced['is_public_acl'] == 1) & (df_enhanced.get('accessibility_score', 0) > 6)
        risk_combinations.append(combo_3.astype(int) * 4)

    if risk_combinations:
        df_enhanced['high_risk_combination'] = sum(risk_combinations)
    else:
        df_enhanced['high_risk_combination'] = 0

    # Configuration inconsistencies
    inconsistency_patterns = []

    # Pattern 1: Public ACL but private policy (contradictory)
    if all(col in df_enhanced.columns for col in ['is_public_acl', 'has_public_policy']):
        pattern_1 = (df_enhanced['is_public_acl'] == 1) & (df_enhanced['has_public_policy'] == 0)
        inconsistency_patterns.append(pattern_1.astype(int))

    # Pattern 2: High security features but public access (unusual)
    if all(col in df_enhanced.columns for col in ['security_features_count', 'has_any_public_access']):
        pattern_2 = (df_enhanced['security_features_count'] >= 3) & (df_enhanced['has_any_public_access'] == 1)
        inconsistency_patterns.append(pattern_2.astype(int))

    if inconsistency_patterns:
        df_enhanced['configuration_inconsistency'] = sum(inconsistency_patterns)
    else:
        df_enhanced['configuration_inconsistency'] = 0

    return df_enhanced


def create_anomaly_indicator_features(df):
    """
    Create features specifically designed to identify anomalous configurations.
    """
    df_enhanced = df.copy()

    # Extreme value indicators
    extreme_indicators = []

    # Extremely high accessibility (public read-write with no blocks)
    if all(col in df_enhanced.columns for col in ['public_access_score', 'public_access_blocks']):
        extreme_1 = (df_enhanced['public_access_score'] >= 5) & (df_enhanced['public_access_blocks'] == 0)
        extreme_indicators.append(extreme_1.astype(int) * 3)

    # Zero security features
    if 'security_features_count' in df_enhanced.columns:
        extreme_2 = (df_enhanced['security_features_count'] == 0)
        extreme_indicators.append(extreme_2.astype(int) * 2)

    # High complexity but low security
    if all(col in df_enhanced.columns for col in ['advanced_features_count', 'security_features_count']):
        extreme_3 = (df_enhanced['advanced_features_count'] >= 2) & (df_enhanced['security_features_count'] <= 1)
        extreme_indicators.append(extreme_3.astype(int) * 2)

    if extreme_indicators:
        df_enhanced['extreme_configuration'] = sum(extreme_indicators)
    else:
        df_enhanced['extreme_configuration'] = 0

    # Composite anomaly score
    anomaly_components = []

    # Add available components
    if 'high_risk_combination' in df_enhanced.columns:
        anomaly_components.append(df_enhanced['high_risk_combination'] * 0.3)

    if 'configuration_inconsistency' in df_enhanced.columns:
        anomaly_components.append(df_enhanced['configuration_inconsistency'] * 0.2)

    if 'extreme_configuration' in df_enhanced.columns:
        anomaly_components.append(df_enhanced['extreme_configuration'] * 0.3)

    if 'vulnerability_score' in df_enhanced.columns:
        # Normalize vulnerability score
        max_vuln = df_enhanced['vulnerability_score'].max()
        if max_vuln > 0:
            normalized_vuln = df_enhanced['vulnerability_score'] / max_vuln
        else:
            normalized_vuln = df_enhanced['vulnerability_score']
        anomaly_components.append(normalized_vuln * 0.2)

    if anomaly_components:
        df_enhanced['composite_anomaly_score'] = sum(anomaly_components)
    else:
        df_enhanced['composite_anomaly_score'] = 0

    return df_enhanced


def apply_feature_scaling(df, method='robust'):
    """
    Apply feature scaling with proper NaN handling.
    """
    # Select only numeric columns for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude metadata columns
    exclude_cols = ['bucket_name', 'source_file', 'assumed_anomalous']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_scaled = df.copy()

    if not scale_cols:
        return df_scaled, None

    # Fill any remaining NaN values before scaling
    for col in scale_cols:
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col].fillna(0)

    # Apply scaling
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    try:
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        return df_scaled, scaler
    except Exception as e:
        log.error(f"Error in feature scaling: {str(e)}")
        # Return unscaled data if scaling fails
        return df_scaled, None


def select_discriminative_features(df, n_features=12):
    """
    Select the most discriminative features for anomaly detection.
    """
    # Priority features that are crucial for security anomaly detection
    priority_features = [
        'has_any_public_access',  # Most important
        'public_access_score',  # Public access strength
        'adjusted_security_score',  # Security after penalties
        'vulnerability_score',  # Vulnerability indicators
        'high_risk_combination',  # High-risk patterns
        'composite_anomaly_score',  # Overall anomaly score
        'security_features_count',  # Count of security features
        'critical_security_gaps',  # Missing critical security
        'public_access_type',  # Type of public access
        'configuration_inconsistency',  # Inconsistent configurations
        'extreme_configuration',  # Extreme patterns
        'insufficient_blocks'  # Insufficient public access blocks
    ]

    # Get available priority features
    available_priority = [col for col in priority_features if col in df.columns]

    # Fill to n_features with other numeric features
    if len(available_priority) < n_features:
        # Get other numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['bucket_name', 'source_file', 'assumed_anomalous']
        other_features = [col for col in numeric_cols
                          if col not in exclude_cols and col not in available_priority]

        # Add features with highest variance
        if other_features and len(df) > 1:
            try:
                variances = df[other_features].var()
                high_variance_features = variances.sort_values(ascending=False).head(
                    n_features - len(available_priority)
                ).index.tolist()
                available_priority.extend(high_variance_features)
            except Exception as e:
                log.warning(f"Could not calculate feature variances: {str(e)}")

    # Ensure we don't exceed n_features
    selected_features = available_priority[:n_features]

    # Remove features with zero variance (if we have enough samples)
    if len(df) > 1:
        final_features = []
        for feature in selected_features:
            if feature in df.columns:
                try:
                    if df[feature].var() > 0:
                        final_features.append(feature)
                except:
                    # Keep feature if variance calculation fails
                    final_features.append(feature)
        selected_features = final_features

    log.info(f"Selected {len(selected_features)} discriminative features: {selected_features}")

    return selected_features


def engineer_features_pipeline(df):
    """
    Complete feature engineering pipeline with robust error handling.
    """
    log.info("Starting feature engineering pipeline...")

    try:
        # Step 1: Create numerical features
        df = create_numerical_features(df)
        log.info(f"After numerical features: {df.shape[1]} columns")

        # Step 2: Create security score features
        df = create_security_score_features(df)
        log.info(f"After security score features: {df.shape[1]} columns")

        # Step 3: Create interaction features
        df = create_interaction_features(df)
        log.info(f"After interaction features: {df.shape[1]} columns")

        # Step 4: Create anomaly indicator features
        df = create_anomaly_indicator_features(df)
        log.info(f"After anomaly indicator features: {df.shape[1]} columns")

        # Step 5: Handle any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['bucket_name', 'source_file']:
                df[col] = df[col].fillna(0)

        # Step 6: Select discriminative features
        discriminative_features = select_discriminative_features(df, n_features=12)

        # Step 7: Apply feature scaling
        feature_df = df[discriminative_features + ['bucket_name', 'source_file']].copy()
        df_scaled, scaler = apply_feature_scaling(feature_df)

        log.info("Feature engineering pipeline completed successfully")

        return df_scaled, discriminative_features, scaler

    except Exception as e:
        log.error(f"Error in feature engineering pipeline: {str(e)}")
        # Return basic features if pipeline fails
        basic_features = ['is_public_acl', 'has_public_policy', 'security_score', 'accessibility_score']
        available_basic = [col for col in basic_features if col in df.columns]

        if available_basic:
            feature_df = df[available_basic + ['bucket_name', 'source_file']].copy()
            # Fill NaN values
            for col in available_basic:
                feature_df[col] = feature_df[col].fillna(0)

            df_scaled, scaler = apply_feature_scaling(feature_df)
            return df_scaled, available_basic, scaler
        else:
            # Last resort: return original dataframe
            return df, [], None