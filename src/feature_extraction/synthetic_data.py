#!/usr/bin/env python3
"""
synthetic_data.py - Generate synthetic data for S3 bucket anomaly detection

This module provides functions for generating synthetic S3 bucket data to augment
small datasets for better unsupervised learning.
"""

import numpy as np
import pandas as pd
import logging
import random

# Setup logging
log = logging.getLogger("rich")


def generate_synthetic_data(sample_count=50, base_features=None):
    """
    Generate synthetic S3 bucket data with realistic patterns.

    Args:
        sample_count: Number of synthetic examples to generate
        base_features: DataFrame of real features to use as reference (optional)

    Returns:
        List of dictionaries containing synthetic bucket features
    """
    synthetic_features = []

    # Define templates for different security patterns
    templates = {
        'high_security': {
            'is_public_acl': 0,
            'has_public_policy': 0,
            'versioning_enabled': 1,
            'logging_enabled': 1,
            'encryption_enabled': 1,
            'public_access_blocks': lambda: np.random.randint(3, 5),  # 3 or 4 blocks
            'secure_transport': 1,
            'security_score': lambda: np.random.uniform(7.5, 10.0),
            'accessibility_score': lambda: np.random.uniform(0, 2.0),
            'object_lock_enabled': lambda: np.random.choice([0, 1], p=[0.7, 0.3]),
            'replication_enabled': lambda: np.random.choice([0, 1], p=[0.6, 0.4]),
            'lifecycle_rules': lambda: np.random.randint(0, 4)
        },
        'medium_security': {
            'is_public_acl': 0,
            'has_public_policy': lambda: np.random.choice([0, 1], p=[0.8, 0.2]),
            'versioning_enabled': lambda: np.random.choice([0, 1], p=[0.4, 0.6]),
            'logging_enabled': lambda: np.random.choice([0, 1], p=[0.6, 0.4]),
            'encryption_enabled': lambda: np.random.choice([0, 1], p=[0.3, 0.7]),
            'public_access_blocks': lambda: np.random.randint(1, 4),
            'secure_transport': lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
            'security_score': lambda: np.random.uniform(4.0, 7.5),
            'accessibility_score': lambda: np.random.uniform(1.0, 5.0),
            'object_lock_enabled': lambda: np.random.choice([0, 1], p=[0.9, 0.1]),
            'replication_enabled': lambda: np.random.choice([0, 1], p=[0.8, 0.2]),
            'lifecycle_rules': lambda: np.random.randint(0, 3)
        },
        'low_security': {
            'is_public_acl': lambda: np.random.choice([0, 1], p=[0.3, 0.7]),
            'has_public_policy': lambda: np.random.choice([0, 1], p=[0.3, 0.7]),
            'versioning_enabled': lambda: np.random.choice([0, 1], p=[0.8, 0.2]),
            'logging_enabled': 0,
            'encryption_enabled': lambda: np.random.choice([0, 1], p=[0.7, 0.3]),
            'public_access_blocks': lambda: np.random.randint(0, 2),
            'secure_transport': 0,
            'security_score': lambda: np.random.uniform(1.0, 4.0),
            'accessibility_score': lambda: np.random.uniform(5.0, 10.0),
            'object_lock_enabled': 0,
            'replication_enabled': 0,
            'lifecycle_rules': lambda: np.random.randint(0, 2)
        },
        'extreme_risk': {
            'is_public_acl': 1,
            'has_public_policy': 1,
            'versioning_enabled': 0,
            'logging_enabled': 0,
            'encryption_enabled': 0,
            'public_access_blocks': 0,
            'secure_transport': 0,
            'security_score': lambda: np.random.uniform(0, 1.0),
            'accessibility_score': lambda: np.random.uniform(8.0, 10.0),
            'object_lock_enabled': 0,
            'replication_enabled': 0,
            'lifecycle_rules': 0
        },
        'anomalous_mix': {  # Unusual combinations that may be anomalies
            'is_public_acl': 1,
            'has_public_policy': 0,
            'versioning_enabled': 1,
            'logging_enabled': 1,
            'encryption_enabled': 1,
            'public_access_blocks': lambda: np.random.randint(2, 4),
            'secure_transport': 1,
            'security_score': lambda: np.random.uniform(4.0, 7.0),
            'accessibility_score': lambda: np.random.uniform(5.0, 8.0),
            'object_lock_enabled': lambda: np.random.choice([0, 1]),
            'replication_enabled': 1,
            'lifecycle_rules': lambda: np.random.randint(1, 5)
        }
    }

    # Additional features not specific to security patterns
    additional_features = {
        'cors_enabled': lambda: np.random.choice([0, 1], p=[0.8, 0.2]),
        'website_enabled': lambda: np.random.choice([0, 1], p=[0.7, 0.3]),
        'intelligent_tiering_enabled': lambda: np.random.choice([0, 1], p=[0.9, 0.1]),
        'analytics_enabled': lambda: np.random.choice([0, 1], p=[0.95, 0.05]),
        'inventory_enabled': lambda: np.random.choice([0, 1], p=[0.9, 0.1]),
        'accelerate_enabled': lambda: np.random.choice([0, 1], p=[0.95, 0.05]),
        'bucket_policy_complexity': lambda: np.random.randint(0, 5)
    }

    # Define the ACL options based on is_public_acl
    acl_options = {
        0: ['private'],
        1: ['public-read', 'public-read-write']
    }

    # Probability distribution for template selection
    if base_features is not None and not base_features.empty:
        # Try to infer a reasonable distribution from real data
        # Start with default weights
        template_weights = {
            'high_security': 0.35,
            'medium_security': 0.35,
            'low_security': 0.2,
            'extreme_risk': 0.05,
            'anomalous_mix': 0.05
        }

        # Adjust based on real data if possible
        if 'security_score' in base_features.columns:
            # Calculate percentage of high, medium, low security buckets
            high_sec = (base_features['security_score'] > 7.5).mean()
            med_sec = ((base_features['security_score'] <= 7.5) &
                       (base_features['security_score'] > 4.0)).mean()
            low_sec = (base_features['security_score'] <= 4.0).mean()

            # Update weights if we have valid percentages
            if not (np.isnan(high_sec) or np.isnan(med_sec) or np.isnan(low_sec)):
                template_weights['high_security'] = high_sec
                template_weights['medium_security'] = med_sec
                template_weights['low_security'] = low_sec

                # Keep small but non-zero probabilities for anomalies
                template_weights['extreme_risk'] = max(0.05, min(0.1, low_sec / 2))
                template_weights['anomalous_mix'] = max(0.05, min(0.1, med_sec / 2))

                # Normalize weights to sum to 1
                weight_sum = sum(template_weights.values())
                for key in template_weights:
                    template_weights[key] /= weight_sum
    else:
        # Default weights without reference data
        template_weights = {
            'high_security': 0.35,
            'medium_security': 0.35,
            'low_security': 0.2,
            'extreme_risk': 0.05,
            'anomalous_mix': 0.05
        }

    # Generate synthetic samples
    for i in range(sample_count):
        # Choose template based on weights
        template_name = np.random.choice(
            list(template_weights.keys()),
            p=list(template_weights.values())
        )
        template = templates[template_name]

        # Start with a base feature set
        synthetic = {
            'bucket_name': f"synthetic_{template_name}_{i}",
            'source_file': f"synthetic/{template_name}_{i}.tf"
        }

        # Apply template values
        for key, value in template.items():
            if callable(value):
                synthetic[key] = value()
            else:
                synthetic[key] = value

        # Add additional features
        for key, value_func in additional_features.items():
            synthetic[key] = value_func()

        # Set ACL based on is_public_acl
        is_public = synthetic.get('is_public_acl', 0)
        synthetic['acl'] = np.random.choice(acl_options[is_public])

        # Calculate risk ratio if needed
        if 'security_score' in synthetic and 'accessibility_score' in synthetic:
            synthetic['risk_ratio'] = (
                    synthetic['accessibility_score'] /
                    max(0.1, synthetic['security_score'])  # Avoid division by zero
            )

        # Add to results
        synthetic_features.append(synthetic)

    log.info(f"[green]Generated {len(synthetic_features)} synthetic bucket features[/green]")
    return synthetic_features


def augment_with_perturbations(df, perturbation_factor=1.5):
    """
    Augment real data by adding perturbations to existing samples.
    This is different from generating completely synthetic data.

    Args:
        df: DataFrame with bucket features
        perturbation_factor: Controls the number of perturbations per real sample

    Returns:
        DataFrame with perturbed samples
    """
    if df.empty:
        return pd.DataFrame()

    num_samples = len(df)
    num_perturbations = int(num_samples * perturbation_factor)

    # If dataset is very small, create at least 10 perturbations
    num_perturbations = max(num_perturbations, min(10, num_samples * 5))

    log.info(f"[cyan]Creating {num_perturbations} perturbed samples from {num_samples} real samples[/cyan]")

    # Select columns for perturbation
    binary_cols = []
    numeric_cols = []
    categorical_cols = []
    excluded_cols = ['bucket_name', 'source_file']  # Don't perturb these

    for col in df.columns:
        if col in excluded_cols:
            continue

        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif set(df[col].unique()).issubset({0, 1}):
            binary_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            numeric_cols.append(col)

    perturbed_samples = []

    for _ in range(num_perturbations):
        # Select a random sample to perturb
        sample_idx = np.random.randint(0, num_samples)
        original_sample = df.iloc[sample_idx].copy()

        # Create a perturbed version
        perturbed_sample = original_sample.copy()

        # Give it a new name
        perturbed_sample['bucket_name'] = f"{original_sample['bucket_name']}_perturbed_{_}"
        perturbed_sample['source_file'] = f"{original_sample['source_file']}_perturbed"

        # Perturb binary features (flip with small probability)
        for col in binary_cols:
            if np.random.random() < 0.2:  # 20% chance to flip
                perturbed_sample[col] = 1 - original_sample[col]

        # Perturb numeric features (add small random noise)
        for col in numeric_cols:
            # Skip highly correlated scores that should be derived, not perturbed
            if col in ['security_score', 'risk_ratio']:
                continue

            original_value = original_sample[col]

            # Determine scale of perturbation based on the column's range
            col_std = df[col].std()
            if pd.isna(col_std) or col_std == 0:
                # If std is 0 or NaN, use a very small perturbation
                scale = 0.1
            else:
                scale = col_std * 0.3  # 30% of std as scale

            # Add random noise
            perturbed_value = original_value + np.random.normal(0, scale)

            # Ensure non-negative for most features
            if col.endswith('_enabled') or col.endswith('_blocks') or col.endswith('_rules'):
                perturbed_value = max(0, perturbed_value)
                # For integer features, round to nearest integer
                perturbed_value = round(perturbed_value)

            perturbed_sample[col] = perturbed_value

        # For categorical features like ACL, maintain consistency with is_public_acl
        if 'acl' in categorical_cols and 'is_public_acl' in binary_cols:
            # If is_public_acl was flipped, update ACL accordingly
            if original_sample['is_public_acl'] != perturbed_sample['is_public_acl']:
                if perturbed_sample['is_public_acl'] == 1:
                    perturbed_sample['acl'] = np.random.choice(['public-read', 'public-read-write'])
                else:
                    perturbed_sample['acl'] = 'private'

        # Add to results
        perturbed_samples.append(perturbed_sample)

    # Convert to DataFrame
    perturbed_df = pd.DataFrame(perturbed_samples)

    # Recalculate derived columns to maintain consistency
    if 'security_score' in perturbed_df.columns and 'accessibility_score' in perturbed_df.columns:
        perturbed_df['risk_ratio'] = (
                perturbed_df['accessibility_score'] /
                perturbed_df['security_score'].clip(lower=0.1)  # Avoid division by zero
        )

    return perturbed_df


def generate_anomalous_samples(df, num_anomalies=5):
    """
    Generate specific anomalous samples that don't follow normal patterns.

    Args:
        df: DataFrame with bucket features
        num_anomalies: Number of anomalous samples to generate

    Returns:
        DataFrame with anomalous samples
    """
    if df.empty:
        return pd.DataFrame()

    log.info(f"[cyan]Generating {num_anomalies} specifically anomalous samples[/cyan]")

    # Define specific anomaly patterns
    anomaly_patterns = [
        # Public bucket with all security features enabled (rare combination)
        {
            'is_public_acl': 1,
            'has_public_policy': 1,
            'versioning_enabled': 1,
            'logging_enabled': 1,
            'encryption_enabled': 1,
            'public_access_blocks': 4,  # All blocks enabled despite public ACL (inconsistent)
            'security_score': lambda: np.random.uniform(7.0, 9.0),  # High security despite public
            'accessibility_score': lambda: np.random.uniform(7.0, 9.0)  # High accessibility despite blocks
        },
        # Bucket with extreme policy complexity but minimal security
        {
            'is_public_acl': 0,
            'has_public_policy': 1,
            'bucket_policy_complexity': lambda: np.random.randint(8, 15),  # Very complex policy
            'versioning_enabled': 0,
            'logging_enabled': 0,
            'encryption_enabled': 0,
            'security_score': lambda: np.random.uniform(1.0, 3.0)
        },
        # Website bucket without CORS (unusual for a website)
        {
            'is_public_acl': 1,
            'website_enabled': 1,
            'cors_enabled': 0,  # No CORS for a website bucket (unusual)
            'secure_transport': 0,
            'security_score': lambda: np.random.uniform(2.0, 4.0)
        },
        # Extreme feature imbalance - all advanced features but no basic security
        {
            'versioning_enabled': 0,
            'logging_enabled': 0,
            'encryption_enabled': 0,
            'intelligent_tiering_enabled': 1,
            'analytics_enabled': 1,
            'inventory_enabled': 1,
            'replication_enabled': 1,
            'accelerate_enabled': 1,
            'security_score': lambda: np.random.uniform(1.0, 3.0)
        },
        # Private bucket with website config but no proper permissions
        {
            'is_public_acl': 0,
            'has_public_policy': 0,
            'website_enabled': 1,  # Website enabled but not accessible (misconfiguration)
            'accessibility_score': lambda: np.random.uniform(0.0, 2.0)  # Low accessibility despite website
        }
    ]

    anomalous_samples = []

    # Generate samples based on the patterns
    for i in range(num_anomalies):
        # Select a pattern (cycling through them)
        pattern = anomaly_patterns[i % len(anomaly_patterns)]

        # Start with a random real sample as base
        base_idx = np.random.randint(0, len(df))
        sample = df.iloc[base_idx].copy()

        # Apply anomaly pattern
        for key, value in pattern.items():
            if callable(value):
                sample[key] = value()
            else:
                sample[key] = value

        # Update name and source
        sample['bucket_name'] = f"anomalous_sample_{i}"
        sample['source_file'] = f"synthetic/anomalous_{i}.tf"

        # Set ACL based on is_public_acl
        if 'is_public_acl' in sample:
            if sample['is_public_acl'] == 1:
                sample['acl'] = np.random.choice(['public-read', 'public-read-write'])
            else:
                sample['acl'] = 'private'

        # Recalculate risk ratio
        if 'security_score' in sample and 'accessibility_score' in sample:
            sample['risk_ratio'] = (
                    sample['accessibility_score'] /
                    max(0.1, sample['security_score'])
            )

        anomalous_samples.append(sample)

    # Convert to DataFrame
    anomalous_df = pd.DataFrame(anomalous_samples)

    return anomalous_df