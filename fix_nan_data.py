#!/usr/bin/env python3
"""
Quick fix script to clean NaN values from existing processed data.
Run this before running the main pipeline to fix immediate NaN issues.
"""

import os
import numpy as np
import pandas as pd
import logging
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_nan_in_arrays(output_dir):
    """Fix NaN values in existing numpy arrays."""

    features_dir = f"{output_dir}/features"

    if not os.path.exists(features_dir):
        console.print(f"[red]Features directory not found: {features_dir}[/red]")
        return

    # Files to fix
    array_files = ['X_train.npy', 'X_test.npy', 'X_val.npy']

    for filename in array_files:
        filepath = os.path.join(features_dir, filename)

        if os.path.exists(filepath):
            try:
                # Load array
                data = np.load(filepath)
                console.print(f"[cyan]Processing {filename}: shape {data.shape}[/cyan]")

                # Check for NaN values
                nan_count = np.isnan(data).sum()
                if nan_count > 0:
                    console.print(f"[yellow]Found {nan_count} NaN values in {filename}[/yellow]")

                    # Replace NaN with 0
                    data_clean = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

                    # Save cleaned data
                    np.save(filepath, data_clean)
                    console.print(f"[green]Fixed NaN values in {filename}[/green]")
                else:
                    console.print(f"[green]No NaN values found in {filename}[/green]")

            except Exception as e:
                console.print(f"[red]Error processing {filename}: {str(e)}[/red]")
        else:
            console.print(f"[yellow]File not found: {filepath}[/yellow]")

    # Fix raw features CSV
    raw_features_path = os.path.join(features_dir, "raw_features.csv")
    if os.path.exists(raw_features_path):
        try:
            df = pd.read_csv(raw_features_path)
            console.print(f"[cyan]Processing raw_features.csv: shape {df.shape}[/cyan]")

            # Check for NaN in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            nan_counts = df[numeric_cols].isnull().sum()
            total_nans = nan_counts.sum()

            if total_nans > 0:
                console.print(f"[yellow]Found {total_nans} NaN values in raw_features.csv[/yellow]")

                # Fill NaN values
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if col.endswith('_enabled') or col in ['is_public_acl', 'has_public_policy']:
                            df[col] = df[col].fillna(0)
                        elif col.endswith('_score'):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(0)

                # Save cleaned data
                df.to_csv(raw_features_path, index=False)
                console.print(f"[green]Fixed NaN values in raw_features.csv[/green]")
            else:
                console.print(f"[green]No NaN values found in raw_features.csv[/green]")

        except Exception as e:
            console.print(f"[red]Error processing raw_features.csv: {str(e)}[/red]")


def create_synthetic_data_if_needed(output_dir):
    """Create synthetic data if no real data exists."""

    features_dir = f"{output_dir}/features"

    # Check if we have any data
    has_data = False
    for filename in ['X_train.npy', 'X_test.npy']:
        if os.path.exists(os.path.join(features_dir, filename)):
            has_data = True
            break

    if not has_data:
        console.print("[yellow]No training data found. Creating synthetic data...[/yellow]")

        # Create synthetic features based on your Terraform files
        synthetic_features = []

        # Analyze your actual files to create realistic synthetic data
        # Private configurations (from correct folder)
        for i in range(11):  # 11 files in correct folder
            feature = {
                'bucket_name': f'private_bucket_{i}',
                'source_file': f'correct/private_{i}.tf',
                'acl': 'private',
                'is_public_acl': 0,
                'has_public_policy': 0,
                'versioning_enabled': np.random.choice([0, 1], p=[0.6, 0.4]),
                'logging_enabled': np.random.choice([0, 1], p=[0.7, 0.3]),
                'encryption_enabled': np.random.choice([0, 1], p=[0.5, 0.5]),
                'lifecycle_rules': np.random.randint(0, 3),
                'secure_transport': np.random.choice([0, 1], p=[0.8, 0.2]),
                'bucket_policy_complexity': np.random.randint(0, 2),
                'public_access_blocks': 4,  # All blocks enabled for private
                'cors_enabled': 0,
                'website_enabled': 0,
                'replication_enabled': 0,
                'object_lock_enabled': 0,
                'intelligent_tiering_enabled': 0,
                'analytics_enabled': 0,
                'inventory_enabled': 0,
                'accelerate_enabled': 0,
                'assumed_anomalous': 0  # Normal
            }

            # Calculate derived features
            security_features = [
                feature['versioning_enabled'],
                feature['logging_enabled'],
                feature['encryption_enabled'],
                feature['secure_transport'],
                1,  # private ACL
                1  # no public policy
            ]
            feature['security_score'] = sum(security_features) / len(security_features) * 10
            feature['accessibility_score'] = 0  # Private buckets have low accessibility
            feature['risk_ratio'] = 0.1

            synthetic_features.append(feature)

        # Public configurations (from misconfig folder)
        public_acls = ['public-read', 'public-read-write']

        for i in range(17):  # 17 files in misconfig folder
            acl_type = np.random.choice(public_acls)
            feature = {
                'bucket_name': f'public_bucket_{i}',
                'source_file': f'misconfig/public_{i}.tf',
                'acl': acl_type,
                'is_public_acl': 1,
                'has_public_policy': np.random.choice([0, 1], p=[0.6, 0.4]),
                'versioning_enabled': np.random.choice([0, 1], p=[0.9, 0.1]),  # Usually disabled
                'logging_enabled': np.random.choice([0, 1], p=[0.8, 0.2]),  # Usually disabled
                'encryption_enabled': np.random.choice([0, 1], p=[0.7, 0.3]),  # Sometimes enabled
                'lifecycle_rules': np.random.randint(0, 2),
                'secure_transport': 0,  # Usually no secure transport for public
                'bucket_policy_complexity': np.random.randint(0, 3),
                'public_access_blocks': np.random.randint(0, 2),  # Few or no blocks
                'cors_enabled': np.random.choice([0, 1], p=[0.7, 0.3]),
                'website_enabled': np.random.choice([0, 1], p=[0.6, 0.4]),
                'replication_enabled': 0,
                'object_lock_enabled': 0,
                'intelligent_tiering_enabled': 0,
                'analytics_enabled': 0,
                'inventory_enabled': 0,
                'accelerate_enabled': 0,
                'assumed_anomalous': 1  # Anomalous
            }

            # Calculate derived features
            security_features = [
                feature['versioning_enabled'],
                feature['logging_enabled'],
                feature['encryption_enabled'],
                feature['secure_transport'],
                0,  # public ACL penalty
                0 if feature['has_public_policy'] == 1 else 1  # policy penalty
            ]
            feature['security_score'] = sum(security_features) / len(security_features) * 10
            feature['accessibility_score'] = 8 + np.random.uniform(0, 2)  # High accessibility
            feature['risk_ratio'] = feature['accessibility_score'] / max(0.1, feature['security_score'])

            synthetic_features.append(feature)

        # Convert to DataFrame
        df = pd.DataFrame(synthetic_features)

        # Save raw features
        df.to_csv(os.path.join(features_dir, 'raw_features.csv'), index=False)

        # Create feature matrix
        feature_cols = [col for col in df.columns if
                        col not in ['bucket_name', 'acl', 'source_file', 'assumed_anomalous']]
        X = df[feature_cols].values
        y = df['assumed_anomalous'].values

        # Simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val = X_test[:len(X_test) // 2]  # Use half of test as validation
        X_test = X_test[len(X_test) // 2:]
        y_val = y_test[:len(y_test) // 2]
        y_test = y_test[len(y_test) // 2:]

        # Apply scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        # Save arrays
        np.save(os.path.join(features_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(features_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(features_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(features_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(features_dir, 'y_val.npy'), y_val)

        # Save feature names
        pd.DataFrame({'feature_name': feature_cols}).to_csv(
            os.path.join(features_dir, 'feature_names.csv'), index=False
        )

        console.print(f"[green]Created synthetic data with {len(X_train)} training samples[/green]")


def main():
    """Main function to fix NaN issues."""
    output_dir = "data/output"

    # Create output directories
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    console.print("[bold blue]Fixing NaN issues in anomaly detection pipeline[/bold blue]")

    # Fix NaN values in existing data
    fix_nan_in_arrays(output_dir)

    # Create synthetic data if needed
    create_synthetic_data_if_needed(output_dir)

    console.print("[green]NaN fix completed! You can now run the main pipeline.[/green]")


if __name__ == "__main__":
    main()