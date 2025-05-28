#!/usr/bin/env python3
"""
terraform_parser.py - Parse Terraform files to extract S3 bucket configurations

This module handles parsing Terraform files to extract S3 bucket configurations and
security-related attributes with improved handling of modern Terraform syntax.
"""

import os
import re
import json
import logging
import glob
import hcl2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.logging import RichHandler

# Setup logging
log = logging.getLogger("rich")
console = Console()


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("data/terraform/correct", exist_ok=True)
    os.makedirs("data/terraform/misconfig", exist_ok=True)
    os.makedirs("data/output/plots", exist_ok=True)
    os.makedirs("data/output/models", exist_ok=True)
    os.makedirs("data/output/features", exist_ok=True)


def extract_s3_with_regex(content, file_path):
    """Extract S3 bucket details using regex when HCL parsing fails."""
    bucket_features = []

    # Find all S3 bucket resources
    bucket_pattern = r'resource\s+"aws_s3_bucket"\s+"([^"]+)"\s*{([^}]*(?:{[^}]*}[^}]*)*)}'
    bucket_matches = re.finditer(bucket_pattern, content, re.MULTILINE | re.DOTALL)

    buckets_found = {}

    for match in bucket_matches:
        resource_name = match.group(1)
        bucket_block = match.group(2)

        # Extract bucket name
        bucket_name_match = re.search(r'bucket\s*=\s*"([^"]+)"', bucket_block)
        bucket_name = bucket_name_match.group(1) if bucket_name_match else resource_name

        # Initialize bucket features with defaults
        bucket_config = {
            'bucket_name': bucket_name,
            'source_file': file_path,
            'acl': 'private',  # Default ACL
            'is_public_acl': 0,
            'has_public_policy': 0,
            'versioning_enabled': 0,
            'logging_enabled': 0,
            'encryption_enabled': 0,
            'lifecycle_rules': 0,
            'secure_transport': 0,
            'bucket_policy_complexity': 0,
            'public_access_blocks': 0,
            'cors_enabled': 0,
            'website_enabled': 0,
            'replication_enabled': 0,
            'object_lock_enabled': 0,
            'intelligent_tiering_enabled': 0,
            'analytics_enabled': 0,
            'inventory_enabled': 0,
            'accelerate_enabled': 0
        }

        # Check for inline ACL in bucket resource
        acl_match = re.search(r'acl\s*=\s*"([^"]+)"', bucket_block)
        if acl_match:
            bucket_config['acl'] = acl_match.group(1)
            bucket_config['is_public_acl'] = 1 if acl_match.group(1) in ['public-read', 'public-read-write'] else 0

        buckets_found[resource_name] = bucket_config

    # Now look for separate ACL resources
    acl_pattern = r'resource\s+"aws_s3_bucket_acl"\s+"([^"]+)"\s*{([^}]*(?:{[^}]*}[^}]*)*)}'
    acl_matches = re.finditer(acl_pattern, content, re.MULTILINE | re.DOTALL)

    for match in acl_matches:
        acl_resource_name = match.group(1)
        acl_block = match.group(2)

        # Find which bucket this ACL applies to
        bucket_ref_match = re.search(r'bucket\s*=\s*aws_s3_bucket\.([^.]+)\.id', acl_block)
        if bucket_ref_match:
            bucket_resource_name = bucket_ref_match.group(1)
            if bucket_resource_name in buckets_found:
                # Extract ACL value
                acl_match = re.search(r'acl\s*=\s*"([^"]+)"', acl_block)
                if acl_match:
                    acl_value = acl_match.group(1)
                    buckets_found[bucket_resource_name]['acl'] = acl_value
                    buckets_found[bucket_resource_name]['is_public_acl'] = 1 if acl_value in ['public-read',
                                                                                              'public-read-write'] else 0

    # Look for bucket policies
    policy_pattern = r'resource\s+"aws_s3_bucket_policy"\s+"([^"]+)"\s*{([^}]*(?:{[^}]*}[^}]*)*)}'
    policy_matches = re.finditer(policy_pattern, content, re.MULTILINE | re.DOTALL)

    for match in policy_matches:
        policy_resource_name = match.group(1)
        policy_block = match.group(2)

        # Find which bucket this policy applies to
        bucket_ref_match = re.search(r'bucket\s*=\s*aws_s3_bucket\.([^.]+)\.id', policy_block)
        if bucket_ref_match:
            bucket_resource_name = bucket_ref_match.group(1)
            if bucket_resource_name in buckets_found:
                # Extract policy
                policy_match = re.search(r'policy\s*=\s*jsonencode\s*\(\s*{([^}]+(?:{[^}]*}[^}]*)*)\s*}\s*\)',
                                         policy_block, re.DOTALL)
                if policy_match:
                    policy_content = policy_match.group(1)
                    # Check if policy allows public access
                    if 'Principal' in policy_content and '"*"' in policy_content and 'Allow' in policy_content:
                        buckets_found[bucket_resource_name]['has_public_policy'] = 1
                        buckets_found[bucket_resource_name]['bucket_policy_complexity'] = 1
                    elif 'Deny' in policy_content:
                        buckets_found[bucket_resource_name]['secure_transport'] = 1
                        buckets_found[bucket_resource_name]['bucket_policy_complexity'] = 1

    # Look for encryption configurations
    encryption_pattern = r'resource\s+"aws_s3_bucket_server_side_encryption_configuration"\s+"([^"]+)"\s*{([^}]*(?:{[^}]*}[^}]*)*)}'
    encryption_matches = re.finditer(encryption_pattern, content, re.MULTILINE | re.DOTALL)

    for match in encryption_matches:
        encryption_block = match.group(2)
        # Find which bucket this applies to
        bucket_ref_match = re.search(r'bucket\s*=\s*aws_s3_bucket\.([^.]+)\.id', encryption_block)
        if bucket_ref_match:
            bucket_resource_name = bucket_ref_match.group(1)
            if bucket_resource_name in buckets_found:
                buckets_found[bucket_resource_name]['encryption_enabled'] = 1

    # Look for logging configurations
    logging_pattern = r'resource\s+"aws_s3_bucket_logging"\s+"([^"]+)"\s*{([^}]*(?:{[^}]*}[^}]*)*)}'
    logging_matches = re.finditer(logging_pattern, content, re.MULTILINE | re.DOTALL)

    for match in logging_matches:
        logging_block = match.group(2)
        # Find which bucket this applies to
        bucket_ref_match = re.search(r'bucket\s*=\s*aws_s3_bucket\.([^.]+)\.id', logging_block)
        if bucket_ref_match:
            bucket_resource_name = bucket_ref_match.group(1)
            if bucket_resource_name in buckets_found:
                buckets_found[bucket_resource_name]['logging_enabled'] = 1

    # Convert to list and calculate derived features
    for bucket_name, bucket_config in buckets_found.items():
        # Calculate security score
        security_features = [
            bucket_config['versioning_enabled'],
            bucket_config['logging_enabled'],
            bucket_config['encryption_enabled'],
            bucket_config['secure_transport'],
            bucket_config['public_access_blocks'] / 4,  # Normalize
            1 if bucket_config['is_public_acl'] == 0 else 0,  # Reward private ACL
            1 if bucket_config['has_public_policy'] == 0 else 0  # Reward private policy
        ]
        bucket_config['security_score'] = sum(security_features) / len(security_features) * 10

        # Calculate accessibility score
        accessibility_features = [
            bucket_config['is_public_acl'] * 4,  # Public ACL major factor
            bucket_config['has_public_policy'] * 4,  # Public policy major factor
            bucket_config['website_enabled'] * 2,
            bucket_config['cors_enabled'] * 1,
            (4 - bucket_config['public_access_blocks'])  # Lack of blocks increases accessibility
        ]
        bucket_config['accessibility_score'] = sum(accessibility_features)

        # Risk ratio
        bucket_config['risk_ratio'] = bucket_config['accessibility_score'] / max(0.1, bucket_config['security_score'])

        bucket_features.append(bucket_config)

    log.info(f"Extracted {len(bucket_features)} S3 bucket configurations from {file_path}")
    return bucket_features


def parse_tf_file(file_path):
    """
    Parse Terraform file to extract S3 bucket features with improved handling.
    """
    bucket_features = []

    try:
        # Skip state files
        if file_path.endswith('.tfstate') or file_path.endswith('.backup'):
            log.debug(f"Skipping state file: {file_path}")
            return bucket_features

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip empty files
        if not content.strip():
            log.debug(f"Skipping empty file: {file_path}")
            return bucket_features

        # Check if file contains S3 bucket resources
        if 'aws_s3_bucket' not in content:
            log.debug(f"No S3 bucket resources found in {file_path}")
            return bucket_features

        # Try HCL parsing first
        try:
            parsed_content = hcl2.loads(content)
            bucket_features = extract_from_hcl(parsed_content, file_path)
            if bucket_features:
                log.debug(f"Successfully parsed {file_path} with HCL2, found {len(bucket_features)} buckets")
                return bucket_features
        except Exception as e:
            log.debug(f"HCL parsing failed for {file_path}: {str(e)}")

        # Fall back to regex parsing
        log.debug(f"Using regex parsing for {file_path}")
        bucket_features = extract_s3_with_regex(content, file_path)

    except Exception as e:
        log.error(f"Error parsing {file_path}: {str(e)}")

    return bucket_features


def extract_from_hcl(parsed_content, file_path):
    """
    Extract bucket features from HCL parsed content.
    """
    bucket_features = []

    if 'resource' not in parsed_content:
        return bucket_features

    resources = parsed_content['resource']
    buckets = {}

    # Process S3 bucket resources
    if 'aws_s3_bucket' in resources:
        s3_resources = resources['aws_s3_bucket']
        if isinstance(s3_resources, dict):
            for resource_name, attributes in s3_resources.items():
                bucket_name = attributes.get('bucket', resource_name)

                bucket_config = {
                    'bucket_name': bucket_name,
                    'source_file': file_path,
                    'acl': attributes.get('acl', 'private'),
                    'is_public_acl': 0,
                    'has_public_policy': 0,
                    'versioning_enabled': 0,
                    'logging_enabled': 0,
                    'encryption_enabled': 0,
                    'lifecycle_rules': 0,
                    'secure_transport': 0,
                    'bucket_policy_complexity': 0,
                    'public_access_blocks': 0,
                    'cors_enabled': 0,
                    'website_enabled': 0,
                    'replication_enabled': 0,
                    'object_lock_enabled': 0,
                    'intelligent_tiering_enabled': 0,
                    'analytics_enabled': 0,
                    'inventory_enabled': 0,
                    'accelerate_enabled': 0
                }

                # Check ACL
                acl = attributes.get('acl', 'private')
                if isinstance(acl, str):
                    bucket_config['acl'] = acl
                    bucket_config['is_public_acl'] = 1 if acl in ['public-read', 'public-read-write'] else 0

                buckets[resource_name] = bucket_config

    # Process other S3 resources (ACL, policy, etc.)
    for resource_type, resource_items in resources.items():
        if resource_type.startswith('aws_s3_bucket_') and resource_type != 'aws_s3_bucket':
            if isinstance(resource_items, dict):
                for resource_name, attributes in resource_items.items():
                    bucket_ref = attributes.get('bucket')
                    target_bucket = None

                    # Find target bucket
                    if isinstance(bucket_ref, dict):
                        for ref_type, ref_id in bucket_ref.items():
                            if ref_type == 'aws_s3_bucket' and ref_id in buckets:
                                target_bucket = buckets[ref_id]
                                break
                    elif isinstance(bucket_ref, str):
                        # Look for bucket reference pattern
                        for bucket_id, bucket_data in buckets.items():
                            if bucket_data['bucket_name'] == bucket_ref:
                                target_bucket = bucket_data
                                break

                    if target_bucket:
                        # Update bucket based on resource type
                        if resource_type == 'aws_s3_bucket_acl':
                            acl = attributes.get('acl', 'private')
                            target_bucket['acl'] = acl
                            target_bucket['is_public_acl'] = 1 if acl in ['public-read', 'public-read-write'] else 0

                        elif resource_type == 'aws_s3_bucket_policy':
                            policy = attributes.get('policy')
                            if policy:
                                target_bucket['has_public_policy'] = 1 if is_public_policy(policy) else 0
                                target_bucket['bucket_policy_complexity'] = 1
                                target_bucket['secure_transport'] = 1 if has_secure_transport_condition(policy) else 0

                        elif resource_type == 'aws_s3_bucket_server_side_encryption_configuration':
                            target_bucket['encryption_enabled'] = 1

                        elif resource_type == 'aws_s3_bucket_logging':
                            target_bucket['logging_enabled'] = 1

    # Convert to list and calculate derived features
    for bucket_name, bucket_config in buckets.items():
        # Calculate security score
        security_features = [
            bucket_config['versioning_enabled'],
            bucket_config['logging_enabled'],
            bucket_config['encryption_enabled'],
            bucket_config['secure_transport'],
            bucket_config['public_access_blocks'] / 4,  # Normalize
            1 if bucket_config['is_public_acl'] == 0 else 0,  # Reward private ACL
            1 if bucket_config['has_public_policy'] == 0 else 0  # Reward private policy
        ]
        bucket_config['security_score'] = sum(security_features) / len(security_features) * 10

        # Calculate accessibility score
        accessibility_features = [
            bucket_config['is_public_acl'] * 4,  # Public ACL major factor
            bucket_config['has_public_policy'] * 4,  # Public policy major factor
            bucket_config['website_enabled'] * 2,
            bucket_config['cors_enabled'] * 1,
            (4 - bucket_config['public_access_blocks'])  # Lack of blocks increases accessibility
        ]
        bucket_config['accessibility_score'] = sum(accessibility_features)

        # Risk ratio
        bucket_config['risk_ratio'] = bucket_config['accessibility_score'] / max(0.1, bucket_config['security_score'])

        bucket_features.append(bucket_config)

    return bucket_features


def is_public_policy(policy_doc):
    """
    Analyze a bucket policy to determine if it allows public access.
    """
    if not policy_doc:
        return False

    # Handle different policy formats
    if isinstance(policy_doc, str):
        try:
            policy_doc = json.loads(policy_doc)
        except json.JSONDecodeError:
            # Check for public indicators in string
            public_indicators = [
                '"Principal": "*"',
                '"Principal":"*"',
                '"Effect": "Allow"'
            ]
            return any(indicator in policy_doc for indicator in public_indicators)

    if not isinstance(policy_doc, dict):
        return False

    # Check for Statement key
    if "Statement" not in policy_doc:
        return False

    statements = policy_doc["Statement"]
    if not isinstance(statements, list):
        statements = [statements]

    # Analyze each statement
    for statement in statements:
        if not isinstance(statement, dict):
            continue

        # Check if Effect is Allow
        effect = statement.get("Effect", "").lower()
        if effect != "allow":
            continue

        # Check Principal for public access
        principal = statement.get("Principal")
        if principal == "*":
            return True

        if isinstance(principal, dict):
            for key, value in principal.items():
                if value == "*" or (isinstance(value, list) and "*" in value):
                    return True

    return False


def has_secure_transport_condition(policy_doc):
    """
    Check if a bucket policy includes a secure transport condition.
    """
    if not policy_doc:
        return False

    if isinstance(policy_doc, str):
        try:
            policy_doc = json.loads(policy_doc)
        except json.JSONDecodeError:
            return '"SecureTransport"' in policy_doc

    if not isinstance(policy_doc, dict):
        return False

    if "Statement" not in policy_doc:
        return False

    statements = policy_doc["Statement"]
    if not isinstance(statements, list):
        statements = [statements]

    for statement in statements:
        if not isinstance(statement, dict):
            continue

        condition = statement.get("Condition")
        if isinstance(condition, dict):
            bool_equals = condition.get("Bool", {})
            if isinstance(bool_equals, dict):
                if bool_equals.get("aws:SecureTransport") == "false":
                    return True

    return False


def parse_terraform_files(input_dir):
    """
    Parse all Terraform files in the given directory to extract S3 bucket features.
    """
    all_features = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        # Find all Terraform files
        tf_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.tf') and not file.endswith('.tfstate'):
                    tf_files.append(os.path.join(root, file))

        if not tf_files:
            log.warning("No Terraform files found. Creating sample files...")
            create_sample_terraform_files()
            # Look for samples again
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.tf'):
                        tf_files.append(os.path.join(root, file))

        log.info(f"Found {len(tf_files)} Terraform files to process")

        # Process files with progress bar
        process_task = progress.add_task("[cyan]Processing Terraform files...", total=len(tf_files))

        for file_path in tf_files:
            bucket_features = parse_tf_file(file_path)
            all_features.extend(bucket_features)
            progress.update(process_task, advance=1)

    log.info(f"Processed {len(tf_files)} Terraform files, extracted features for {len(all_features)} S3 buckets")

    # Ensure we have some features
    if not all_features:
        log.warning("No bucket features extracted. Creating synthetic data...")
        all_features = create_synthetic_features()

    return all_features


def create_synthetic_features():
    """
    Create synthetic bucket features if no real data is available.
    """
    synthetic_features = []

    # Create some private buckets
    for i in range(8):
        feature = {
            'bucket_name': f'synthetic-private-{i}',
            'source_file': f'synthetic/private_{i}.tf',
            'acl': 'private',
            'is_public_acl': 0,
            'has_public_policy': 0,
            'versioning_enabled': 1 if i % 2 == 0 else 0,
            'logging_enabled': 1 if i % 3 == 0 else 0,
            'encryption_enabled': 1 if i % 2 == 1 else 0,
            'lifecycle_rules': i % 3,
            'secure_transport': 1 if i % 4 == 0 else 0,
            'bucket_policy_complexity': 0,
            'public_access_blocks': 4,
            'cors_enabled': 0,
            'website_enabled': 0,
            'replication_enabled': 0,
            'object_lock_enabled': 0,
            'intelligent_tiering_enabled': 0,
            'analytics_enabled': 0,
            'inventory_enabled': 0,
            'accelerate_enabled': 0,
            'security_score': 7.0 + i * 0.5,
            'accessibility_score': 0.0,
            'risk_ratio': 0.0
        }
        synthetic_features.append(feature)

    # Create some public buckets
    for i in range(5):
        acl_type = 'public-read' if i % 2 == 0 else 'public-read-write'
        feature = {
            'bucket_name': f'synthetic-public-{i}',
            'source_file': f'synthetic/public_{i}.tf',
            'acl': acl_type,
            'is_public_acl': 1,
            'has_public_policy': 1 if i % 2 == 1 else 0,
            'versioning_enabled': 0,
            'logging_enabled': 1 if i == 0 else 0,  # One public bucket with logging
            'encryption_enabled': 1 if i == 1 else 0,  # One public bucket with encryption
            'lifecycle_rules': 0,
            'secure_transport': 0,
            'bucket_policy_complexity': 1 if i % 2 == 1 else 0,
            'public_access_blocks': 0,
            'cors_enabled': 1 if i % 3 == 0 else 0,
            'website_enabled': 1 if i % 2 == 0 else 0,
            'replication_enabled': 0,
            'object_lock_enabled': 0,
            'intelligent_tiering_enabled': 0,
            'analytics_enabled': 0,
            'inventory_enabled': 0,
            'accelerate_enabled': 0,
            'security_score': 1.0 + i * 0.5,
            'accessibility_score': 8.0 + i * 0.5,
            'risk_ratio': 8.0 / max(0.1, 1.0 + i * 0.5)
        }
        synthetic_features.append(feature)

    return synthetic_features


def create_sample_terraform_files():
    """Create sample Terraform files for testing if none exist."""
    os.makedirs("data/terraform/correct", exist_ok=True)
    os.makedirs("data/terraform/misconfig", exist_ok=True)

    # Sample 1: Secure bucket configuration
    with open("data/terraform/correct/secure_bucket.tf", "w") as f:
        f.write("""
resource "aws_s3_bucket" "secure_bucket" {
  bucket = "my-secure-bucket"
}

resource "aws_s3_bucket_acl" "secure_bucket_acl" {
  bucket = aws_s3_bucket.secure_bucket.id
  acl    = "private"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secure_bucket_encryption" {
  bucket = aws_s3_bucket.secure_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
""")

    # Sample 2: Public bucket configuration
    with open("data/terraform/misconfig/public_bucket.tf", "w") as f:
        f.write("""
resource "aws_s3_bucket" "public_website" {
  bucket = "my-public-website"
}

resource "aws_s3_bucket_acl" "public_website_acl" {
  bucket = aws_s3_bucket.public_website.id
  acl    = "public-read"
}

resource "aws_s3_bucket_policy" "public_website_policy" {
  bucket = aws_s3_bucket.public_website.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = "*",
      Action    = "s3:GetObject",
      Resource  = "${aws_s3_bucket.public_website.arn}/*"
    }]
  })
}
""")

    log.info("Created sample Terraform files")