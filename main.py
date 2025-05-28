#!/usr/bin/env python3
"""
Main script for anomaly detection in IaC scripts.
Orchestrates the entire pipeline: feature extraction, model training, and visualization.
Updated with better error handling and robustness.
"""

import os
import argparse
import logging
import sys
from rich.console import Console
from rich.panel import Panel

# Import pipeline components
from extract_features import extract_features_main
from train_models import train_models_main
from generate_visualizations import generate_visualizations_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("anomaly_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("anomaly_detector")
console = Console()


def create_directories():
    """Create all necessary directories for the pipeline."""
    directories = [
        "data/terraform/correct",
        "data/terraform/misconfig",
        "data/output/features",
        "data/output/models",
        "data/output/plots",
        "data/output/plots/features",
        "data/output/plots/anomaly_scores",
        "data/output/plots/anomaly_detection",
        "data/output/plots/dimension_reduction",
        "data/output/plots/model_comparison"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def check_terraform_files(terraform_dir):
    """Check if Terraform files exist and provide guidance."""
    correct_dir = os.path.join(terraform_dir, "correct")
    misconfig_dir = os.path.join(terraform_dir, "misconfig")

    correct_files = []
    misconfig_files = []

    if os.path.exists(correct_dir):
        correct_files = [f for f in os.listdir(correct_dir) if f.endswith('.tf')]

    if os.path.exists(misconfig_dir):
        misconfig_files = [f for f in os.listdir(misconfig_dir) if f.endswith('.tf')]

    console.print(f"[cyan]Found {len(correct_files)} correct configuration files[/cyan]")
    console.print(f"[cyan]Found {len(misconfig_files)} misconfiguration files[/cyan]")

    if len(correct_files) == 0 and len(misconfig_files) == 0:
        console.print("[yellow]No Terraform files found. The pipeline will generate synthetic data.[/yellow]")

    return len(correct_files) + len(misconfig_files) > 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection in IaC Scripts using Unsupervised Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run full pipeline with defaults
  python main.py --skip-extraction                 # Skip feature extraction
  python main.py --models isolation_forest,autoencoder  # Train specific models
  python main.py --output-dir custom_output        # Use custom output directory
        """
    )

    parser.add_argument(
        "--terraform-dir", type=str, default="data/terraform",
        help="Directory containing Terraform files (default: data/terraform)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/output",
        help="Directory to store output files (default: data/output)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip feature extraction phase"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training phase"
    )
    parser.add_argument(
        "--skip-visualization", action="store_true",
        help="Skip visualization generation phase"
    )
    parser.add_argument(
        "--models", type=str, default="isolation_forest,one_class_svm,autoencoder,local_outlier_factor",
        help="Comma-separated list of models to train (default: all)"
    )
    parser.add_argument(
        "--fix-nan", action="store_true",
        help="Run NaN fix before starting the pipeline"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def run_nan_fix(output_dir):
    """Run the NaN fix script."""
    try:
        console.print("[yellow]Running NaN fix...[/yellow]")
        from fix_nan_data import main as fix_nan_main
        fix_nan_main()
        console.print("[green]NaN fix completed successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error running NaN fix: {str(e)}[/red]")
        logger.error(f"NaN fix failed: {str(e)}")


def validate_environment():
    """Validate that required dependencies are available."""
    required_packages = [
        'sklearn', 'numpy', 'pandas', 'matplotlib', 'seaborn',
        'rich', 'plotly', 'hcl2', 'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        console.print(f"[red]Missing required packages: {', '.join(missing_packages)}[/red]")
        console.print("[yellow]Please install missing packages using pip install -r requirements.txt[/yellow]")
        return False

    return True


def main():
    """Main function to orchestrate the anomaly detection pipeline."""
    console.print(Panel(
        "[bold blue]Anomaly Detection in IaC Scripts[/bold blue]\n"
        "Detecting Cloud Misconfigurations Using Unsupervised Learning",
        style="blue"
    ))

    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate environment
    if not validate_environment():
        console.print("[red]Environment validation failed. Please install required dependencies.[/red]")
        sys.exit(1)

    # Create necessary directories
    create_directories()

    # Check Terraform files
    has_tf_files = check_terraform_files(args.terraform_dir)

    # Run NaN fix if requested
    if args.fix_nan:
        run_nan_fix(args.output_dir)

    # Parse models list
    models_list = [model.strip() for model in args.models.split(',') if model.strip()]
    valid_models = ['isolation_forest', 'one_class_svm', 'autoencoder', 'local_outlier_factor']
    models_list = [model for model in models_list if model in valid_models]

    if not models_list:
        console.print("[yellow]No valid models specified. Using default models.[/yellow]")
        models_list = valid_models

    console.print(f"[cyan]Training models: {', '.join(models_list)}[/cyan]")

    try:
        # Phase 1: Feature Extraction
        if not args.skip_extraction:
            console.print(Panel("Phase 1: Feature Extraction", style="bold green"))

            try:
                extract_features_main(args.terraform_dir, args.output_dir)
                console.print("[green]✓ Feature extraction completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]✗ Feature extraction failed: {str(e)}[/red]")
                logger.error(f"Feature extraction failed: {str(e)}")

                # Try to run NaN fix and continue
                console.print("[yellow]Attempting to fix data issues...[/yellow]")
                run_nan_fix(args.output_dir)
        else:
            console.print("[yellow]Skipping feature extraction phase[/yellow]")

        # Phase 2: Model Training
        if not args.skip_training:
            console.print(Panel("Phase 2: Model Training", style="bold cyan"))

            try:
                train_models_main(args.output_dir, models_list)
                console.print("[green]✓ Model training completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]✗ Model training failed: {str(e)}[/red]")
                logger.error(f"Model training failed: {str(e)}")

                # Continue to visualization if possible
                console.print("[yellow]Continuing to visualization phase...[/yellow]")
        else:
            console.print("[yellow]Skipping model training phase[/yellow]")

        # Phase 3: Visualization Generation
        if not args.skip_visualization:
            console.print(Panel("Phase 3: Visualization Generation", style="bold magenta"))

            try:
                generate_visualizations_main(args.output_dir, models_list)
                console.print("[green]✓ Visualization generation completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]✗ Visualization generation failed: {str(e)}[/red]")
                logger.error(f"Visualization generation failed: {str(e)}")
        else:
            console.print("[yellow]Skipping visualization generation phase[/yellow]")

        # Final summary
        console.print(Panel(
            "[bold green]Anomaly Detection Pipeline Completed![/bold green]\n\n"
            f"Results saved to: {args.output_dir}\n"
            f"Models trained: {', '.join(models_list)}\n"
            f"Check the output directory for:\n"
            f"  • Feature analysis: {args.output_dir}/features/\n"
            f"  • Model results: {args.output_dir}/models/\n"
            f"  • Visualizations: {args.output_dir}/plots/",
            style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error in pipeline: {str(e)}[/red]")
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()