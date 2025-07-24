#!/usr/bin/env python3
"""
Command-line interface for the Virtual Cell Challenge toolkit.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .preprocessing import preprocess_perturbation_data, validate_data_format
from .data_loading import ExperimentConfig
from . import __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_command(args):
    """Validate data format command."""
    import anndata
    
    logger.info(f"Validating data format for {args.input}")
    adata = anndata.read_h5ad(args.input)
    
    validation_results = validate_data_format(adata)
    
    print("🔍 Data Format Validation Results:")
    print(f"   Valid: {'✅' if validation_results['is_valid'] else '❌'}")
    
    if validation_results['errors']:
        print("   Errors:")
        for error in validation_results['errors']:
            print(f"      • {error}")
    
    if validation_results['warnings']:
        print("   Warnings:")
        for warning in validation_results['warnings']:
            print(f"      • {warning}")
    
    print("\n📊 Data Info:")
    for key, value in validation_results['data_info'].items():
        print(f"   • {key}: {value}")


def preprocess_command(args):
    """Preprocess data command."""
    logger.info(f"Preprocessing data from {args.input}")
    
    results = preprocess_perturbation_data(
        adata_path=args.input,
        output_path=args.output,
        perturbation_column=args.pert_col,
        control_label=args.control_label,
        residual_expression=args.residual_expression,
        cell_residual_expression=args.cell_residual_expression,
        min_cells=args.min_cells
    )
    
    print("🧬 Preprocessing Results:")
    print(f"   • Initial cells: {results['initial_stats']['n_cells']:,}")
    print(f"   • Final cells: {results['final_stats']['n_cells']:,}")
    print(f"   • Cells removed: {results['cells_removed']:,}")
    print(f"   • Retention rate: {results['filtering_efficiency']['cell_retention_rate']:.1%}")


def config_command(args):
    """Validate configuration command."""
    logger.info(f"Validating configuration {args.config}")
    
    try:
        config = ExperimentConfig.from_toml(args.config)
        config.validate()
        
        print("✅ Configuration is valid!")
        print("\n📋 Configuration Summary:")
        print(config.summary())
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)


def info_command(args):
    """Display information about the Virtual Cell Challenge."""
    print(f"""
🧬 Virtual Cell Challenge Toolkit v{__version__}

The Virtual Cell Challenge represents an ambitious effort to build AI-powered 
models that can predict how cells respond to various perturbations.

Key Components:
• STATE Model: State Transition and Embedding model
• Cell-Load Library: PyTorch-based data loading framework  
• Cell_Eval Framework: Comprehensive evaluation metrics

Datasets:
• Tahoe-100M: 100M cells from drug perturbation experiments
• scBaseCount: 230M+ cells across organisms and tissues
• Replogle: High-quality genetic perturbation screens

For more information, visit: https://github.com/your-repo/virtual-cell-challenge
    """)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Virtual Cell Challenge toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Virtual Cell Challenge v{__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate', 
        help='Validate data format'
    )
    validate_parser.add_argument(
        'input', 
        help='Input H5AD file to validate'
    )
    validate_parser.set_defaults(func=validate_command)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess', 
        help='Preprocess perturbation data'
    )
    preprocess_parser.add_argument(
        'input', 
        help='Input H5AD file'
    )
    preprocess_parser.add_argument(
        '-o', '--output', 
        help='Output H5AD file (optional)'
    )
    preprocess_parser.add_argument(
        '--pert-col', 
        default='gene',
        help='Perturbation column name (default: gene)'
    )
    preprocess_parser.add_argument(
        '--control-label', 
        default='non-targeting',
        help='Control label (default: non-targeting)'
    )
    preprocess_parser.add_argument(
        '--residual-expression', 
        type=float, 
        default=0.30,
        help='Residual expression threshold (default: 0.30)'
    )
    preprocess_parser.add_argument(
        '--cell-residual-expression', 
        type=float, 
        default=0.50,
        help='Cell residual expression threshold (default: 0.50)'
    )
    preprocess_parser.add_argument(
        '--min-cells', 
        type=int, 
        default=30,
        help='Minimum cells per perturbation (default: 30)'
    )
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config', 
        help='Validate TOML configuration'
    )
    config_parser.add_argument(
        'config', 
        help='TOML configuration file'
    )
    config_parser.set_defaults(func=config_command)
    
    # Info command
    info_parser = subparsers.add_parser(
        'info', 
        help='Display toolkit information'
    )
    info_parser.set_defaults(func=info_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 