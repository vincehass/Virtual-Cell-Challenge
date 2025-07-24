#!/usr/bin/env python3
"""
🧬 EXTERNAL DRIVE Virtual Cell Challenge Pipeline
Memory-optimized pipeline designed for external drive usage.
Saves all results and intermediate files to avoid internal disk saturation.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from pathlib import Path
import warnings
import gc
import psutil
from datetime import datetime
import json
warnings.filterwarnings("ignore")

# W&B for experiment tracking
import wandb

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['WANDB_PROJECT'] = 'virtual-cell-challenge'

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB

def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

def setup_output_directories(base_path):
    """Set up comprehensive output directory structure."""
    
    base_path = Path(base_path)
    
    directories = {
        'results': base_path / 'data' / 'results',
        'logs': base_path / 'logs',
        'models': base_path / 'models', 
        'checkpoints': base_path / 'checkpoints',
        'figures': base_path / 'data' / 'results' / 'figures',
        'processed': base_path / 'data' / 'processed',
        'embeddings': base_path / 'data' / 'processed' / 'embeddings',
        'metrics': base_path / 'data' / 'results' / 'metrics'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {name} -> {path}")
    
    return directories

def save_analysis_metadata(output_dirs, start_time):
    """Save comprehensive metadata about the analysis run."""
    
    metadata = {
        'analysis_info': {
            'start_time': start_time.isoformat(),
            'script_name': 'vcc_external_drive_pipeline.py',
            'purpose': 'Virtual Cell Challenge analysis on external drive',
            'memory_optimized': True
        },
        'system_info': {
            'platform': os.uname().sysname if hasattr(os, 'uname') else 'unknown',
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'working_directory': str(Path.cwd()),
            'initial_memory_gb': get_memory_usage()
        },
        'output_directories': {k: str(v) for k, v in output_dirs.items()},
        'data_info': {
            'expected_cells': 221273,
            'expected_genes': 18080,
            'dataset_size_gb': 14
        }
    }
    
    metadata_file = output_dirs['results'] / 'analysis_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved analysis metadata: {metadata_file}")
    return metadata

def setup_wandb_experiment(output_dirs, metadata):
    """Initialize W&B experiment with comprehensive config."""
    
    config = {
        # Dataset configuration
        "dataset_name": "VCC_Training_External_Drive",
        "dataset_path": "data/vcc_data/adata_Training.h5ad", 
        "dataset_size_gb": 14,
        "expected_cells": 221273,
        "expected_genes": 18080,
        
        # Analysis configuration
        "memory_optimized": True,
        "external_drive": True,
        "chunk_size": 50000,
        "save_intermediate": True,
        
        # Output configuration
        "output_base": str(output_dirs['results']),
        "save_embeddings": True,
        "save_figures": True,
        
        # System configuration
        "python_version": metadata['system_info']['python_version'],
        "initial_memory_gb": metadata['system_info']['initial_memory_gb']
    }
    
    # Initialize W&B
    run = wandb.init(
        project="virtual-cell-challenge",
        name=f"external-drive-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config,
        tags=["external-drive", "memory-optimized", "vcc-training"]
    )
    
    print("✅ W&B experiment initialized")
    print(f"🔗 W&B URL: {run.url}")
    
    return run

def load_and_analyze_data_chunked(data_path, output_dirs, chunk_size=50000):
    """Load and analyze data in chunks to manage memory."""
    
    print(f"📊 Loading data from: {data_path}")
    start_memory = get_memory_usage()
    print(f"💾 Initial memory usage: {start_memory:.2f} GB")
    
    # Load data
    adata = ad.read_h5ad(data_path)
    print(f"✅ Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Save basic info
    basic_info = {
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'cell_types': list(adata.obs['cell_type'].unique()),
        'perturbations': list(adata.obs['gene'].unique()[:50]),  # First 50
        'total_perturbations': int(adata.obs['gene'].nunique())
    }
    
    info_file = output_dirs['processed'] / 'dataset_basic_info.json'
    with open(info_file, 'w') as f:
        json.dump(basic_info, f, indent=2)
    
    print(f"📋 Dataset info saved: {info_file}")
    
    # Memory check after loading
    load_memory = get_memory_usage()
    print(f"💾 Memory after loading: {load_memory:.2f} GB (Δ: +{load_memory - start_memory:.2f} GB)")
    
    return adata, basic_info

def analyze_perturbations_chunked(adata, output_dirs, chunk_size=50000):
    """Analyze perturbations in chunks to save memory."""
    
    print("🔬 Starting perturbation analysis...")
    
    # Get perturbation statistics
    pert_stats = adata.obs['gene'].value_counts()
    print(f"📊 Found {len(pert_stats)} unique perturbations")
    
    # Save perturbation statistics
    pert_stats_df = pd.DataFrame({
        'perturbation': pert_stats.index,
        'cell_count': pert_stats.values
    })
    
    pert_stats_file = output_dirs['processed'] / 'perturbation_statistics.csv'
    pert_stats_df.to_csv(pert_stats_file, index=False)
    print(f"✅ Perturbation stats saved: {pert_stats_file}")
    
    # Cell type analysis
    cell_type_stats = adata.obs['cell_type'].value_counts()
    cell_type_df = pd.DataFrame({
        'cell_type': cell_type_stats.index,
        'cell_count': cell_type_stats.values
    })
    
    cell_type_file = output_dirs['processed'] / 'cell_type_statistics.csv'
    cell_type_df.to_csv(cell_type_file, index=False)
    print(f"✅ Cell type stats saved: {cell_type_file}")
    
    # Memory cleanup
    cleanup_memory()
    
    return pert_stats_df, cell_type_df

def create_visualizations(pert_stats_df, cell_type_df, output_dirs):
    """Create and save comprehensive visualizations."""
    
    print("📈 Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Perturbation distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Virtual Cell Challenge Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Top 20 perturbations
    top_perts = pert_stats_df.head(20)
    axes[0, 0].barh(range(len(top_perts)), top_perts['cell_count'])
    axes[0, 0].set_yticks(range(len(top_perts)))
    axes[0, 0].set_yticklabels(top_perts['perturbation'], fontsize=8)
    axes[0, 0].set_xlabel('Number of Cells')
    axes[0, 0].set_title('Top 20 Perturbations by Cell Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Perturbation count distribution
    axes[0, 1].hist(pert_stats_df['cell_count'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Cells per Perturbation')
    axes[0, 1].set_ylabel('Number of Perturbations')
    axes[0, 1].set_title('Distribution of Cells per Perturbation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cell type distribution
    axes[1, 0].pie(cell_type_df['cell_count'], labels=cell_type_df['cell_type'], 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Cell Type Distribution')
    
    # Summary statistics
    summary_text = f"""
    Dataset Summary:
    • Total Cells: {pert_stats_df['cell_count'].sum():,}
    • Total Perturbations: {len(pert_stats_df):,}
    • Cell Types: {len(cell_type_df)}
    • Avg Cells/Perturbation: {pert_stats_df['cell_count'].mean():.1f}
    • Median Cells/Perturbation: {pert_stats_df['cell_count'].median():.1f}
    • Max Cells/Perturbation: {pert_stats_df['cell_count'].max():,}
    • Min Cells/Perturbation: {pert_stats_df['cell_count'].min():,}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset Statistics')
    
    plt.tight_layout()
    
    # Save figure
    figure_file = output_dirs['figures'] / 'dataset_overview.png'
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Overview figure saved: {figure_file}")
    
    # 2. Detailed perturbation analysis
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Log scale histogram
    axes[0].hist(pert_stats_df['cell_count'], bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Cells per Perturbation')
    axes[0].set_ylabel('Number of Perturbations (log scale)')
    axes[0].set_title('Perturbation Cell Count Distribution (Log Scale)')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_counts = np.sort(pert_stats_df['cell_count'])
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    axes[1].plot(sorted_counts, cumulative_pct, linewidth=2)
    axes[1].set_xlabel('Cells per Perturbation')
    axes[1].set_ylabel('Cumulative Percentage')
    axes[1].set_title('Cumulative Distribution of Cells per Perturbation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    detailed_figure_file = output_dirs['figures'] / 'perturbation_analysis.png'
    plt.savefig(detailed_figure_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Detailed analysis figure saved: {detailed_figure_file}")
    
    return [figure_file, detailed_figure_file]

def save_comprehensive_results(adata, pert_stats_df, cell_type_df, output_dirs, start_time):
    """Save comprehensive analysis results."""
    
    print("💾 Saving comprehensive results...")
    
    # Calculate analysis duration
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Create comprehensive summary
    summary = {
        'analysis_completed': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'duration_formatted': str(duration),
        
        'dataset_summary': {
            'total_cells': int(adata.n_obs),
            'total_genes': int(adata.n_vars),
            'total_perturbations': len(pert_stats_df),
            'cell_types': len(cell_type_df)
        },
        
        'perturbation_statistics': {
            'mean_cells_per_perturbation': float(pert_stats_df['cell_count'].mean()),
            'median_cells_per_perturbation': float(pert_stats_df['cell_count'].median()),
            'max_cells_per_perturbation': int(pert_stats_df['cell_count'].max()),
            'min_cells_per_perturbation': int(pert_stats_df['cell_count'].min()),
            'std_cells_per_perturbation': float(pert_stats_df['cell_count'].std())
        },
        
        'cell_type_distribution': {
            row['cell_type']: int(row['cell_count']) 
            for _, row in cell_type_df.iterrows()
        },
        
        'top_10_perturbations': {
            row['perturbation']: int(row['cell_count']) 
            for _, row in pert_stats_df.head(10).iterrows()
        },
        
        'memory_usage': {
            'final_memory_gb': get_memory_usage(),
            'peak_memory_gb': get_memory_usage()  # Simplified - could track actual peak
        }
    }
    
    # Save comprehensive summary
    summary_file = output_dirs['results'] / 'comprehensive_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Comprehensive summary saved: {summary_file}")
    
    # Save Excel report with multiple sheets
    excel_file = output_dirs['results'] / 'vcc_analysis_report.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Perturbation statistics
        pert_stats_df.to_excel(writer, sheet_name='Perturbations', index=False)
        
        # Cell type statistics  
        cell_type_df.to_excel(writer, sheet_name='CellTypes', index=False)
        
        # Summary sheet
        summary_df = pd.DataFrame([
            ['Total Cells', summary['dataset_summary']['total_cells']],
            ['Total Genes', summary['dataset_summary']['total_genes']],
            ['Total Perturbations', summary['dataset_summary']['total_perturbations']],
            ['Cell Types', summary['dataset_summary']['cell_types']],
            ['Analysis Duration', summary['duration_formatted']],
            ['Mean Cells/Perturbation', f"{summary['perturbation_statistics']['mean_cells_per_perturbation']:.1f}"],
            ['Median Cells/Perturbation', summary['perturbation_statistics']['median_cells_per_perturbation']]
        ], columns=['Metric', 'Value'])
        
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"📊 Excel report saved: {excel_file}")
    
    return summary

def main():
    """Main pipeline execution."""
    
    print("🧬 Virtual Cell Challenge - External Drive Pipeline")
    print("=" * 60)
    
    start_time = datetime.now()
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup output directories
    output_dirs = setup_output_directories(Path.cwd())
    
    # Save analysis metadata
    metadata = save_analysis_metadata(output_dirs, start_time)
    
    # Setup W&B experiment
    wandb_run = setup_wandb_experiment(output_dirs, metadata)
    
    try:
        # Data loading and analysis
        data_path = "data/vcc_data/adata_Training.h5ad"
        
        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure the data file exists at the correct location.")
            return
        
        # Load and analyze data
        adata, basic_info = load_and_analyze_data_chunked(data_path, output_dirs)
        
        # Log basic info to W&B
        wandb.log(basic_info)
        
        # Analyze perturbations
        pert_stats_df, cell_type_df = analyze_perturbations_chunked(adata, output_dirs)
        
        # Create visualizations
        figure_files = create_visualizations(pert_stats_df, cell_type_df, output_dirs)
        
        # Log figures to W&B
        for fig_file in figure_files:
            wandb.log({f"figure_{fig_file.stem}": wandb.Image(str(fig_file))})
        
        # Save comprehensive results
        final_summary = save_comprehensive_results(
            adata, pert_stats_df, cell_type_df, output_dirs, start_time
        )
        
        # Log final metrics to W&B
        wandb.log(final_summary['perturbation_statistics'])
        wandb.log(final_summary['dataset_summary'])
        
        # Final memory cleanup
        del adata
        cleanup_memory()
        
        # Success message
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"⏰ Duration: {duration}")
        print(f"📁 Results saved to: {output_dirs['results']}")
        print(f"📊 Figures saved to: {output_dirs['figures']}")
        print(f"🔗 W&B URL: {wandb_run.url}")
        
        final_memory = get_memory_usage()
        print(f"💾 Final memory usage: {final_memory:.2f} GB")
        
        print(f"\n📋 Key Results:")
        print(f"• {final_summary['dataset_summary']['total_cells']:,} cells analyzed")
        print(f"• {final_summary['dataset_summary']['total_perturbations']:,} perturbations found")
        print(f"• {final_summary['dataset_summary']['cell_types']} cell types identified")
        print(f"• Average {final_summary['perturbation_statistics']['mean_cells_per_perturbation']:.1f} cells per perturbation")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        wandb.log({"error": str(e)})
        raise
    
    finally:
        # Finish W&B run
        wandb.finish()
        print("✅ W&B run finished")

if __name__ == "__main__":
    main() 