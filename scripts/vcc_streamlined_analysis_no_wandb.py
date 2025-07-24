#!/usr/bin/env python3
"""
🧬 STREAMLINED Virtual Cell Challenge Analysis (No W&B)
Using existing processed data for fast development and analysis.
Optimized for quick results without W&B complexity.
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

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB

def setup_results_directories():
    """Set up result directories in existing structure."""
    
    base_path = Path.cwd()
    
    directories = {
        'results': base_path / 'data' / 'results',
        'figures': base_path / 'data' / 'results' / 'figures',
        'metrics': base_path / 'data' / 'results' / 'metrics',
        'reports': base_path / 'data' / 'results' / 'reports'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ready: {name} -> {path}")
    
    return directories

def analyze_available_datasets():
    """Analyze what processed datasets are available."""
    
    processed_dir = Path("data/processed")
    datasets = {}
    
    print("🔍 Scanning available processed datasets...")
    
    if processed_dir.exists():
        for file in processed_dir.glob("*.h5ad"):
            try:
                size_gb = file.stat().st_size / (1024**3)
                datasets[file.stem] = {
                    'path': file,
                    'size_gb': size_gb,
                    'recommended': size_gb < 2.0  # Recommend files under 2GB
                }
                status = "🟢 RECOMMENDED" if size_gb < 2.0 else "🟡 LARGE" if size_gb < 10.0 else "🔴 VERY LARGE"
                print(f"  {status} {file.name} ({size_gb:.1f} GB)")
            except Exception as e:
                print(f"  ❌ Error reading {file.name}: {e}")
    
    return datasets

def load_dataset_smart(dataset_path, sample_size=None):
    """Smart dataset loading with optional sampling."""
    
    print(f"📊 Loading dataset: {dataset_path}")
    start_memory = get_memory_usage()
    
    # Load data
    adata = ad.read_h5ad(dataset_path)
    
    original_size = adata.n_obs
    
    # Optional sampling for very large datasets
    if sample_size and adata.n_obs > sample_size:
        print(f"🎲 Sampling {sample_size:,} cells from {original_size:,} cells")
        sample_indices = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[sample_indices, :].copy()
    
    load_memory = get_memory_usage()
    
    print(f"✅ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"💾 Memory usage: {load_memory:.2f} GB (+{load_memory - start_memory:.2f} GB)")
    
    return adata

def comprehensive_analysis(adata, dataset_name, output_dirs):
    """Perform comprehensive analysis on the dataset."""
    
    print(f"🔬 Analyzing {dataset_name}...")
    
    # Basic dataset info
    dataset_info = {
        'name': dataset_name,
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'cell_types': list(adata.obs.get('cell_type', pd.Series([])).unique()) if 'cell_type' in adata.obs else [],
        'perturbations': list(adata.obs.get('gene', pd.Series([])).unique()[:20]) if 'gene' in adata.obs else [],  # First 20
        'columns': list(adata.obs.columns),
        'memory_gb': get_memory_usage()
    }
    
    print(f"📋 Dataset: {dataset_info['n_cells']:,} cells, {dataset_info['n_genes']:,} genes")
    print(f"📋 Available columns: {', '.join(dataset_info['columns'])}")
    
    # Analyze perturbations if available
    results = {'dataset_info': dataset_info}
    
    if 'gene' in adata.obs:
        print("🧬 Analyzing perturbations...")
        pert_stats = adata.obs['gene'].value_counts()
        
        results['perturbation_analysis'] = {
            'total_perturbations': len(pert_stats),
            'mean_cells_per_pert': float(pert_stats.mean()),
            'median_cells_per_pert': float(pert_stats.median()),
            'max_cells_per_pert': int(pert_stats.max()),
            'min_cells_per_pert': int(pert_stats.min())
        }
        
        print(f"  📊 {len(pert_stats)} perturbations found")
        print(f"  📈 Avg: {pert_stats.mean():.1f} cells/perturbation")
    
    # Analyze cell types if available
    if 'cell_type' in adata.obs:
        print("🦠 Analyzing cell types...")
        cell_type_stats = adata.obs['cell_type'].value_counts()
        
        results['cell_type_analysis'] = {
            'total_cell_types': len(cell_type_stats),
            'cell_type_distribution': dict(cell_type_stats.head(10))
        }
        
        print(f"  🦠 {len(cell_type_stats)} cell types found")
    
    # Gene expression analysis
    print("🧬 Analyzing gene expression...")
    
    # Basic expression stats
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix
        mean_expr = np.array(adata.X.mean(axis=0)).flatten()
        total_expr = np.array(adata.X.sum(axis=1)).flatten()
    else:
        # Dense matrix
        mean_expr = adata.X.mean(axis=0)
        total_expr = adata.X.sum(axis=1)
    
    results['expression_analysis'] = {
        'mean_genes_per_cell': float(np.mean(total_expr)),
        'median_genes_per_cell': float(np.median(total_expr)),
        'mean_expression_per_gene': float(np.mean(mean_expr)),
        'highly_expressed_genes': int(np.sum(mean_expr > np.percentile(mean_expr, 95)))
    }
    
    print(f"  📊 Mean expression/cell: {np.mean(total_expr):.1f}")
    print(f"  📊 Highly expressed genes: {int(np.sum(mean_expr > np.percentile(mean_expr, 95)))}")
    
    return results

def create_comprehensive_visualizations(adata, dataset_name, results, output_dirs):
    """Create comprehensive visualizations."""
    
    print(f"📈 Creating visualizations for {dataset_name}...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Virtual Cell Challenge Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Perturbation analysis (if available)
    if 'gene' in adata.obs:
        pert_stats = adata.obs['gene'].value_counts()
        top_perts = pert_stats.head(15)
        
        axes[0, 0].barh(range(len(top_perts)), top_perts.values)
        axes[0, 0].set_yticks(range(len(top_perts)))
        axes[0, 0].set_yticklabels(top_perts.index, fontsize=8)
        axes[0, 0].set_xlabel('Number of Cells')
        axes[0, 0].set_title('Top 15 Perturbations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perturbation distribution
        axes[0, 1].hist(pert_stats.values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Cells per Perturbation')
        axes[0, 1].set_ylabel('Number of Perturbations')
        axes[0, 1].set_title('Perturbation Size Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No perturbation\ndata available', 
                       ha='center', va='center', fontsize=12)
        axes[0, 0].set_title('Perturbation Analysis')
        axes[0, 1].text(0.5, 0.5, 'No perturbation\ndata available', 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Perturbation Distribution')
    
    # 2. Cell type analysis (if available)
    if 'cell_type' in adata.obs:
        cell_type_stats = adata.obs['cell_type'].value_counts()
        if len(cell_type_stats) <= 10:
            axes[0, 2].pie(cell_type_stats.values, labels=cell_type_stats.index, 
                          autopct='%1.1f%%', startangle=90)
        else:
            top_types = cell_type_stats.head(10)
            axes[0, 2].bar(range(len(top_types)), top_types.values)
            axes[0, 2].set_xticks(range(len(top_types)))
            axes[0, 2].set_xticklabels(top_types.index, rotation=45, ha='right')
            axes[0, 2].set_ylabel('Number of Cells')
        axes[0, 2].set_title('Cell Type Distribution')
    else:
        axes[0, 2].text(0.5, 0.5, 'No cell type\ndata available', 
                       ha='center', va='center', fontsize=12)
        axes[0, 2].set_title('Cell Type Analysis')
    
    # 3. Expression analysis
    if hasattr(adata.X, 'toarray'):
        total_expr = np.array(adata.X.sum(axis=1)).flatten()
        mean_expr = np.array(adata.X.mean(axis=0)).flatten()
    else:
        total_expr = adata.X.sum(axis=1)
        mean_expr = adata.X.mean(axis=0)
    
    # Total expression per cell
    axes[1, 0].hist(total_expr, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Total Expression per Cell')
    axes[1, 0].set_ylabel('Number of Cells')
    axes[1, 0].set_title('Expression Distribution per Cell')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean expression per gene
    axes[1, 1].hist(mean_expr, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Mean Expression per Gene')
    axes[1, 1].set_ylabel('Number of Genes')
    axes[1, 1].set_title('Expression Distribution per Gene')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    summary_text = f"""
Dataset Summary:
• Cells: {results['dataset_info']['n_cells']:,}
• Genes: {results['dataset_info']['n_genes']:,}
• Cell Types: {len(results['dataset_info']['cell_types'])}
• Perturbations: {len(results['dataset_info']['perturbations'])}

Expression Stats:
• Mean expr/cell: {results['expression_analysis']['mean_genes_per_cell']:.1f}
• Median expr/cell: {results['expression_analysis']['median_genes_per_cell']:.1f}
• High expr genes: {results['expression_analysis']['highly_expressed_genes']}

Memory: {results['dataset_info']['memory_gb']:.2f} GB
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save figure
    figure_file = output_dirs['figures'] / f'{dataset_name}_comprehensive_analysis.png'
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved: {figure_file}")
    
    return figure_file

def save_results(results, dataset_name, output_dirs):
    """Save comprehensive results."""
    
    print(f"💾 Saving results for {dataset_name}...")
    
    # Save JSON results
    results_file = output_dirs['results'] / f'{dataset_name}_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create Excel report
    excel_file = output_dirs['reports'] / f'{dataset_name}_report.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = [
            ['Dataset Name', dataset_name],
            ['Total Cells', results['dataset_info']['n_cells']],
            ['Total Genes', results['dataset_info']['n_genes']],
            ['Cell Types', len(results['dataset_info']['cell_types'])],
            ['Memory Usage (GB)', f"{results['dataset_info']['memory_gb']:.2f}"]
        ]
        
        if 'perturbation_analysis' in results:
            summary_data.extend([
                ['Total Perturbations', results['perturbation_analysis']['total_perturbations']],
                ['Mean Cells/Perturbation', f"{results['perturbation_analysis']['mean_cells_per_pert']:.1f}"],
                ['Median Cells/Perturbation', results['perturbation_analysis']['median_cells_per_pert']]
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Expression stats
        expr_data = [
            ['Mean Genes per Cell', results['expression_analysis']['mean_genes_per_cell']],
            ['Median Genes per Cell', results['expression_analysis']['median_genes_per_cell']],
            ['Mean Expression per Gene', results['expression_analysis']['mean_expression_per_gene']],
            ['Highly Expressed Genes', results['expression_analysis']['highly_expressed_genes']]
        ]
        
        expr_df = pd.DataFrame(expr_data, columns=['Metric', 'Value'])
        expr_df.to_excel(writer, sheet_name='Expression', index=False)
    
    print(f"✅ Results saved: {results_file}")
    print(f"📊 Excel report saved: {excel_file}")
    
    return results_file, excel_file

def main():
    """Main analysis pipeline."""
    
    print("🧬 Virtual Cell Challenge - Streamlined Analysis (No W&B)")
    print("=" * 60)
    
    start_time = datetime.now()
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    output_dirs = setup_results_directories()
    
    # Analyze available datasets
    datasets = analyze_available_datasets()
    
    if not datasets:
        print("❌ No processed datasets found in data/processed/")
        print("Please ensure you have .h5ad files in the data/processed/ directory")
        return
    
    # Select dataset for analysis
    print(f"\n🎯 Available datasets:")
    recommended = [name for name, info in datasets.items() if info['recommended']]
    
    if recommended:
        print(f"\n🟢 Recommended (small datasets for quick analysis):")
        for name in recommended:
            info = datasets[name]
            print(f"  • {name} ({info['size_gb']:.1f} GB)")
        
        # Use the smallest recommended dataset
        selected_dataset = min(recommended, key=lambda x: datasets[x]['size_gb'])
        print(f"\n🎯 Auto-selecting: {selected_dataset}")
    else:
        # Use the smallest available dataset
        selected_dataset = min(datasets.keys(), key=lambda x: datasets[x]['size_gb'])
        print(f"\n🎯 Auto-selecting smallest: {selected_dataset}")
    
    dataset_info = datasets[selected_dataset]
    
    try:
        # Load and analyze dataset
        sample_size = 50000 if dataset_info['size_gb'] > 5.0 else None  # Sample large datasets
        adata = load_dataset_smart(dataset_info['path'], sample_size)
        
        # Perform analysis
        results = comprehensive_analysis(adata, selected_dataset, output_dirs)
        
        # Create visualizations
        figure_file = create_comprehensive_visualizations(adata, selected_dataset, results, output_dirs)
        
        # Save results
        results_file, excel_file = save_results(results, selected_dataset, output_dirs)
        
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"⏰ Duration: {duration}")
        print(f"📊 Dataset: {selected_dataset}")
        print(f"📋 Results: {results_file}")
        print(f"📈 Visualization: {figure_file}")
        print(f"📊 Excel Report: {excel_file}")
        
        final_memory = get_memory_usage()
        print(f"💾 Final memory usage: {final_memory:.2f} GB")
        
        print(f"\n📋 Key Findings:")
        print(f"• {results['dataset_info']['n_cells']:,} cells analyzed")
        print(f"• {results['dataset_info']['n_genes']:,} genes analyzed")
        if 'perturbation_analysis' in results:
            print(f"• {results['perturbation_analysis']['total_perturbations']} perturbations found")
        print(f"• {len(results['dataset_info']['cell_types'])} cell types identified")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if 'adata' in locals():
            del adata
        gc.collect()
        print("✅ Analysis complete and cleaned up")

if __name__ == "__main__":
    main() 