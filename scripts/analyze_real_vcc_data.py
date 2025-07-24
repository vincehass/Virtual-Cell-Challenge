#!/usr/bin/env python3
"""
Comprehensive analysis of real Virtual Cell Challenge data.

This script demonstrates the complete pipeline using the user's actual VCC dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from pathlib import Path

# Set environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import our Virtual Cell Challenge toolkit
import sys
sys.path.append('src')

from virtual_cell_challenge.preprocessing import (
    preprocess_perturbation_data, 
    analyze_perturbation_quality,
    validate_data_format
)
from virtual_cell_challenge.analysis import analyze_dataset
from virtual_cell_challenge.evaluation import CellEvalMetrics

def adapt_vcc_data_format(adata):
    """
    Adapt the VCC data format to match our pipeline expectations.
    
    Your data uses:
    - 'target_gene' instead of 'gene'
    - 'batch' instead of 'gem_group'
    - No 'cell_type' column
    """
    print("🔧 Adapting data format for pipeline compatibility...")
    
    # Create a copy to avoid modifying original
    adata_adapted = adata.copy()
    
    # Map column names
    if 'target_gene' in adata_adapted.obs.columns:
        adata_adapted.obs['gene'] = adata_adapted.obs['target_gene']
    
    if 'batch' in adata_adapted.obs.columns:
        adata_adapted.obs['gem_group'] = adata_adapted.obs['batch']
    
    # Add a default cell type since your data appears to be single cell type
    adata_adapted.obs['cell_type'] = 'primary_cells'  # Or whatever cell type this is
    
    # Add gene names from the separate file if available
    try:
        gene_names_df = pd.read_csv('data/vcc_data/gene_names.csv')
        # The CSV seems to have gene names as column headers, need to process this
        gene_names = gene_names_df.columns.tolist()
        if len(gene_names) == len(adata_adapted.var):
            adata_adapted.var['gene_name'] = gene_names
            print(f"✅ Added {len(gene_names)} gene names")
        else:
            print(f"⚠️ Gene names count mismatch: {len(gene_names)} names vs {len(adata_adapted.var)} genes")
    except Exception as e:
        print(f"⚠️ Could not load gene names: {e}")
        # Use gene_id as gene_name fallback
        adata_adapted.var['gene_name'] = adata_adapted.var['gene_id']
    
    print("✅ Data format adaptation complete!")
    return adata_adapted

def compute_basic_embeddings(adata):
    """Compute basic embeddings since they're not present in the data."""
    print("🧮 Computing basic embeddings...")
    
    # Compute PCA
    import scanpy as sc
    sc.settings.verbosity = 1
    
    # Basic preprocessing for PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Compute PCA
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    
    # Create HVG embedding (subset of highly variable genes)
    hvg_genes = adata.var['highly_variable']
    if hvg_genes.sum() > 0:
        X_hvg = adata.X[:, hvg_genes]
        if hasattr(X_hvg, 'toarray'):
            X_hvg = X_hvg.toarray()
        adata.obsm['X_hvg'] = X_hvg
        print(f"✅ Created X_hvg embedding: {X_hvg.shape}")
    
    # Compute UMAP for visualization
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    
    print("✅ Embeddings computed successfully!")
    return adata

def main():
    """Main analysis pipeline for real VCC data."""
    print("🧬 Virtual Cell Challenge - Real Data Analysis")
    print("=" * 60)
    
    # Load your real data
    print("📊 Loading your real VCC data...")
    adata = ad.read_h5ad('data/vcc_data/adata_Training.h5ad')
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Adapt data format
    adata_adapted = adapt_vcc_data_format(adata)
    
    # Compute embeddings
    adata_processed = compute_basic_embeddings(adata_adapted)
    
    # Data validation
    print("\n🔍 Validating data format...")
    validation = validate_data_format(adata_processed)
    
    print(f"✅ Validation results:")
    print(f"   • Valid: {'✅' if validation['is_valid'] else '❌'}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"   • Warning: {warning}")
    
    # Comprehensive dataset analysis
    print("\n📈 Performing comprehensive analysis...")
    analysis_results = analyze_dataset(
        adata_processed,
        perturbation_column='gene',
        cell_type_column='cell_type',
        batch_column='gem_group',
        control_label='non-targeting'
    )
    
    # Print key results
    overview = analysis_results['overview']
    perturbations = analysis_results.get('perturbations', {})
    quality = analysis_results.get('quality', {})
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   • Dataset: {overview['n_cells']:,} cells, {overview['n_genes']:,} genes")
    print(f"   • Sparsity: {overview['sparsity']:.1%}")
    
    if perturbations:
        print(f"   • Perturbations: {perturbations['total_perturbations']}")
        print(f"   • Control cells: {perturbations['control_cells']:,}")
        print(f"   • Treatment cells: {perturbations['treatment_cells']:,}")
    
    if quality and 'effectiveness_rate' in quality:
        print(f"   • Perturbation effectiveness: {quality['effectiveness_rate']:.1%}")
        print(f"   • Mean knockdown: {quality['mean_knockdown_percent']:.1f}%")
    
    # Quality control analysis
    print("\n🔬 Quality control analysis...")
    try:
        quality_report = analyze_perturbation_quality(
            adata_processed, 
            perturbation_column='gene',
            control_label='non-targeting'
        )
        
        print(f"Quality analysis results:")
        print(f"   • Perturbations analyzed: {len(quality_report)}")
        print(f"   • Effective perturbations: {quality_report['is_effective'].sum()}")
        print(f"   • Effectiveness rate: {quality_report['is_effective'].mean():.1%}")
        print(f"   • Mean knockdown: {quality_report['knockdown_percent'].mean():.1f}%")
        
        # Top performers
        top_performers = quality_report.nlargest(5, 'knockdown_percent')
        print(f"\n🥇 Top 5 performing perturbations:")
        for _, row in top_performers.iterrows():
            print(f"   • {row['perturbation']}: {row['knockdown_percent']:.1f}% knockdown")
            
    except Exception as e:
        print(f"⚠️ Quality analysis failed: {e}")
    
    # Save processed data
    print(f"\n💾 Saving processed data...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "vcc_training_processed.h5ad"
    adata_processed.write(output_path)
    print(f"✅ Processed data saved to: {output_path}")
    
    # Create basic visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # Perturbation distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Real VCC Data Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Perturbation counts
    pert_counts = adata_processed.obs['gene'].value_counts()
    top_perts = pert_counts.head(20)
    axes[0, 0].barh(range(len(top_perts)), top_perts.values)
    axes[0, 0].set_yticks(range(len(top_perts)))
    axes[0, 0].set_yticklabels(top_perts.index, fontsize=8)
    axes[0, 0].set_xlabel('Cell Count')
    axes[0, 0].set_title('Top 20 Perturbations')
    
    # Plot 2: Batch distribution
    batch_counts = adata_processed.obs['gem_group'].value_counts()
    axes[0, 1].hist(batch_counts.values, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Cells per Batch')
    axes[0, 1].set_ylabel('Number of Batches')
    axes[0, 1].set_title('Batch Distribution')
    
    # Plot 3: UMAP if available
    if 'X_umap' in adata_processed.obsm:
        umap_coords = adata_processed.obsm['X_umap']
        scatter = axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                   c=adata_processed.obs['gem_group'].astype('category').cat.codes,
                                   alpha=0.6, s=0.5)
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('UMAP 2')
        axes[1, 0].set_title('UMAP Embedding (colored by batch)')
    
    # Plot 4: Expression statistics
    if hasattr(adata_processed.X, 'toarray'):
        expr_data = adata_processed.X.toarray()
    else:
        expr_data = adata_processed.X
    
    cell_totals = np.array(expr_data.sum(axis=1)).flatten()
    gene_detection = np.array((expr_data > 0).sum(axis=1)).flatten()
    axes[1, 1].scatter(cell_totals, gene_detection, alpha=0.6, s=0.5)
    axes[1, 1].set_xlabel('Total UMI Count')
    axes[1, 1].set_ylabel('Genes Detected')
    axes[1, 1].set_title('Expression Statistics')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "vcc_analysis_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_path}")
    plt.show()
    
    print(f"\n🎉 Analysis complete!")
    print(f"✅ Your real VCC data is ready for virtual cell modeling!")
    print(f"\n📁 Generated files:")
    print(f"   • Processed data: {output_path}")
    print(f"   • Analysis plots: {plot_path}")
    print(f"   • Configuration: config/real_data_config.toml")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Use the processed data for training virtual cell models")
    print(f"   2. Implement STATE model architecture")
    print(f"   3. Run zero-shot and few-shot learning experiments")
    print(f"   4. Evaluate using Cell_Eval metrics")

if __name__ == "__main__":
    main() 