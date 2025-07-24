"""
Complete preprocessing pipeline for single-cell perturbation data.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import anndata
import pandas as pd

from .quality_control import (
    filter_on_target_knockdown,
    set_var_index_to_col,
    analyze_perturbation_quality,
    suspected_discrete_torch,
    suspected_log_torch,
)

logger = logging.getLogger(__name__)


def preprocess_perturbation_data(
    adata_path: str,
    output_path: Optional[str] = None,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    cell_residual_expression: float = 0.50,
    min_cells: int = 30,
    layer: Optional[str] = None,
    var_gene_name: str = "gene_name",
    return_quality_report: bool = True,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for perturbation data.
    
    Args:
        adata_path: Path to input AnnData file
        output_path: Path to save filtered data (optional)
        perturbation_column: Column with perturbation information
        control_label: Label for control cells
        residual_expression: Perturbation-level threshold
        cell_residual_expression: Cell-level threshold  
        min_cells: Minimum cells per perturbation
        layer: Data layer to use
        var_gene_name: Gene name column in var
        return_quality_report: Whether to return detailed quality metrics
        
    Returns:
        Dictionary with processed data and quality metrics
    """
    logger.info(f"Starting preprocessing pipeline for {adata_path}")
    
    # 1. Load data
    logger.info("Loading data...")
    adata = anndata.read_h5ad(adata_path)
    
    initial_stats = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_perturbations': len(adata.obs[perturbation_column].unique()),
    }
    
    logger.info(f"Initial data: {initial_stats['n_cells']} cells, {initial_stats['n_genes']} genes, {initial_stats['n_perturbations']} perturbations")
    
    # 2. Set gene names as index (if needed)
    logger.info("Setting gene names as index...")
    adata = set_var_index_to_col(adata, col=var_gene_name)
    
    # 3. Analyze data quality before filtering
    quality_report = None
    if return_quality_report:
        logger.info("Analyzing perturbation quality...")
        quality_report = analyze_perturbation_quality(
            adata, perturbation_column, control_label, residual_expression, layer
        )
        
        effective_perts = quality_report['is_effective'].sum()
        total_perts = len(quality_report)
        logger.info(f"Quality analysis: {effective_perts}/{total_perts} perturbations show effective knockdown")
    
    # 4. Apply quality control filtering
    logger.info("Applying quality control filtering...")
    filtered_adata = filter_on_target_knockdown(
        adata=adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        residual_expression=residual_expression,
        cell_residual_expression=cell_residual_expression,
        min_cells=min_cells,
        layer=layer,
        var_gene_name=var_gene_name,
    )
    
    final_stats = {
        'n_cells': filtered_adata.n_obs,
        'n_genes': filtered_adata.n_vars,
        'n_perturbations': len(filtered_adata.obs[perturbation_column].unique()),
    }
    
    # 5. Calculate filtering statistics
    cells_removed = initial_stats['n_cells'] - final_stats['n_cells']
    perts_removed = initial_stats['n_perturbations'] - final_stats['n_perturbations']
    
    logger.info(f"Filtering results:")
    logger.info(f"  Cells: {final_stats['n_cells']} ({cells_removed} removed, {100*cells_removed/initial_stats['n_cells']:.1f}%)")
    logger.info(f"  Perturbations: {final_stats['n_perturbations']} ({perts_removed} removed)")
    
    # 6. Save filtered data (if output path provided)
    if output_path:
        logger.info(f"Saving filtered data to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        filtered_adata.write_h5ad(output_path)
    
    # 7. Prepare results
    results = {
        'filtered_adata': filtered_adata,
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'cells_removed': cells_removed,
        'perturbations_removed': perts_removed,
        'filtering_efficiency': {
            'cell_retention_rate': final_stats['n_cells'] / initial_stats['n_cells'],
            'perturbation_retention_rate': final_stats['n_perturbations'] / initial_stats['n_perturbations'],
        }
    }
    
    if quality_report is not None:
        results['quality_report'] = quality_report
        
    logger.info("Preprocessing pipeline completed successfully")
    return results


def validate_data_format(adata: anndata.AnnData) -> Dict[str, Any]:
    """
    Validate that data is in the expected format for cell-load pipeline.
    
    Args:
        adata: AnnData object to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'data_info': {}
    }
    
    # Check required obs columns
    required_obs = ['gene', 'cell_type', 'gem_group']
    for col in required_obs:
        if col not in adata.obs.columns:
            validation_results['errors'].append(f"Missing required obs column: {col}")
            validation_results['is_valid'] = False
    
    # Check var columns
    if 'gene_name' not in adata.var.columns:
        validation_results['warnings'].append("Missing 'gene_name' column in var")
    
    # Check embeddings
    if 'X_hvg' not in adata.obsm:
        validation_results['warnings'].append("Missing 'X_hvg' embedding in obsm")
    
    if 'X_state' not in adata.obsm:
        validation_results['warnings'].append("Missing 'X_state' embedding in obsm")
    
    # Analyze data characteristics
    if hasattr(adata.X, 'toarray'):  # sparse matrix
        sample_data = adata.X[:100].toarray()
    else:
        sample_data = adata.X[:100]
    
    import torch
    sample_tensor = torch.tensor(sample_data, dtype=torch.float32)
    
    validation_results['data_info'] = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'is_sparse': hasattr(adata.X, 'toarray'),
        'appears_discrete': suspected_discrete_torch(sample_tensor),
        'appears_log_transformed': suspected_log_torch(sample_tensor),
        'perturbations': list(adata.obs['gene'].unique()) if 'gene' in adata.obs.columns else [],
        'cell_types': list(adata.obs['cell_type'].unique()) if 'cell_type' in adata.obs.columns else [],
        'batches': list(adata.obs['gem_group'].unique()) if 'gem_group' in adata.obs.columns else [],
    }
    
    return validation_results


def create_dataset_summary(adata: anndata.AnnData, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive summary of the dataset.
    
    Args:
        adata: AnnData object
        save_path: Optional path to save summary CSV
        
    Returns:
        DataFrame with dataset summary
    """
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'metric': 'Total cells',
        'value': adata.n_obs,
        'category': 'overview'
    })
    
    summary_data.append({
        'metric': 'Total genes',
        'value': adata.n_vars,
        'category': 'overview'
    })
    
    if 'gene' in adata.obs.columns:
        # Perturbation statistics
        perturbations = adata.obs['gene'].value_counts()
        summary_data.append({
            'metric': 'Total perturbations',
            'value': len(perturbations),
            'category': 'perturbations'
        })
        
        summary_data.append({
            'metric': 'Mean cells per perturbation',
            'value': perturbations.mean(),
            'category': 'perturbations'
        })
        
        summary_data.append({
            'metric': 'Median cells per perturbation',
            'value': perturbations.median(),
            'category': 'perturbations'
        })
    
    if 'cell_type' in adata.obs.columns:
        # Cell type statistics
        cell_types = adata.obs['cell_type'].value_counts()
        summary_data.append({
            'metric': 'Total cell types',
            'value': len(cell_types),
            'category': 'cell_types'
        })
        
        summary_data.append({
            'metric': 'Mean cells per type',
            'value': cell_types.mean(),
            'category': 'cell_types'
        })
    
    if 'gem_group' in adata.obs.columns:
        # Batch statistics
        batches = adata.obs['gem_group'].value_counts()
        summary_data.append({
            'metric': 'Total batches',
            'value': len(batches),
            'category': 'batches'
        })
        
        summary_data.append({
            'metric': 'Mean cells per batch',
            'value': batches.mean(),
            'category': 'batches'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        logger.info(f"Dataset summary saved to {save_path}")
    
    return summary_df 