"""
Dataset analysis tools for single-cell perturbation data.

Provides comprehensive statistical analysis and biological insights.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_dataset(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    cell_type_column: str = "cell_type", 
    batch_column: str = "gem_group",
    control_label: str = "non-targeting",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of a single-cell perturbation dataset.
    
    Args:
        adata: AnnData object
        perturbation_column: Column with perturbation information
        cell_type_column: Column with cell type information
        batch_column: Column with batch information
        control_label: Label for control perturbations
        save_path: Optional path to save detailed results
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info("Starting comprehensive dataset analysis...")
    
    results = {
        "overview": {},
        "perturbations": {},
        "cell_types": {},
        "batches": {},
        "quality": {},
        "expression": {},
        "embeddings": {},
        "recommendations": []
    }
    
    # Basic overview
    results["overview"] = _analyze_overview(adata)
    
    # Perturbation analysis
    results["perturbations"] = _analyze_perturbations(
        adata, perturbation_column, control_label
    )
    
    # Cell type analysis
    results["cell_types"] = _analyze_cell_types(
        adata, cell_type_column, perturbation_column
    )
    
    # Batch analysis
    results["batches"] = _analyze_batches(
        adata, batch_column, perturbation_column, cell_type_column
    )
    
    # Quality control analysis
    results["quality"] = _analyze_quality_control(
        adata, perturbation_column, control_label
    )
    
    # Expression analysis
    results["expression"] = _analyze_expression_patterns(adata)
    
    # Embedding analysis
    results["embeddings"] = _analyze_embeddings(adata)
    
    # Generate recommendations
    results["recommendations"] = _generate_recommendations(results)
    
    logger.info("Dataset analysis completed successfully")
    return results


def _analyze_overview(adata: anndata.AnnData) -> Dict[str, Any]:
    """Analyze basic dataset overview statistics."""
    overview = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_umi_counts": int(adata.X.sum()) if hasattr(adata.X, 'sum') else None,
        "mean_genes_per_cell": float(np.mean((adata.X > 0).sum(axis=1))),
        "mean_umi_per_cell": float(np.mean(adata.X.sum(axis=1))),
        "sparsity": float(1.0 - (adata.X > 0).sum() / (adata.n_obs * adata.n_vars)),
        "has_raw": adata.raw is not None,
        "layers": list(adata.layers.keys()) if adata.layers else [],
        "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
    }
    
    return overview


def _analyze_perturbations(
    adata: anndata.AnnData, 
    perturbation_column: str, 
    control_label: str
) -> Dict[str, Any]:
    """Analyze perturbation distribution and characteristics."""
    if perturbation_column not in adata.obs.columns:
        return {"error": f"Column '{perturbation_column}' not found"}
    
    perturbations = adata.obs[perturbation_column]
    pert_counts = perturbations.value_counts()
    
    analysis = {
        "total_perturbations": len(pert_counts),
        "control_perturbations": 1 if control_label in pert_counts.index else 0,
        "treatment_perturbations": len(pert_counts) - (1 if control_label in pert_counts.index else 0),
        "cells_per_perturbation": {
            "mean": float(pert_counts.mean()),
            "median": float(pert_counts.median()),
            "min": int(pert_counts.min()),
            "max": int(pert_counts.max()),
            "std": float(pert_counts.std())
        },
        "control_cells": int(pert_counts.get(control_label, 0)),
        "treatment_cells": int(pert_counts.drop(control_label, errors='ignore').sum()),
        "top_perturbations": pert_counts.head(10).to_dict(),
    }
    
    return analysis


def _analyze_cell_types(
    adata: anndata.AnnData,
    cell_type_column: str,
    perturbation_column: str
) -> Dict[str, Any]:
    """Analyze cell type distribution and perturbation coverage."""
    if cell_type_column not in adata.obs.columns:
        return {"error": f"Column '{cell_type_column}' not found"}
    
    cell_types = adata.obs[cell_type_column]
    ct_counts = cell_types.value_counts()
    
    analysis = {
        "total_cell_types": len(ct_counts),
        "cells_per_type": {
            "mean": float(ct_counts.mean()),
            "median": float(ct_counts.median()),
            "min": int(ct_counts.min()),
            "max": int(ct_counts.max()),
        },
        "cell_type_distribution": ct_counts.to_dict()
    }
    
    return analysis


def _analyze_batches(
    adata: anndata.AnnData,
    batch_column: str,
    perturbation_column: str,
    cell_type_column: str
) -> Dict[str, Any]:
    """Analyze batch effects and distribution."""
    if batch_column not in adata.obs.columns:
        return {"error": f"Column '{batch_column}' not found"}
    
    batches = adata.obs[batch_column]
    batch_counts = batches.value_counts()
    
    analysis = {
        "total_batches": len(batch_counts),
        "cells_per_batch": {
            "mean": float(batch_counts.mean()),
            "median": float(batch_counts.median()),
            "min": int(batch_counts.min()),
            "max": int(batch_counts.max()),
        },
        "batch_distribution": batch_counts.to_dict()
    }
    
    return analysis


def _analyze_quality_control(
    adata: anndata.AnnData,
    perturbation_column: str,
    control_label: str
) -> Dict[str, Any]:
    """Analyze data quality and perturbation effectiveness."""
    try:
        from ..preprocessing import analyze_perturbation_quality
        
        quality_report = analyze_perturbation_quality(
            adata, perturbation_column, control_label
        )
        
        if len(quality_report) == 0:
            return {"error": "No perturbations found for quality analysis"}
        
        analysis = {
            "total_perturbations_analyzed": len(quality_report),
            "effective_perturbations": int(quality_report['is_effective'].sum()),
            "effectiveness_rate": float(quality_report['is_effective'].mean()),
            "mean_knockdown_percent": float(quality_report['knockdown_percent'].mean()),
            "median_knockdown_percent": float(quality_report['knockdown_percent'].median()),
        }
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Quality control analysis failed: {e}")
        return {"error": str(e)}


def _analyze_expression_patterns(adata: anndata.AnnData) -> Dict[str, Any]:
    """Analyze gene expression patterns and statistics."""
    analysis = {}
    
    # Basic expression statistics
    if hasattr(adata.X, 'toarray'):  # sparse matrix
        expr_data = adata.X.toarray()
    else:
        expr_data = adata.X
    
    analysis["expression_stats"] = {
        "mean_expression": float(np.mean(expr_data)),
        "median_expression": float(np.median(expr_data)),
        "std_expression": float(np.std(expr_data)),
        "min_expression": float(np.min(expr_data)),
        "max_expression": float(np.max(expr_data)),
        "zeros_fraction": float(np.mean(expr_data == 0))
    }
    
    return analysis


def _analyze_embeddings(adata: anndata.AnnData) -> Dict[str, Any]:
    """Analyze available embeddings and dimensionality reductions."""
    analysis = {
        "available_embeddings": list(adata.obsm.keys()) if adata.obsm else [],
        "embedding_analysis": {}
    }
    
    for embedding_name in analysis["available_embeddings"]:
        embedding = adata.obsm[embedding_name]
        
        emb_analysis = {
            "dimensions": embedding.shape[1],
            "mean_values": embedding.mean(axis=0).tolist()[:5],  # First 5 dims
            "std_values": embedding.std(axis=0).tolist()[:5],
        }
        
        analysis["embedding_analysis"][embedding_name] = emb_analysis
    
    return analysis


def _generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate analysis recommendations based on results."""
    recommendations = []
    
    # Data size recommendations
    n_cells = results["overview"]["n_cells"]
    n_genes = results["overview"]["n_genes"]
    
    if n_cells < 1000:
        recommendations.append("Dataset is quite small (<1000 cells). Consider combining with other datasets.")
    elif n_cells > 100000:
        recommendations.append("Large dataset (>100K cells). Consider subsampling for initial analysis.")
    
    # Quality control recommendations
    if "quality" in results and "effectiveness_rate" in results["quality"]:
        effectiveness = results["quality"]["effectiveness_rate"]
        if effectiveness < 0.5:
            recommendations.append("Low perturbation effectiveness (<50%). Review quality control parameters.")
    
    return recommendations


def create_dataset_summary(
    adata: anndata.AnnData,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a concise dataset summary table.
    
    Args:
        adata: AnnData object
        save_path: Optional path to save summary CSV
        
    Returns:
        DataFrame with dataset summary
    """
    analysis_results = analyze_dataset(adata)
    
    summary_data = []
    
    # Basic metrics
    overview = analysis_results["overview"]
    summary_data.extend([
        {"metric": "Total cells", "value": overview["n_cells"], "category": "overview"},
        {"metric": "Total genes", "value": overview["n_genes"], "category": "overview"},
        {"metric": "Mean UMI per cell", "value": f"{overview['mean_umi_per_cell']:.0f}", "category": "overview"},
        {"metric": "Data sparsity", "value": f"{overview['sparsity']:.1%}", "category": "overview"},
    ])
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        logger.info(f"Dataset summary saved to {save_path}")
    
    return summary_df 