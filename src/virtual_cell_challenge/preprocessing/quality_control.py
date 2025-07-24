"""
Quality control utilities for single-cell perturbation data.

Replicates the functionality from cell-load/utils/data_utils.py
"""

import warnings
import logging
from typing import Union, Optional

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _mean(expr) -> float:
    """Return the mean of a dense or sparse 1-D/2-D slice."""
    if sp.issparse(expr):
        return float(expr.mean())
    return float(np.asarray(expr).mean())


def suspected_discrete_torch(x: torch.Tensor, n_cells: int = 100) -> bool:
    """
    Check if data appears to be discrete/raw counts by examining row sums.
    Adapted from validate_normlog function for PyTorch tensors.
    
    Args:
        x: Expression data tensor
        n_cells: Number of cells to check
        
    Returns:
        True if data appears to be discrete counts
    """
    top_n = min(x.shape[0], n_cells)
    rowsum = x[:top_n].sum(dim=1)

    # Check if row sums are integers (fractional part == 0)
    frac_part = rowsum - rowsum.floor()
    return torch.all(torch.abs(frac_part) < 1e-7)


def suspected_log_torch(x: torch.Tensor) -> bool:
    """
    Check if the data is log transformed already.
    
    Args:
        x: Expression data tensor
        
    Returns:
        True if data appears to be log-transformed
    """
    global_max = x.max()
    return global_max.item() < 15.0


def set_var_index_to_col(adata: anndata.AnnData, col: str) -> anndata.AnnData:
    """
    Set the var index to use gene names from a specific column.
    
    Args:
        adata: AnnData object
        col: Column name in adata.var to use as index
        
    Returns:
        AnnData with updated var index
    """
    if col in adata.var.columns:
        adata.var.index = adata.var[col].astype(str)
    else:
        logger.warning(f"Column '{col}' not found in adata.var")
    return adata


def is_on_target_knockdown(
    adata: anndata.AnnData,
    target_gene: str,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    layer: Optional[str] = None,
) -> bool:
    """
    Check if a perturbation shows effective knockdown.
    
    True if average expression of target_gene in perturbed cells is below
    residual_expression × (average expression in control cells).

    Args:
        adata: AnnData object
        target_gene: Gene symbol to check
        perturbation_column: Column in adata.obs holding perturbation identities
        control_label: Category in perturbation_column marking control cells
        residual_expression: Residual fraction (0-1). 0.30 → 70% knockdown
        layer: Use this matrix in adata.layers instead of adata.X

    Returns:
        True if knockdown is effective
        
    Raises:
        KeyError: target_gene not present in adata.var_names
        ValueError: No perturbed cells for target_gene, or control mean is zero
    """
    if target_gene == control_label:
        # Never evaluate the control itself
        return False

    if target_gene not in adata.var_names:
        logger.warning(f"Gene {target_gene!r} not found in adata.var_names")
        return False

    gene_idx = adata.var_names.get_loc(target_gene)
    X = adata.layers[layer] if layer is not None else adata.X

    control_cells = adata.obs[perturbation_column] == control_label
    perturbed_cells = adata.obs[perturbation_column] == target_gene

    if not perturbed_cells.any():
        raise ValueError(f"No cells labelled with perturbation {target_gene!r}.")

    try:
        control_mean = _mean(X[control_cells, gene_idx])
    except Exception:
        control_cells = (adata.obs[perturbation_column] == control_label).values
        control_mean = _mean(X[control_cells, gene_idx])

    if control_mean == 0:
        raise ValueError(
            f"Mean {target_gene!r} expression in control cells is zero; "
            "cannot compute knock-down ratio."
        )

    try:
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])
    except Exception:
        perturbed_cells = (adata.obs[perturbation_column] == target_gene).values
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])

    knockdown_ratio = perturbed_mean / control_mean
    return knockdown_ratio < residual_expression


def filter_on_target_knockdown(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,  # perturbation-level threshold
    cell_residual_expression: float = 0.50,  # cell-level threshold
    min_cells: int = 30,  # minimum cells/perturbation
    layer: Optional[str] = None,
    var_gene_name: str = "gene_name",
) -> anndata.AnnData:
    """
    Filter perturbation data based on knockdown effectiveness.
    
    Three-stage filtering process:
    1. Keep perturbations whose average knock-down ≥ (1-residual_expression)
    2. Within those, keep only cells whose knock-down ≥ (1-cell_residual_expression)  
    3. Discard perturbations that have < min_cells cells remaining after stages 1-2
    
    Control cells are always preserved regardless of these criteria.

    Args:
        adata: AnnData object
        perturbation_column: Column in adata.obs holding perturbation identities
        control_label: Category in perturbation_column marking control cells
        residual_expression: Perturbation-level threshold (0.30 = 70% knockdown required)
        cell_residual_expression: Cell-level threshold (0.50 = 50% knockdown per cell)
        min_cells: Minimum cells per perturbation after filtering
        layer: Use this matrix in adata.layers instead of adata.X
        var_gene_name: Column in adata.var containing gene names
        
    Returns:
        AnnData view satisfying all three criteria
    """
    # Prep
    adata_ = set_var_index_to_col(adata.copy(), col=var_gene_name)
    X = adata_.layers[layer] if layer is not None else adata_.X
    perts = adata_.obs[perturbation_column]
    control_cells = (perts == control_label).values

    # Stage 1: perturbation filter
    perts_to_keep = [control_label]  # always keep controls
    for pert in perts.unique():
        if pert == control_label:
            continue
        if is_on_target_knockdown(
            adata_,
            target_gene=pert,
            perturbation_column=perturbation_column,
            control_label=control_label,
            residual_expression=residual_expression,
            layer=layer,
        ):
            perts_to_keep.append(pert)

    logger.info(f"Stage 1: Keeping {len(perts_to_keep)-1}/{len(perts.unique())-1} perturbations with effective knockdown")

    # Stage 2: cell filter
    keep_mask = np.zeros(adata_.n_obs, dtype=bool)
    keep_mask[control_cells] = True  # retain all controls

    # Cache control means to avoid recomputation
    control_mean_cache = {}

    for pert in perts_to_keep:
        if pert == control_label:
            continue

        if pert not in adata_.var_names:
            continue

        gene_idx = adata_.var_names.get_loc(pert)

        # Control mean for this gene
        if pert not in control_mean_cache:
            try:
                ctrl_mean = _mean(X[control_cells, gene_idx])
            except Exception:
                logger.error(f"Error computing control mean for {pert}")
                continue
            if ctrl_mean == 0:
                logger.warning(f"Mean {pert!r} expression in control cells is zero")
                continue
            control_mean_cache[pert] = ctrl_mean
        else:
            ctrl_mean = control_mean_cache[pert]

        pert_cells = (perts == pert).values
        # Handle sparse matrices properly
        expr_vals = (
            X[pert_cells, gene_idx].toarray().flatten()
            if sp.issparse(X)
            else X[pert_cells, gene_idx]
        )
        ratios = expr_vals / ctrl_mean
        keep_mask[pert_cells] = ratios < cell_residual_expression

    logger.info(f"Stage 2: Keeping {keep_mask.sum()}/{len(keep_mask)} cells with effective knockdown")

    # Stage 3: minimum-cell filter
    initial_cells = keep_mask.sum()
    for pert in perts.unique():
        if pert == control_label:
            continue
        # Cells of this perturbation still kept after stages 1-2
        pert_mask = (perts == pert).values & keep_mask
        if pert_mask.sum() < min_cells:
            keep_mask[pert_mask] = False  # drop them

    logger.info(f"Stage 3: Keeping {keep_mask.sum()}/{initial_cells} cells after minimum cell filter")

    # Return view with all criteria satisfied
    return adata_[keep_mask]


def analyze_perturbation_quality(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyze the quality of all perturbations in the dataset.
    
    Args:
        adata: AnnData object
        perturbation_column: Column in adata.obs holding perturbation identities
        control_label: Category marking control cells
        residual_expression: Threshold for effective knockdown
        layer: Matrix to use for analysis
        
    Returns:
        DataFrame with perturbation quality metrics
    """
    perturbations = adata.obs[perturbation_column].unique()
    results = []
    
    for pert in perturbations:
        if pert == control_label:
            continue
            
        try:
            is_effective = is_on_target_knockdown(
                adata, pert, perturbation_column, control_label, residual_expression, layer
            )
            
            # Get cell counts
            pert_cells = (adata.obs[perturbation_column] == pert).sum()
            
            # Get expression data if gene exists
            if pert in adata.var_names:
                gene_idx = adata.var_names.get_loc(pert)
                X = adata.layers[layer] if layer is not None else adata.X
                
                control_cells = adata.obs[perturbation_column] == control_label
                perturbed_cells = adata.obs[perturbation_column] == pert
                
                control_mean = _mean(X[control_cells, gene_idx])
                perturbed_mean = _mean(X[perturbed_cells, gene_idx])
                knockdown_ratio = perturbed_mean / control_mean if control_mean > 0 else np.nan
            else:
                control_mean = perturbed_mean = knockdown_ratio = np.nan
            
            results.append({
                'perturbation': pert,
                'n_cells': pert_cells,
                'is_effective': is_effective,
                'control_mean': control_mean,
                'perturbed_mean': perturbed_mean,
                'knockdown_ratio': knockdown_ratio,
                'knockdown_percent': (1 - knockdown_ratio) * 100 if not np.isnan(knockdown_ratio) else np.nan,
            })
            
        except Exception as e:
            logger.warning(f"Error analyzing perturbation {pert}: {e}")
            results.append({
                'perturbation': pert,
                'n_cells': (adata.obs[perturbation_column] == pert).sum(),
                'is_effective': False,
                'control_mean': np.nan,
                'perturbed_mean': np.nan,
                'knockdown_ratio': np.nan,
                'knockdown_percent': np.nan,
            })
    
    return pd.DataFrame(results) 