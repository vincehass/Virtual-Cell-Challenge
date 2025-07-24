"""
Perturbation dataset class for loading single-cell perturbation data.

Replicates the functionality from cell-load/dataset.py
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Dataset for loading single-cell perturbation data from H5AD files.
    
    Handles sparse/dense expression matrices, embeddings, and metadata.
    Supports multiple cell types and perturbations within a single dataset.
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        embed_key: Optional[str] = "X_hvg",
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        split_pert_type_comb: Optional[str] = None,
        split_idx: Optional[np.ndarray] = None,
        output_space: str = "embedding",
        **kwargs
    ):
        """
        Initialize perturbation dataset.
        
        Args:
            dataset_path: Path to H5AD file or directory
            embed_key: Key for embedding in adata.obsm (None for raw expression)
            pert_col: Column name for perturbation information
            cell_type_key: Column name for cell type
            batch_col: Column name for batch information
            control_pert: Label for control perturbations
            split_pert_type_comb: Specific perturbation-celltype combination for this split
            split_idx: Indices for this dataset split
            output_space: "embedding" or "gene" for output format
        """
        self.dataset_path = Path(dataset_path)
        self.embed_key = embed_key
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.split_pert_type_comb = split_pert_type_comb
        self.split_idx = split_idx
        self.output_space = output_space
        
        # Load data
        self._load_data()
        
        # Apply split filtering if specified
        if self.split_idx is not None:
            self._apply_split_filter()
            
        logger.info(f"Dataset loaded: {len(self)} cells from {self.dataset_path}")
    
    def _load_data(self):
        """Load AnnData object from file."""
        if self.dataset_path.is_file():
            self.adata = anndata.read_h5ad(self.dataset_path)
        else:
            # Handle directory with multiple files
            h5ad_files = list(self.dataset_path.glob("*.h5ad"))
            if not h5ad_files:
                raise ValueError(f"No H5AD files found in {self.dataset_path}")
            
            # Load and concatenate multiple files
            adatas = []
            for file in h5ad_files:
                adata = anndata.read_h5ad(file)
                adatas.append(adata)
            
            self.adata = anndata.concat(adatas, join="outer")
            
        # Validate required columns
        self._validate_data()
        
        # Cache frequently accessed data
        self._cache_metadata()
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_obs = [self.pert_col, self.cell_type_key, self.batch_col]
        missing_cols = [col for col in required_obs if col not in self.adata.obs.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in obs: {missing_cols}")
        
        if self.embed_key and self.embed_key not in self.adata.obsm:
            logger.warning(f"Embedding key '{self.embed_key}' not found in obsm")
    
    def _cache_metadata(self):
        """Cache frequently accessed metadata for performance."""
        self.perturbations = self.adata.obs[self.pert_col].values
        self.cell_types = self.adata.obs[self.cell_type_key].values
        self.batches = self.adata.obs[self.batch_col].values
        
        # Create mapping of perturbations to indices
        self.pert_to_indices = {}
        for pert in np.unique(self.perturbations):
            self.pert_to_indices[pert] = np.where(self.perturbations == pert)[0]
        
        # Get unique values
        self.unique_perts = np.unique(self.perturbations)
        self.unique_cell_types = np.unique(self.cell_types)
        self.unique_batches = np.unique(self.batches)
    
    def _apply_split_filter(self):
        """Apply split filtering to dataset."""
        self.adata = self.adata[self.split_idx]
        self._cache_metadata()
    
    def __len__(self) -> int:
        """Return number of cells in dataset."""
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index - mainly for compatibility, use fetch methods instead."""
        return {
            'cell_idx': idx,
            'perturbation': self.get_perturbation(idx),
            'cell_type': self.get_cell_type(idx),
            'batch': self.get_batch(idx),
        }
    
    def get_perturbation(self, idx: int) -> str:
        """Get perturbation for cell at index."""
        return self.perturbations[idx]
    
    def get_cell_type(self, idx: int) -> str:
        """Get cell type for cell at index."""
        return self.cell_types[idx]
    
    def get_batch(self, idx: int) -> str:
        """Get batch for cell at index."""
        return self.batches[idx]
    
    def get_all_perturbations(self, indices: np.ndarray) -> np.ndarray:
        """Get perturbations for multiple indices."""
        return self.perturbations[indices]
    
    def get_all_cell_types(self, indices: np.ndarray) -> np.ndarray:
        """Get cell types for multiple indices."""
        return self.cell_types[indices]
    
    def get_all_batches(self, indices: np.ndarray) -> np.ndarray:
        """Get batches for multiple indices."""
        return self.batches[indices]
    
    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        """Fetch raw gene expression for cell at index."""
        expr = self.adata.X[idx]
        
        # Handle sparse matrices
        if sp.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()
        
        return torch.tensor(expr, dtype=torch.float32)
    
    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        """Fetch embedding/obsm data for cell at index."""
        if key not in self.adata.obsm:
            raise KeyError(f"Key '{key}' not found in obsm")
        
        expr = self.adata.obsm[key][idx]
        
        # Handle sparse matrices
        if sp.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()
        
        return torch.tensor(expr, dtype=torch.float32)
    
    def get_expression_data(self, idx: int) -> torch.Tensor:
        """Get expression data based on embed_key setting."""
        if self.embed_key:
            return self.fetch_obsm_expression(idx, self.embed_key)
        else:
            return self.fetch_gene_expression(idx)
    
    def get_perturbation_indices(self, perturbation: str) -> np.ndarray:
        """Get all indices for a specific perturbation."""
        return self.pert_to_indices.get(perturbation, np.array([]))
    
    def get_control_indices(self) -> np.ndarray:
        """Get all control cell indices."""
        return self.get_perturbation_indices(self.control_pert)
    
    def get_perturbed_indices(self) -> np.ndarray:
        """Get all non-control cell indices."""
        all_indices = np.arange(len(self))
        control_indices = self.get_control_indices()
        return np.setdiff1d(all_indices, control_indices)
    
    def filter_by_cell_type(self, cell_type: str) -> np.ndarray:
        """Get indices for specific cell type."""
        return np.where(self.cell_types == cell_type)[0]
    
    def filter_by_batch(self, batch: str) -> np.ndarray:
        """Get indices for specific batch."""
        return np.where(self.batches == batch)[0]
    
    def get_cell_barcode(self, idx: int) -> Optional[str]:
        """Get cell barcode if available."""
        if 'barcode' in self.adata.obs.columns:
            return self.adata.obs['barcode'].iloc[idx]
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        return {
            'n_cells': len(self),
            'n_genes': self.adata.n_vars,
            'n_perturbations': len(self.unique_perts),
            'n_cell_types': len(self.unique_cell_types),
            'n_batches': len(self.unique_batches),
            'perturbations': list(self.unique_perts),
            'cell_types': list(self.unique_cell_types),
            'control_label': self.control_pert,
            'embed_key': self.embed_key,
            'has_embeddings': bool(self.embed_key and self.embed_key in self.adata.obsm),
        } 