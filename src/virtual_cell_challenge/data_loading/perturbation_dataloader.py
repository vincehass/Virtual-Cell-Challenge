"""
PyTorch Lightning data module for perturbation experiments.

Replicates the functionality from cell-load/data_modules.py
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any

import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .perturbation_dataset import PerturbationDataset
from .mapping_strategies import BaseMappingStrategy, RandomMappingStrategy, BatchMappingStrategy

logger = logging.getLogger(__name__)


class PerturbationDataModule(L.LightningDataModule):
    """
    Lightning data module for perturbation experiments.
    
    Handles multiple datasets, train/val/test splits, zero-shot and few-shot learning,
    and control cell mapping strategies.
    """
    
    def __init__(
        self,
        toml_config_path: str,
        embed_key: Optional[str] = "X_hvg",
        output_space: str = "embedding",
        basal_mapping_strategy: str = "random",
        n_basal_samples: int = 1,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        should_yield_control_cells: bool = False,
        barcode: bool = False,
        perturbation_features_file: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize perturbation data module.
        
        Args:
            toml_config_path: Path to TOML configuration file
            embed_key: Key for embedding in adata.obsm (None for raw expression)
            output_space: "embedding" or "gene" for output format
            basal_mapping_strategy: "random" or "batch" for control mapping
            n_basal_samples: Number of control cells per perturbation
            pert_col: Column name for perturbation information
            cell_type_key: Column name for cell type
            batch_col: Column name for batch information
            control_pert: Label for control perturbations
            should_yield_control_cells: Whether to include control cells in batches
            barcode: Whether to include cell barcodes
            perturbation_features_file: Path to perturbation embeddings
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory in data loaders
        """
        super().__init__()
        
        # Configuration
        self.config = ExperimentConfig.from_toml(toml_config_path)
        self.embed_key = embed_key
        self.output_space = output_space
        self.basal_mapping_strategy = basal_mapping_strategy
        self.n_basal_samples = n_basal_samples
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.should_yield_control_cells = should_yield_control_cells
        self.barcode = barcode
        self.perturbation_features_file = perturbation_features_file
        
        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Initialize mapping strategy
        self.mapping_strategy = self._create_mapping_strategy()
        
        # Storage for datasets and splits
        self.datasets = {}
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
        # Perturbation features (if provided)
        self.perturbation_features = None
        if self.perturbation_features_file:
            self.perturbation_features = torch.load(self.perturbation_features_file)
    
    def _create_mapping_strategy(self) -> BaseMappingStrategy:
        """Create mapping strategy based on configuration."""
        if self.basal_mapping_strategy == "random":
            return RandomMappingStrategy(
                n_basal_samples=self.n_basal_samples,
                random_state=42
            )
        elif self.basal_mapping_strategy == "batch":
            return BatchMappingStrategy(
                n_basal_samples=self.n_basal_samples,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown mapping strategy: {self.basal_mapping_strategy}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test splits."""
        logger.info("Setting up data module...")
        
        # Validate configuration
        self.config.validate()
        
        # Load all datasets
        self._load_datasets()
        
        # Create splits
        self._create_splits()
        
        # Register splits with mapping strategy
        self._register_mapping_strategy()
        
        logger.info(f"Setup complete: {len(self.train_datasets)} train, {len(self.val_datasets)} val, {len(self.test_datasets)} test")
    
    def _load_datasets(self):
        """Load all datasets from configuration."""
        for dataset_name, dataset_path in self.config.datasets.items():
            logger.info(f"Loading dataset: {dataset_name} from {dataset_path}")
            
            dataset = PerturbationDataset(
                dataset_path=dataset_path,
                embed_key=self.embed_key,
                pert_col=self.pert_col,
                cell_type_key=self.cell_type_key,
                batch_col=self.batch_col,
                control_pert=self.control_pert,
                output_space=self.output_space
            )
            
            self.datasets[dataset_name] = dataset
            logger.info(f"Dataset {dataset_name}: {dataset.summary()}")
    
    def _create_splits(self):
        """Create train/val/test splits based on configuration."""
        for dataset_name, dataset in self.datasets.items():
            
            # Get zeroshot and fewshot configurations for this dataset
            zeroshot_celltypes = self.config.get_zeroshot_celltypes(dataset_name)
            fewshot_celltypes = self.config.get_fewshot_celltypes(dataset_name)
            
            # Create splits for each cell type in the dataset
            for cell_type in dataset.unique_cell_types:
                self._create_splits_for_celltype(
                    dataset_name, dataset, cell_type, 
                    zeroshot_celltypes, fewshot_celltypes
                )
    
    def _create_splits_for_celltype(
        self, 
        dataset_name: str, 
        dataset: PerturbationDataset,
        cell_type: str,
        zeroshot_celltypes: Dict[str, str],
        fewshot_celltypes: Dict[str, Dict[str, List[str]]]
    ):
        """Create splits for a specific cell type within a dataset."""
        
        # Get all indices for this cell type
        celltype_indices = dataset.filter_by_cell_type(cell_type)
        
        # Check if this cell type is in zeroshot
        if cell_type in zeroshot_celltypes:
            split = zeroshot_celltypes[cell_type]
            split_dataset = self._create_split_dataset(dataset, celltype_indices, f"{dataset_name}.{cell_type}")
            self._add_to_split(split_dataset, split)
            return
        
        # Check if this cell type is in fewshot
        if cell_type in fewshot_celltypes:
            fewshot_config = fewshot_celltypes[cell_type]
            self._create_fewshot_splits(dataset, celltype_indices, fewshot_config, f"{dataset_name}.{cell_type}")
            return
        
        # Default: add entire cell type to training
        if dataset_name in self.config.training:
            split_dataset = self._create_split_dataset(dataset, celltype_indices, f"{dataset_name}.{cell_type}")
            self._add_to_split(split_dataset, "train")
    
    def _create_fewshot_splits(
        self,
        dataset: PerturbationDataset,
        celltype_indices: np.ndarray,
        fewshot_config: Dict[str, List[str]],
        split_name: str
    ):
        """Create few-shot splits for a cell type."""
        
        # Get perturbations for this cell type
        celltype_perts = dataset.get_all_perturbations(celltype_indices)
        
        # Create masks for each split
        train_mask = np.ones(len(celltype_indices), dtype=bool)
        
        for split, perturbations in fewshot_config.items():
            # Find indices for these perturbations
            split_mask = np.isin(celltype_perts, perturbations)
            split_indices = celltype_indices[split_mask]
            
            if len(split_indices) > 0:
                split_dataset = self._create_split_dataset(dataset, split_indices, f"{split_name}.{split}")
                self._add_to_split(split_dataset, split)
                
                # Remove from training
                train_mask &= ~split_mask
        
        # Add remaining cells to training
        train_indices = celltype_indices[train_mask]
        if len(train_indices) > 0:
            train_dataset = self._create_split_dataset(dataset, train_indices, f"{split_name}.train")
            self._add_to_split(train_dataset, "train")
    
    def _create_split_dataset(self, base_dataset: PerturbationDataset, indices: np.ndarray, split_name: str) -> PerturbationDataset:
        """Create a dataset for a specific split."""
        split_dataset = PerturbationDataset(
            dataset_path=base_dataset.dataset_path,
            embed_key=base_dataset.embed_key,
            pert_col=base_dataset.pert_col,
            cell_type_key=base_dataset.cell_type_key,
            batch_col=base_dataset.batch_col,
            control_pert=base_dataset.control_pert,
            split_idx=indices,
            output_space=base_dataset.output_space,
            split_pert_type_comb=split_name
        )
        return split_dataset
    
    def _add_to_split(self, dataset: PerturbationDataset, split: str):
        """Add dataset to appropriate split list."""
        if split == "train":
            self.train_datasets.append(dataset)
        elif split == "val":
            self.val_datasets.append(dataset)
        elif split == "test":
            self.test_datasets.append(dataset)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _register_mapping_strategy(self):
        """Register all splits with the mapping strategy."""
        for split, datasets in [("train", self.train_datasets), ("val", self.val_datasets), ("test", self.test_datasets)]:
            for dataset in datasets:
                # Get perturbed and control indices
                perturbed_indices = dataset.get_perturbed_indices()
                control_indices = dataset.get_control_indices()
                
                # Register with mapping strategy
                self.mapping_strategy.register_split_indices(
                    dataset, split, perturbed_indices, control_indices
                )
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if not self.train_datasets:
            raise ValueError("No training datasets available")
        
        combined_dataset = CombinedPerturbationDataset(
            self.train_datasets, self.mapping_strategy, "train",
            barcode=self.barcode,
            perturbation_features=self.perturbation_features
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if not self.val_datasets:
            return None
        
        combined_dataset = CombinedPerturbationDataset(
            self.val_datasets, self.mapping_strategy, "val",
            barcode=self.barcode,
            perturbation_features=self.perturbation_features
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if not self.test_datasets:
            return None
        
        combined_dataset = CombinedPerturbationDataset(
            self.test_datasets, self.mapping_strategy, "test",
            barcode=self.barcode,
            perturbation_features=self.perturbation_features
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for perturbation data."""
        # Stack tensors
        batch_dict = {}
        
        # Handle tensor fields
        tensor_fields = ['pert_cell_emb', 'ctrl_cell_emb', 'pert_emb']
        for field in tensor_fields:
            if field in batch[0]:
                batch_dict[field] = torch.stack([item[field] for item in batch])
        
        # Handle list fields
        list_fields = ['pert_name', 'cell_type']
        if self.barcode:
            list_fields.extend(['pert_cell_barcode', 'ctrl_cell_barcode'])
        
        for field in list_fields:
            if field in batch[0]:
                batch_dict[field] = [item[field] for item in batch]
        
        # Handle batch field
        if 'batch' in batch[0]:
            batch_dict['batch'] = torch.tensor([item['batch'] for item in batch])
        
        return batch_dict


class CombinedPerturbationDataset(Dataset):
    """
    Combined dataset that merges multiple perturbation datasets and handles mapping.
    """
    
    def __init__(
        self,
        datasets: List[PerturbationDataset],
        mapping_strategy: BaseMappingStrategy,
        split: str,
        barcode: bool = False,
        perturbation_features: Optional[Dict] = None
    ):
        self.datasets = datasets
        self.mapping_strategy = mapping_strategy
        self.split = split
        self.barcode = barcode
        self.perturbation_features = perturbation_features
        
        # Create global index mapping
        self._create_index_mapping()
        
        # Create perturbation vocabulary
        self._create_perturbation_vocab()
    
    def _create_index_mapping(self):
        """Create mapping from global index to (dataset_idx, local_idx)."""
        self.index_mapping = []
        self.dataset_offsets = [0]
        
        for dataset_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                self.index_mapping.append((dataset_idx, local_idx))
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(dataset))
    
    def _create_perturbation_vocab(self):
        """Create vocabulary of all perturbations across datasets."""
        all_perts = set()
        for dataset in self.datasets:
            all_perts.update(dataset.unique_perts)
        
        self.pert_vocab = sorted(list(all_perts))
        self.pert_to_idx = {pert: idx for idx, pert in enumerate(self.pert_vocab)}
    
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by global index."""
        dataset_idx, local_idx = self.index_mapping[idx]
        dataset = self.datasets[dataset_idx]
        
        # Get perturbation info
        pert_name = dataset.get_perturbation(local_idx)
        cell_type = dataset.get_cell_type(local_idx)
        batch = dataset.get_batch(local_idx)
        
        # Get expressions using mapping strategy
        pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
            dataset, self.split, local_idx
        )
        
        # Create perturbation embedding (one-hot or feature-based)
        if self.perturbation_features and pert_name in self.perturbation_features:
            pert_emb = torch.tensor(self.perturbation_features[pert_name], dtype=torch.float32)
        else:
            # One-hot encoding
            pert_emb = torch.zeros(len(self.pert_vocab))
            if pert_name in self.pert_to_idx:
                pert_emb[self.pert_to_idx[pert_name]] = 1.0
        
        result = {
            'pert_cell_emb': pert_expr,
            'ctrl_cell_emb': ctrl_expr,
            'pert_emb': pert_emb,
            'pert_name': pert_name,
            'cell_type': cell_type,
            'batch': batch,
        }
        
        # Add barcodes if requested
        if self.barcode:
            result['pert_cell_barcode'] = dataset.get_cell_barcode(local_idx) or f"cell_{idx}"
            result['ctrl_cell_barcode'] = dataset.get_cell_barcode(ctrl_idx) if ctrl_idx is not None else f"ctrl_{idx}"
        
        return result 