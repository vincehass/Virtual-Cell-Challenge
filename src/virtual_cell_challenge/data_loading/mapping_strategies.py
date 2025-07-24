"""
Mapping strategies for pairing perturbed cells with control cells.

Replicates the functionality from cell-load/mapping_strategies/
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .perturbation_dataset import PerturbationDataset

logger = logging.getLogger(__name__)


class BaseMappingStrategy(ABC):
    """
    Abstract base class for mapping a perturbed cell to one or more control cells.
    Each strategy can store internal data structures that assist in retrieving
    control indices for a perturbed cell.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        random_state: int = 42,
        n_basal_samples: int = 1,
        stage: str = "train",
        **kwargs,
    ):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_basal_samples = n_basal_samples
        self.name = name
        self.stage = stage
        self.map_controls = kwargs.get("map_controls", False)

    @abstractmethod
    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Called once per split (train/val/test) to initialize or compute
        the mapping information for that split.
        """
        pass

    @abstractmethod
    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns the control indices for a given perturbed index in a particular split.
        """
        pass

    @abstractmethod
    def get_control_index(self, dataset, split, perturbed_idx) -> Optional[int]:
        """
        Returns a single control index for a given perturbed index.
        """
        pass

    def get_mapped_expressions(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        Returns perturbed expression, control expression, and control index.
        """
        # Get expression(s) based on embed_key
        if dataset.embed_key:
            control_index = self.get_control_index(dataset, split, perturbed_idx)
            pert_expr = dataset.fetch_obsm_expression(perturbed_idx, dataset.embed_key)
            if control_index is None:
                ctrl_expr = torch.zeros_like(pert_expr)
            else:
                ctrl_expr = dataset.fetch_obsm_expression(control_index, dataset.embed_key)
            return pert_expr, ctrl_expr, control_index
        else:
            control_index = self.get_control_index(dataset, split, perturbed_idx)
            pert_expr = dataset.fetch_gene_expression(perturbed_idx)
            if control_index is None:
                ctrl_expr = torch.zeros_like(pert_expr)
            else:
                ctrl_expr = dataset.fetch_gene_expression(control_index)
            return pert_expr, ctrl_expr, control_index


class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control cell(s) drawn from the same plate.
    Ensures that only control cells with the same cell type as the perturbed cell are considered.
    """

    def __init__(self, name="random", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)

        # Map cell type -> list of control indices for each split
        self.split_control_pool = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

        # Initialize Python's random module with the same seed
        random.seed(random_state)

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        For the given split, group all control indices by their cell type.
        """
        # Get cell types for all control indices
        cell_types = dataset.get_all_cell_types(control_indices)

        # Group by cell type and store the control indices
        for ct in np.unique(cell_types):
            ct_mask = cell_types == ct
            ct_indices = control_indices[ct_mask]

            if ct not in self.split_control_pool[split]:
                self.split_control_pool[split][ct] = list(ct_indices)
            else:
                self.split_control_pool[split][ct].extend(ct_indices)

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns random control indices for a given perturbed index.
        """
        cell_type = dataset.get_cell_type(perturbed_idx)
        
        if cell_type not in self.split_control_pool[split]:
            return np.array([])
        
        control_pool = self.split_control_pool[split][cell_type]
        if not control_pool:
            return np.array([])
        
        n_samples = min(self.n_basal_samples, len(control_pool))
        return np.array(self.rng.choice(control_pool, n_samples, replace=False))

    def get_control_index(self, dataset, split, perturbed_idx) -> Optional[int]:
        """
        Returns a single random control index.
        """
        indices = self.get_control_indices(dataset, split, perturbed_idx)
        return indices[0] if len(indices) > 0 else None


class BatchMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to control cells within the same experimental batch.
    """

    def __init__(self, name="batch", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)

        # Map (batch, cell_type) -> list of control indices for each split
        self.split_control_pool = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        For the given split, group all control indices by batch and cell type.
        """
        # Get batches and cell types for all control indices
        batches = dataset.get_all_batches(control_indices)
        cell_types = dataset.get_all_cell_types(control_indices)

        # Group by (batch, cell_type) combination
        for batch, ct in zip(batches, cell_types):
            key = (batch, ct)
            matching_mask = (batches == batch) & (cell_types == ct)
            matching_indices = control_indices[matching_mask]

            if key not in self.split_control_pool[split]:
                self.split_control_pool[split][key] = list(matching_indices)
            else:
                self.split_control_pool[split][key].extend(matching_indices)

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns control indices from the same batch and cell type.
        """
        batch = dataset.get_batch(perturbed_idx)
        cell_type = dataset.get_cell_type(perturbed_idx)
        key = (batch, cell_type)
        
        if key not in self.split_control_pool[split]:
            return np.array([])
        
        control_pool = self.split_control_pool[split][key]
        if not control_pool:
            return np.array([])
        
        n_samples = min(self.n_basal_samples, len(control_pool))
        return np.array(self.rng.choice(control_pool, n_samples, replace=False))

    def get_control_index(self, dataset, split, perturbed_idx) -> Optional[int]:
        """
        Returns a single control index from the same batch.
        """
        indices = self.get_control_indices(dataset, split, perturbed_idx)
        return indices[0] if len(indices) > 0 else None 