"""
Configuration module for perturbation experiments from TOML files.

Replicates the functionality from cell-load/config.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, List

import toml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for perturbation experiments from TOML file."""

    # Dataset paths
    datasets: Dict[str, str]

    # Training datasets (entire datasets)
    training: Dict[str, str]

    # Zeroshot cell types (dataset.celltype -> split)
    zeroshot: Dict[str, str]

    # Fewshot perturbation assignments (dataset.celltype -> {split: [perts]})
    fewshot: Dict[str, Dict[str, List[str]]]

    @classmethod
    def from_toml(cls, toml_path: str) -> "ExperimentConfig":
        """Load configuration from TOML file."""
        with open(toml_path, "r") as f:
            config = toml.load(f)

        return cls(
            datasets=config.get("datasets", {}),
            training=config.get("training", {}),
            zeroshot=config.get("zeroshot", {}),
            fewshot=config.get("fewshot", {}),
        )

    def get_all_datasets(self) -> Set[str]:
        """Get all dataset names referenced in config."""
        datasets = set(self.training.keys())

        # Extract dataset names from zeroshot keys (format: "dataset.celltype")
        for key in self.zeroshot.keys():
            dataset = key.split(".")[0]
            datasets.add(dataset)

        # Extract dataset names from fewshot keys
        for key in self.fewshot.keys():
            dataset = key.split(".")[0]
            datasets.add(dataset)

        return datasets

    def get_zeroshot_celltypes(self, dataset: str) -> Dict[str, str]:
        """Get zeroshot cell types for a dataset and their target splits."""
        result = {}
        for key, split in self.zeroshot.items():
            if key.startswith(f"{dataset}."):
                celltype = key.split(".", 1)[1]
                result[celltype] = split
        return result

    def get_fewshot_celltypes(self, dataset: str) -> Dict[str, Dict[str, List[str]]]:
        """Get fewshot cell types for a dataset and their perturbation assignments."""
        result = {}
        for key, pert_config in self.fewshot.items():
            if key.startswith(f"{dataset}."):
                celltype = key.split(".", 1)[1]
                result[celltype] = pert_config
                logger.info(f"{dataset}.{celltype}: {dict((k, len(v)) for k, v in pert_config.items())}")
        return result

    def validate(self) -> None:
        """Validate configuration consistency."""
        all_datasets = self.get_all_datasets()

        # Check that all referenced datasets have paths
        missing_paths = all_datasets - set(self.datasets.keys())
        if missing_paths:
            raise ValueError(f"Missing dataset paths for: {missing_paths}")

        # Check that dataset paths exist
        for dataset, path in self.datasets.items():
            logger.info(f"Dataset {dataset}: {path}")
            if not Path(path).exists():
                logger.warning(f"Dataset path does not exist: {path}")

        # Validate splits are valid
        valid_splits = {"train", "val", "test"}
        for split in self.zeroshot.values():
            if split not in valid_splits:
                raise ValueError(
                    f"Invalid split '{split}'. Must be one of {valid_splits}"
                )

        logger.info("Configuration validation passed")

    def summary(self) -> str:
        """Return a summary of the configuration."""
        summary = []
        summary.append("Experiment Configuration Summary:")
        summary.append(f"  Datasets: {len(self.datasets)}")
        for name, path in self.datasets.items():
            summary.append(f"    {name}: {path}")
        
        summary.append(f"  Training: {len(self.training)} datasets")
        summary.append(f"  Zero-shot: {len(self.zeroshot)} cell types")
        summary.append(f"  Few-shot: {len(self.fewshot)} cell type/perturbation combinations")
        
        return "\n".join(summary) 