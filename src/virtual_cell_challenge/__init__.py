"""
Virtual Cell Challenge: Single-Cell Data Loading & Analysis

A comprehensive toolkit for understanding and reproducing the single-cell 
perturbation data loading pipeline from the Arc Institute's Virtual Cell Challenge.
"""

__version__ = "0.1.0"
__author__ = "Single Cell Challenge Team"
__email__ = "contact@virtualcellchallenge.org"

from .data_loading import PerturbationDataModule, PerturbationDataset
from .preprocessing import filter_on_target_knockdown, preprocess_perturbation_data
from .evaluation import CellEvalMetrics, evaluate_predictions
from .analysis import analyze_dataset, visualize_perturbations

__all__ = [
    # Data loading
    "PerturbationDataModule",
    "PerturbationDataset", 
    
    # Preprocessing
    "filter_on_target_knockdown",
    "preprocess_perturbation_data",
    
    # Evaluation
    "CellEvalMetrics", 
    "evaluate_predictions",
    
    # Analysis
    "analyze_dataset",
    "visualize_perturbations",
] 