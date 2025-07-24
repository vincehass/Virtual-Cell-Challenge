"""
Cell_Eval framework for evaluating virtual cell models.

Implements biologically relevant metrics beyond simple expression correlation,
focusing on differential expression accuracy and perturbation strength.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    mean_squared_error as sklearn_mse,
    r2_score,
    accuracy_score,
    roc_auc_score
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class CellEvalMetrics:
    """
    Cell_Eval framework for comprehensive virtual cell model evaluation.
    
    Implements metrics that assess biological relevance:
    - Differential expression accuracy
    - Perturbation strength estimation
    - Direction accuracy (up/down regulation)
    - Top-K gene recovery
    - Effect size correlation
    """
    
    def __init__(
        self,
        control_label: str = "non-targeting",
        significance_threshold: float = 0.05,
        effect_size_threshold: float = 0.5,
        top_k_genes: int = 100
    ):
        """
        Initialize Cell_Eval metrics.
        
        Args:
            control_label: Label for control perturbations
            significance_threshold: P-value threshold for differential expression
            effect_size_threshold: Minimum effect size for meaningful changes
            top_k_genes: Number of top genes to consider for ranking metrics
        """
        self.control_label = control_label
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold
        self.top_k_genes = top_k_genes
    
    def evaluate_predictions(
        self,
        predicted: Union[torch.Tensor, np.ndarray],
        actual: Union[torch.Tensor, np.ndarray],
        perturbations: List[str],
        controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
        gene_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of predictions using Cell_Eval metrics.
        
        Args:
            predicted: Predicted expression values (n_cells, n_genes)
            actual: Actual expression values (n_cells, n_genes)
            perturbations: List of perturbation names for each cell
            controls: Control expression values (optional)
            gene_names: Gene names (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy for easier handling
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        if controls is not None and isinstance(controls, torch.Tensor):
            controls = controls.detach().cpu().numpy()
        
        metrics = {}
        
        # Basic correlation metrics
        metrics.update(self._basic_correlation_metrics(predicted, actual))
        
        # Differential expression metrics
        if controls is not None:
            de_metrics = self._differential_expression_metrics(
                predicted, actual, perturbations, controls, gene_names
            )
            metrics.update(de_metrics)
        
        # Per-perturbation analysis
        pert_metrics = self._per_perturbation_metrics(predicted, actual, perturbations)
        metrics.update(pert_metrics)
        
        return metrics
    
    def _basic_correlation_metrics(
        self, 
        predicted: np.ndarray, 
        actual: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic correlation and distance metrics."""
        metrics = {}
        
        # Flatten for global correlation
        pred_flat = predicted.flatten()
        actual_flat = actual.flatten()
        
        # Remove NaN/Inf values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(actual_flat)
        pred_clean = pred_flat[valid_mask]
        actual_clean = actual_flat[valid_mask]
        
        if len(pred_clean) == 0:
            logger.warning("No valid predictions for correlation calculation")
            return {"pearson_r": 0.0, "spearman_r": 0.0, "mse": float('inf')}
        
        # Pearson correlation
        if len(pred_clean) > 1:
            pearson_r, _ = stats.pearsonr(pred_clean, actual_clean)
            metrics["pearson_r"] = float(pearson_r) if not np.isnan(pearson_r) else 0.0
            
            # Spearman correlation
            spearman_r, _ = stats.spearmanr(pred_clean, actual_clean)
            metrics["spearman_r"] = float(spearman_r) if not np.isnan(spearman_r) else 0.0
        else:
            metrics["pearson_r"] = 0.0
            metrics["spearman_r"] = 0.0
        
        # Mean squared error
        metrics["mse"] = float(sklearn_mse(actual_clean, pred_clean))
        
        # R-squared
        r2 = r2_score(actual_clean, pred_clean)
        metrics["r2"] = float(r2) if not np.isnan(r2) else 0.0
        
        # Mean absolute error
        metrics["mae"] = float(np.mean(np.abs(pred_clean - actual_clean)))
        
        return metrics
    
    def _differential_expression_metrics(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        perturbations: List[str],
        controls: np.ndarray,
        gene_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute differential expression accuracy metrics."""
        metrics = {}
        unique_perts = [p for p in set(perturbations) if p != self.control_label]
        
        if not unique_perts:
            return metrics
        
        de_accuracies = []
        direction_accuracies = []
        effect_correlations = []
        
        for pert in unique_perts:
            # Get indices for this perturbation
            pert_mask = np.array(perturbations) == pert
            if not np.any(pert_mask):
                continue
            
            pert_pred = predicted[pert_mask]
            pert_actual = actual[pert_mask]
            
            # Average across cells for this perturbation
            pred_mean = np.mean(pert_pred, axis=0)
            actual_mean = np.mean(pert_actual, axis=0)
            control_mean = np.mean(controls, axis=0)
            
            # Calculate differential expression
            de_metrics = self._calculate_de_for_perturbation(
                pred_mean, actual_mean, control_mean
            )
            
            if de_metrics:
                de_accuracies.append(de_metrics['de_accuracy'])
                direction_accuracies.append(de_metrics['direction_accuracy'])
                effect_correlations.append(de_metrics['effect_correlation'])
        
        # Aggregate metrics
        if de_accuracies:
            metrics["de_accuracy"] = float(np.mean(de_accuracies))
            metrics["direction_accuracy"] = float(np.mean(direction_accuracies))
            metrics["effect_correlation"] = float(np.mean(effect_correlations))
        
        return metrics
    
    def _calculate_de_for_perturbation(
        self,
        pred_mean: np.ndarray,
        actual_mean: np.ndarray,
        control_mean: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Calculate DE metrics for a single perturbation."""
        
        # Calculate fold changes
        actual_fc = actual_mean - control_mean
        pred_fc = pred_mean - control_mean
        
        # Identify significantly changed genes (ground truth)
        actual_changed = np.abs(actual_fc) > self.effect_size_threshold
        
        if not np.any(actual_changed):
            return None
        
        # Predict significantly changed genes
        pred_changed = np.abs(pred_fc) > self.effect_size_threshold
        
        # DE accuracy: fraction of truly changed genes correctly identified
        de_accuracy = np.sum(actual_changed & pred_changed) / np.sum(actual_changed)
        
        # Direction accuracy: fraction with correct up/down regulation
        correct_direction = (actual_fc * pred_fc) > 0
        direction_accuracy = np.sum(correct_direction & actual_changed) / np.sum(actual_changed)
        
        # Effect size correlation
        effect_correlation = stats.pearsonr(actual_fc, pred_fc)[0]
        if np.isnan(effect_correlation):
            effect_correlation = 0.0
        
        return {
            'de_accuracy': de_accuracy,
            'direction_accuracy': direction_accuracy,
            'effect_correlation': effect_correlation
        }
    
    def _per_perturbation_metrics(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        perturbations: List[str]
    ) -> Dict[str, float]:
        """Compute per-perturbation analysis metrics."""
        metrics = {}
        unique_perts = [p for p in set(perturbations) if p != self.control_label]
        
        pert_correlations = []
        pert_mses = []
        
        for pert in unique_perts:
            pert_mask = np.array(perturbations) == pert
            if not np.any(pert_mask):
                continue
            
            pert_pred = predicted[pert_mask]
            pert_actual = actual[pert_mask]
            
            # Average across cells
            pred_mean = np.mean(pert_pred, axis=0)
            actual_mean = np.mean(pert_actual, axis=0)
            
            # Per-perturbation correlation
            if len(pred_mean) > 1:
                correlation = stats.pearsonr(pred_mean, actual_mean)[0]
                if not np.isnan(correlation):
                    pert_correlations.append(correlation)
            
            # Per-perturbation MSE
            mse = np.mean((pred_mean - actual_mean) ** 2)
            pert_mses.append(mse)
        
        if pert_correlations:
            metrics["mean_pert_correlation"] = float(np.mean(pert_correlations))
            metrics["std_pert_correlation"] = float(np.std(pert_correlations))
        
        if pert_mses:
            metrics["mean_pert_mse"] = float(np.mean(pert_mses))
            metrics["std_pert_mse"] = float(np.std(pert_mses))
        
        return metrics
    
    def top_k_gene_recovery(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        controls: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate top-K gene recovery rate.
        
        Measures how well the model identifies the most differentially expressed genes.
        """
        if k is None:
            k = self.top_k_genes
        
        # Calculate actual differential expression
        actual_de = actual - controls
        actual_rankings = np.argsort(np.abs(actual_de), axis=1)[:, -k:]
        
        # Calculate predicted differential expression
        pred_de = predicted - controls
        pred_rankings = np.argsort(np.abs(pred_de), axis=1)[:, -k:]
        
        # Calculate overlap
        overlaps = []
        for i in range(len(actual_rankings)):
            overlap = len(set(actual_rankings[i]) & set(pred_rankings[i]))
            overlaps.append(overlap / k)
        
        return float(np.mean(overlaps))
    
    def perturbation_strength_correlation(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        controls: np.ndarray
    ) -> float:
        """
        Calculate correlation between predicted and actual perturbation strengths.
        
        Perturbation strength is measured as the magnitude of change from control.
        """
        # Calculate perturbation strengths
        actual_strength = np.linalg.norm(actual - controls, axis=1)
        pred_strength = np.linalg.norm(predicted - controls, axis=1)
        
        # Calculate correlation
        if len(actual_strength) > 1:
            correlation = stats.pearsonr(actual_strength, pred_strength)[0]
            return float(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0


def evaluate_predictions(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    perturbations: List[str],
    controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
    gene_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function for evaluating predictions with Cell_Eval metrics.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values
        perturbations: List of perturbation names
        controls: Control expression values (optional)
        gene_names: Gene names (optional)
        **kwargs: Additional arguments for CellEvalMetrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = CellEvalMetrics(**kwargs)
    return evaluator.evaluate_predictions(
        predicted, actual, perturbations, controls, gene_names
    )


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple models using Cell_Eval metrics.
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        metric_names: Specific metrics to compare (optional)
        
    Returns:
        DataFrame with model comparison
    """
    if not results:
        return pd.DataFrame()
    
    # Get all available metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    if metric_names:
        all_metrics = [m for m in all_metrics if m in metric_names]
    else:
        all_metrics = sorted(list(all_metrics))
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, model_results in results.items():
        row = {"model": model_name}
        for metric in all_metrics:
            row[metric] = model_results.get(metric, np.nan)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.set_index("model", inplace=True)
    
    return df 