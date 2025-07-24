"""
Additional evaluation metrics for virtual cell models.

Provides standard machine learning metrics and biological relevance measures
that complement the Cell_Eval framework.
"""

import logging
from typing import Union, List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cosine as scipy_cosine
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support
)

logger = logging.getLogger(__name__)


def pearson_correlation(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        per_sample: If True, calculate correlation per sample/cell
        
    Returns:
        Correlation coefficient(s)
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    if per_sample:
        correlations = []
        for i in range(predicted.shape[0]):
            if len(predicted[i]) > 1:
                corr, _ = stats.pearsonr(predicted[i], actual[i])
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        return np.array(correlations)
    else:
        # Global correlation
        pred_flat = predicted.flatten()
        actual_flat = actual.flatten()
        
        # Remove invalid values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(actual_flat)
        pred_clean = pred_flat[valid_mask]
        actual_clean = actual_flat[valid_mask]
        
        if len(pred_clean) > 1:
            corr, _ = stats.pearsonr(pred_clean, actual_clean)
            return float(corr) if not np.isnan(corr) else 0.0
        else:
            return 0.0


def cosine_similarity(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate cosine similarity.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        per_sample: If True, calculate similarity per sample/cell
        
    Returns:
        Cosine similarity score(s)
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    if per_sample:
        similarities = []
        for i in range(predicted.shape[0]):
            # Avoid zero vectors
            if np.linalg.norm(predicted[i]) > 0 and np.linalg.norm(actual[i]) > 0:
                sim = 1 - scipy_cosine(predicted[i], actual[i])
                similarities.append(sim if not np.isnan(sim) else 0.0)
            else:
                similarities.append(0.0)
        return np.array(similarities)
    else:
        # Global similarity
        pred_flat = predicted.flatten()
        actual_flat = actual.flatten()
        
        if np.linalg.norm(pred_flat) > 0 and np.linalg.norm(actual_flat) > 0:
            sim = 1 - scipy_cosine(pred_flat, actual_flat)
            return float(sim) if not np.isnan(sim) else 0.0
        else:
            return 0.0


def mean_squared_error_metric(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate mean squared error.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        per_sample: If True, calculate MSE per sample/cell
        
    Returns:
        MSE score(s)
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    if per_sample:
        mses = []
        for i in range(predicted.shape[0]):
            mse = mean_squared_error(actual[i], predicted[i])
            mses.append(float(mse))
        return np.array(mses)
    else:
        return float(mean_squared_error(actual.flatten(), predicted.flatten()))


def top_k_gene_recovery(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    controls: Union[torch.Tensor, np.ndarray],
    k: int = 100
) -> float:
    """
    Calculate top-K gene recovery rate.
    
    Measures how well the model identifies the most differentially expressed genes.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values
        controls: Control expression values
        k: Number of top genes to consider
        
    Returns:
        Recovery rate (0-1)
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    
    # Calculate differential expression
    actual_de = np.abs(actual - controls)
    pred_de = np.abs(predicted - controls)
    
    # Get top-K genes for each sample
    recovery_rates = []
    for i in range(len(actual_de)):
        actual_top_k = set(np.argsort(actual_de[i])[-k:])
        pred_top_k = set(np.argsort(pred_de[i])[-k:])
        
        overlap = len(actual_top_k & pred_top_k)
        recovery_rate = overlap / k
        recovery_rates.append(recovery_rate)
    
    return float(np.mean(recovery_rates))


def direction_accuracy(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    controls: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.1
) -> float:
    """
    Calculate direction accuracy for gene expression changes.
    
    Measures the fraction of genes with correct up/down regulation direction.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values
        controls: Control expression values
        threshold: Minimum change threshold to consider
        
    Returns:
        Direction accuracy (0-1)
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    
    # Calculate fold changes
    actual_fc = actual - controls
    pred_fc = predicted - controls
    
    # Only consider genes with significant changes
    significant_mask = np.abs(actual_fc) > threshold
    
    if not np.any(significant_mask):
        return 0.0
    
    # Calculate direction accuracy
    correct_direction = (actual_fc * pred_fc) > 0
    accuracy = np.mean(correct_direction[significant_mask])
    
    return float(accuracy)


def differential_expression_metrics(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    controls: Union[torch.Tensor, np.ndarray],
    significance_threshold: float = 0.05,
    effect_size_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive differential expression metrics.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values
        controls: Control expression values
        significance_threshold: P-value threshold (not used currently)
        effect_size_threshold: Minimum effect size for significance
        
    Returns:
        Dictionary of DE metrics
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    
    metrics = {}
    
    # Calculate fold changes
    actual_fc = actual - controls
    pred_fc = predicted - controls
    
    # Identify truly differentially expressed genes
    true_de = np.abs(actual_fc) > effect_size_threshold
    pred_de = np.abs(pred_fc) > effect_size_threshold
    
    # Calculate per-sample metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i in range(len(actual_fc)):
        if np.any(true_de[i]) or np.any(pred_de[i]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_de[i], pred_de[i], average='binary', zero_division=0
            )
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
    
    if precision_scores:
        metrics['de_precision'] = float(np.mean(precision_scores))
        metrics['de_recall'] = float(np.mean(recall_scores))
        metrics['de_f1'] = float(np.mean(f1_scores))
    else:
        metrics['de_precision'] = 0.0
        metrics['de_recall'] = 0.0
        metrics['de_f1'] = 0.0
    
    # Effect size correlation
    effect_correlations = []
    for i in range(len(actual_fc)):
        if len(actual_fc[i]) > 1:
            corr = pearson_correlation(pred_fc[i], actual_fc[i])
            effect_correlations.append(corr)
    
    if effect_correlations:
        metrics['effect_size_correlation'] = float(np.mean(effect_correlations))
    else:
        metrics['effect_size_correlation'] = 0.0
    
    # Direction accuracy
    metrics['direction_accuracy'] = direction_accuracy(predicted, actual, controls)
    
    # Top-K gene recovery
    metrics['top_100_recovery'] = top_k_gene_recovery(predicted, actual, controls, k=100)
    metrics['top_50_recovery'] = top_k_gene_recovery(predicted, actual, controls, k=50)
    metrics['top_20_recovery'] = top_k_gene_recovery(predicted, actual, controls, k=20)
    
    return metrics


def perturbation_strength_metrics(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    controls: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate perturbation strength-related metrics.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values  
        controls: Control expression values
        
    Returns:
        Dictionary of perturbation strength metrics
    """
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    
    metrics = {}
    
    # Calculate perturbation strengths (L2 norm of change)
    actual_strength = np.linalg.norm(actual - controls, axis=1)
    pred_strength = np.linalg.norm(predicted - controls, axis=1)
    
    # Strength correlation
    if len(actual_strength) > 1:
        strength_corr = pearson_correlation(pred_strength, actual_strength)
        metrics['strength_correlation'] = strength_corr
    else:
        metrics['strength_correlation'] = 0.0
    
    # Strength MSE
    metrics['strength_mse'] = float(mean_squared_error(actual_strength, pred_strength))
    
    # Strength MAE
    metrics['strength_mae'] = float(mean_absolute_error(actual_strength, pred_strength))
    
    # Relative strength error
    relative_errors = np.abs(pred_strength - actual_strength) / (actual_strength + 1e-8)
    metrics['relative_strength_error'] = float(np.mean(relative_errors))
    
    return metrics


def batch_evaluation_metrics(
    predicted: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    perturbations: List[str],
    controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
    control_label: str = "non-targeting"
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for a batch of predictions.
    
    Args:
        predicted: Predicted expression values
        actual: Actual expression values
        perturbations: List of perturbation names
        controls: Control expression values (optional)
        control_label: Label for control perturbations
        
    Returns:
        Dictionary of comprehensive metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['pearson_correlation'] = pearson_correlation(predicted, actual)
    metrics['cosine_similarity'] = cosine_similarity(predicted, actual)
    metrics['mse'] = mean_squared_error_metric(predicted, actual)
    
    # Per-sample metrics
    per_sample_pearson = pearson_correlation(predicted, actual, per_sample=True)
    per_sample_cosine = cosine_similarity(predicted, actual, per_sample=True)
    per_sample_mse = mean_squared_error_metric(predicted, actual, per_sample=True)
    
    metrics['mean_per_sample_pearson'] = float(np.mean(per_sample_pearson))
    metrics['std_per_sample_pearson'] = float(np.std(per_sample_pearson))
    metrics['mean_per_sample_cosine'] = float(np.mean(per_sample_cosine))
    metrics['mean_per_sample_mse'] = float(np.mean(per_sample_mse))
    
    # If controls are provided, calculate DE metrics
    if controls is not None:
        de_metrics = differential_expression_metrics(predicted, actual, controls)
        metrics.update(de_metrics)
        
        strength_metrics = perturbation_strength_metrics(predicted, actual, controls)
        metrics.update(strength_metrics)
    
    # Per-perturbation analysis
    unique_perts = [p for p in set(perturbations) if p != control_label]
    if unique_perts:
        pert_correlations = []
        
        for pert in unique_perts:
            pert_mask = np.array(perturbations) == pert
            if np.any(pert_mask):
                pert_pred = predicted[pert_mask] if isinstance(predicted, np.ndarray) else predicted[pert_mask].detach().cpu().numpy()
                pert_actual = actual[pert_mask] if isinstance(actual, np.ndarray) else actual[pert_mask].detach().cpu().numpy()
                
                # Average across cells for this perturbation
                pred_mean = np.mean(pert_pred, axis=0)
                actual_mean = np.mean(pert_actual, axis=0)
                
                corr = pearson_correlation(pred_mean, actual_mean)
                pert_correlations.append(corr)
        
        if pert_correlations:
            metrics['mean_perturbation_correlation'] = float(np.mean(pert_correlations))
            metrics['std_perturbation_correlation'] = float(np.std(pert_correlations))
    
    return metrics 