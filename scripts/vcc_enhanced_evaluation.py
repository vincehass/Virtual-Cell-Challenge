#!/usr/bin/env python3
"""
🚀 Virtual Cell Challenge - Enhanced Evaluation with Robust Metrics
Addresses critical findings from initial analysis: larger samples, cross-validation, per-perturbation analysis.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from pathlib import Path
import warnings
from datetime import datetime
import json
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from scipy.stats import gaussian_kde, chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RobustBiologicalEvaluator:
    """
    Enhanced evaluator with robust statistical methods and larger sample sizes.
    """
    
    def __init__(self):
        self.results = {}
        self.cv_results = {}
    
    def perturbation_discrimination_robust(self, y_pred, y_true, all_perturbed, perturbation_labels, 
                                         bootstrap_samples=100):
        """
        Robust perturbation discrimination with bootstrap confidence intervals.
        """
        print("🔍 Computing robust perturbation discrimination with bootstrap...")
        
        scores = []
        per_perturbation_scores = {}
        unique_perts = np.unique(perturbation_labels)
        
        # Main calculation
        for i, (pred, true, true_pert) in enumerate(zip(y_pred, y_true, perturbation_labels)):
            distances = np.sum(np.abs(all_perturbed - pred), axis=1)
            true_distance = np.sum(np.abs(true - pred))
            rank = np.sum(distances < true_distance)
            pdisc = rank / len(all_perturbed)
            scores.append(pdisc)
            
            if true_pert not in per_perturbation_scores:
                per_perturbation_scores[true_pert] = []
            per_perturbation_scores[true_pert].append(pdisc)
        
        # Bootstrap confidence intervals
        bootstrap_scores = []
        for _ in range(bootstrap_samples):
            bootstrap_indices = np.random.choice(len(scores), len(scores), replace=True)
            bootstrap_score = np.mean([scores[i] for i in bootstrap_indices])
            bootstrap_scores.append(1 - 2 * bootstrap_score)  # Convert to final score
        
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        # Per-perturbation detailed analysis
        per_pert_detailed = {}
        for pert, pert_scores in per_perturbation_scores.items():
            if len(pert_scores) >= 3:  # Minimum for meaningful statistics
                per_pert_detailed[pert] = {
                    'mean': np.mean(pert_scores),
                    'std': np.std(pert_scores),
                    'count': len(pert_scores),
                    'median': np.median(pert_scores),
                    'iqr': np.percentile(pert_scores, 75) - np.percentile(pert_scores, 25),
                    'performance_tier': 'high' if np.mean(pert_scores) < 0.3 else 'medium' if np.mean(pert_scores) < 0.6 else 'low'
                }
        
        return {
            'overall_score': 1 - 2 * np.mean(scores),
            'raw_score': np.mean(scores),
            'std': np.std(scores),
            'confidence_interval': (ci_lower, ci_upper),
            'individual_scores': scores,
            'per_perturbation': per_pert_detailed,
            'bootstrap_distribution': bootstrap_scores,
            'evaluation_power': len(scores),
            'unique_perturbations': len(unique_perts)
        }
    
    def differential_expression_comprehensive(self, y_pred, y_true, control_cells, gene_names, 
                                            effect_size_threshold=0.2):
        """
        Comprehensive differential expression analysis with effect size filtering.
        """
        print("📊 Computing comprehensive differential expression analysis...")
        
        n_genes = y_pred.shape[1]
        
        # Robust mean calculations
        control_mean = np.mean(control_cells, axis=0)
        control_std = np.std(control_cells, axis=0)
        true_mean = np.mean(y_true, axis=0)
        pred_mean = np.mean(y_pred, axis=0)
        
        # Multiple statistical approaches
        true_lfc = np.log2((true_mean + 1e-8) / (control_mean + 1e-8))
        pred_lfc = np.log2((pred_mean + 1e-8) / (control_mean + 1e-8))
        
        # Robust correlation measures
        finite_mask = np.isfinite(true_lfc) & np.isfinite(pred_lfc)
        valid_genes = np.sum(finite_mask)
        
        if valid_genes > 10:
            # Pearson correlation
            pearson_corr = np.corrcoef(true_lfc[finite_mask], pred_lfc[finite_mask])[0, 1]
            
            # Spearman correlation (rank-based, more robust)
            spearman_corr = stats.spearmanr(true_lfc[finite_mask], pred_lfc[finite_mask])[0]
            
            # Robust correlation (using median absolute deviation)
            def robust_correlation(x, y):
                x_centered = x - np.median(x)
                y_centered = y - np.median(y)
                mad_x = np.median(np.abs(x_centered))
                mad_y = np.median(np.abs(y_centered))
                if mad_x > 0 and mad_y > 0:
                    return np.median(x_centered * y_centered) / (mad_x * mad_y)
                return 0.0
            
            robust_corr = robust_correlation(true_lfc[finite_mask], pred_lfc[finite_mask])
        else:
            pearson_corr = spearman_corr = robust_corr = 0.0
        
        # Gene-level detailed analysis
        de_genes = []
        significant_true = 0
        significant_pred = 0
        correctly_predicted = 0
        
        for gene_idx in range(min(n_genes, len(gene_names))):
            try:
                # Statistical tests
                _, p_true = stats.mannwhitneyu(
                    y_true[:, gene_idx], control_cells[:, gene_idx], alternative='two-sided'
                )
                _, p_pred = stats.mannwhitneyu(
                    y_pred[:, gene_idx], control_cells[:, gene_idx], alternative='two-sided'
                )
                
                # Effect sizes
                true_effect = (true_mean[gene_idx] - control_mean[gene_idx]) / (control_std[gene_idx] + 1e-8)
                pred_effect = (pred_mean[gene_idx] - control_mean[gene_idx]) / (control_std[gene_idx] + 1e-8)
                
                # Classification
                is_sig_true = p_true < 0.05 and abs(true_effect) > effect_size_threshold
                is_sig_pred = p_pred < 0.05 and abs(pred_effect) > effect_size_threshold
                
                if is_sig_true:
                    significant_true += 1
                if is_sig_pred:
                    significant_pred += 1
                if is_sig_true and is_sig_pred:
                    correctly_predicted += 1
                
                de_genes.append({
                    'gene_idx': gene_idx,
                    'gene_name': gene_names[gene_idx],
                    'true_lfc': true_lfc[gene_idx],
                    'pred_lfc': pred_lfc[gene_idx],
                    'p_true': p_true,
                    'p_pred': p_pred,
                    'true_effect_size': true_effect,
                    'pred_effect_size': pred_effect,
                    'significant_true': is_sig_true,
                    'significant_pred': is_sig_pred,
                    'correctly_predicted': is_sig_true and is_sig_pred
                })
                
            except Exception as e:
                de_genes.append({
                    'gene_idx': gene_idx,
                    'gene_name': gene_names[gene_idx],
                    'error': str(e)
                })
        
        # Performance metrics
        precision = correctly_predicted / max(significant_pred, 1)
        recall = correctly_predicted / max(significant_true, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'correlations': {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'robust': robust_corr
            },
            'de_performance': {
                'significant_true': significant_true,
                'significant_pred': significant_pred,
                'correctly_predicted': correctly_predicted,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'genes': de_genes,
            'valid_genes': valid_genes,
            'total_genes': n_genes,
            'true_lfc': true_lfc,
            'pred_lfc': pred_lfc
        }
    
    def cross_validation_analysis(self, data, models, n_folds=5, stratify_by='perturbation'):
        """
        Comprehensive cross-validation analysis with stratification.
        """
        print(f"🔄 Running {n_folds}-fold cross-validation analysis...")
        
        perturbed_cells = data['perturbed_cells']
        perturbed_labels = data['perturbed_labels']
        control_cells = data['control_cells']
        gene_names = data['gene_names']
        
        # Create stratified folds
        unique_perts = np.unique(perturbed_labels)
        
        # Ensure minimum samples per perturbation for stratification
        valid_perts = []
        valid_indices = []
        
        for pert in unique_perts:
            pert_indices = np.where(perturbed_labels == pert)[0]
            if len(pert_indices) >= n_folds:  # Minimum requirement for stratification
                valid_perts.append(pert)
                valid_indices.extend(pert_indices)
        
        print(f"  📊 Using {len(valid_perts)} perturbations with sufficient samples for CV")
        
        if len(valid_indices) < 100:
            print("  ⚠️  Insufficient data for robust cross-validation")
            return None
        
        # Prepare stratified data
        cv_data = perturbed_cells[valid_indices]
        cv_labels = perturbed_labels[valid_indices]
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        for model_name, model_info in models.items():
            if 'predictor' not in model_info:
                continue
                
            cv_results[model_name] = {
                'fold_scores': [],
                'fold_de_corr': [],
                'fold_f1_scores': []
            }
        
        # Run cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(cv_data, cv_labels)):
            print(f"    🔄 Fold {fold_idx + 1}/{n_folds}...")
            
            test_data = cv_data[test_idx]
            test_labels = cv_labels[test_idx]
            
            for model_name, model_info in models.items():
                if 'predictor' not in model_info:
                    continue
                
                try:
                    # Generate predictions for this fold
                    predictions = model_info['predictor'](test_data)
                    
                    if predictions.ndim == 1:
                        predictions = predictions.reshape(1, -1)
                    if len(predictions) != len(test_data):
                        predictions = np.tile(predictions[0], (len(test_data), 1))
                    
                    # Evaluate perturbation discrimination
                    pdisc_results = self.perturbation_discrimination_robust(
                        predictions, test_data, cv_data, test_labels, bootstrap_samples=50
                    )
                    
                    # Evaluate differential expression
                    de_results = self.differential_expression_comprehensive(
                        predictions, test_data, control_cells, gene_names
                    )
                    
                    # Store fold results
                    cv_results[model_name]['fold_scores'].append(pdisc_results['overall_score'])
                    cv_results[model_name]['fold_de_corr'].append(de_results['correlations']['pearson'])
                    cv_results[model_name]['fold_f1_scores'].append(de_results['de_performance']['f1_score'])
                    
                except Exception as e:
                    print(f"      ❌ {model_name} failed on fold {fold_idx + 1}: {e}")
                    cv_results[model_name]['fold_scores'].append(0.0)
                    cv_results[model_name]['fold_de_corr'].append(0.0)
                    cv_results[model_name]['fold_f1_scores'].append(0.0)
        
        # Compute cross-validation statistics
        for model_name in cv_results:
            scores = cv_results[model_name]['fold_scores']
            de_corrs = cv_results[model_name]['fold_de_corr']
            f1_scores = cv_results[model_name]['fold_f1_scores']
            
            cv_results[model_name]['statistics'] = {
                'pdisc_mean': np.mean(scores),
                'pdisc_std': np.std(scores),
                'pdisc_ci': (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
                'de_corr_mean': np.mean(de_corrs),
                'de_corr_std': np.std(de_corrs),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores)
            }
        
        return cv_results

class EnhancedVCCAnalyzer:
    """
    Enhanced Virtual Cell Challenge analyzer with robust evaluation methods.
    """
    
    def __init__(self, output_dir, project_name="virtual-cell-enhanced-eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = RobustBiologicalEvaluator()
        self.project_name = project_name
        self.wandb_run = None
        
    def initialize_wandb(self, config=None):
        """Initialize W&B with enhanced configuration."""
        try:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=f"enhanced-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config or {},
                reinit=True
            )
            print("✅ W&B initialized for enhanced evaluation")
            return True
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            return False
    
    def load_and_prepare_enhanced_data(self, max_cells=50000, max_genes=3000):
        """Load and prepare data with enhanced sampling strategy."""
        print("🔬 Loading data with enhanced sampling strategy...")
        
        # Load the largest available dataset
        dataset_paths = [
            "data/processed/vcc_training_processed.h5ad",
            "data/processed/vcc_train_memory_fixed.h5ad", 
            "data/processed/vcc_complete_memory_fixed.h5ad"
        ]
        
        adata = None
        for path in dataset_paths:
            if Path(path).exists():
                print(f"📊 Loading: {path}")
                try:
                    adata = ad.read_h5ad(path)
                    print(f"✅ Loaded dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
                    continue
        
        if adata is None:
            raise ValueError("No valid dataset found")
        
        # Enhanced sampling strategy: stratified by perturbation
        if 'gene' in adata.obs.columns:
            perturbation_labels = adata.obs['gene'].values
            unique_perts = np.unique(perturbation_labels)
            
            # Stratified sampling to ensure representation of all perturbations
            sampled_indices = []
            samples_per_pert = max_cells // len(unique_perts)
            
            for pert in unique_perts:
                pert_indices = np.where(perturbation_labels == pert)[0]
                if len(pert_indices) > 0:
                    n_sample = min(samples_per_pert, len(pert_indices))
                    selected = np.random.choice(pert_indices, n_sample, replace=False)
                    sampled_indices.extend(selected)
            
            # Add random samples if we haven't reached max_cells
            remaining = max_cells - len(sampled_indices)
            if remaining > 0:
                all_indices = set(range(adata.shape[0]))
                unused_indices = list(all_indices - set(sampled_indices))
                if len(unused_indices) >= remaining:
                    additional = np.random.choice(unused_indices, remaining, replace=False)
                    sampled_indices.extend(additional)
            
            adata = adata[sampled_indices].copy()
            print(f"📈 Stratified sampling: {adata.shape[0]} cells selected")
        else:
            # Random sampling fallback
            if adata.shape[0] > max_cells:
                sample_indices = np.random.choice(adata.shape[0], max_cells, replace=False)
                adata = adata[sample_indices].copy()
        
        # Gene selection (top variable)
        if adata.shape[1] > max_genes:
            if hasattr(adata.X, 'toarray'):
                gene_var = np.var(adata.X.toarray(), axis=0)
            else:
                gene_var = np.var(adata.X, axis=0)
            
            top_gene_indices = np.argsort(gene_var)[-max_genes:]
            adata = adata[:, top_gene_indices].copy()
        
        print(f"📈 Final dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Prepare data with enhanced processing
        if hasattr(adata.X, 'toarray'):
            expression_data = adata.X.toarray()
        else:
            expression_data = adata.X
        
        expression_data = np.log1p(expression_data)
        
        # Enhanced perturbation identification
        if 'gene' in adata.obs.columns:
            perturbation_labels = adata.obs['gene'].values
            unique_perts = np.unique(perturbation_labels)
            
            # Improved control identification
            control_keywords = ['non-targeting', 'control', 'DMSO', 'untreated', 'mock', 'vehicle', 'neg']
            control_mask = np.zeros(len(perturbation_labels), dtype=bool)
            
            for keyword in control_keywords:
                keyword_mask = np.array([keyword.lower() in str(label).lower() for label in perturbation_labels])
                control_mask |= keyword_mask
            
            if control_mask.sum() == 0:
                unique, counts = np.unique(perturbation_labels, return_counts=True)
                most_common = unique[np.argmax(counts)]
                control_mask = perturbation_labels == most_common
                print(f"🎯 Using most common as control: {most_common}")
            
            control_cells = expression_data[control_mask]
            perturbed_cells = expression_data[~control_mask]
            perturbed_labels = perturbation_labels[~control_mask]
        else:
            n_control = len(expression_data) // 3
            control_cells = expression_data[:n_control]
            perturbed_cells = expression_data[n_control:]
            perturbed_labels = np.array(['unknown_pert'] * len(perturbed_cells))
        
        return {
            'all_expression': expression_data,
            'control_cells': control_cells,
            'perturbed_cells': perturbed_cells,
            'perturbed_labels': perturbed_labels,
            'gene_names': adata.var_names.tolist(),
            'perturbation_labels': perturbation_labels,
            'unique_perturbations': len(np.unique(perturbed_labels)),
            'n_control': len(control_cells),
            'n_perturbed': len(perturbed_cells)
        }
    
    def create_enhanced_models(self, data):
        """Create enhanced model suite with improved baselines."""
        print("🧠 Creating enhanced model suite...")
        
        control_cells = data['control_cells']
        perturbed_cells = data['perturbed_cells']
        
        models = {}
        
        # 1. Enhanced statistical model with perturbation-specific effects
        print("  📊 Creating enhanced statistical models...")
        per_pert_effects = {}
        unique_perts = np.unique(data['perturbed_labels'])
        
        for pert in unique_perts:
            pert_mask = data['perturbed_labels'] == pert
            if pert_mask.sum() >= 5:  # Minimum samples for reliable statistics
                pert_cells = data['perturbed_cells'][pert_mask]
                pert_mean = np.mean(pert_cells, axis=0)
                control_mean = np.mean(control_cells, axis=0)
                per_pert_effects[pert] = pert_mean / (control_mean + 1e-8)
        
        def enhanced_statistical_model(x):
            # Use random perturbation effects
            selected_pert = np.random.choice(list(per_pert_effects.keys()))
            fold_changes = per_pert_effects[selected_pert]
            
            base_pred = x * fold_changes
            # Add realistic biological noise
            noise_scale = 0.1 * np.std(base_pred, axis=1, keepdims=True)
            biological_noise = np.random.normal(0, noise_scale, base_pred.shape)
            return base_pred + biological_noise
        
        models['enhanced_statistical'] = {
            'name': 'Enhanced Statistical Model',
            'predictor': enhanced_statistical_model,
            'description': f'Perturbation-specific effects from {len(per_pert_effects)} perturbations'
        }
        
        # 2. Multi-scale PCA model
        print("  🔍 Creating multi-scale PCA model...")
        pca_models = {}
        for n_comp in [50, 100, 200]:
            if n_comp < min(control_cells.shape):
                pca = PCA(n_components=n_comp)
                combined_data = np.vstack([control_cells, perturbed_cells])
                pca.fit(combined_data)
                pca_models[n_comp] = pca
        
        def multiscale_pca_model(x):
            predictions = []
            for n_comp, pca in pca_models.items():
                x_pca = pca.transform(x)
                # Add perturbation effect in PCA space
                pert_direction = np.random.normal(0, 0.1, x_pca.shape[1])
                perturbed_pca = x_pca + pert_direction
                pred = pca.inverse_transform(perturbed_pca)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        
        models['multiscale_pca'] = {
            'name': 'Multi-scale PCA Model',
            'predictor': multiscale_pca_model,
            'description': f'Ensemble of PCA models with {list(pca_models.keys())} components'
        }
        
        # 3. K-NN perturbation model
        print("  🔗 Creating K-NN perturbation model...")
        if len(perturbed_cells) > 50:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit KNN on perturbed cells
            knn = NearestNeighbors(n_neighbors=min(10, len(perturbed_cells) // 5))
            knn.fit(control_cells)
            
            def knn_perturbation_model(x):
                predictions = []
                for cell in x:
                    # Find nearest control cells
                    distances, indices = knn.kneighbors([cell])
                    nearest_controls = control_cells[indices[0]]
                    
                    # Find corresponding perturbation effects
                    effects = []
                    for ctrl in nearest_controls:
                        # Find perturbed cells similar to this control
                        ctrl_distances = np.sum((perturbed_cells - ctrl) ** 2, axis=1)
                        closest_pert_idx = np.argmin(ctrl_distances)
                        effect = perturbed_cells[closest_pert_idx] - ctrl
                        effects.append(effect)
                    
                    # Average effect and apply
                    mean_effect = np.mean(effects, axis=0)
                    pred = cell + mean_effect
                    predictions.append(pred)
                
                return np.array(predictions)
            
            models['knn_perturbation'] = {
                'name': 'K-NN Perturbation Model',
                'predictor': knn_perturbation_model,
                'description': 'Nearest neighbor-based perturbation effects'
            }
        
        # 4. Identity baseline (for comparison)
        models['identity'] = {
            'name': 'Identity Baseline',
            'predictor': lambda x: x.copy(),
            'description': 'No perturbation effect (baseline)'
        }
        
        return models
    
    def run_enhanced_evaluation(self, data, models, large_sample_size=5000):
        """Run enhanced evaluation with large sample sizes and cross-validation."""
        print(f"📊 Running enhanced evaluation with {large_sample_size} samples...")
        
        perturbed_cells = data['perturbed_cells']
        control_cells = data['control_cells']
        perturbed_labels = data['perturbed_labels']
        gene_names = data['gene_names']
        
        # Large sample evaluation
        eval_size = min(large_sample_size, len(perturbed_cells))
        eval_indices = np.random.choice(len(perturbed_cells), eval_size, replace=False)
        eval_perturbed = perturbed_cells[eval_indices]
        eval_labels = perturbed_labels[eval_indices]
        
        print(f"  📈 Large sample evaluation: {eval_size} cells")
        
        # Single large evaluation
        large_sample_results = {}
        for model_name, model_info in models.items():
            print(f"    🔬 Evaluating {model_info['name']}...")
            
            try:
                predictions = model_info['predictor'](eval_perturbed)
                
                if predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)
                if len(predictions) != len(eval_perturbed):
                    predictions = np.tile(predictions[0], (len(eval_perturbed), 1))
                
                # Robust perturbation discrimination
                pdisc_results = self.evaluator.perturbation_discrimination_robust(
                    predictions, eval_perturbed, perturbed_cells, eval_labels
                )
                
                # Comprehensive differential expression
                de_results = self.evaluator.differential_expression_comprehensive(
                    predictions, eval_perturbed, control_cells, gene_names
                )
                
                large_sample_results[model_name] = {
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'perturbation_discrimination': pdisc_results,
                    'differential_expression': de_results,
                    'evaluation_size': eval_size
                }
                
                # Log to W&B
                if self.wandb_run:
                    model_key = model_name.replace(' ', '_').lower()
                    wandb.log({
                        f"large_eval/{model_key}/pdisc_score": pdisc_results['overall_score'],
                        f"large_eval/{model_key}/pdisc_ci_lower": pdisc_results['confidence_interval'][0],
                        f"large_eval/{model_key}/pdisc_ci_upper": pdisc_results['confidence_interval'][1],
                        f"large_eval/{model_key}/de_pearson": de_results['correlations']['pearson'],
                        f"large_eval/{model_key}/de_spearman": de_results['correlations']['spearman'],
                        f"large_eval/{model_key}/de_f1_score": de_results['de_performance']['f1_score']
                    })
                
                print(f"      ✅ PDisc: {pdisc_results['overall_score']:.3f} "
                      f"[{pdisc_results['confidence_interval'][0]:.3f}, {pdisc_results['confidence_interval'][1]:.3f}]")
                print(f"      ✅ DE Corr: {de_results['correlations']['pearson']:.3f} "
                      f"(Spearman: {de_results['correlations']['spearman']:.3f})")
                print(f"      ✅ DE F1: {de_results['de_performance']['f1_score']:.3f}")
                
            except Exception as e:
                large_sample_results[model_name] = {
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'error': str(e)
                }
                print(f"      ❌ Failed: {e}")
        
        # Cross-validation analysis
        print(f"  🔄 Running cross-validation analysis...")
        cv_results = self.evaluator.cross_validation_analysis(data, models, n_folds=5)
        
        if cv_results:
            print(f"    ✅ Cross-validation completed for {len(cv_results)} models")
            
            # Log CV results to W&B
            if self.wandb_run:
                for model_name, cv_result in cv_results.items():
                    if 'statistics' in cv_result:
                        stats = cv_result['statistics']
                        model_key = model_name.replace(' ', '_').lower()
                        wandb.log({
                            f"cv/{model_key}/pdisc_mean": stats['pdisc_mean'],
                            f"cv/{model_key}/pdisc_std": stats['pdisc_std'],
                            f"cv/{model_key}/de_corr_mean": stats['de_corr_mean'],
                            f"cv/{model_key}/de_corr_std": stats['de_corr_std'],
                            f"cv/{model_key}/f1_mean": stats['f1_mean'],
                            f"cv/{model_key}/f1_std": stats['f1_std']
                        })
        
        return {
            'large_sample_results': large_sample_results,
            'cross_validation_results': cv_results,
            'evaluation_config': {
                'large_sample_size': eval_size,
                'cv_folds': 5 if cv_results else 0,
                'total_perturbations': data['unique_perturbations']
            }
        }

def main():
    """
    Main enhanced evaluation addressing critical findings from initial analysis.
    """
    print("🚀 Virtual Cell Challenge - Enhanced Evaluation with Robust Metrics")
    print("=" * 80)
    print("🎯 Addressing: Large samples, Cross-validation, Per-perturbation analysis")
    print("📊 Goal: Robust performance assessment for winning strategy")
    print()
    
    start_time = datetime.now()
    
    # Initialize enhanced analyzer
    output_dir = Path("data/results/enhanced_evaluation")
    analyzer = EnhancedVCCAnalyzer(output_dir)
    
    # Initialize W&B
    config = {
        "analysis_type": "enhanced_robust_evaluation",
        "evaluation_size": "5000_cells",
        "cross_validation": "5_fold_stratified",
        "bootstrap_samples": "100_per_metric"
    }
    analyzer.initialize_wandb(config)
    
    # Load and prepare enhanced data
    enhanced_data = analyzer.load_and_prepare_enhanced_data(max_cells=50000, max_genes=3000)
    print(f"✅ Enhanced data prepared: {enhanced_data['n_perturbed']} perturbed, {enhanced_data['n_control']} control")
    print(f"✅ Unique perturbations: {enhanced_data['unique_perturbations']}")
    
    # Create enhanced model suite
    enhanced_models = analyzer.create_enhanced_models(enhanced_data)
    print(f"✅ Enhanced models created: {len(enhanced_models)} models")
    
    # Run enhanced evaluation
    enhanced_results = analyzer.run_enhanced_evaluation(enhanced_data, enhanced_models, large_sample_size=5000)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 Enhanced Evaluation Complete!")
    print(f"⏰ Duration: {duration}")
    print(f"📁 Results: {output_dir}")
    
    # Key insights
    large_results = enhanced_results['large_sample_results']
    cv_results = enhanced_results['cross_validation_results']
    
    print(f"\n🔍 Enhanced Evaluation Insights:")
    print(f"📊 Large Sample Size: {enhanced_results['evaluation_config']['large_sample_size']:,} cells")
    print(f"🔄 Cross-Validation: {enhanced_results['evaluation_config']['cv_folds']} folds")
    print(f"🎯 Total Perturbations: {enhanced_results['evaluation_config']['total_perturbations']}")
    
    # Show robust performance metrics
    print(f"\n📈 Model Performance (Large Sample):")
    for model_name, result in large_results.items():
        if 'perturbation_discrimination' in result:
            pdisc = result['perturbation_discrimination']
            de = result['differential_expression']
            
            print(f"  🧠 {result['name']}:")
            print(f"    📍 PDisc: {pdisc['overall_score']:.3f} [{pdisc['confidence_interval'][0]:.3f}, {pdisc['confidence_interval'][1]:.3f}]")
            print(f"    📊 DE Pearson: {de['correlations']['pearson']:.3f}")
            print(f"    🎯 DE F1: {de['de_performance']['f1_score']:.3f}")
    
    if cv_results:
        print(f"\n🔄 Cross-Validation Results:")
        for model_name, cv_result in cv_results.items():
            if 'statistics' in cv_result:
                stats = cv_result['statistics']
                print(f"  🧠 {model_name}:")
                print(f"    📍 PDisc: {stats['pdisc_mean']:.3f} ± {stats['pdisc_std']:.3f}")
                print(f"    📊 DE Corr: {stats['de_corr_mean']:.3f} ± {stats['de_corr_std']:.3f}")
    
    if analyzer.wandb_run:
        print(f"\n🌐 W&B Dashboard: {analyzer.wandb_run.url}")
        wandb.finish()

if __name__ == "__main__":
    main() 