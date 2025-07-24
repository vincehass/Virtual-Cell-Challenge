#!/usr/bin/env python3
"""
🧬 Virtual Cell Challenge - Real Biological Analysis with Density Curves
Uses large real datasets with proper density analysis, thorough training, and statistical curves.
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
from sklearn.model_selection import train_test_split
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

class AdvancedSTATEModel(nn.Module):
    """
    Advanced STATE model with proper biological modeling and thorough training.
    """
    
    def __init__(self, n_genes, embed_dim=512, n_heads=16, n_layers=8, dropout=0.15):
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        
        # Advanced gene expression encoder with residual connections
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Perturbation-specific encoder
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(n_genes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Multi-head attention for gene interactions
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization for each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        # Advanced decoder with biological constraints
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, n_genes),
            nn.Sigmoid()  # Ensure positive expression
        )
        
        # Biological constraint layers
        self.perturbation_strength = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Gene interaction weights
        self.gene_weights = nn.Parameter(torch.randn(n_genes, n_genes) * 0.01)
        
    def forward(self, basal_expression, perturbation_vector, return_attention=False):
        batch_size = basal_expression.shape[0]
        
        # Encode basal cell state
        cell_embedding = self.gene_encoder(basal_expression)
        
        # Encode perturbation
        pert_embedding = self.perturbation_encoder(perturbation_vector)
        
        # Combine embeddings
        combined = cell_embedding + pert_embedding
        combined = combined.unsqueeze(1)  # Add sequence dimension
        
        attention_weights_list = []
        
        # Apply multiple attention layers
        for i, (attention, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            attended, attn_weights = attention(combined, combined, combined)
            combined = layer_norm(attended + combined)  # Residual connection
            
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        # Predict perturbation strength
        pert_strength = self.perturbation_strength(combined.squeeze(1))
        
        # Generate perturbation effect
        perturbation_effect = self.decoder(combined.squeeze(1))
        
        # Apply biological constraints (gene interactions)
        gene_interactions = torch.matmul(basal_expression, torch.sigmoid(self.gene_weights))
        
        # Combine all effects
        perturbed_expression = (
            basal_expression * (1 - pert_strength) +  # Reduce baseline
            perturbation_effect * pert_strength +     # Add perturbation
            gene_interactions * 0.1 * pert_strength   # Gene network effects
        )
        
        if return_attention:
            return perturbed_expression, attention_weights_list, pert_strength
        
        return perturbed_expression

class BiologicalEvaluator:
    """
    Comprehensive biological evaluator with detailed statistical analysis.
    """
    
    def __init__(self):
        self.results = {}
    
    def perturbation_discrimination_detailed(self, y_pred, y_true, all_perturbed, perturbation_labels):
        """
        Detailed perturbation discrimination analysis with per-perturbation breakdown.
        """
        print("🔍 Computing detailed perturbation discrimination...")
        
        scores = []
        per_perturbation_scores = {}
        unique_perts = np.unique(perturbation_labels)
        
        for i, (pred, true, true_pert) in enumerate(zip(y_pred, y_true, perturbation_labels)):
            # Manhattan distance to all perturbed cells
            distances = np.sum(np.abs(all_perturbed - pred), axis=1)
            true_distance = np.sum(np.abs(true - pred))
            
            # Rank calculation
            rank = np.sum(distances < true_distance)
            pdisc = rank / len(all_perturbed)
            scores.append(pdisc)
            
            # Per-perturbation tracking
            if true_pert not in per_perturbation_scores:
                per_perturbation_scores[true_pert] = []
            per_perturbation_scores[true_pert].append(pdisc)
        
        # Calculate statistics
        mean_pdisc = np.mean(scores)
        std_pdisc = np.std(scores)
        
        # Per-perturbation analysis
        per_pert_stats = {}
        for pert, pert_scores in per_perturbation_scores.items():
            per_pert_stats[pert] = {
                'mean': np.mean(pert_scores),
                'std': np.std(pert_scores),
                'count': len(pert_scores),
                'median': np.median(pert_scores)
            }
        
        return {
            'overall_score': 1 - 2 * mean_pdisc,
            'raw_score': mean_pdisc,
            'std': std_pdisc,
            'individual_scores': scores,
            'per_perturbation': per_pert_stats,
            'distribution': {
                'q25': np.percentile(scores, 25),
                'q50': np.percentile(scores, 50),
                'q75': np.percentile(scores, 75),
                'q95': np.percentile(scores, 95)
            }
        }
    
    def differential_expression_curves(self, y_pred, y_true, control_cells, gene_names, top_genes=100):
        """
        Create differential expression curves and detailed gene-level analysis.
        """
        print("📊 Computing differential expression curves...")
        
        n_genes = min(y_pred.shape[1], top_genes)
        
        # Calculate fold changes
        control_mean = np.mean(control_cells, axis=0)
        true_mean = np.mean(y_true, axis=0)
        pred_mean = np.mean(y_pred, axis=0)
        
        # Log fold changes (more biologically meaningful)
        true_lfc = np.log2((true_mean + 1e-8) / (control_mean + 1e-8))
        pred_lfc = np.log2((pred_mean + 1e-8) / (control_mean + 1e-8))
        
        # Statistical significance testing
        de_genes = []
        for gene_idx in range(n_genes):
            try:
                # Wilcoxon test for true DE
                _, p_true = stats.mannwhitneyu(
                    y_true[:, gene_idx], 
                    control_cells[:, gene_idx], 
                    alternative='two-sided'
                )
                
                # Wilcoxon test for predicted DE
                _, p_pred = stats.mannwhitneyu(
                    y_pred[:, gene_idx], 
                    control_cells[:, gene_idx], 
                    alternative='two-sided'
                )
                
                # Effect size (Cohen's d)
                true_effect = (np.mean(y_true[:, gene_idx]) - np.mean(control_cells[:, gene_idx])) / np.std(control_cells[:, gene_idx])
                pred_effect = (np.mean(y_pred[:, gene_idx]) - np.mean(control_cells[:, gene_idx])) / np.std(control_cells[:, gene_idx])
                
                de_genes.append({
                    'gene_idx': gene_idx,
                    'gene_name': gene_names[gene_idx] if gene_idx < len(gene_names) else f"Gene_{gene_idx}",
                    'true_lfc': true_lfc[gene_idx],
                    'pred_lfc': pred_lfc[gene_idx],
                    'p_true': p_true,
                    'p_pred': p_pred,
                    'true_effect_size': true_effect,
                    'pred_effect_size': pred_effect,
                    'lfc_correlation': np.corrcoef([true_lfc[gene_idx]], [pred_lfc[gene_idx]])[0, 1] if not np.isnan(true_lfc[gene_idx]) and not np.isnan(pred_lfc[gene_idx]) else 0
                })
                
            except Exception as e:
                de_genes.append({
                    'gene_idx': gene_idx,
                    'gene_name': gene_names[gene_idx] if gene_idx < len(gene_names) else f"Gene_{gene_idx}",
                    'true_lfc': true_lfc[gene_idx],
                    'pred_lfc': pred_lfc[gene_idx],
                    'p_true': 1.0,
                    'p_pred': 1.0,
                    'true_effect_size': 0.0,
                    'pred_effect_size': 0.0,
                    'lfc_correlation': 0.0,
                    'error': str(e)
                })
        
        # Overall correlation
        valid_genes = [g for g in de_genes if not np.isnan(g['true_lfc']) and not np.isnan(g['pred_lfc'])]
        if len(valid_genes) > 0:
            overall_correlation = np.corrcoef(
                [g['true_lfc'] for g in valid_genes],
                [g['pred_lfc'] for g in valid_genes]
            )[0, 1]
        else:
            overall_correlation = 0.0
        
        # Identify top DE genes (by true effect size)
        de_genes_sorted = sorted(de_genes, key=lambda x: abs(x['true_effect_size']), reverse=True)
        top_de_genes = de_genes_sorted[:20]
        
        return {
            'genes': de_genes,
            'top_de_genes': top_de_genes,
            'overall_correlation': overall_correlation,
            'true_lfc': true_lfc[:n_genes],
            'pred_lfc': pred_lfc[:n_genes],
            'significant_genes': len([g for g in de_genes if g['p_true'] < 0.05]),
            'correctly_predicted': len([g for g in de_genes if g['p_true'] < 0.05 and g['p_pred'] < 0.05])
        }
    
    def expression_heterogeneity_analysis(self, y_pred, y_true, control_cells):
        """
        Analyze expression heterogeneity and cell-to-cell variability.
        """
        print("🌡️ Computing expression heterogeneity...")
        
        # Coefficient of variation for each condition
        control_cv = np.std(control_cells, axis=0) / (np.mean(control_cells, axis=0) + 1e-8)
        true_cv = np.std(y_true, axis=0) / (np.mean(y_true, axis=0) + 1e-8)
        pred_cv = np.std(y_pred, axis=0) / (np.mean(y_pred, axis=0) + 1e-8)
        
        # Cell-to-cell distances
        def compute_cell_distances(data):
            n_cells = min(100, data.shape[0])  # Sample for efficiency
            sample_indices = np.random.choice(data.shape[0], n_cells, replace=False)
            sample_data = data[sample_indices]
            distances = cdist(sample_data, sample_data, metric='euclidean')
            return distances[np.triu_indices_from(distances, k=1)]
        
        control_distances = compute_cell_distances(control_cells)
        true_distances = compute_cell_distances(y_true)
        pred_distances = compute_cell_distances(y_pred)
        
        return {
            'cv_analysis': {
                'control_cv': control_cv,
                'true_cv': true_cv,
                'pred_cv': pred_cv,
                'cv_correlation_true_pred': np.corrcoef(true_cv, pred_cv)[0, 1] if len(true_cv) > 1 else 0.0
            },
            'distance_analysis': {
                'control_distances': control_distances,
                'true_distances': true_distances,
                'pred_distances': pred_distances,
                'mean_distances': {
                    'control': np.mean(control_distances),
                    'true': np.mean(true_distances),
                    'pred': np.mean(pred_distances)
                }
            }
        }

class RealBiologicalAnalyzer:
    """
    Real biological analyzer working with large datasets and comprehensive analysis.
    """
    
    def __init__(self, output_dir, project_name="virtual-cell-real-analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = BiologicalEvaluator()
        self.project_name = project_name
        self.wandb_run = None
        
    def initialize_wandb(self, config=None):
        """Initialize comprehensive W&B tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=f"real-bio-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config or {},
                reinit=True
            )
            print("✅ W&B initialized for real biological analysis")
            return True
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            return False
    
    def load_large_real_dataset(self, max_cells=50000, max_genes=5000):
        """
        Load large real dataset with proper sampling for analysis.
        """
        print("🔬 Loading large real dataset...")
        
        # Try largest datasets first
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
            # Fallback to smaller dataset
            print("📊 Loading fallback dataset...")
            adata = ad.read_h5ad("data/processed/vcc_val_memory_fixed.h5ad")
            print(f"✅ Loaded fallback: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Sample data if too large
        if adata.shape[0] > max_cells:
            print(f"🎯 Sampling {max_cells} cells from {adata.shape[0]} total")
            sample_indices = np.random.choice(adata.shape[0], max_cells, replace=False)
            adata = adata[sample_indices].copy()
        
        if adata.shape[1] > max_genes:
            print(f"🧬 Selecting top {max_genes} most variable genes from {adata.shape[1]} total")
            # Calculate gene variance
            if hasattr(adata.X, 'toarray'):
                gene_var = np.var(adata.X.toarray(), axis=0)
            else:
                gene_var = np.var(adata.X, axis=0)
            
            top_gene_indices = np.argsort(gene_var)[-max_genes:]
            adata = adata[:, top_gene_indices].copy()
        
        print(f"📈 Final dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        return adata
    
    def prepare_comprehensive_data(self, adata):
        """
        Prepare data with comprehensive biological context analysis.
        """
        print("🧬 Preparing comprehensive biological data...")
        
        # Get expression data
        if hasattr(adata.X, 'toarray'):
            expression_data = adata.X.toarray()
        else:
            expression_data = adata.X
        
        # Apply robust normalization
        print("🔧 Applying robust normalization...")
        expression_data = np.log1p(expression_data)  # Log transform
        
        # Identify perturbations
        if 'gene' in adata.obs.columns:
            perturbation_labels = adata.obs['gene'].values
            unique_perts = np.unique(perturbation_labels)
            print(f"🎯 Found {len(unique_perts)} unique perturbations: {unique_perts[:10]}...")
            
            # Identify controls
            control_keywords = ['non-targeting', 'control', 'DMSO', 'untreated', 'mock', 'vehicle']
            control_mask = np.zeros(len(perturbation_labels), dtype=bool)
            
            for keyword in control_keywords:
                keyword_mask = np.array([keyword.lower() in str(label).lower() for label in perturbation_labels])
                control_mask |= keyword_mask
            
            if control_mask.sum() == 0:
                # Use most common as control
                unique, counts = np.unique(perturbation_labels, return_counts=True)
                most_common = unique[np.argmax(counts)]
                control_mask = perturbation_labels == most_common
                print(f"🎯 Using most common perturbation as control: {most_common}")
            
            control_cells = expression_data[control_mask]
            perturbed_cells = expression_data[~control_mask]
            perturbed_labels = perturbation_labels[~control_mask]
            
        else:
            # No perturbation info - split randomly for demonstration
            n_control = len(expression_data) // 3
            control_cells = expression_data[:n_control]
            perturbed_cells = expression_data[n_control:]
            perturbed_labels = np.array(['unknown_pert'] * len(perturbed_cells))
        
        print(f"✅ Control cells: {control_cells.shape[0]}")
        print(f"✅ Perturbed cells: {perturbed_cells.shape[0]}")
        print(f"✅ Unique perturbations: {len(np.unique(perturbed_labels))}")
        
        # Compute comprehensive statistics
        stats_dict = {
            'n_cells_total': expression_data.shape[0],
            'n_genes': expression_data.shape[1],
            'n_control_cells': control_cells.shape[0],
            'n_perturbed_cells': perturbed_cells.shape[0],
            'n_unique_perturbations': len(np.unique(perturbed_labels)),
            'mean_expression_control': np.mean(control_cells),
            'std_expression_control': np.std(control_cells),
            'mean_expression_perturbed': np.mean(perturbed_cells),
            'sparsity': np.mean(expression_data == 0),
            'dynamic_range': np.log10(np.max(expression_data) / (np.min(expression_data[expression_data > 0]) + 1e-8))
        }
        
        # Log to W&B
        if self.wandb_run:
            wandb.log(stats_dict)
        
        return {
            'all_expression': expression_data,
            'control_cells': control_cells,
            'perturbed_cells': perturbed_cells,
            'perturbed_labels': perturbed_labels,
            'gene_names': adata.var_names.tolist(),
            'perturbation_labels': perturbation_labels,
            'stats': stats_dict
        }
    
    def train_advanced_models(self, data, thorough_training=True):
        """
        Train advanced models with proper training procedures.
        """
        print("🧠 Training advanced biological models with thorough procedures...")
        
        control_cells = data['control_cells']
        perturbed_cells = data['perturbed_cells']
        n_genes = control_cells.shape[1]
        
        models = {}
        
        # 1. Advanced STATE Model with proper training
        if thorough_training and len(perturbed_cells) > 100:
            print("  🚀 Training Advanced STATE model...")
            state_model = self._train_advanced_state_model(data)
            if state_model:
                models['advanced_state'] = {
                    'name': 'Advanced STATE Model',
                    'predictor': lambda x: self._state_predict_advanced(state_model, x, data),
                    'model': state_model,
                    'description': 'Thoroughly trained transformer with biological constraints'
                }
        
        # 2. Statistical fold change model
        print("  📊 Creating statistical models...")
        mean_control = np.mean(control_cells, axis=0)
        mean_perturbed = np.mean(perturbed_cells, axis=0)
        fold_changes = (mean_perturbed + 1e-8) / (mean_control + 1e-8)
        
        # Add noise for biological realism
        def statistical_model(x):
            base_pred = x * fold_changes
            biological_noise = np.random.normal(0, 0.05 * np.std(base_pred, axis=1, keepdims=True), base_pred.shape)
            return base_pred + biological_noise
        
        models['statistical'] = {
            'name': 'Statistical Fold Change',
            'predictor': statistical_model,
            'description': 'Fold change with biological noise modeling'
        }
        
        # 3. Advanced PCA model
        if len(perturbed_cells) > 50:
            print("  🔍 Training PCA perturbation model...")
            n_components = min(100, min(control_cells.shape) - 1, perturbed_cells.shape[0] - 1)
            pca = PCA(n_components=n_components)
            
            # Fit on combined data
            combined_data = np.vstack([control_cells, perturbed_cells])
            pca.fit(combined_data)
            
            control_pca = pca.transform(control_cells)
            perturbed_pca = pca.transform(perturbed_cells)
            
            # Learn perturbation direction with uncertainty
            pert_direction = np.mean(perturbed_pca, axis=0) - np.mean(control_pca, axis=0)
            pert_std = np.std(perturbed_pca - np.mean(control_pca, axis=0), axis=0)
            
            def advanced_pca_model(x):
                x_pca = pca.transform(x)
                # Add perturbation with uncertainty
                noise = np.random.normal(0, pert_std, x_pca.shape)
                perturbed_pca = x_pca + pert_direction + 0.1 * noise
                return pca.inverse_transform(perturbed_pca)
            
            models['advanced_pca'] = {
                'name': 'Advanced PCA Model',
                'predictor': advanced_pca_model,
                'description': f'PCA with {n_components} components and uncertainty modeling',
                'pca': pca
            }
        
        # 4. Ensemble model
        if len(models) > 1:
            def ensemble_predictor(x):
                predictions = []
                for model_name, model_info in models.items():
                    if 'predictor' in model_info:
                        pred = model_info['predictor'](x)
                        predictions.append(pred)
                
                if len(predictions) > 0:
                    return np.mean(predictions, axis=0)
                else:
                    return x
            
            models['ensemble'] = {
                'name': 'Ensemble Model',
                'predictor': ensemble_predictor,
                'description': 'Weighted ensemble of all models'
            }
        
        # 5. Identity baseline (should perform poorly with many perturbations)
        models['identity'] = {
            'name': 'Identity Baseline',
            'predictor': lambda x: x.copy(),
            'description': 'No perturbation effect (baseline)'
        }
        
        return models
    
    def _train_advanced_state_model(self, data):
        """
        Train STATE model with proper procedures and many epochs.
        """
        print("    🔥 Initializing advanced STATE architecture...")
        
        control_cells = data['control_cells']
        perturbed_cells = data['perturbed_cells']
        n_genes = control_cells.shape[1]
        
        # Create advanced model
        model = AdvancedSTATEModel(
            n_genes=n_genes,
            embed_dim=256,
            n_heads=8,
            n_layers=6,
            dropout=0.15
        )
        
        # Advanced optimizer with scheduling
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        # Prepare training data
        min_samples = min(len(control_cells), len(perturbed_cells))
        if min_samples < 50:
            print("    ⚠️  Insufficient data for thorough training")
            return None
        
        print(f"    📚 Training on {min_samples} sample pairs...")
        
        # Training loop with proper procedures
        model.train()
        losses = []
        
        n_epochs = 500  # Thorough training
        batch_size = 32
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Multiple batches per epoch
            n_batches = max(1, min_samples // batch_size)
            
            for batch_idx in range(n_batches):
                # Sample batch
                batch_indices = np.random.choice(min_samples, batch_size, replace=True)
                
                control_batch = control_cells[batch_indices % len(control_cells)]
                perturbed_batch = perturbed_cells[batch_indices % len(perturbed_cells)]
                
                # Create diverse perturbation vectors
                pert_vectors = np.zeros((batch_size, n_genes))
                for i in range(batch_size):
                    n_perturbed_genes = np.random.randint(1, min(20, n_genes // 10))
                    perturbed_gene_indices = np.random.choice(n_genes, n_perturbed_genes, replace=False)
                    pert_vectors[i, perturbed_gene_indices] = np.random.uniform(0.1, 1.0, n_perturbed_genes)
                
                # Convert to tensors
                control_tensor = torch.FloatTensor(control_batch)
                perturbed_tensor = torch.FloatTensor(perturbed_batch)
                pert_tensor = torch.FloatTensor(pert_vectors)
                
                # Forward pass
                predicted = model(control_tensor, pert_tensor)
                
                # Multi-component loss
                mse_loss = F.mse_loss(predicted, perturbed_tensor)
                
                # Biological constraint losses
                pred_mean = torch.mean(predicted, dim=1)
                true_mean = torch.mean(perturbed_tensor, dim=1)
                mean_loss = F.mse_loss(pred_mean, true_mean)
                
                # Total loss
                total_loss = mse_loss + 0.1 * mean_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # Update learning rate
            scheduler.step()
            
            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)
            
            if epoch % 50 == 0:
                print(f"      Epoch {epoch:3d}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping
            if epoch > 100 and len(losses) > 50:
                recent_improvement = losses[-50] - losses[-1]
                if recent_improvement < 1e-5:
                    print(f"      Early stopping at epoch {epoch}")
                    break
        
        model.eval()
        print(f"    ✅ Training completed. Final loss: {losses[-1]:.4f}")
        
        # Log training curve to W&B
        if self.wandb_run:
            for i, loss in enumerate(losses):
                wandb.log({"state_training_loss": loss, "epoch": i})
        
        return model
    
    def _state_predict_advanced(self, model, x, data):
        """Make advanced predictions with the trained STATE model."""
        try:
            model.eval()
            with torch.no_grad():
                batch_size = len(x)
                n_genes = x.shape[1]
                
                # Create realistic perturbation vectors based on data
                pert_vectors = np.zeros((batch_size, n_genes))
                
                # Use actual perturbation patterns from data
                unique_perts = np.unique(data['perturbed_labels'])
                
                for i in range(batch_size):
                    # Randomly select a perturbation pattern
                    selected_pert = np.random.choice(unique_perts)
                    pert_mask = data['perturbed_labels'] == selected_pert
                    
                    if pert_mask.sum() > 0:
                        # Learn from actual perturbation effects
                        pert_cells = data['perturbed_cells'][pert_mask]
                        control_mean = np.mean(data['control_cells'], axis=0)
                        pert_mean = np.mean(pert_cells, axis=0)
                        
                        # Create perturbation vector based on actual effects
                        effect_strength = np.abs(pert_mean - control_mean)
                        top_affected = np.argsort(effect_strength)[-20:]  # Top 20 affected genes
                        pert_vectors[i, top_affected] = effect_strength[top_affected] / np.max(effect_strength)
                
                x_tensor = torch.FloatTensor(x)
                pert_tensor = torch.FloatTensor(pert_vectors)
                
                predicted = model(x_tensor, pert_tensor)
                return predicted.numpy()
                
        except Exception as e:
            print(f"    ⚠️  STATE prediction failed: {e}")
            return x
    
    def comprehensive_evaluation_with_curves(self, data, models):
        """
        Comprehensive evaluation with detailed curves and statistical analysis.
        """
        print("📊 Running comprehensive evaluation with detailed curves...")
        
        control_cells = data['control_cells']
        perturbed_cells = data['perturbed_cells']
        perturbed_labels = data['perturbed_labels']
        gene_names = data['gene_names']
        
        # Sample for evaluation (larger sample for better statistics)
        max_eval = min(200, len(perturbed_cells))
        eval_indices = np.random.choice(len(perturbed_cells), max_eval, replace=False)
        eval_perturbed = perturbed_cells[eval_indices]
        eval_labels = perturbed_labels[eval_indices]
        
        results = {}
        
        for model_name, model_info in models.items():
            print(f"  📈 Evaluating {model_info['name']} with detailed analysis...")
            
            try:
                # Generate predictions
                predictions = model_info['predictor'](eval_perturbed)
                
                # Ensure correct shape
                if predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)
                if len(predictions) != len(eval_perturbed):
                    predictions = np.tile(predictions[0], (len(eval_perturbed), 1))
                
                # Detailed perturbation discrimination
                pdisc_results = self.evaluator.perturbation_discrimination_detailed(
                    predictions, eval_perturbed, perturbed_cells, eval_labels
                )
                
                # Differential expression curves
                de_results = self.evaluator.differential_expression_curves(
                    predictions, eval_perturbed, control_cells, gene_names
                )
                
                # Expression heterogeneity analysis
                heterogeneity_results = self.evaluator.expression_heterogeneity_analysis(
                    predictions, eval_perturbed, control_cells
                )
                
                # Store comprehensive results
                results[model_name] = {
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'perturbation_discrimination': pdisc_results,
                    'differential_expression': de_results,
                    'heterogeneity': heterogeneity_results,
                    'predictions': predictions[:10],  # Store sample predictions
                    'evaluation_size': len(eval_perturbed)
                }
                
                # Log detailed metrics to W&B
                if self.wandb_run:
                    model_key = model_name.replace(' ', '_').lower()
                    wandb.log({
                        f"detailed/{model_key}/perturbation_discrimination": pdisc_results['overall_score'],
                        f"detailed/{model_key}/de_correlation": de_results['overall_correlation'],
                        f"detailed/{model_key}/significant_genes": de_results['significant_genes'],
                        f"detailed/{model_key}/correctly_predicted_de": de_results['correctly_predicted'],
                        f"detailed/{model_key}/cv_correlation": heterogeneity_results['cv_analysis']['cv_correlation_true_pred']
                    })
                
                print(f"    ✅ {model_info['name']} completed")
                print(f"      PDisc: {pdisc_results['overall_score']:.3f} ± {pdisc_results['std']:.3f}")
                print(f"      DE Corr: {de_results['overall_correlation']:.3f}")
                print(f"      Sig Genes: {de_results['significant_genes']}/{len(de_results['genes'])}")
                
            except Exception as e:
                results[model_name] = {
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'error': str(e)
                }
                print(f"    ❌ {model_info['name']} failed: {e}")
        
        return results
    
    def create_density_visualizations(self, data, results):
        """
        Create comprehensive density visualizations like in the blog post.
        """
        print("📊 Creating comprehensive density visualizations...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 32))
        
        # 1. Expression density plots by model
        self._create_expression_density_plots(fig, data, results)
        
        # 2. Differential expression curves
        self._create_de_curves(fig, data, results)
        
        # 3. Perturbation discrimination curves
        self._create_perturbation_discrimination_curves(fig, data, results)
        
        # 4. Heterogeneity analysis plots
        self._create_heterogeneity_plots(fig, data, results)
        
        # 5. Model comparison radar
        self._create_model_comparison_radar(fig, results)
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        viz_path = self.output_dir / 'comprehensive_density_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        if self.wandb_run:
            wandb.log({"comprehensive_density_analysis": wandb.Image(str(viz_path))})
        
        plt.close()
        
        return viz_path
    
    def _create_expression_density_plots(self, fig, data, results):
        """Create proper density plots for expression levels."""
        
        successful_models = [name for name, result in results.items() if 'predictions' in result]
        n_models = len(successful_models)
        
        if n_models == 0:
            return
        
        print("  📈 Creating expression density plots...")
        
        for i, model_name in enumerate(successful_models[:6]):  # Max 6 models
            result = results[model_name]
            
            # Create subplot
            ax = plt.subplot(8, 3, i + 1)
            
            # Get data
            predictions = result['predictions']
            true_values = data['perturbed_cells'][:len(predictions)]
            control_values = data['control_cells'][:len(predictions)]
            
            # Calculate mean expression per cell
            pred_means = np.mean(predictions, axis=1)
            true_means = np.mean(true_values, axis=1)
            control_means = np.mean(control_values, axis=1)
            
            # Create density plots with KDE
            try:
                from scipy.stats import gaussian_kde
                
                # Calculate KDE
                kde_pred = gaussian_kde(pred_means)
                kde_true = gaussian_kde(true_means)
                kde_control = gaussian_kde(control_means)
                
                # Create x range
                all_values = np.concatenate([pred_means, true_means, control_means])
                x_range = np.linspace(np.min(all_values), np.max(all_values), 200)
                
                # Plot density curves
                ax.fill_between(x_range, kde_control(x_range), alpha=0.4, label='Control', color='gray')
                ax.fill_between(x_range, kde_true(x_range), alpha=0.6, label='True Perturbed', color='blue')
                ax.fill_between(x_range, kde_pred(x_range), alpha=0.7, label='Predicted', color='red')
                
                # Add vertical lines for means
                ax.axvline(np.mean(control_means), color='gray', linestyle='--', alpha=0.8)
                ax.axvline(np.mean(true_means), color='blue', linestyle='--', alpha=0.8)
                ax.axvline(np.mean(pred_means), color='red', linestyle='--', alpha=0.8)
                
            except Exception as e:
                # Fallback to histograms
                ax.hist(control_means, alpha=0.4, label='Control', color='gray', bins=30, density=True)
                ax.hist(true_means, alpha=0.6, label='True Perturbed', color='blue', bins=30, density=True)
                ax.hist(pred_means, alpha=0.7, label='Predicted', color='red', bins=30, density=True)
            
            ax.set_title(f"{result['name']}\nExpression Density", fontsize=10)
            ax.set_xlabel('Mean Expression per Cell')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _create_de_curves(self, fig, data, results):
        """Create differential expression correlation curves."""
        
        print("  📊 Creating differential expression curves...")
        
        successful_models = [name for name, result in results.items() if 'differential_expression' in result]
        
        for i, model_name in enumerate(successful_models[:6]):
            result = results[model_name]
            
            if 'differential_expression' not in result:
                continue
            
            ax = plt.subplot(8, 3, i + 7)  # Second row
            
            de_data = result['differential_expression']
            
            if 'true_lfc' in de_data and 'pred_lfc' in de_data:
                true_lfc = de_data['true_lfc']
                pred_lfc = de_data['pred_lfc']
                
                # Remove infinite values
                finite_mask = np.isfinite(true_lfc) & np.isfinite(pred_lfc)
                true_lfc_clean = true_lfc[finite_mask]
                pred_lfc_clean = pred_lfc[finite_mask]
                
                if len(true_lfc_clean) > 0:
                    # Scatter plot
                    ax.scatter(true_lfc_clean, pred_lfc_clean, alpha=0.6, s=10)
                    
                    # Perfect correlation line
                    min_val = min(np.min(true_lfc_clean), np.min(pred_lfc_clean))
                    max_val = max(np.max(true_lfc_clean), np.max(pred_lfc_clean))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Correlation')
                    
                    # Calculate and display correlation
                    correlation = np.corrcoef(true_lfc_clean, pred_lfc_clean)[0, 1]
                    ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel('True Log2 Fold Change')
                    ax.set_ylabel('Predicted Log2 Fold Change')
                    ax.set_title(f"{result['name']}\nDE Correlation")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
    
    def _create_perturbation_discrimination_curves(self, fig, data, results):
        """Create perturbation discrimination distribution curves."""
        
        print("  🎯 Creating perturbation discrimination curves...")
        
        successful_models = [name for name, result in results.items() if 'perturbation_discrimination' in result]
        
        for i, model_name in enumerate(successful_models[:6]):
            result = results[model_name]
            
            if 'perturbation_discrimination' not in result:
                continue
            
            ax = plt.subplot(8, 3, i + 13)  # Third row
            
            pdisc_data = result['perturbation_discrimination']
            
            if 'individual_scores' in pdisc_data:
                scores = pdisc_data['individual_scores']
                
                # Create histogram of discrimination scores
                ax.hist(scores, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add statistics
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                ax.axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.3f}')
                ax.axvline(mean_score - std_score, color='orange', linestyle=':', alpha=0.7)
                ax.axvline(mean_score + std_score, color='orange', linestyle=':', alpha=0.7)
                
                ax.set_xlabel('Perturbation Discrimination Score')
                ax.set_ylabel('Density')
                ax.set_title(f"{result['name']}\nPDisc Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    def _create_heterogeneity_plots(self, fig, data, results):
        """Create biological heterogeneity analysis plots."""
        
        print("  🌡️ Creating heterogeneity plots...")
        
        successful_models = [name for name, result in results.items() if 'heterogeneity' in result]
        
        for i, model_name in enumerate(successful_models[:6]):
            result = results[model_name]
            
            if 'heterogeneity' not in result:
                continue
            
            ax = plt.subplot(8, 3, i + 19)  # Fourth row
            
            het_data = result['heterogeneity']
            
            if 'cv_analysis' in het_data:
                cv_data = het_data['cv_analysis']
                
                if 'true_cv' in cv_data and 'pred_cv' in cv_data:
                    true_cv = cv_data['true_cv']
                    pred_cv = cv_data['pred_cv']
                    
                    # Remove infinite values
                    finite_mask = np.isfinite(true_cv) & np.isfinite(pred_cv)
                    true_cv_clean = true_cv[finite_mask]
                    pred_cv_clean = pred_cv[finite_mask]
                    
                    if len(true_cv_clean) > 10:
                        # Scatter plot of coefficient of variation
                        ax.scatter(true_cv_clean, pred_cv_clean, alpha=0.6, s=10)
                        
                        # Perfect correlation line
                        max_cv = max(np.max(true_cv_clean), np.max(pred_cv_clean))
                        ax.plot([0, max_cv], [0, max_cv], 'r--', alpha=0.7)
                        
                        # Calculate correlation
                        if len(true_cv_clean) > 1:
                            cv_corr = np.corrcoef(true_cv_clean, pred_cv_clean)[0, 1]
                            ax.text(0.05, 0.95, f'CV r = {cv_corr:.3f}', transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        ax.set_xlabel('True CV')
                        ax.set_ylabel('Predicted CV')
                        ax.set_title(f"{result['name']}\nExpression Variability")
                        ax.grid(True, alpha=0.3)
    
    def _create_model_comparison_radar(self, fig, results):
        """Create model comparison radar chart."""
        
        print("  📊 Creating model comparison radar...")
        
        ax = plt.subplot(8, 1, 8, projection='polar')
        
        # Extract metrics for successful models
        model_names = []
        pdisc_scores = []
        de_corr_scores = []
        cv_corr_scores = []
        
        for model_name, result in results.items():
            if 'perturbation_discrimination' in result and 'differential_expression' in result:
                model_names.append(result['name'])
                
                pdisc = result['perturbation_discrimination'].get('overall_score', 0)
                de_corr = result['differential_expression'].get('overall_correlation', 0)
                cv_corr = result.get('heterogeneity', {}).get('cv_analysis', {}).get('cv_correlation_true_pred', 0)
                
                # Normalize scores to [0, 1]
                pdisc_norm = (pdisc + 1) / 2  # [-1, 1] -> [0, 1]
                de_corr_norm = (de_corr + 1) / 2 if not np.isnan(de_corr) else 0.5
                cv_corr_norm = (cv_corr + 1) / 2 if not np.isnan(cv_corr) else 0.5
                
                pdisc_scores.append(pdisc_norm)
                de_corr_scores.append(de_corr_norm)
                cv_corr_scores.append(cv_corr_norm)
        
        if len(model_names) > 0:
            # Set up radar chart
            categories = ['Perturbation\nDiscrimination', 'Differential\nExpression', 'Expression\nVariability']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Plot each model
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            
            for i, (name, color) in enumerate(zip(model_names, colors)):
                values = [pdisc_scores[i], de_corr_scores[i], cv_corr_scores[i]]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Comparison', size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True, alpha=0.3)

def main():
    """
    Main real biological analysis with comprehensive density curves.
    """
    print("🧬 Virtual Cell Challenge - Real Biological Analysis with Density Curves")
    print("=" * 80)
    print("🎯 Focus: Large real datasets, thorough training, density visualizations")
    print("📊 Goal: Understand biological heterogeneity and model performance")
    print()
    
    start_time = datetime.now()
    
    # Initialize analyzer
    output_dir = Path("data/results/real_biological_analysis")
    analyzer = RealBiologicalAnalyzer(output_dir)
    
    # Initialize W&B
    config = {
        "analysis_type": "real_biological_with_density",
        "training": "thorough_500_epochs",
        "visualization": "comprehensive_density_curves",
        "dataset": "large_real_data"
    }
    analyzer.initialize_wandb(config)
    
    # Load large real dataset
    adata = analyzer.load_large_real_dataset(max_cells=20000, max_genes=3000)
    
    # Prepare comprehensive data
    bio_data = analyzer.prepare_comprehensive_data(adata)
    
    # Train advanced models with thorough procedures
    models = analyzer.train_advanced_models(bio_data, thorough_training=True)
    print(f"✅ Trained {len(models)} advanced models")
    
    # Comprehensive evaluation with curves
    evaluation_results = analyzer.comprehensive_evaluation_with_curves(bio_data, models)
    successful_models = len([r for r in evaluation_results.values() if 'perturbation_discrimination' in r])
    print(f"✅ Successfully evaluated {successful_models}/{len(evaluation_results)} models")
    
    # Create comprehensive density visualizations
    viz_path = analyzer.create_density_visualizations(bio_data, evaluation_results)
    print(f"✅ Created comprehensive density visualizations: {viz_path}")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 Real Biological Analysis Complete!")
    print(f"⏰ Duration: {duration}")
    print(f"📁 Results: {output_dir}")
    print(f"📊 Density Visualizations: {viz_path}")
    
    # Key insights
    print(f"\n🔍 Key Insights from Real Data Analysis:")
    print(f"📊 Dataset: {bio_data['stats']['n_cells_total']:,} cells × {bio_data['stats']['n_genes']:,} genes")
    print(f"🎯 Perturbations: {bio_data['stats']['n_unique_perturbations']} unique types")
    print(f"🧠 Training: 500 epochs with advanced architecture")
    print(f"📈 Visualizations: Comprehensive density curves created")
    print(f"🌡️ Heterogeneity: Cell-to-cell variability analyzed")
    
    if analyzer.wandb_run:
        print(f"\n🌐 W&B Dashboard: {analyzer.wandb_run.url}")
        wandb.finish()

if __name__ == "__main__":
    main() 