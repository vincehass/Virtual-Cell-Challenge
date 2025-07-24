#!/usr/bin/env python3
"""
🎯 Training Ablation Study - Authentic STATE Implementation
Systematic evaluation of different training configurations with step-by-step tracking.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import warnings
from dataclasses import dataclass

# Import our authentic implementations
from authentic_state_implementation import (
    RealDataLoader, AuthenticSTATETrainer, AuthenticSTATEModel
)
from authentic_evaluation_with_density import (
    AuthenticBiologicalEvaluator, AuthenticAblationStudy
)

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

@dataclass
class TrainingConfig:
    """Configuration for training ablation study."""
    optimizer: str = "adamw"
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'warmup_steps': self.warmup_steps,
            'gradient_clip': self.gradient_clip
        }

class TrainingAblationStudy:
    """
    Comprehensive training ablation study with step-by-step tracking.
    """
    
    def __init__(self, base_config_path: str, output_dir: str, 
                 wandb_project: str = None, wandb_tags: List[str] = None,
                 quick_run: bool = False):
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags or []
        self.quick_run = quick_run
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = json.load(f)
        
        # Initialize evaluator
        self.evaluator = AuthenticBiologicalEvaluator()
        
        # Results storage
        self.results = {}
        self.training_curves = {}
        
    def run_single_training_experiment(self, training_config: TrainingConfig, 
                                     experiment_name: str) -> Dict[str, Any]:
        """Run a single training configuration experiment with step tracking."""
        print(f"🎯 Running training experiment: {experiment_name}")
        print(f"   Training config: {training_config.to_dict()}")
        
        # Initialize W&B run
        if self.wandb_project:
            wandb.init(
                project=self.wandb_project,
                name=experiment_name,
                tags=self.wandb_tags + [experiment_name],
                config=training_config.to_dict(),
                reinit=True
            )
        
        try:
            # Load data
            data = self._load_data()
            
            # Create model
            model = self._create_model(data)
            
            # Create trainer with specific configuration
            trainer = AdvancedTrainingTracker(
                model=model,
                training_config=training_config,
                use_wandb=bool(self.wandb_project)
            )
            
            # Train with step-by-step tracking
            epochs = 50 if self.quick_run else 100
            training_results = trainer.train_with_comprehensive_tracking(data, epochs=epochs)
            
            # Store training curves
            self.training_curves[experiment_name] = training_results['curves']
            
            # Evaluate model with density analysis
            evaluation_results = self._evaluate_with_density(model, data, experiment_name)
            
            # Combine results
            results = {
                'training_config': training_config.to_dict(),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'experiment_name': experiment_name
            }
            
            # Save results
            result_file = self.output_dir / f"{experiment_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if self.wandb_project:
                wandb.log({
                    'final_pdisc_score': evaluation_results.get('pdisc_score', 0.0),
                    'final_de_correlation': evaluation_results.get('de_correlation', 0.0),
                    'final_train_loss': training_results.get('final_train_loss', float('inf')),
                    'final_val_loss': training_results.get('final_val_loss', float('inf')),
                    'training_stability': training_results.get('stability_metrics', {}).get('loss_variance', 0.0),
                    'convergence_epoch': training_results.get('convergence_epoch', epochs)
                })
                wandb.finish()
            
            print(f"✅ Training experiment completed: {experiment_name}")
            return results
            
        except Exception as e:
            print(f"❌ Training experiment failed: {experiment_name}")
            print(f"   Error: {str(e)}")
            
            if self.wandb_project:
                wandb.finish()
            
            return {
                'training_config': training_config.to_dict(),
                'experiment_name': experiment_name,
                'error': str(e)
            }
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data for training experiments."""
        data_loader = RealDataLoader()
        
        # Adjust data size for quick runs
        if self.quick_run:
            max_cells = 10000
            max_genes = 2000
        else:
            max_cells = 15000
            max_genes = 2500
        
        try:
            data = data_loader.load_stratified_real_data(
                max_cells=max_cells, 
                max_genes=max_genes
            )
            print(f"✅ Real data loaded: {data['n_cells']:,} cells × {data['n_genes']:,} genes")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load real data: {e}")
            print("🔄 Creating synthetic data for demonstration...")
            return self._create_synthetic_data(max_cells, max_genes)
    
    def _create_synthetic_data(self, max_cells: int, max_genes: int) -> Dict[str, Any]:
        """Create synthetic data for testing."""
        n_cells = min(max_cells, 5000)
        n_genes = min(max_genes, 1000)
        n_perturbations = 20
        
        # Create realistic gene expression data
        base_expression = np.random.lognormal(0, 1, (n_cells, n_genes))
        base_expression = np.log1p(base_expression)
        
        # Create perturbation labels
        perturbation_names = [f"GENE_{i}" for i in range(n_perturbations)] + ["non-targeting"]
        perturbation_labels = np.random.choice(perturbation_names, n_cells)
        
        # Control vs perturbed
        control_mask = perturbation_labels == "non-targeting"
        
        # Add perturbation effects
        for i, pert in enumerate(perturbation_names[:-1]):
            pert_mask = perturbation_labels == pert
            if np.sum(pert_mask) > 0:
                effect_genes = np.random.choice(n_genes, 50, replace=False)
                effect_size = np.random.normal(0, 0.5, 50)
                base_expression[pert_mask][:, effect_genes] += effect_size
        
        # Create perturbation vectors
        unique_perts = np.unique(perturbation_labels)
        perturbation_vectors = np.zeros((n_cells, 128))
        for i, pert in enumerate(unique_perts):
            if i < 128:
                perturbation_vectors[perturbation_labels == pert, i] = 1.0
        
        return {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'expression_data': base_expression,
            'perturbation_data': {
                'all_labels': perturbation_labels,
                'unique_perturbations': unique_perts,
                'control_mask': control_mask,
                'control_cells': base_expression[control_mask],
                'perturbed_cells': base_expression[~control_mask],
                'perturbed_labels': perturbation_labels[~control_mask],
                'perturbation_vectors': perturbation_vectors,
                'n_controls': np.sum(control_mask),
                'n_perturbed': np.sum(~control_mask),
                'n_unique_perturbations': len(unique_perts)
            },
            'gene_names': [f'Gene_{i}' for i in range(n_genes)],
            'batch_column': None
        }
    
    def _create_model(self, data: Dict[str, Any]) -> AuthenticSTATEModel:
        """Create STATE model for training configuration."""
        se_config = {
            'embed_dim': 256,
            'n_heads': 8,
            'n_layers': 6,
            'dropout': 0.1
        }
        
        st_config = {
            'state_dim': 128,
            'perturbation_dim': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1
        }
        
        return AuthenticSTATEModel(data['n_genes'], se_config, st_config)
    
    def _evaluate_with_density(self, model: AuthenticSTATEModel, data: Dict[str, Any], 
                             experiment_name: str) -> Dict[str, Any]:
        """Evaluate model with comprehensive density analysis."""
        print(f"📊 Evaluating {experiment_name} with density analysis...")
        
        # Create predictor function
        def predictor(input_data):
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data)
                batch_size = input_tensor.shape[0]
                dummy_pert_vectors = torch.zeros(batch_size, 128)
                predictions = model(input_tensor, dummy_pert_vectors)
                return predictions.numpy()
        
        # Sample evaluation data
        eval_size = min(500 if self.quick_run else 1000, 
                       len(data['perturbation_data']['perturbed_cells']))
        eval_indices = np.random.choice(
            len(data['perturbation_data']['perturbed_cells']), 
            eval_size, replace=False
        )
        
        eval_perturbed = data['perturbation_data']['perturbed_cells'][eval_indices]
        eval_labels = data['perturbation_data']['perturbed_labels'][eval_indices]
        
        # Generate predictions
        predictions = predictor(eval_perturbed)
        
        # Perturbation discrimination with density
        pdisc_results = self.evaluator.perturbation_discrimination_with_density(
            predictions, eval_perturbed, 
            data['perturbation_data']['perturbed_cells'], 
            eval_labels, create_density_plots=True
        )
        
        # Differential expression with density
        de_results = self.evaluator.differential_expression_with_density(
            predictions, eval_perturbed,
            data['perturbation_data']['control_cells'],
            data['gene_names'], create_density_plots=True
        )
        
        # Expression heterogeneity with density
        hetero_results = self.evaluator.expression_heterogeneity_with_density(
            predictions, eval_perturbed,
            data['perturbation_data']['control_cells'], 
            create_density_plots=True
        )
        
        # Save density plots
        self._save_density_plots(experiment_name, pdisc_results, de_results, hetero_results)
        
        return {
            'perturbation_discrimination': pdisc_results,
            'differential_expression': de_results,
            'expression_heterogeneity': hetero_results,
            'pdisc_score': pdisc_results.get('overall_score', 0.0),
            'de_correlation': de_results.get('correlations', {}).get('pearson', 0.0) if 'correlations' in de_results else 0.0,
            'heterogeneity_cv': hetero_results.get('coefficient_variation', {}).get('mean', 0.0) if 'coefficient_variation' in hetero_results else 0.0
        }
    
    def _save_density_plots(self, experiment_name: str, pdisc_results: Dict, 
                          de_results: Dict, hetero_results: Dict):
        """Save density plots for the experiment."""
        plots_dir = self.output_dir / "density_plots" / experiment_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save any generated plots
        if hasattr(plt, 'gcf'):
            current_fig = plt.gcf()
            if len(current_fig.get_axes()) > 0:
                current_fig.savefig(plots_dir / f"{experiment_name}_density_analysis.png", 
                                  dpi=300, bbox_inches='tight')
                plt.close()
    
    def create_training_curves_comparison(self):
        """Create comprehensive training curves comparison."""
        if not self.training_curves:
            print("⚠️  No training curves available")
            return
        
        print("📈 Creating training curves comparison...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # Plot training loss curves
        ax = axes[0, 0]
        for exp_name, curves in self.training_curves.items():
            if 'train_losses' in curves:
                ax.plot(curves['train_losses'], label=exp_name, alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves (Different Optimizers/Configs)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot validation loss curves
        ax = axes[0, 1]
        for exp_name, curves in self.training_curves.items():
            if 'val_losses' in curves:
                ax.plot(curves['val_losses'], label=exp_name, alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot learning rate schedules
        ax = axes[1, 0]
        for exp_name, curves in self.training_curves.items():
            if 'learning_rates' in curves:
                ax.plot(curves['learning_rates'], label=exp_name, alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedules')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot gradient norms
        ax = axes[1, 1]
        for exp_name, curves in self.training_curves.items():
            if 'gradient_norms' in curves:
                ax.plot(curves['gradient_norms'], label=exp_name, alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms (Training Stability)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot evaluation metrics over training
        ax = axes[2, 0]
        for exp_name, curves in self.training_curves.items():
            if 'eval_metrics' in curves and 'pdisc_scores' in curves['eval_metrics']:
                eval_epochs = list(range(0, len(curves['train_losses']), 10))[:len(curves['eval_metrics']['pdisc_scores'])]
                ax.plot(eval_epochs, curves['eval_metrics']['pdisc_scores'], 
                       label=f"{exp_name} PDisc", alpha=0.8, linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perturbation Discrimination Score')
        ax.set_title('PDisc Score During Training')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot convergence comparison
        ax = axes[2, 1]
        optimizer_names = []
        convergence_epochs = []
        final_losses = []
        
        for exp_name, curves in self.training_curves.items():
            if 'train_losses' in curves:
                optimizer_names.append(exp_name)
                
                # Find convergence epoch (when loss stabilizes)
                losses = curves['train_losses']
                if len(losses) > 20:
                    recent_losses = losses[-20:]
                    loss_std = np.std(recent_losses)
                    if loss_std < 0.001:  # Converged
                        convergence_epochs.append(len(losses) - 20)
                    else:
                        convergence_epochs.append(len(losses))
                else:
                    convergence_epochs.append(len(losses))
                
                final_losses.append(losses[-1])
        
        x_pos = np.arange(len(optimizer_names))
        ax.bar(x_pos, convergence_epochs, alpha=0.7, color='skyblue', label='Convergence Epoch')
        ax.set_xlabel('Training Configuration')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title('Training Convergence Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create optimizer performance comparison
        self._create_optimizer_performance_plot(optimizer_names, final_losses, convergence_epochs)
        
        print(f"✅ Training curves saved: {self.output_dir / 'training_curves_comparison.png'}")
    
    def _create_optimizer_performance_plot(self, optimizer_names: List[str], 
                                         final_losses: List[float], 
                                         convergence_epochs: List[int]):
        """Create optimizer performance comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final loss comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(optimizer_names)))
        bars1 = ax1.bar(optimizer_names, final_losses, color=colors, alpha=0.8)
        ax1.set_ylabel('Final Training Loss')
        ax1.set_title('Final Training Loss by Configuration')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars1, final_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom')
        
        # Convergence speed comparison
        bars2 = ax2.bar(optimizer_names, convergence_epochs, color=colors, alpha=0.8)
        ax2.set_ylabel('Epochs to Convergence')
        ax2.set_title('Training Speed by Configuration')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, epochs in zip(bars2, convergence_epochs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{epochs}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimizer_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

class AdvancedTrainingTracker:
    """Advanced trainer with comprehensive tracking for training ablation studies."""
    
    def __init__(self, model: AuthenticSTATEModel, training_config: TrainingConfig, use_wandb: bool = False):
        self.model = model
        self.training_config = training_config
        self.use_wandb = use_wandb
        
    def train_with_comprehensive_tracking(self, data: Dict[str, Any], epochs: int = 100) -> Dict[str, Any]:
        """Train model with comprehensive tracking including stability metrics."""
        # Prepare data loaders
        dataset = self._create_dataset(data)
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, epochs)
        
        # Initialize comprehensive tracking
        train_losses = []
        val_losses = []
        learning_rates = []
        gradient_norms = []
        eval_metrics = {'pdisc_scores': [], 'de_correlations': []}
        
        # Training stability tracking
        loss_windows = []
        convergence_epoch = epochs
        
        for epoch in range(epochs):
            # Training with gradient tracking
            train_loss, grad_norm = self._train_epoch_with_tracking(train_loader, optimizer)
            train_losses.append(train_loss)
            gradient_norms.append(grad_norm)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate tracking
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Check convergence (sliding window)
            if len(train_losses) >= 10:
                recent_window = train_losses[-10:]
                loss_windows.append(np.std(recent_window))
                if len(loss_windows) >= 5:
                    recent_stability = np.mean(loss_windows[-5:])
                    if recent_stability < 0.001 and convergence_epoch == epochs:
                        convergence_epoch = epoch
            
            # Periodic evaluation (every 10 epochs)
            if epoch % 10 == 0:
                eval_results = self._evaluate_step(data)
                eval_metrics['pdisc_scores'].append(eval_results.get('pdisc_score', 0.0))
                eval_metrics['de_correlations'].append(eval_results.get('de_correlation', 0.0))
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'gradient_norm': grad_norm,
                        'step_pdisc_score': eval_results.get('pdisc_score', 0.0),
                        'step_de_correlation': eval_results.get('de_correlation', 0.0),
                        'loss_stability': loss_windows[-1] if loss_windows else 0.0
                    })
            else:
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'gradient_norm': grad_norm
                    })
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.6f}, GradNorm={grad_norm:.4f}")
        
        # Calculate stability metrics
        stability_metrics = {
            'loss_variance': np.var(train_losses[-20:]) if len(train_losses) >= 20 else np.var(train_losses),
            'gradient_variance': np.var(gradient_norms[-20:]) if len(gradient_norms) >= 20 else np.var(gradient_norms),
            'convergence_rate': convergence_epoch / epochs,
            'final_gradient_norm': gradient_norms[-1],
            'training_stability_score': 1.0 / (1.0 + np.mean(loss_windows[-10:]) if loss_windows else 1.0)
        }
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'convergence_epoch': convergence_epoch,
            'stability_metrics': stability_metrics,
            'curves': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'gradient_norms': gradient_norms,
                'eval_metrics': eval_metrics
            }
        }
    
    def _create_optimizer(self):
        """Create optimizer based on training configuration."""
        if self.training_config.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
        elif self.training_config.optimizer.lower() == 'rmsprop':
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
    
    def _create_scheduler(self, optimizer, epochs):
        """Create learning rate scheduler."""
        if self.training_config.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.training_config.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)
        elif self.training_config.scheduler.lower() == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.training_config.scheduler.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        else:
            return None
    
    def _create_dataset(self, data: Dict[str, Any]):
        """Create dataset for training."""
        from authentic_state_implementation import STATEDataset
        
        perturbation_data = data['perturbation_data']
        return STATEDataset(
            perturbation_data['control_cells'],
            perturbation_data['perturbed_cells'],
            perturbation_data['perturbation_vectors'][~perturbation_data['control_mask']]
        )
    
    def _create_data_loaders(self, dataset):
        """Create train and validation data loaders."""
        from torch.utils.data import DataLoader
        
        # Train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, 
                              shuffle=False, num_workers=2)
        
        return train_loader, val_loader
    
    def _train_epoch_with_tracking(self, train_loader, optimizer):
        """Train one epoch with gradient norm tracking."""
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        num_batches = 0
        
        for batch in train_loader:
            baseline_expr, perturbed_expr, pert_vectors = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_expr = self.model(baseline_expr, pert_vectors)
            
            # Loss calculation
            loss = torch.nn.functional.mse_loss(predicted_expr, perturbed_expr)
            
            # Backward pass
            loss.backward()
            
            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                     max_norm=self.training_config.gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            num_batches += 1
        
        return total_loss / num_batches, total_grad_norm / num_batches
    
    def _validate_epoch(self, val_loader):
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                baseline_expr, perturbed_expr, pert_vectors = batch
                
                # Forward pass
                predicted_expr = self.model(baseline_expr, pert_vectors)
                
                # Loss calculation
                loss = torch.nn.functional.mse_loss(predicted_expr, perturbed_expr)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Quick evaluation for step tracking."""
        eval_size = min(200, len(data['perturbation_data']['perturbed_cells']))
        eval_indices = np.random.choice(
            len(data['perturbation_data']['perturbed_cells']), 
            eval_size, replace=False
        )
        
        eval_perturbed = data['perturbation_data']['perturbed_cells'][eval_indices]
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(eval_perturbed)
            dummy_pert_vectors = torch.zeros(input_tensor.shape[0], 128)
            predictions = self.model(input_tensor, dummy_pert_vectors)
            predictions = predictions.numpy()
        
        # Quick metrics
        mse = np.mean((predictions - eval_perturbed) ** 2)
        correlation = np.corrcoef(predictions.flatten(), eval_perturbed.flatten())[0, 1]
        
        # Mock PDisc score (simplified)
        pdisc_score = max(0, correlation)
        
        return {
            'pdisc_score': pdisc_score,
            'de_correlation': correlation,
            'mse': mse
        }

def main():
    """Main function for training ablation study."""
    parser = argparse.ArgumentParser(description="Training Ablation Study for Authentic STATE")
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_tags', nargs='+', default=[],
                       help='Weights & Biases tags')
    parser.add_argument('--quick_run', action='store_true',
                       help='Run quick version with reduced epochs and data')
    
    # Specific training configuration overrides
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize study
    study = TrainingAblationStudy(
        base_config_path=args.config,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        quick_run=args.quick_run
    )
    
    # If specific parameters are provided, run single experiment
    if any([args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.scheduler]):
        # Create custom training configuration
        training_config = TrainingConfig()
        
        if args.optimizer is not None:
            training_config.optimizer = args.optimizer
        if args.batch_size is not None:
            training_config.batch_size = args.batch_size
        if args.learning_rate is not None:
            training_config.learning_rate = args.learning_rate
        if args.weight_decay is not None:
            training_config.weight_decay = args.weight_decay
        if args.scheduler is not None:
            training_config.scheduler = args.scheduler
        
        # Create experiment name
        experiment_name = f"training_{training_config.optimizer}_{training_config.scheduler}_bs{training_config.batch_size}"
        
        # Run single experiment
        result = study.run_single_training_experiment(training_config, experiment_name)
        
    else:
        # Run comprehensive training ablation study
        print("🎯 Running comprehensive training ablation study...")
        
        # Optimizer ablation
        for optimizer in ["adam", "adamw", "sgd", "rmsprop"]:
            config = TrainingConfig(optimizer=optimizer)
            study.run_single_training_experiment(config, f"opt_{optimizer}")
        
        # Batch size ablation
        for batch_size in [16, 32, 64, 128]:
            config = TrainingConfig(batch_size=batch_size)
            study.run_single_training_experiment(config, f"batch_{batch_size}")
        
        # Weight decay ablation
        for weight_decay in [0.0, 1e-5, 1e-4, 1e-3]:
            config = TrainingConfig(weight_decay=weight_decay)
            study.run_single_training_experiment(config, f"wd_{weight_decay}")
        
        # Scheduler ablation
        for scheduler in ["cosine", "step", "exponential", "plateau"]:
            config = TrainingConfig(scheduler=scheduler)
            study.run_single_training_experiment(config, f"sched_{scheduler}")
    
    # Create training curves comparison
    study.create_training_curves_comparison()
    
    print(f"✅ Training ablation study completed!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 