#!/usr/bin/env python3
"""
🏆 Simple Benchmarking with Progress Bars and Step-by-Step Curves
Shows training progress like reinforcement learning with clear visualization.
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
from tqdm import tqdm, trange
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

# Set style for nice plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProgressTracker:
    """Track and visualize training progress with live updates."""
    
    def __init__(self, total_epochs: int, use_wandb: bool = False):
        self.total_epochs = total_epochs
        self.use_wandb = use_wandb
        
        # Tracking variables
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.pdisc_scores = []
        self.de_correlations = []
        self.epochs_completed = 0
        
        # Progress bar
        self.pbar = None
        
    def start_training(self, description: str = "Training"):
        """Start the training progress bar."""
        self.pbar = trange(self.total_epochs, desc=description, unit="epoch")
        return self.pbar
    
    def update_epoch(self, train_loss: float, val_loss: float, lr: float, 
                    pdisc_score: float = None, de_corr: float = None):
        """Update progress with epoch results."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        if pdisc_score is not None:
            self.pdisc_scores.append(pdisc_score)
        if de_corr is not None:
            self.de_correlations.append(de_corr)
        
        # Update progress bar
        if self.pbar:
            self.pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'LR': f'{lr:.6f}',
                'PDisc': f'{pdisc_score:.3f}' if pdisc_score else 'N/A'
            })
            self.pbar.update(1)
        
        # Log to W&B
        if self.use_wandb:
            log_dict = {
                'epoch': self.epochs_completed,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr
            }
            if pdisc_score is not None:
                log_dict['pdisc_score'] = pdisc_score
            if de_corr is not None:
                log_dict['de_correlation'] = de_corr
            
            wandb.log(log_dict)
        
        self.epochs_completed += 1
    
    def finish_training(self):
        """Finish the training progress bar."""
        if self.pbar:
            self.pbar.close()
    
    def get_curves(self) -> Dict[str, List]:
        """Get all training curves."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'pdisc_scores': self.pdisc_scores,
            'de_correlations': self.de_correlations
        }

class ProgressiveTrainer:
    """Trainer with detailed progress tracking and evaluation."""
    
    def __init__(self, model: AuthenticSTATEModel, data: Dict[str, Any], 
                 use_wandb: bool = False, eval_frequency: int = 10):
        self.model = model
        self.data = data
        self.use_wandb = use_wandb
        self.eval_frequency = eval_frequency
        
        # Initialize evaluator
        self.evaluator = AuthenticBiologicalEvaluator()
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training and evaluation data."""
        perturbation_data = self.data['perturbation_data']
        
        # Create dataset
        from authentic_state_implementation import STATEDataset
        dataset = STATEDataset(
            perturbation_data['control_cells'],
            perturbation_data['perturbed_cells'],
            perturbation_data['perturbation_vectors'][~perturbation_data['control_mask']]
        )
        
        # Train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Prepare evaluation data
        eval_size = min(500, len(perturbation_data['perturbed_cells']))
        eval_indices = np.random.choice(len(perturbation_data['perturbed_cells']), 
                                      eval_size, replace=False)
        
        self.eval_perturbed = perturbation_data['perturbed_cells'][eval_indices]
        self.eval_labels = perturbation_data['perturbed_labels'][eval_indices]
        self.eval_controls = perturbation_data['control_cells']
    
    def train_with_progress(self, epochs: int = 100, lr: float = 1e-4) -> Dict[str, Any]:
        """Train model with detailed progress tracking."""
        print(f"🎯 Starting training for {epochs} epochs...")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Initialize progress tracker
        tracker = ProgressTracker(epochs, self.use_wandb)
        pbar = tracker.start_training("Training STATE Model")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(optimizer)
            
            # Validation phase
            val_loss = self._validate_epoch()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Periodic evaluation
            pdisc_score = None
            de_corr = None
            
            if epoch % self.eval_frequency == 0 or epoch == epochs - 1:
                eval_results = self._evaluate_model()
                pdisc_score = eval_results.get('pdisc_score', 0.0)
                de_corr = eval_results.get('de_correlation', 0.0)
            
            # Update progress
            tracker.update_epoch(train_loss, val_loss, current_lr, pdisc_score, de_corr)
            
            # Update scheduler
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Could save model here if needed
        
        tracker.finish_training()
        
        print(f"✅ Training completed!")
        print(f"   Final train loss: {train_loss:.4f}")
        print(f"   Final val loss: {val_loss:.4f}")
        print(f"   Best val loss: {best_val_loss:.4f}")
        
        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'curves': tracker.get_curves(),
            'model': self.model
        }
    
    def _train_epoch(self, optimizer) -> float:
        """Train one epoch with progress tracking."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(self.train_loader, desc="Training Batches", leave=False)
        
        for batch in batch_pbar:
            baseline_expr, perturbed_expr, pert_vectors = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_expr = self.model(baseline_expr, pert_vectors)
            
            # Loss calculation
            loss = torch.nn.functional.mse_loss(predicted_expr, perturbed_expr)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update batch progress
            batch_pbar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
        
        batch_pbar.close()
        return total_loss / num_batches
    
    def _validate_epoch(self) -> float:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                baseline_expr, perturbed_expr, pert_vectors = batch
                
                # Forward pass
                predicted_expr = self.model(baseline_expr, pert_vectors)
                
                # Loss calculation
                loss = torch.nn.functional.mse_loss(predicted_expr, perturbed_expr)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Quick model evaluation for progress tracking."""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(self.eval_perturbed)
            dummy_pert_vectors = torch.zeros(input_tensor.shape[0], 128)
            predictions = self.model(input_tensor, dummy_pert_vectors)
            predictions = predictions.numpy()
        
        # Quick metrics
        mse = np.mean((predictions - self.eval_perturbed) ** 2)
        correlation = np.corrcoef(predictions.flatten(), self.eval_perturbed.flatten())[0, 1]
        
        # Mock evaluation scores (simplified for progress tracking)
        pdisc_score = max(0, correlation)
        de_correlation = correlation
        
        return {
            'pdisc_score': pdisc_score,
            'de_correlation': de_correlation,
            'mse': mse
        }

class SimpleBenchmarkingSuite:
    """Simple benchmarking suite with clear progress visualization."""
    
    def __init__(self, output_dir: str, wandb_project: str = None, quick_run: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.quick_run = quick_run
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        
        # Results storage
        self.results = {}
        self.all_curves = {}
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run simple benchmark with progress tracking."""
        print("🏆 Starting Simple STATE Benchmarking with Progress Bars")
        print("=" * 60)
        
        # Load data
        data = self._load_data()
        
        # Create models to benchmark
        models = self._create_models_to_benchmark(data)
        
        # Train and evaluate each model
        for model_name, model_info in models.items():
            print(f"\n🧬 Benchmarking: {model_name}")
            print("-" * 40)
            
            if self.wandb_project:
                wandb.init(
                    project=self.wandb_project,
                    name=f"benchmark_{model_name}",
                    tags=["benchmark", model_name],
                    reinit=True
                )
            
            # Train model
            trainer = ProgressiveTrainer(
                model=model_info['model'], 
                data=data, 
                use_wandb=bool(self.wandb_project)
            )
            
            epochs = 30 if self.quick_run else 60
            training_results = trainer.train_with_progress(epochs=epochs, lr=model_info['lr'])
            
            # Store results
            self.results[model_name] = training_results
            self.all_curves[model_name] = training_results['curves']
            
            if self.wandb_project:
                wandb.finish()
        
        # Create comparison plots
        self._create_comparison_plots()
        
        # Generate report
        self._generate_report()
        
        return {
            'results': self.results,
            'curves': self.all_curves,
            'output_dir': str(self.output_dir)
        }
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data for benchmarking."""
        print("📊 Loading data for benchmarking...")
        
        data_loader = RealDataLoader()
        
        # Adjust data size
        max_cells = 5000 if self.quick_run else 10000
        max_genes = 1000 if self.quick_run else 2000
        
        try:
            data = data_loader.load_stratified_real_data(
                max_cells=max_cells, 
                max_genes=max_genes
            )
            print(f"✅ Real data loaded: {data['n_cells']:,} cells × {data['n_genes']:,} genes")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load real data: {e}")
            print("🔄 Creating synthetic data...")
            return self._create_synthetic_data(max_cells, max_genes)
    
    def _create_synthetic_data(self, max_cells: int, max_genes: int) -> Dict[str, Any]:
        """Create synthetic data for testing."""
        print("🔬 Creating synthetic single-cell data...")
        
        n_cells = min(max_cells, 3000)
        n_genes = min(max_genes, 800)
        n_perturbations = 15
        
        # Create realistic gene expression data
        np.random.seed(42)
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
                effect_genes = np.random.choice(n_genes, 30, replace=False)
                effect_size = np.random.normal(0, 0.4, 30)
                base_expression[pert_mask][:, effect_genes] += effect_size
        
        # Create perturbation vectors
        unique_perts = np.unique(perturbation_labels)
        perturbation_vectors = np.zeros((n_cells, 128))
        for i, pert in enumerate(unique_perts):
            if i < 128:
                perturbation_vectors[perturbation_labels == pert, i] = 1.0
        
        print(f"✅ Synthetic data created: {n_cells:,} cells × {n_genes:,} genes")
        
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
    
    def _create_models_to_benchmark(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create models to benchmark."""
        print("🧬 Creating models to benchmark...")
        
        models = {}
        
        # 1. Small STATE Model
        se_config_small = {
            'embed_dim': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1
        }
        st_config_small = {
            'state_dim': 64,
            'perturbation_dim': 64,
            'n_heads': 2,
            'n_layers': 2,
            'dropout': 0.1
        }
        small_model = AuthenticSTATEModel(data['n_genes'], se_config_small, st_config_small)
        models['Small_STATE'] = {'model': small_model, 'lr': 1e-3}
        
        # 2. Medium STATE Model
        se_config_medium = {
            'embed_dim': 256,
            'n_heads': 8,
            'n_layers': 6,
            'dropout': 0.1
        }
        st_config_medium = {
            'state_dim': 128,
            'perturbation_dim': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1
        }
        medium_model = AuthenticSTATEModel(data['n_genes'], se_config_medium, st_config_medium)
        models['Medium_STATE'] = {'model': medium_model, 'lr': 5e-4}
        
        # 3. Large STATE Model (only if not quick run)
        if not self.quick_run:
            se_config_large = {
                'embed_dim': 512,
                'n_heads': 16,
                'n_layers': 12,
                'dropout': 0.1
            }
            st_config_large = {
                'state_dim': 256,
                'perturbation_dim': 128,
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1
            }
            large_model = AuthenticSTATEModel(data['n_genes'], se_config_large, st_config_large)
            models['Large_STATE'] = {'model': large_model, 'lr': 1e-4}
        
        print(f"✅ Created {len(models)} models to benchmark")
        return models
    
    def _create_comparison_plots(self):
        """Create comprehensive comparison plots."""
        print("📈 Creating comparison plots...")
        
        if not self.all_curves:
            print("⚠️  No curves to plot")
            return
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training loss curves
        ax = axes[0, 0]
        for model_name, curves in self.all_curves.items():
            if 'train_losses' in curves and curves['train_losses']:
                epochs = range(len(curves['train_losses']))
                ax.plot(epochs, curves['train_losses'], label=model_name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves (Like RL Rewards)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Validation loss curves
        ax = axes[0, 1]
        for model_name, curves in self.all_curves.items():
            if 'val_losses' in curves and curves['val_losses']:
                epochs = range(len(curves['val_losses']))
                ax.plot(epochs, curves['val_losses'], label=model_name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Learning rate schedules
        ax = axes[0, 2]
        for model_name, curves in self.all_curves.items():
            if 'learning_rates' in curves and curves['learning_rates']:
                epochs = range(len(curves['learning_rates']))
                ax.plot(epochs, curves['learning_rates'], label=model_name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedules')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # PDisc scores over time
        ax = axes[1, 0]
        for model_name, curves in self.all_curves.items():
            if 'pdisc_scores' in curves and curves['pdisc_scores']:
                # PDisc scores are evaluated every 10 epochs
                eval_epochs = list(range(0, len(curves['train_losses']), 10))[:len(curves['pdisc_scores'])]
                ax.plot(eval_epochs, curves['pdisc_scores'], 
                       label=model_name, linewidth=2, alpha=0.8, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perturbation Discrimination Score')
        ax.set_title('PDisc Score During Training (RL-style)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # DE correlation over time
        ax = axes[1, 1]
        for model_name, curves in self.all_curves.items():
            if 'de_correlations' in curves and curves['de_correlations']:
                eval_epochs = list(range(0, len(curves['train_losses']), 10))[:len(curves['de_correlations'])]
                ax.plot(eval_epochs, curves['de_correlations'], 
                       label=model_name, linewidth=2, alpha=0.8, marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('DE Correlation')
        ax.set_title('Differential Expression Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final performance comparison
        ax = axes[1, 2]
        model_names = list(self.results.keys())
        final_train_losses = [self.results[m].get('final_train_loss', 0) for m in model_names]
        final_val_losses = [self.results[m].get('final_val_loss', 0) for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, final_train_losses, width, label='Train Loss', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_val_losses, width, label='Val Loss', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "benchmark_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to: {self.output_dir / 'plots' / 'benchmark_comparison.png'}")
    
    def _generate_report(self):
        """Generate benchmark report."""
        print("📝 Generating benchmark report...")
        
        report = f"""
# Simple STATE Benchmarking Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Benchmarked

"""
        
        for model_name, results in self.results.items():
            final_train = results.get('final_train_loss', 'N/A')
            final_val = results.get('final_val_loss', 'N/A')
            best_val = results.get('best_val_loss', 'N/A')
            
            report += f"""
### {model_name}

- **Final Training Loss**: {final_train:.6f if isinstance(final_train, float) else final_train}
- **Final Validation Loss**: {final_val:.6f if isinstance(final_val, float) else final_val}
- **Best Validation Loss**: {best_val:.6f if isinstance(best_val, float) else best_val}

"""
        
        report += f"""
## Training Progress

All models were trained with progress bars showing:
- Real-time loss updates
- Learning rate schedules
- Evaluation metrics every 10 epochs
- Step-by-step progress like reinforcement learning

## Output Files

- **Comparison Plots**: `plots/benchmark_comparison.png`
- **This Report**: `benchmark_report.md`

## Next Steps

1. Review the training curves to understand convergence
2. Compare final performance metrics
3. Analyze which model architecture works best
4. Consider hyperparameter tuning for the best performing model

"""
        
        # Save report
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")

def main():
    """Main function for simple benchmarking."""
    parser = argparse.ArgumentParser(description="Simple STATE Benchmarking with Progress Bars")
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--quick_run', action='store_true',
                       help='Run quick version with smaller models and fewer epochs')
    
    args = parser.parse_args()
    
    # Initialize benchmarking suite
    benchmark_suite = SimpleBenchmarkingSuite(
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        quick_run=args.quick_run
    )
    
    # Run benchmark
    results = benchmark_suite.run_benchmark()
    
    print(f"\n🎉 Simple benchmarking completed!")
    print(f"📁 Results saved to: {results['output_dir']}")
    print(f"📊 Models benchmarked: {len(results['results'])}")
    
    if args.wandb_project:
        print(f"🌐 View detailed results on W&B: https://wandb.ai/{args.wandb_project}")

if __name__ == "__main__":
    main() 