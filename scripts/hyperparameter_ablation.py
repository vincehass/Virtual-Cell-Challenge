#!/usr/bin/env python3
"""
🔧 Hyperparameter Ablation Study - Authentic STATE Implementation
Systematic evaluation of different hyperparameters for optimal performance.
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
class HyperparameterConfig:
    """Configuration for hyperparameter ablation study."""
    learning_rate: float = 1e-4
    embed_dim: int = 512
    n_heads: int = 16
    n_layers: int = 12
    dropout: float = 0.1
    batch_size: int = 64
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'embed_dim': self.embed_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'warmup_steps': self.warmup_steps
        }

class HyperparameterAblationStudy:
    """
    Comprehensive hyperparameter ablation study for STATE models.
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
        self.best_config = None
        self.best_score = -float('inf')
        
    def load_data(self) -> Dict[str, Any]:
        """Load data for hyperparameter study."""
        print("🔬 Loading data for hyperparameter ablation...")
        
        data_loader = RealDataLoader()
        
        # Adjust data size for quick runs
        if self.quick_run:
            max_cells = 5000
            max_genes = 1000
        else:
            max_cells = self.base_config['data']['max_cells']
            max_genes = self.base_config['data']['max_genes']
        
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
    
    def run_single_experiment(self, hyperparams: HyperparameterConfig, 
                            data: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Run a single hyperparameter experiment."""
        print(f"🧪 Running experiment: {experiment_name}")
        print(f"   Hyperparameters: {hyperparams.to_dict()}")
        
        # Initialize W&B run
        if self.wandb_project:
            wandb.init(
                project=self.wandb_project,
                name=experiment_name,
                tags=self.wandb_tags + [experiment_name],
                config=hyperparams.to_dict(),
                reinit=True
            )
        
        try:
            # Create model configuration
            se_config = {
                'embed_dim': hyperparams.embed_dim,
                'n_heads': hyperparams.n_heads,
                'n_layers': hyperparams.n_layers,
                'dropout': hyperparams.dropout
            }
            
            st_config = {
                'state_dim': hyperparams.embed_dim // 2,
                'perturbation_dim': 128,
                'n_heads': max(hyperparams.n_heads // 2, 2),
                'n_layers': max(hyperparams.n_layers // 2, 2),
                'dropout': hyperparams.dropout
            }
            
            # Create and train model
            model = AuthenticSTATEModel(data['n_genes'], se_config, st_config)
            
            # Create custom trainer
            trainer = CustomHyperparameterTrainer(
                model=model,
                hyperparams=hyperparams,
                use_wandb=bool(self.wandb_project)
            )
            
            # Train model
            epochs = 50 if self.quick_run else 100
            training_results = trainer.train(data, epochs=epochs)
            
            # Evaluate model
            evaluation_results = self._evaluate_model(model, data, hyperparams)
            
            # Combine results
            results = {
                'hyperparameters': hyperparams.to_dict(),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'experiment_name': experiment_name
            }
            
            # Save results
            result_file = self.output_dir / f"{experiment_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Update best configuration
            overall_score = evaluation_results.get('overall_score', -float('inf'))
            if overall_score > self.best_score:
                self.best_score = overall_score
                self.best_config = hyperparams
            
            if self.wandb_project:
                wandb.log({
                    'final_overall_score': overall_score,
                    'final_train_loss': training_results.get('final_train_loss', float('inf')),
                    'final_val_loss': training_results.get('final_val_loss', float('inf'))
                })
                wandb.finish()
            
            print(f"✅ Experiment completed: {experiment_name}")
            print(f"   Overall Score: {overall_score:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment failed: {experiment_name}")
            print(f"   Error: {str(e)}")
            
            if self.wandb_project:
                wandb.finish()
            
            return {
                'hyperparameters': hyperparams.to_dict(),
                'experiment_name': experiment_name,
                'error': str(e)
            }
    
    def _evaluate_model(self, model: AuthenticSTATEModel, data: Dict[str, Any], 
                       hyperparams: HyperparameterConfig) -> Dict[str, Any]:
        """Evaluate trained model."""
        print("📊 Evaluating model...")
        
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
        
        # Perturbation discrimination
        pdisc_results = self.evaluator.perturbation_discrimination_with_density(
            predictions, eval_perturbed, 
            data['perturbation_data']['perturbed_cells'], 
            eval_labels, create_density_plots=False
        )
        
        # Differential expression
        de_results = self.evaluator.differential_expression_with_density(
            predictions, eval_perturbed,
            data['perturbation_data']['control_cells'],
            data['gene_names'], create_density_plots=False
        )
        
        # Calculate overall score
        pdisc_score = pdisc_results.get('overall_score', 0.0)
        de_correlation = de_results.get('correlations', {}).get('pearson', 0.0) if 'correlations' in de_results else 0.0
        de_f1 = de_results.get('differential_expression', {}).get('f1_score', 0.0) if 'differential_expression' in de_results else 0.0
        
        # Weighted overall score
        overall_score = 0.5 * pdisc_score + 0.3 * de_correlation + 0.2 * de_f1
        
        return {
            'perturbation_discrimination': pdisc_results,
            'differential_expression': de_results,
            'overall_score': overall_score,
            'pdisc_score': pdisc_score,
            'de_correlation': de_correlation,
            'de_f1': de_f1
        }
    
    def run_learning_rate_ablation(self, data: Dict[str, Any], 
                                 learning_rates: List[float]) -> Dict[str, Any]:
        """Run learning rate ablation study."""
        print("🎯 Running learning rate ablation...")
        
        results = {}
        base_hyperparams = HyperparameterConfig()
        
        for lr in learning_rates:
            hyperparams = HyperparameterConfig(
                learning_rate=lr,
                embed_dim=base_hyperparams.embed_dim,
                n_heads=base_hyperparams.n_heads,
                n_layers=base_hyperparams.n_layers
            )
            
            experiment_name = f"lr_{lr}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        return results
    
    def run_architecture_ablation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run architecture ablation study."""
        print("🏗️  Running architecture ablation...")
        
        results = {}
        
        # Embedding dimension ablation
        for embed_dim in [128, 256, 512, 1024]:
            hyperparams = HyperparameterConfig(
                embed_dim=embed_dim,
                n_heads=min(embed_dim // 32, 16)  # Adjust heads based on embedding dim
            )
            
            experiment_name = f"embed_{embed_dim}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        # Number of layers ablation
        for n_layers in [3, 6, 9, 12]:
            hyperparams = HyperparameterConfig(n_layers=n_layers)
            
            experiment_name = f"layers_{n_layers}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        # Number of heads ablation
        for n_heads in [4, 8, 16, 32]:
            hyperparams = HyperparameterConfig(n_heads=n_heads)
            
            experiment_name = f"heads_{n_heads}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        return results
    
    def run_regularization_ablation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run regularization ablation study."""
        print("🛡️  Running regularization ablation...")
        
        results = {}
        
        # Dropout ablation
        for dropout in [0.0, 0.1, 0.2, 0.3]:
            hyperparams = HyperparameterConfig(dropout=dropout)
            
            experiment_name = f"dropout_{dropout}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        # Weight decay ablation
        for weight_decay in [0.0, 1e-5, 1e-4, 1e-3]:
            hyperparams = HyperparameterConfig(weight_decay=weight_decay)
            
            experiment_name = f"wd_{weight_decay}"
            result = self.run_single_experiment(hyperparams, data, experiment_name)
            results[experiment_name] = result
        
        return results
    
    def create_analysis_report(self) -> None:
        """Create comprehensive analysis report."""
        print("📋 Creating hyperparameter analysis report...")
        
        # Collect all results
        all_results = []
        for result_file in self.output_dir.glob("*_results.json"):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    if 'error' not in result:
                        all_results.append(result)
            except Exception as e:
                print(f"⚠️  Failed to load {result_file}: {e}")
        
        if not all_results:
            print("❌ No valid results found for analysis")
            return
        
        # Create analysis
        df_results = pd.DataFrame([
            {
                'experiment': r['experiment_name'],
                'learning_rate': r['hyperparameters']['learning_rate'],
                'embed_dim': r['hyperparameters']['embed_dim'],
                'n_heads': r['hyperparameters']['n_heads'],
                'n_layers': r['hyperparameters']['n_layers'],
                'dropout': r['hyperparameters']['dropout'],
                'weight_decay': r['hyperparameters']['weight_decay'],
                'overall_score': r['evaluation_results']['overall_score'],
                'pdisc_score': r['evaluation_results']['pdisc_score'],
                'de_correlation': r['evaluation_results']['de_correlation'],
                'de_f1': r['evaluation_results']['de_f1'],
                'final_train_loss': r['training_results'].get('final_train_loss', np.nan),
                'final_val_loss': r['training_results'].get('final_val_loss', np.nan)
            } for r in all_results
        ])
        
        # Save results table
        df_results.to_csv(self.output_dir / "hyperparameter_results.csv", index=False)
        
        # Create visualizations
        self._create_hyperparameter_plots(df_results)
        
        # Generate report
        report_path = self.output_dir / "hyperparameter_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report(df_results))
        
        print(f"✅ Analysis report created: {report_path}")
    
    def _create_hyperparameter_plots(self, df: pd.DataFrame) -> None:
        """Create hyperparameter analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Overall score vs hyperparameters
        hyperparams = ['learning_rate', 'embed_dim', 'n_heads', 'n_layers', 'dropout', 'weight_decay']
        
        for i, param in enumerate(hyperparams):
            row, col = i // 3, i % 3
            
            # Group by hyperparameter and plot
            grouped = df.groupby(param)['overall_score'].agg(['mean', 'std']).reset_index()
            
            axes[row, col].errorbar(grouped[param], grouped['mean'], 
                                  yerr=grouped['std'], marker='o', capsize=5)
            axes[row, col].set_xlabel(param)
            axes[row, col].set_ylabel('Overall Score')
            axes[row, col].set_title(f'Overall Score vs {param}')
            axes[row, col].grid(True, alpha=0.3)
            
            if param == 'learning_rate':
                axes[row, col].set_xscale('log')
            elif param == 'weight_decay':
                axes[row, col].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_cols = ['learning_rate', 'embed_dim', 'n_heads', 'n_layers', 
                          'dropout', 'weight_decay', 'overall_score']
        corr_matrix = df[correlation_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Hyperparameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / "hyperparameter_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, df: pd.DataFrame) -> str:
        """Generate markdown analysis report."""
        best_result = df.loc[df['overall_score'].idxmax()]
        
        report = f"""# Hyperparameter Ablation Analysis Report

## Summary
- **Total Experiments**: {len(df)}
- **Best Overall Score**: {best_result['overall_score']:.4f}
- **Best Configuration**: {best_result['experiment']}

## Best Hyperparameters
- **Learning Rate**: {best_result['learning_rate']}
- **Embedding Dimension**: {best_result['embed_dim']}
- **Number of Heads**: {best_result['n_heads']}
- **Number of Layers**: {best_result['n_layers']}
- **Dropout**: {best_result['dropout']}
- **Weight Decay**: {best_result['weight_decay']}

## Performance Metrics
- **Perturbation Discrimination**: {best_result['pdisc_score']:.4f}
- **DE Correlation**: {best_result['de_correlation']:.4f}
- **DE F1 Score**: {best_result['de_f1']:.4f}

## Top 5 Configurations
{df.nlargest(5, 'overall_score')[['experiment', 'overall_score', 'learning_rate', 'embed_dim', 'n_heads', 'n_layers']].to_markdown(index=False)}

## Parameter Analysis

### Learning Rate
- **Range Tested**: {df['learning_rate'].min()} - {df['learning_rate'].max()}
- **Best Value**: {df.loc[df['overall_score'].idxmax(), 'learning_rate']}
- **Correlation with Performance**: {df['learning_rate'].corr(df['overall_score']):.3f}

### Architecture Parameters
- **Embedding Dimension Range**: {df['embed_dim'].min()} - {df['embed_dim'].max()}
- **Number of Layers Range**: {df['n_layers'].min()} - {df['n_layers'].max()}
- **Number of Heads Range**: {df['n_heads'].min()} - {df['n_heads'].max()}

### Regularization
- **Dropout Range**: {df['dropout'].min()} - {df['dropout'].max()}
- **Weight Decay Range**: {df['weight_decay'].min()} - {df['weight_decay'].max()}

## Recommendations
1. Use learning rate: {best_result['learning_rate']}
2. Optimal embedding dimension: {best_result['embed_dim']}
3. Recommended number of layers: {best_result['n_layers']}
4. Optimal dropout: {best_result['dropout']}

## Files Generated
- `hyperparameter_results.csv`: Complete results table
- `hyperparameter_analysis.png`: Performance vs hyperparameters plots
- `hyperparameter_correlation.png`: Correlation heatmap
"""
        return report

class CustomHyperparameterTrainer:
    """Custom trainer for hyperparameter experiments."""
    
    def __init__(self, model: AuthenticSTATEModel, hyperparams: HyperparameterConfig, use_wandb: bool = False):
        self.model = model
        self.hyperparams = hyperparams
        self.use_wandb = use_wandb
        
    def train(self, data: Dict[str, Any], epochs: int = 100) -> Dict[str, Any]:
        """Train model with specific hyperparameters."""
        # Prepare data loaders
        dataset = self._create_dataset(data)
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Create optimizer
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, epochs)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparams.batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.hyperparams.batch_size, 
                              shuffle=False, num_workers=2)
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """Create optimizer based on hyperparameters."""
        if self.hyperparams.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), 
                                  lr=self.hyperparams.learning_rate,
                                  weight_decay=self.hyperparams.weight_decay)
        elif self.hyperparams.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(),
                                   lr=self.hyperparams.learning_rate,
                                   weight_decay=self.hyperparams.weight_decay)
        elif self.hyperparams.optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(),
                                 lr=self.hyperparams.learning_rate,
                                 weight_decay=self.hyperparams.weight_decay,
                                 momentum=0.9)
        else:
            return torch.optim.AdamW(self.model.parameters(),
                                   lr=self.hyperparams.learning_rate,
                                   weight_decay=self.hyperparams.weight_decay)
    
    def _create_scheduler(self, optimizer, epochs):
        """Create learning rate scheduler."""
        if self.hyperparams.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.hyperparams.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)
        elif self.hyperparams.scheduler.lower() == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            return None
    
    def _train_epoch(self, train_loader, optimizer):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
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

def main():
    """Main function for hyperparameter ablation study."""
    parser = argparse.ArgumentParser(description="Hyperparameter Ablation Study for Authentic STATE")
    
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
    
    # Specific hyperparameter overrides
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize study
    study = HyperparameterAblationStudy(
        base_config_path=args.config,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        quick_run=args.quick_run
    )
    
    # Load data
    data = study.load_data()
    
    # If specific hyperparameters are provided, run single experiment
    if any([args.learning_rate, args.embed_dim, args.n_heads, args.n_layers, 
           args.dropout, args.weight_decay, args.batch_size, args.optimizer, args.scheduler]):
        
        # Create custom hyperparameter configuration
        hyperparams = HyperparameterConfig()
        
        if args.learning_rate is not None:
            hyperparams.learning_rate = args.learning_rate
        if args.embed_dim is not None:
            hyperparams.embed_dim = args.embed_dim
        if args.n_heads is not None:
            hyperparams.n_heads = args.n_heads
        if args.n_layers is not None:
            hyperparams.n_layers = args.n_layers
        if args.dropout is not None:
            hyperparams.dropout = args.dropout
        if args.weight_decay is not None:
            hyperparams.weight_decay = args.weight_decay
        if args.batch_size is not None:
            hyperparams.batch_size = args.batch_size
        if args.optimizer is not None:
            hyperparams.optimizer = args.optimizer
        if args.scheduler is not None:
            hyperparams.scheduler = args.scheduler
        
        # Create experiment name
        experiment_name = "_".join([
            f"lr_{hyperparams.learning_rate}",
            f"embed_{hyperparams.embed_dim}",
            f"heads_{hyperparams.n_heads}",
            f"layers_{hyperparams.n_layers}",
            f"dropout_{hyperparams.dropout}"
        ])
        
        # Run single experiment
        result = study.run_single_experiment(hyperparams, data, experiment_name)
        
    else:
        # Run comprehensive ablation study
        print("🔧 Running comprehensive hyperparameter ablation study...")
        
        # Learning rate ablation
        study.run_learning_rate_ablation(data, [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        
        # Architecture ablation
        study.run_architecture_ablation(data)
        
        # Regularization ablation
        study.run_regularization_ablation(data)
    
    # Create analysis report
    study.create_analysis_report()
    
    print(f"✅ Hyperparameter ablation study completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    if study.best_config:
        print(f"🏆 Best configuration found:")
        print(f"   Score: {study.best_score:.4f}")
        print(f"   Config: {study.best_config.to_dict()}")

if __name__ == "__main__":
    main() 