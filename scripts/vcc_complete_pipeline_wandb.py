#!/usr/bin/env python3
"""
🧬 COMPLETE Virtual Cell Challenge Pipeline with W&B Logging

This script provides a comprehensive step-by-step analysis and preparation 
of the VCC dataset with full experiment tracking in Weights & Biases.

Steps:
1. Data Loading & Validation
2. Dataset Analysis & QC
3. Data Preprocessing
4. Embedding Computation  
5. Training Split Creation
6. Model Preparation
7. Evaluation Setup
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# W&B for experiment tracking
import wandb

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['WANDB_PROJECT'] = 'virtual-cell-challenge'

# Import our Virtual Cell Challenge toolkit
import sys
sys.path.append('src')

from virtual_cell_challenge.preprocessing import (
    preprocess_perturbation_data, 
    analyze_perturbation_quality,
    validate_data_format
)
from virtual_cell_challenge.analysis import analyze_dataset
from virtual_cell_challenge.evaluation import CellEvalMetrics
from virtual_cell_challenge.data_loading import PerturbationDataset

def setup_wandb_experiment():
    """Initialize W&B experiment with comprehensive config."""
    
    config = {
        # Dataset configuration
        "dataset_name": "VCC_Training_Real",
        "dataset_path": "data/vcc_data/adata_Training.h5ad", 
        "dataset_size_gb": 14,
        "expected_cells": 221273,
        "expected_genes": 18080,
        
        # Pipeline configuration
        "pipeline_version": "1.0.0",
        "perturbation_column": "target_gene",
        "batch_column": "batch", 
        "control_label": "non-targeting",
        "cell_type": "primary_cells",  # Single cell type experiment
        
        # Processing configuration
        "normalize_total": 1e4,
        "log_transform": True,
        "hvg_n_top_genes": 2000,
        "pca_n_components": 50,
        "umap_n_neighbors": 15,
        "umap_n_components": 2,
        
        # Training configuration
        "val_perturbations": ["TMSB4X", "PRCP", "TADA1"],
        "test_perturbations": ["HIRA", "IGF2R", "NCK2", "MED13", "MED12"],
        "random_seed": 42,
        
        # Hardware
        "compute_environment": "CPU",
        "python_version": "3.9",
    }
    
    # Initialize W&B
    run = wandb.init(
        project="virtual-cell-challenge",
        name="vcc-complete-pipeline",
        notes="Complete VCC data analysis and model preparation pipeline",
        tags=["data-analysis", "preprocessing", "real-data", "vcc"],
        config=config
    )
    
    return run, config

def log_step_start(step_name, step_description):
    """Log the start of a pipeline step."""
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"📝 {step_description}")
    print(f"{'='*60}")
    
    wandb.log({
        f"step_{step_name.lower().replace(' ', '_')}_started": True,
        "current_step": step_name,
    })
    
    return time.time()

def log_step_complete(step_name, start_time, metrics=None):
    """Log the completion of a pipeline step."""
    duration = time.time() - start_time
    
    print(f"✅ {step_name} completed in {duration:.2f}s")
    
    log_data = {
        f"step_{step_name.lower().replace(' ', '_')}_duration": duration,
        f"step_{step_name.lower().replace(' ', '_')}_completed": True,
    }
    
    if metrics:
        for key, value in metrics.items():
            log_data[f"{step_name.lower().replace(' ', '_')}_{key}"] = value
    
    wandb.log(log_data)

def step1_data_loading_validation(config):
    """Step 1: Load and validate the VCC dataset."""
    start_time = log_step_start("Data Loading", "Loading and validating the complete VCC dataset")
    
    # Load dataset
    print("📊 Loading VCC training dataset...")
    load_start = time.time()
    adata = ad.read_h5ad(config["dataset_path"])
    load_time = time.time() - load_start
    
    print(f"✅ Dataset loaded in {load_time:.2f}s")
    print(f"📏 Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Validate dataset structure
    print("🔍 Validating dataset structure...")
    
    # Check required columns
    required_obs_columns = [config["perturbation_column"], config["batch_column"], "guide_id"]
    missing_columns = [col for col in required_obs_columns if col not in adata.obs.columns]
    
    validation_results = {
        "cells_count": int(adata.n_obs),
        "genes_count": int(adata.n_vars),
        "load_time_seconds": load_time,
        "missing_columns": missing_columns,
        "has_required_columns": len(missing_columns) == 0,
        "sparsity": float(1.0 - (adata.X > 0).sum() / (adata.n_obs * adata.n_vars)),
        "memory_usage_gb": float(adata.X.data.nbytes / (1024**3)) if hasattr(adata.X, 'data') else 0,
    }
    
    # Log basic statistics
    perturbation_counts = adata.obs[config["perturbation_column"]].value_counts()
    batch_counts = adata.obs[config["batch_column"]].value_counts()
    
    validation_results.update({
        "n_perturbations": len(perturbation_counts),
        "n_batches": len(batch_counts),
        "control_cells": int(perturbation_counts.get(config["control_label"], 0)),
        "treatment_cells": int(perturbation_counts.drop(config["control_label"], errors='ignore').sum()),
        "cells_per_perturbation_mean": float(perturbation_counts.mean()),
        "cells_per_perturbation_std": float(perturbation_counts.std()),
        "cells_per_batch_mean": float(batch_counts.mean()),
        "cells_per_batch_std": float(batch_counts.std()),
    })
    
    # Create and log perturbation distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top perturbations
    top_perts = perturbation_counts.head(20)
    ax1.barh(range(len(top_perts)), top_perts.values)
    ax1.set_yticks(range(len(top_perts)))
    ax1.set_yticklabels(top_perts.index, fontsize=8)
    ax1.set_xlabel('Cell Count')
    ax1.set_title('Top 20 Perturbations')
    
    # Batch distribution
    ax2.hist(batch_counts.values, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Cells per Batch')
    ax2.set_ylabel('Number of Batches')
    ax2.set_title('Batch Distribution')
    
    plt.tight_layout()
    wandb.log({"data_validation/perturbation_distribution": wandb.Image(fig)})
    plt.close()
    
    log_step_complete("Data Loading", start_time, validation_results)
    return adata, validation_results

def step2_dataset_analysis(adata, config):
    """Step 2: Comprehensive dataset analysis."""
    start_time = log_step_start("Dataset Analysis", "Performing comprehensive analysis of dataset characteristics")
    
    # Basic analysis
    print("📈 Computing dataset statistics...")
    
    # Perturbation analysis
    perturbations = adata.obs[config["perturbation_column"]]
    pert_counts = perturbations.value_counts()
    
    # Expression analysis
    if hasattr(adata.X, 'toarray'):
        expr_data = adata.X.toarray()
    else:
        expr_data = adata.X
        
    # Sample for memory efficiency
    sample_size = min(10000, adata.n_obs)
    sample_indices = np.random.choice(adata.n_obs, size=sample_size, replace=False)
    expr_sample = expr_data[sample_indices]
    
    analysis_metrics = {
        "mean_umi_per_cell": float(np.mean(expr_sample.sum(axis=1))),
        "median_umi_per_cell": float(np.median(expr_sample.sum(axis=1))),
        "mean_genes_per_cell": float(np.mean((expr_sample > 0).sum(axis=1))),
        "median_genes_per_cell": float(np.median((expr_sample > 0).sum(axis=1))),
        "mean_expression_level": float(np.mean(expr_sample[expr_sample > 0])),
        "median_expression_level": float(np.median(expr_sample[expr_sample > 0])),
        "zero_fraction": float(np.mean(expr_sample == 0)),
    }
    
    # Perturbation diversity
    perturbation_metrics = {
        "gini_coefficient_perturbations": float(calculate_gini_coefficient(pert_counts.values)),
        "perturbation_entropy": float(calculate_entropy(pert_counts.values)),
        "min_cells_per_perturbation": int(pert_counts.min()),
        "max_cells_per_perturbation": int(pert_counts.max()),
    }
    
    analysis_metrics.update(perturbation_metrics)
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VCC Dataset Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # UMI distribution
    cell_totals = expr_sample.sum(axis=1)
    axes[0, 0].hist(cell_totals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Total UMI Count')
    axes[0, 0].set_ylabel('Number of Cells')
    axes[0, 0].set_title('UMI Count Distribution')
    axes[0, 0].axvline(analysis_metrics["mean_umi_per_cell"], color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Gene detection
    gene_detection = (expr_sample > 0).sum(axis=1)
    axes[0, 1].hist(gene_detection, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Genes Detected')
    axes[0, 1].set_ylabel('Number of Cells')
    axes[0, 1].set_title('Gene Detection Distribution')
    axes[0, 1].axvline(analysis_metrics["mean_genes_per_cell"], color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # UMI vs Gene detection
    axes[0, 2].scatter(cell_totals, gene_detection, alpha=0.6, s=1)
    axes[0, 2].set_xlabel('Total UMI Count')
    axes[0, 2].set_ylabel('Genes Detected')
    axes[0, 2].set_title('UMI vs Gene Detection')
    
    # Perturbation size distribution
    axes[1, 0].hist(pert_counts.values, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Cells per Perturbation')
    axes[1, 0].set_ylabel('Number of Perturbations')
    axes[1, 0].set_title('Perturbation Size Distribution')
    axes[1, 0].axvline(pert_counts.mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    # Expression level distribution
    nonzero_expr = expr_sample[expr_sample > 0]
    axes[1, 1].hist(nonzero_expr, bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[1, 1].set_xlabel('Expression Level (log counts)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Expression Level Distribution')
    axes[1, 1].set_yscale('log')
    
    # Batch effects
    batch_counts = adata.obs[config["batch_column"]].value_counts()
    axes[1, 2].hist(batch_counts.values, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Cells per Batch')
    axes[1, 2].set_ylabel('Number of Batches')
    axes[1, 2].set_title('Batch Size Distribution')
    
    plt.tight_layout()
    wandb.log({"dataset_analysis/comprehensive_plots": wandb.Image(fig)})
    plt.close()
    
    log_step_complete("Dataset Analysis", start_time, analysis_metrics)
    return analysis_metrics

def step3_data_preprocessing(adata, config):
    """Step 3: Data preprocessing and quality control."""
    start_time = log_step_start("Data Preprocessing", "Preprocessing data for downstream analysis")
    
    # Create working copy
    adata_processed = adata.copy()
    
    print("🔧 Adapting data format...")
    # Adapt column names for our pipeline
    adata_processed.obs['gene'] = adata_processed.obs[config["perturbation_column"]]
    adata_processed.obs['gem_group'] = adata_processed.obs[config["batch_column"]]
    adata_processed.obs['cell_type'] = config["cell_type"]
    adata_processed.var['gene_name'] = adata_processed.var['gene_id']
    
    print("📊 Computing basic statistics...")
    # Basic QC metrics
    adata_processed.var['n_cells'] = (adata_processed.X > 0).sum(axis=0).A1 if hasattr(adata_processed.X, 'A1') else (adata_processed.X > 0).sum(axis=0)
    adata_processed.obs['n_genes'] = (adata_processed.X > 0).sum(axis=1).A1 if hasattr(adata_processed.X, 'A1') else (adata_processed.X > 0).sum(axis=1)
    adata_processed.obs['total_counts'] = adata_processed.X.sum(axis=1).A1 if hasattr(adata_processed.X, 'A1') else adata_processed.X.sum(axis=1)
    
    preprocessing_metrics = {
        "cells_before_qc": int(adata_processed.n_obs),
        "genes_before_qc": int(adata_processed.n_vars),
        "mean_genes_per_cell_before": float(adata_processed.obs['n_genes'].mean()),
        "mean_counts_per_cell_before": float(adata_processed.obs['total_counts'].mean()),
    }
    
    # Quality control filtering
    print("🔍 Applying quality control filters...")
    
    # Filter cells with too few genes
    min_genes = 200
    cell_filter = adata_processed.obs['n_genes'] >= min_genes
    print(f"Filtering cells with < {min_genes} genes: {(~cell_filter).sum()} cells removed")
    
    # Filter genes expressed in too few cells
    min_cells = 10
    gene_filter = adata_processed.var['n_cells'] >= min_cells
    print(f"Filtering genes in < {min_cells} cells: {(~gene_filter).sum()} genes removed")
    
    # Apply filters
    adata_processed = adata_processed[cell_filter, gene_filter].copy()
    
    preprocessing_metrics.update({
        "cells_after_qc": int(adata_processed.n_obs),
        "genes_after_qc": int(adata_processed.n_vars),
        "cells_filtered": int((~cell_filter).sum()),
        "genes_filtered": int((~gene_filter).sum()),
    })
    
    print("✅ Quality control filtering complete")
    print(f"   Cells: {preprocessing_metrics['cells_before_qc']:,} → {preprocessing_metrics['cells_after_qc']:,}")
    print(f"   Genes: {preprocessing_metrics['genes_before_qc']:,} → {preprocessing_metrics['genes_after_qc']:,}")
    
    # Validate data format for our pipeline
    print("🔍 Validating processed data format...")
    try:
        validation = validate_data_format(adata_processed)
        preprocessing_metrics.update({
            "validation_passed": validation["is_valid"],
            "validation_warnings": len(validation.get("warnings", [])),
        })
        
        if validation["warnings"]:
            print("⚠️ Validation warnings:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")
        preprocessing_metrics["validation_error"] = str(e)
    
    log_step_complete("Data Preprocessing", start_time, preprocessing_metrics)
    return adata_processed, preprocessing_metrics

def step4_embedding_computation(adata_processed, config):
    """Step 4: Compute embeddings for analysis and modeling."""
    start_time = log_step_start("Embedding Computation", "Computing PCA, HVG, and UMAP embeddings")
    
    import scanpy as sc
    sc.settings.verbosity = 1
    
    embedding_metrics = {}
    
    print("🧮 Step 4.1: Normalization and log transformation...")
    norm_start = time.time()
    
    # Normalize to 10,000 reads per cell
    sc.pp.normalize_total(adata_processed, target_sum=config["normalize_total"])
    
    # Log transform
    sc.pp.log1p(adata_processed)
    
    norm_time = time.time() - norm_start
    embedding_metrics["normalization_time"] = norm_time
    print(f"   ✅ Normalization completed in {norm_time:.2f}s")
    
    print("🧮 Step 4.2: Highly variable gene selection...")
    hvg_start = time.time()
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(
        adata_processed, 
        n_top_genes=config["hvg_n_top_genes"],
        min_mean=0.0125, 
        max_mean=3, 
        min_disp=0.5
    )
    
    hvg_time = time.time() - hvg_start
    n_hvg = adata_processed.var['highly_variable'].sum()
    embedding_metrics.update({
        "hvg_selection_time": hvg_time,
        "n_highly_variable_genes": int(n_hvg),
        "hvg_fraction": float(n_hvg / adata_processed.n_vars),
    })
    print(f"   ✅ Selected {n_hvg:,} highly variable genes in {hvg_time:.2f}s")
    
    print("🧮 Step 4.3: PCA computation...")
    pca_start = time.time()
    
    # Compute PCA
    sc.pp.pca(adata_processed, n_comps=config["pca_n_components"], use_highly_variable=True)
    
    pca_time = time.time() - pca_start
    embedding_metrics.update({
        "pca_computation_time": pca_time,
        "pca_components": config["pca_n_components"],
        "pca_variance_ratio": float(adata_processed.uns['pca']['variance_ratio'].sum()),
    })
    print(f"   ✅ PCA completed in {pca_time:.2f}s")
    print(f"   📊 Explained variance: {embedding_metrics['pca_variance_ratio']:.3f}")
    
    print("🧮 Step 4.4: Creating HVG embedding...")
    hvg_start = time.time()
    
    # Create HVG embedding for model training
    hvg_genes = adata_processed.var['highly_variable']
    X_hvg = adata_processed.X[:, hvg_genes]
    if hasattr(X_hvg, 'toarray'):
        X_hvg = X_hvg.toarray()
    adata_processed.obsm['X_hvg'] = X_hvg
    
    hvg_embed_time = time.time() - hvg_start
    embedding_metrics.update({
        "hvg_embedding_time": hvg_embed_time,
        "hvg_embedding_shape": X_hvg.shape,
    })
    print(f"   ✅ HVG embedding created: {X_hvg.shape} in {hvg_embed_time:.2f}s")
    
    print("🧮 Step 4.5: UMAP computation...")
    umap_start = time.time()
    
    # Compute neighbors for UMAP
    sc.pp.neighbors(adata_processed, n_neighbors=config["umap_n_neighbors"], n_pcs=40)
    
    # Compute UMAP  
    sc.tl.umap(adata_processed, n_components=config["umap_n_components"])
    
    umap_time = time.time() - umap_start
    embedding_metrics.update({
        "umap_computation_time": umap_time,
        "umap_components": config["umap_n_components"],
        "umap_neighbors": config["umap_n_neighbors"],
    })
    print(f"   ✅ UMAP completed in {umap_time:.2f}s")
    
    # Create embedding visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA plot
    pca_coords = adata_processed.obsm['X_pca']
    scatter1 = axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1], 
                              c=adata_processed.obs['gem_group'].astype('category').cat.codes,
                              alpha=0.6, s=0.5, rasterized=True)
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].set_title('PCA Embedding (colored by batch)')
    
    # UMAP plot
    umap_coords = adata_processed.obsm['X_umap']
    scatter2 = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                              c=adata_processed.obs['gem_group'].astype('category').cat.codes,
                              alpha=0.6, s=0.5, rasterized=True)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title('UMAP Embedding (colored by batch)')
    
    # PCA variance explained
    variance_ratio = adata_processed.uns['pca']['variance_ratio']
    axes[2].plot(range(1, len(variance_ratio) + 1), np.cumsum(variance_ratio), 'bo-')
    axes[2].set_xlabel('Principal Component')
    axes[2].set_ylabel('Cumulative Variance Explained')
    axes[2].set_title('PCA Variance Explained')
    axes[2].grid(True)
    
    plt.tight_layout()
    wandb.log({"embeddings/pca_umap_analysis": wandb.Image(fig)})
    plt.close()
    
    log_step_complete("Embedding Computation", start_time, embedding_metrics)
    return adata_processed, embedding_metrics

def step5_training_splits(adata_processed, config):
    """Step 5: Create training, validation, and test splits."""
    start_time = log_step_start("Training Splits", "Creating data splits for model training and evaluation")
    
    print("🎯 Creating perturbation-based splits...")
    
    # Get perturbation counts
    perturbations = adata_processed.obs['gene']
    pert_counts = perturbations.value_counts()
    
    # Define splits
    val_perts = config["val_perturbations"]
    test_perts = config["test_perturbations"]
    control_pert = config["control_label"]
    
    # Create split masks
    train_mask = ~perturbations.isin(val_perts + test_perts)
    val_mask = perturbations.isin(val_perts)
    test_mask = perturbations.isin(test_perts)
    
    split_metrics = {
        "train_cells": int(train_mask.sum()),
        "val_cells": int(val_mask.sum()),
        "test_cells": int(test_mask.sum()),
        "train_perturbations": len(perturbations[train_mask].unique()),
        "val_perturbations": len(val_perts),
        "test_perturbations": len(test_perts),
        "control_cells_train": int((perturbations == control_pert)[train_mask].sum()),
        "control_cells_val": int((perturbations == control_pert)[val_mask].sum()),
        "control_cells_test": int((perturbations == control_pert)[test_mask].sum()),
    }
    
    print(f"📊 Split Summary:")
    print(f"   Train: {split_metrics['train_cells']:,} cells, {split_metrics['train_perturbations']} perturbations")
    print(f"   Val:   {split_metrics['val_cells']:,} cells, {split_metrics['val_perturbations']} perturbations")
    print(f"   Test:  {split_metrics['test_cells']:,} cells, {split_metrics['test_perturbations']} perturbations")
    
    # Create split datasets
    adata_train = adata_processed[train_mask].copy()
    adata_val = adata_processed[val_mask].copy()
    adata_test = adata_processed[test_mask].copy()
    
    # Save split information
    split_info = {
        "train_perturbations": list(perturbations[train_mask].unique()),
        "val_perturbations": val_perts,
        "test_perturbations": test_perts,
        "split_strategy": "perturbation_based",
        "control_perturbation": control_pert,
    }
    
    # Create split visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Split size comparison
    splits = ['Train', 'Val', 'Test']
    cell_counts = [split_metrics['train_cells'], split_metrics['val_cells'], split_metrics['test_cells']]
    colors = ['blue', 'orange', 'green']
    
    ax1.bar(splits, cell_counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Data Split Sizes')
    for i, count in enumerate(cell_counts):
        ax1.text(i, count + 1000, f'{count:,}', ha='center', va='bottom')
    
    # Perturbation counts in each split
    pert_counts_splits = [split_metrics['train_perturbations'], 
                         split_metrics['val_perturbations'], 
                         split_metrics['test_perturbations']]
    
    ax2.bar(splits, pert_counts_splits, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Perturbations')
    ax2.set_title('Perturbations per Split')
    for i, count in enumerate(pert_counts_splits):
        ax2.text(i, count + 0.5, f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    wandb.log({"training_splits/split_summary": wandb.Image(fig)})
    plt.close()
    
    # Save processed datasets
    print("💾 Saving split datasets...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    save_start = time.time()
    adata_train.write(output_dir / "vcc_train.h5ad")
    adata_val.write(output_dir / "vcc_val.h5ad")
    adata_test.write(output_dir / "vcc_test.h5ad")
    adata_processed.write(output_dir / "vcc_complete_processed.h5ad")
    
    # Save split info as JSON
    import json
    with open(output_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    save_time = time.time() - save_start
    split_metrics["save_time"] = save_time
    
    print(f"✅ Datasets saved in {save_time:.2f}s")
    
    log_step_complete("Training Splits", start_time, split_metrics)
    return adata_train, adata_val, adata_test, split_metrics

def step6_model_preparation(adata_train, config):
    """Step 6: Prepare data loaders and model components."""
    start_time = log_step_start("Model Preparation", "Setting up data loaders and model preparation")
    
    print("🏗️ Testing data loading pipeline...")
    
    # Test our PerturbationDataset
    try:
        dataset = PerturbationDataset(
            dataset_path="data/processed/vcc_train.h5ad",
            embed_key="X_hvg",
            pert_col="gene",
            cell_type_key="cell_type",
            batch_col="gem_group",
            control_pert=config["control_label"],
            output_space="embedding"
        )
        
        print(f"✅ Dataset loaded: {len(dataset):,} cells")
        
        # Test data access
        sample_idx = np.random.randint(0, len(dataset))
        expr = dataset.get_expression_data(sample_idx)
        perturbation = dataset.get_perturbation(sample_idx)
        
        model_prep_metrics = {
            "dataset_size": len(dataset),
            "embedding_dim": expr.shape[0],
            "control_cells": len(dataset.get_control_indices()),
            "treatment_cells": len(dataset.get_perturbed_indices()),
            "unique_perturbations": len(dataset.unique_perts),
            "data_loader_test_passed": True,
        }
        
        print(f"📊 Model preparation metrics:")
        print(f"   Dataset size: {model_prep_metrics['dataset_size']:,} cells")
        print(f"   Embedding dim: {model_prep_metrics['embedding_dim']:,}")
        print(f"   Control cells: {model_prep_metrics['control_cells']:,}")
        print(f"   Treatment cells: {model_prep_metrics['treatment_cells']:,}")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        model_prep_metrics = {
            "data_loader_test_passed": False,
            "error": str(e)
        }
    
    # Create model architecture summary
    model_config = {
        "input_dim": expr.shape[0] if 'expr' in locals() else config["hvg_n_top_genes"],
        "hidden_dims": [512, 256, 128],
        "output_dim": expr.shape[0] if 'expr' in locals() else config["hvg_n_top_genes"],
        "dropout_rate": 0.1,
        "activation": "relu",
        "batch_norm": True,
    }
    
    model_prep_metrics.update(model_config)
    
    print("🧠 Model architecture prepared:")
    print(f"   Input dim: {model_config['input_dim']}")
    print(f"   Hidden dims: {model_config['hidden_dims']}")
    print(f"   Output dim: {model_config['output_dim']}")
    
    log_step_complete("Model Preparation", start_time, model_prep_metrics)
    return model_prep_metrics

def step7_evaluation_setup(config):
    """Step 7: Set up evaluation metrics and framework."""
    start_time = log_step_start("Evaluation Setup", "Configuring evaluation metrics and Cell_Eval framework")
    
    print("📏 Setting up Cell_Eval metrics...")
    
    # Initialize Cell_Eval metrics
    try:
        evaluator = CellEvalMetrics(
            control_label=config["control_label"],
            significance_threshold=0.05,
            effect_size_threshold=0.5,
            top_k_genes=100
        )
        
        eval_metrics = {
            "cell_eval_initialized": True,
            "control_label": config["control_label"],
            "significance_threshold": 0.05,
            "effect_size_threshold": 0.5,
            "top_k_genes": 100,
            "available_metrics": [
                "pearson_correlation",
                "spearman_correlation", 
                "de_accuracy",
                "direction_accuracy",
                "effect_correlation",
                "top_k_recovery",
                "perturbation_strength_correlation"
            ]
        }
        
        print("✅ Cell_Eval framework configured")
        print("📊 Available metrics:")
        for metric in eval_metrics["available_metrics"]:
            print(f"   • {metric}")
            
    except Exception as e:
        print(f"❌ Evaluation setup failed: {e}")
        eval_metrics = {
            "cell_eval_initialized": False,
            "error": str(e)
        }
    
    log_step_complete("Evaluation Setup", start_time, eval_metrics)
    return eval_metrics

def calculate_gini_coefficient(values):
    """Calculate Gini coefficient for inequality measurement."""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def calculate_entropy(values):
    """Calculate Shannon entropy."""
    values = np.array(values)
    probs = values / values.sum()
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

def main():
    """Main pipeline execution."""
    print("🧬 VIRTUAL CELL CHALLENGE - COMPLETE PIPELINE")
    print("=" * 60)
    print("📝 Step-by-step analysis with W&B logging")
    print("🎯 Goal: Prepare VCC data for virtual cell modeling")
    print("=" * 60)
    
    # Initialize experiment tracking
    run, config = setup_wandb_experiment()
    
    try:
        # Pipeline execution
        pipeline_start = time.time()
        
        # Step 1: Data Loading & Validation
        adata, validation_results = step1_data_loading_validation(config)
        
        # Step 2: Dataset Analysis  
        analysis_metrics = step2_dataset_analysis(adata, config)
        
        # Step 3: Data Preprocessing
        adata_processed, preprocessing_metrics = step3_data_preprocessing(adata, config)
        
        # Step 4: Embedding Computation
        adata_processed, embedding_metrics = step4_embedding_computation(adata_processed, config)
        
        # Step 5: Training Splits
        adata_train, adata_val, adata_test, split_metrics = step5_training_splits(adata_processed, config)
        
        # Step 6: Model Preparation
        model_metrics = step6_model_preparation(adata_train, config)
        
        # Step 7: Evaluation Setup
        eval_metrics = step7_evaluation_setup(config)
        
        # Pipeline completion
        total_time = time.time() - pipeline_start
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        # Final summary metrics
        summary_metrics = {
            "pipeline_total_time": total_time,
            "pipeline_success": True,
            "final_train_cells": split_metrics["train_cells"],
            "final_val_cells": split_metrics["val_cells"],
            "final_test_cells": split_metrics["test_cells"],
            "final_embedding_dim": model_metrics.get("embedding_dim", 0),
            "final_perturbations": analysis_metrics.get("n_perturbations", 0),
        }
        
        wandb.log(summary_metrics)
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   • Training cells: {summary_metrics['final_train_cells']:,}")
        print(f"   • Validation cells: {summary_metrics['final_val_cells']:,}")
        print(f"   • Test cells: {summary_metrics['final_test_cells']:,}")
        print(f"   • Embedding dimension: {summary_metrics['final_embedding_dim']:,}")
        print(f"   • Total perturbations: {summary_metrics['final_perturbations']}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Train STATE model on processed data")
        print(f"   2. Implement zero-shot and few-shot learning")
        print(f"   3. Evaluate with Cell_Eval metrics")
        print(f"   4. Compare with baselines")
        
        print(f"\n📁 Generated files:")
        print(f"   • data/processed/vcc_complete_processed.h5ad")
        print(f"   • data/processed/vcc_train.h5ad")
        print(f"   • data/processed/vcc_val.h5ad")
        print(f"   • data/processed/vcc_test.h5ad")
        print(f"   • data/processed/split_info.json")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        wandb.log({"pipeline_success": False, "error": str(e)})
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 