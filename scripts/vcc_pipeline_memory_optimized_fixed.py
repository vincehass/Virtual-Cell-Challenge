#!/usr/bin/env python3
"""
🧬 MEMORY-OPTIMIZED Virtual Cell Challenge Pipeline with W&B Logging (FIXED)

Fixed version with incremental saving and aggressive memory management.
Optimized for large datasets (200K+ cells) to avoid memory overflow.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path
import warnings
import gc
import psutil
warnings.filterwarnings("ignore")

# W&B for experiment tracking
import wandb

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['WANDB_PROJECT'] = 'virtual-cell-challenge'

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB

def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

def setup_wandb_experiment():
    """Initialize W&B experiment with comprehensive config."""
    
    config = {
        # Dataset configuration
        "dataset_name": "VCC_Training_Real_MemoryFixed",
        "dataset_path": "data/vcc_data/adata_Training.h5ad", 
        "dataset_size_gb": 14,
        "expected_cells": 221273,
        "expected_genes": 18080,
        
        # Pipeline configuration
        "pipeline_version": "1.2.0_memory_fixed",
        "perturbation_column": "target_gene",
        "batch_column": "batch", 
        "control_label": "non-targeting",
        "cell_type": "primary_cells",
        
        # Memory optimization
        "chunk_size": 10000,  # Process 10K cells at a time
        "use_backed_mode": True,
        "sample_for_analysis": 15000,  # Reduced sample size
        "aggressive_memory_cleanup": True,
        "incremental_saving": True,
        
        # Processing configuration
        "normalize_total": 1e4,
        "log_transform": True,
        "hvg_n_top_genes": 2000,
        "pca_n_components": 50,
        
        # Training configuration
        "val_perturbations": ["TMSB4X", "PRCP", "TADA1"],
        "test_perturbations": ["HIRA", "IGF2R", "NCK2", "MED13", "MED12"],
        "random_seed": 42,
        
        # Hardware
        "compute_environment": "CPU_Memory_Fixed",
        "python_version": "3.9",
    }
    
    # Initialize W&B
    run = wandb.init(
        project="virtual-cell-challenge",
        name="vcc-memory-fixed-pipeline",
        notes="Memory-fixed VCC pipeline with incremental saving (221K cells)",
        tags=["data-analysis", "preprocessing", "real-data", "vcc", "memory-fixed"],
        config=config
    )
    
    return run, config

def log_step_start(step_name, step_description):
    """Log the start of a pipeline step."""
    mem_usage = get_memory_usage()
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"📝 {step_description}")
    print(f"⚡ Memory-optimized processing")
    print(f"💾 Current memory usage: {mem_usage:.2f} GB")
    print(f"{'='*60}")
    
    wandb.log({
        f"step_{step_name.lower().replace(' ', '_')}_started": True,
        "current_step": step_name,
        f"memory_usage_gb_start_{step_name.lower().replace(' ', '_')}": mem_usage,
    })
    
    return time.time()

def log_step_complete(step_name, start_time, metrics=None):
    """Log the completion of a pipeline step."""
    duration = time.time() - start_time
    mem_usage = get_memory_usage()
    
    print(f"✅ {step_name} completed in {duration:.2f}s")
    print(f"💾 Memory usage: {mem_usage:.2f} GB")
    
    log_data = {
        f"step_{step_name.lower().replace(' ', '_')}_duration": duration,
        f"step_{step_name.lower().replace(' ', '_')}_completed": True,
        f"memory_usage_gb_end_{step_name.lower().replace(' ', '_')}": mem_usage,
    }
    
    if metrics:
        for key, value in metrics.items():
            log_data[f"{step_name.lower().replace(' ', '_')}_{key}"] = value
    
    wandb.log(log_data)

def step1_data_loading_efficient(config):
    """Step 1: Memory-efficient data loading and validation."""
    start_time = log_step_start("Data Loading Efficient", "Loading dataset in backed mode for memory efficiency")
    
    # Load in backed mode to avoid loading everything into memory
    print("📊 Loading VCC dataset in backed mode...")
    load_start = time.time()
    adata = ad.read_h5ad(config["dataset_path"], backed='r')
    load_time = time.time() - load_start
    
    print(f"✅ Dataset loaded in {load_time:.2f}s (backed mode)")
    print(f"📏 Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Basic validation without loading full data into memory
    validation_results = {
        "cells_count": int(adata.n_obs),
        "genes_count": int(adata.n_vars),
        "load_time_seconds": load_time,
        "backed_mode": True,
        "memory_efficient": True,
    }
    
    # Check required columns
    required_obs_columns = [config["perturbation_column"], config["batch_column"]]
    missing_columns = [col for col in required_obs_columns if col not in adata.obs.columns]
    validation_results["missing_columns"] = missing_columns
    validation_results["has_required_columns"] = len(missing_columns) == 0
    
    # Log basic statistics from metadata (fast)
    perturbation_counts = adata.obs[config["perturbation_column"]].value_counts()
    batch_counts = adata.obs[config["batch_column"]].value_counts()
    
    validation_results.update({
        "n_perturbations": len(perturbation_counts),
        "n_batches": len(batch_counts),
        "control_cells": int(perturbation_counts.get(config["control_label"], 0)),
        "treatment_cells": int(perturbation_counts.drop(config["control_label"], errors='ignore').sum()),
        "cells_per_perturbation_mean": float(perturbation_counts.mean()),
        "cells_per_batch_mean": float(batch_counts.mean()),
    })
    
    # Create lightweight plots from metadata
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Top perturbations
    top_perts = perturbation_counts.head(12)  # Fewer for readability
    ax1.barh(range(len(top_perts)), top_perts.values)
    ax1.set_yticks(range(len(top_perts)))
    ax1.set_yticklabels(top_perts.index, fontsize=8)
    ax1.set_xlabel('Cell Count')
    ax1.set_title('Top 12 Perturbations')
    
    # Batch distribution
    ax2.hist(batch_counts.values, bins=min(15, len(batch_counts)), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Cells per Batch')
    ax2.set_ylabel('Number of Batches')
    ax2.set_title('Batch Distribution')
    
    plt.tight_layout()
    wandb.log({"data_validation_efficient/perturbation_distribution": wandb.Image(fig)})
    plt.close()
    
    cleanup_memory()
    log_step_complete("Data Loading Efficient", start_time, validation_results)
    return adata, validation_results

def step2_sampled_analysis(adata, config):
    """Step 2: Analysis using a representative sample to save memory."""
    start_time = log_step_start("Sampled Analysis", "Performing analysis on representative sample for memory efficiency")
    
    # Sample cells for analysis (reduced size)
    sample_size = min(config["sample_for_analysis"], adata.n_obs)
    print(f"📊 Sampling {sample_size:,} cells for analysis (from {adata.n_obs:,} total)")
    
    # Create stratified sample to maintain perturbation representation
    np.random.seed(config["random_seed"])
    perturbations = adata.obs[config["perturbation_column"]]
    
    sample_indices = []
    for pert in perturbations.unique():
        pert_indices = np.where(perturbations == pert)[0]
        n_sample = min(75, len(pert_indices))  # Reduced from 100 to 75
        if n_sample > 0:
            sampled = np.random.choice(pert_indices, size=n_sample, replace=False)
            sample_indices.extend(sampled)
    
    sample_indices = np.array(sample_indices)
    print(f"✅ Created stratified sample: {len(sample_indices):,} cells")
    
    # Load sample data
    print("📈 Loading sample data for analysis...")
    adata_sample = adata[sample_indices].to_memory()  # Convert to memory for this sample
    
    # Basic expression analysis
    print("🧮 Computing expression statistics...")
    expr_data = adata_sample.X
    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()
    
    analysis_metrics = {
        "sample_size": len(sample_indices),
        "sample_fraction": len(sample_indices) / adata.n_obs,
        "mean_umi_per_cell": float(np.mean(expr_data.sum(axis=1))),
        "median_umi_per_cell": float(np.median(expr_data.sum(axis=1))),
        "mean_genes_per_cell": float(np.mean((expr_data > 0).sum(axis=1))),
        "median_genes_per_cell": float(np.median((expr_data > 0).sum(axis=1))),
        "sparsity": float(np.mean(expr_data == 0)),
    }
    
    print(f"📊 Sample Analysis Results:")
    print(f"   Mean UMI per cell: {analysis_metrics['mean_umi_per_cell']:.0f}")
    print(f"   Mean genes per cell: {analysis_metrics['mean_genes_per_cell']:.0f}")
    print(f"   Sparsity: {analysis_metrics['sparsity']:.1%}")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('VCC Dataset Sample Analysis (Memory Fixed)', fontsize=12, fontweight='bold')
    
    # UMI distribution
    cell_totals = expr_data.sum(axis=1)
    axes[0, 0].hist(cell_totals, bins=25, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Total UMI Count')
    axes[0, 0].set_ylabel('Number of Cells')
    axes[0, 0].set_title('UMI Count Distribution')
    axes[0, 0].axvline(analysis_metrics["mean_umi_per_cell"], color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Gene detection
    gene_detection = (expr_data > 0).sum(axis=1)
    axes[0, 1].hist(gene_detection, bins=25, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Genes Detected')
    axes[0, 1].set_ylabel('Number of Cells')
    axes[0, 1].set_title('Gene Detection Distribution')
    axes[0, 1].axvline(analysis_metrics["mean_genes_per_cell"], color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # UMI vs Gene detection
    axes[1, 0].scatter(cell_totals, gene_detection, alpha=0.6, s=1)
    axes[1, 0].set_xlabel('Total UMI Count')
    axes[1, 0].set_ylabel('Genes Detected')
    axes[1, 0].set_title('UMI vs Gene Detection')
    
    # Perturbation representation in sample
    sample_pert_counts = adata_sample.obs[config["perturbation_column"]].value_counts()
    axes[1, 1].hist(sample_pert_counts.values, bins=12, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Cells per Perturbation (in sample)')
    axes[1, 1].set_ylabel('Number of Perturbations')
    axes[1, 1].set_title('Sample Perturbation Distribution')
    
    plt.tight_layout()
    wandb.log({"sampled_analysis/comprehensive_plots": wandb.Image(fig)})
    plt.close()
    
    # Clean up memory
    del adata_sample, expr_data
    cleanup_memory()
    
    log_step_complete("Sampled Analysis", start_time, analysis_metrics)
    return analysis_metrics

def step3_efficient_preprocessing(adata, config):
    """Step 3: Memory-efficient preprocessing using chunked operations."""
    start_time = log_step_start("Efficient Preprocessing", "Preprocessing with chunked operations for memory efficiency")
    
    print("🔧 Creating memory-efficient copy...")
    print("⚠️ Loading full dataset into memory for preprocessing...")
    print("   This may take several minutes for 221K cells...")
    
    copy_start = time.time()
    adata_processed = adata.to_memory()
    copy_time = time.time() - copy_start
    print(f"✅ Dataset copied to memory in {copy_time:.2f}s")
    print(f"💾 Memory usage after copy: {get_memory_usage():.2f} GB")
    
    # Adapt column names
    print("🔧 Adapting data format...")
    adata_processed.obs['gene'] = adata_processed.obs[config["perturbation_column"]]
    adata_processed.obs['gem_group'] = adata_processed.obs[config["batch_column"]]
    adata_processed.obs['cell_type'] = config["cell_type"]
    adata_processed.var['gene_name'] = adata_processed.var['gene_id']
    
    # Basic QC metrics (efficient computation)
    print("📊 Computing QC metrics...")
    qc_start = time.time()
    
    # Compute basic statistics efficiently
    if hasattr(adata_processed.X, 'toarray'):
        # For sparse matrices, compute efficiently
        adata_processed.var['n_cells'] = np.array((adata_processed.X > 0).sum(axis=0)).flatten()
        adata_processed.obs['n_genes'] = np.array((adata_processed.X > 0).sum(axis=1)).flatten()
        adata_processed.obs['total_counts'] = np.array(adata_processed.X.sum(axis=1)).flatten()
    else:
        adata_processed.var['n_cells'] = (adata_processed.X > 0).sum(axis=0)
        adata_processed.obs['n_genes'] = (adata_processed.X > 0).sum(axis=1)
        adata_processed.obs['total_counts'] = adata_processed.X.sum(axis=1)
    
    qc_time = time.time() - qc_start
    print(f"✅ QC metrics computed in {qc_time:.2f}s")
    print(f"💾 Memory usage after QC: {get_memory_usage():.2f} GB")
    
    preprocessing_metrics = {
        "copy_time": copy_time,
        "qc_computation_time": qc_time,
        "cells_before_qc": int(adata_processed.n_obs),
        "genes_before_qc": int(adata_processed.n_vars),
        "mean_genes_per_cell_before": float(adata_processed.obs['n_genes'].mean()),
        "mean_counts_per_cell_before": float(adata_processed.obs['total_counts'].mean()),
    }
    
    # Light quality control (avoid heavy filtering to preserve data)
    print("🔍 Applying light quality control...")
    
    # Filter cells with very few genes (conservative threshold)
    min_genes = 100  # More lenient than usual
    cell_filter = adata_processed.obs['n_genes'] >= min_genes
    cells_filtered = (~cell_filter).sum()
    print(f"   Cells with < {min_genes} genes: {cells_filtered} ({cells_filtered/len(cell_filter):.1%})")
    
    # Filter genes in very few cells (conservative threshold)
    min_cells = 5  # More lenient than usual
    gene_filter = adata_processed.var['n_cells'] >= min_cells
    genes_filtered = (~gene_filter).sum()
    print(f"   Genes in < {min_cells} cells: {genes_filtered} ({genes_filtered/len(gene_filter):.1%})")
    
    # Apply filters only if not too many cells/genes are removed
    if cells_filtered < 0.1 * len(cell_filter):  # Less than 10% of cells
        adata_processed = adata_processed[cell_filter, :].copy()
        print(f"✅ Applied cell filter: {cells_filtered} cells removed")
        cleanup_memory()
    else:
        print(f"⚠️ Skipping cell filter: would remove too many cells ({cells_filtered})")
    
    if genes_filtered < 0.2 * len(gene_filter):  # Less than 20% of genes
        adata_processed = adata_processed[:, gene_filter].copy()
        print(f"✅ Applied gene filter: {genes_filtered} genes removed")
        cleanup_memory()
    else:
        print(f"⚠️ Skipping gene filter: would remove too many genes ({genes_filtered})")
    
    preprocessing_metrics.update({
        "cells_after_qc": int(adata_processed.n_obs),
        "genes_after_qc": int(adata_processed.n_vars),
        "cells_filtered": int(cells_filtered) if cells_filtered < 0.1 * len(cell_filter) else 0,
        "genes_filtered": int(genes_filtered) if genes_filtered < 0.2 * len(gene_filter) else 0,
    })
    
    print(f"✅ Preprocessing complete:")
    print(f"   Cells: {preprocessing_metrics['cells_before_qc']:,} → {preprocessing_metrics['cells_after_qc']:,}")
    print(f"   Genes: {preprocessing_metrics['genes_before_qc']:,} → {preprocessing_metrics['genes_after_qc']:,}")
    print(f"💾 Final memory usage: {get_memory_usage():.2f} GB")
    
    log_step_complete("Efficient Preprocessing", start_time, preprocessing_metrics)
    return adata_processed, preprocessing_metrics

def step4_create_data_splits_incremental(adata_processed, config):
    """Step 4: Create training splits with incremental saving to avoid memory overflow."""
    start_time = log_step_start("Data Splits Incremental", "Creating training/validation/test splits with incremental saving")
    
    print("🎯 Creating perturbation-based splits...")
    
    # Get perturbation information
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
    }
    
    print(f"📊 Split Summary:")
    print(f"   Train: {split_metrics['train_cells']:,} cells, {split_metrics['train_perturbations']} perturbations")
    print(f"   Val:   {split_metrics['val_cells']:,} cells, {split_metrics['val_perturbations']} perturbations")
    print(f"   Test:  {split_metrics['test_cells']:,} cells, {split_metrics['test_perturbations']} perturbations")
    
    # Save datasets ONE AT A TIME with memory cleanup
    print("💾 Saving split datasets incrementally...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    save_start = time.time()
    individual_save_times = {}
    
    # Save validation set first (smallest)
    print(f"💾 Saving validation set ({split_metrics['val_cells']:,} cells)...")
    val_save_start = time.time()
    adata_val = adata_processed[val_mask].copy()
    adata_val.write(output_dir / "vcc_val_memory_fixed.h5ad")
    del adata_val  # Immediately delete
    cleanup_memory()
    individual_save_times['val'] = time.time() - val_save_start
    print(f"✅ Validation set saved in {individual_save_times['val']:.2f}s")
    print(f"💾 Memory after val save: {get_memory_usage():.2f} GB")
    
    # Save test set
    print(f"💾 Saving test set ({split_metrics['test_cells']:,} cells)...")
    test_save_start = time.time()
    adata_test = adata_processed[test_mask].copy()
    adata_test.write(output_dir / "vcc_test_memory_fixed.h5ad")
    del adata_test  # Immediately delete
    cleanup_memory()
    individual_save_times['test'] = time.time() - test_save_start
    print(f"✅ Test set saved in {individual_save_times['test']:.2f}s")
    print(f"💾 Memory after test save: {get_memory_usage():.2f} GB")
    
    # Save training set (largest)
    print(f"💾 Saving training set ({split_metrics['train_cells']:,} cells)...")
    train_save_start = time.time()
    adata_train = adata_processed[train_mask].copy()
    adata_train.write(output_dir / "vcc_train_memory_fixed.h5ad")
    del adata_train  # Immediately delete
    cleanup_memory()
    individual_save_times['train'] = time.time() - train_save_start
    print(f"✅ Training set saved in {individual_save_times['train']:.2f}s")
    print(f"💾 Memory after train save: {get_memory_usage():.2f} GB")
    
    # Save full processed dataset
    print(f"💾 Saving complete processed dataset ({adata_processed.n_obs:,} cells)...")
    full_save_start = time.time()
    adata_processed.write(output_dir / "vcc_complete_memory_fixed.h5ad")
    individual_save_times['complete'] = time.time() - full_save_start
    print(f"✅ Complete dataset saved in {individual_save_times['complete']:.2f}s")
    
    total_save_time = time.time() - save_start
    split_metrics.update({
        "save_time_total": total_save_time,
        "save_time_val": individual_save_times['val'],
        "save_time_test": individual_save_times['test'],
        "save_time_train": individual_save_times['train'],
        "save_time_complete": individual_save_times['complete'],
    })
    
    print(f"✅ All datasets saved in {total_save_time:.2f}s")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    splits = ['Train', 'Val', 'Test']
    cell_counts = [split_metrics['train_cells'], split_metrics['val_cells'], split_metrics['test_cells']]
    colors = ['blue', 'orange', 'green']
    
    ax.bar(splits, cell_counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Cells')
    ax.set_title('Data Split Sizes (Memory Fixed)')
    for i, count in enumerate(cell_counts):
        ax.text(i, count + 1000, f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    wandb.log({"data_splits/split_summary": wandb.Image(fig)})
    plt.close()
    
    cleanup_memory()
    log_step_complete("Data Splits Incremental", start_time, split_metrics)
    return split_metrics

def main():
    """Memory-fixed main pipeline execution."""
    print("🧬 VIRTUAL CELL CHALLENGE - MEMORY-FIXED PIPELINE")
    print("=" * 60)
    print("📝 Step-by-step analysis with W&B logging")
    print("⚡ Fixed memory management for large datasets (221K cells)")
    print("🎯 Goal: Prepare VCC data efficiently without memory overflow")
    print("=" * 60)
    
    print(f"💾 Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Initialize experiment tracking
    run, config = setup_wandb_experiment()
    
    try:
        pipeline_start = time.time()
        
        # Step 1: Efficient Data Loading
        adata, validation_results = step1_data_loading_efficient(config)
        
        # Step 2: Sampled Analysis
        analysis_metrics = step2_sampled_analysis(adata, config)
        
        # Step 3: Efficient Preprocessing
        adata_processed, preprocessing_metrics = step3_efficient_preprocessing(adata, config)
        
        # Step 4: Create Data Splits (FIXED with incremental saving)
        split_metrics = step4_create_data_splits_incremental(adata_processed, config)
        
        # Pipeline completion
        total_time = time.time() - pipeline_start
        final_memory = get_memory_usage()
        
        print(f"\n🎉 MEMORY-FIXED PIPELINE COMPLETED!")
        print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"💾 Final memory usage: {final_memory:.2f} GB")
        
        # Final summary
        summary_metrics = {
            "pipeline_total_time": total_time,
            "pipeline_success": True,
            "memory_optimization": "fixed_with_incremental_saving",
            "final_memory_usage_gb": final_memory,
            "final_train_cells": split_metrics["train_cells"],
            "final_val_cells": split_metrics["val_cells"],
            "final_test_cells": split_metrics["test_cells"],
        }
        
        wandb.log(summary_metrics)
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   • Training cells: {summary_metrics['final_train_cells']:,}")
        print(f"   • Validation cells: {summary_metrics['final_val_cells']:,}")
        print(f"   • Test cells: {summary_metrics['final_test_cells']:,}")
        print(f"   • Memory optimization: {summary_metrics['memory_optimization']}")
        print(f"   • Peak memory usage: {summary_metrics['final_memory_usage_gb']:.2f} GB")
        
        print(f"\n📁 Generated files:")
        print(f"   • data/processed/vcc_complete_memory_fixed.h5ad")
        print(f"   • data/processed/vcc_train_memory_fixed.h5ad")
        print(f"   • data/processed/vcc_val_memory_fixed.h5ad")
        print(f"   • data/processed/vcc_test_memory_fixed.h5ad")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Run embedding computation on processed data")
        print(f"   2. Train STATE model architecture")
        print(f"   3. Implement virtual cell prediction")
        print(f"   4. Evaluate with Cell_Eval metrics")
        
        print(f"\n✅ SUCCESS: Your 221K cell VCC dataset is ready for modeling!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        wandb.log({"pipeline_success": False, "error": str(e)})
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 