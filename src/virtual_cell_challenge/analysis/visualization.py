"""
Visualization tools for single-cell perturbation data analysis.

Provides comprehensive plotting functions for data exploration and quality assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anndata

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def visualize_perturbations(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    cell_type_column: str = "cell_type",
    batch_column: str = "gem_group",
    control_label: str = "non-targeting",
    save_path: Optional[str] = None,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive perturbation visualizations.
    
    Args:
        adata: AnnData object
        perturbation_column: Column with perturbation information
        cell_type_column: Column with cell type information 
        batch_column: Column with batch information
        control_label: Label for control perturbations
        save_path: Optional path to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with plot objects
    """
    plots = {}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Perturbation Dataset Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Perturbation distribution
    plots['perturbation_dist'] = _plot_perturbation_distribution(
        adata, perturbation_column, control_label, axes[0, 0]
    )
    
    # Plot 2: Cell type distribution
    plots['celltype_dist'] = _plot_cell_type_distribution(
        adata, cell_type_column, axes[0, 1]
    )
    
    # Plot 3: Batch distribution
    plots['batch_dist'] = _plot_batch_distribution(
        adata, batch_column, axes[0, 2]
    )
    
    # Plot 4: Perturbation-cell type heatmap
    plots['pert_celltype_heatmap'] = _plot_perturbation_celltype_heatmap(
        adata, perturbation_column, cell_type_column, axes[1, 0]
    )
    
    # Plot 5: Expression statistics
    plots['expression_stats'] = _plot_expression_statistics(
        adata, axes[1, 1]
    )
    
    # Plot 6: Quality metrics (if available)
    plots['quality_metrics'] = _plot_quality_metrics(
        adata, perturbation_column, control_label, axes[1, 2]
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Perturbation plots saved to {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plots


def plot_quality_control(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    save_path: Optional[str] = None,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Create quality control visualizations.
    
    Args:
        adata: AnnData object
        perturbation_column: Column with perturbation information
        control_label: Label for control perturbations
        save_path: Optional path to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with plot objects
    """
    from ..preprocessing import analyze_perturbation_quality
    
    plots = {}
    
    try:
        # Get quality report
        quality_report = analyze_perturbation_quality(
            adata, perturbation_column, control_label
        )
        
        if len(quality_report) == 0:
            logger.warning("No perturbations found for quality analysis")
            return plots
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quality Control Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Knockdown effectiveness
        plots['knockdown_dist'] = _plot_knockdown_distribution(quality_report, axes[0, 0])
        
        # Plot 2: Cell count vs effectiveness
        plots['cells_vs_effectiveness'] = _plot_cells_vs_effectiveness(quality_report, axes[0, 1])
        
        # Plot 3: Expression levels comparison
        plots['expression_comparison'] = _plot_expression_comparison(quality_report, axes[1, 0])
        
        # Plot 4: QC summary statistics
        plots['qc_summary'] = _plot_qc_summary(quality_report, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Quality control plots saved to {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Quality control plotting failed: {e}")
        return {}
    
    return plots


def plot_embedding_spaces(
    adata: anndata.AnnData,
    embedding_keys: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plots: bool = True,
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Visualize embedding spaces (PCA, UMAP, t-SNE, etc.).
    
    Args:
        adata: AnnData object
        embedding_keys: List of embedding keys to plot (None for all available)
        color_by: Column to color points by
        save_path: Optional path to save plots
        show_plots: Whether to display plots
        interactive: Whether to create interactive plots
        
    Returns:
        Dictionary with plot objects
    """
    plots = {}
    
    if not adata.obsm:
        logger.warning("No embeddings found in adata.obsm")
        return plots
    
    available_embeddings = list(adata.obsm.keys())
    if embedding_keys is None:
        embedding_keys = available_embeddings
    else:
        embedding_keys = [key for key in embedding_keys if key in available_embeddings]
    
    if not embedding_keys:
        logger.warning("No valid embedding keys found")
        return plots
    
    if interactive:
        plots = _create_interactive_embedding_plots(adata, embedding_keys, color_by)
    else:
        plots = _create_static_embedding_plots(adata, embedding_keys, color_by, show_plots)
    
    if save_path and not interactive:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embedding plots saved to {save_path}")
    
    return plots


def plot_perturbation_effects(
    adata: anndata.AnnData,
    perturbations: List[str],
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    top_genes: int = 20,
    save_path: Optional[str] = None,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Visualize specific perturbation effects.
    
    Args:
        adata: AnnData object
        perturbations: List of perturbations to analyze
        perturbation_column: Column with perturbation information
        control_label: Label for control perturbations
        top_genes: Number of top differentially expressed genes to show
        save_path: Optional path to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with plot objects
    """
    plots = {}
    
    n_perts = len(perturbations)
    if n_perts == 0:
        logger.warning("No perturbations provided")
        return plots
    
    # Create figure
    fig, axes = plt.subplots(n_perts, 2, figsize=(15, 5 * n_perts))
    if n_perts == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Perturbation Effects Analysis', fontsize=16, fontweight='bold')
    
    for i, pert in enumerate(perturbations):
        if pert not in adata.obs[perturbation_column].values:
            logger.warning(f"Perturbation '{pert}' not found in data")
            continue
        
        # Plot differential expression
        plots[f'{pert}_de'] = _plot_differential_expression(
            adata, pert, perturbation_column, control_label, top_genes, axes[i, 0]
        )
        
        # Plot expression comparison
        plots[f'{pert}_comparison'] = _plot_perturbation_expression_comparison(
            adata, pert, perturbation_column, control_label, axes[i, 1]
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Perturbation effect plots saved to {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plots


# Helper functions for specific plot types

def _plot_perturbation_distribution(adata, perturbation_column, control_label, ax):
    """Plot perturbation distribution."""
    pert_counts = adata.obs[perturbation_column].value_counts()
    
    # Separate control and treatments
    if control_label in pert_counts.index:
        control_count = pert_counts[control_label]
        treatment_counts = pert_counts.drop(control_label)
    else:
        control_count = 0
        treatment_counts = pert_counts
    
    # Plot histogram of treatment counts
    ax.hist(treatment_counts.values, bins=min(20, len(treatment_counts)), 
            alpha=0.7, edgecolor='black')
    ax.axvline(control_count, color='red', linestyle='--', 
               label=f'Control: {control_count}')
    ax.set_xlabel('Cells per Perturbation')
    ax.set_ylabel('Number of Perturbations')
    ax.set_title('Perturbation Distribution')
    ax.legend()
    
    return ax


def _plot_cell_type_distribution(adata, cell_type_column, ax):
    """Plot cell type distribution."""
    if cell_type_column not in adata.obs.columns:
        ax.text(0.5, 0.5, f"Column '{cell_type_column}' not found", 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Cell Type Distribution')
        return ax
    
    ct_counts = adata.obs[cell_type_column].value_counts()
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(ct_counts)))
    wedges, texts, autotexts = ax.pie(ct_counts.values, labels=ct_counts.index, 
                                      autopct='%1.1f%%', colors=colors)
    ax.set_title('Cell Type Distribution')
    
    return ax


def _plot_batch_distribution(adata, batch_column, ax):
    """Plot batch distribution."""
    if batch_column not in adata.obs.columns:
        ax.text(0.5, 0.5, f"Column '{batch_column}' not found", 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Batch Distribution')
        return ax
    
    batch_counts = adata.obs[batch_column].value_counts()
    
    # Plot bar chart
    ax.bar(range(len(batch_counts)), batch_counts.values)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Batch Distribution')
    ax.set_xticks(range(len(batch_counts)))
    ax.set_xticklabels(batch_counts.index, rotation=45)
    
    return ax


def _plot_perturbation_celltype_heatmap(adata, perturbation_column, cell_type_column, ax):
    """Plot perturbation-cell type heatmap."""
    if perturbation_column not in adata.obs.columns or cell_type_column not in adata.obs.columns:
        ax.text(0.5, 0.5, "Required columns not found", 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Perturbation-Cell Type Heatmap')
        return ax
    
    cross_tab = pd.crosstab(adata.obs[perturbation_column], adata.obs[cell_type_column])
    
    # Plot heatmap (sample top perturbations if too many)
    if len(cross_tab) > 20:
        top_perts = cross_tab.sum(axis=1).nlargest(20).index
        cross_tab = cross_tab.loc[top_perts]
    
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Perturbation-Cell Type Matrix')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Perturbation')
    
    return ax


def _plot_expression_statistics(adata, ax):
    """Plot expression statistics."""
    if hasattr(adata.X, 'toarray'):
        expr_data = adata.X.toarray()
    else:
        expr_data = adata.X
    
    # Calculate statistics
    cell_totals = np.array(expr_data.sum(axis=1)).flatten()
    gene_detection = np.array((expr_data > 0).sum(axis=1)).flatten()
    
    # Create scatter plot
    ax.scatter(cell_totals, gene_detection, alpha=0.6)
    ax.set_xlabel('Total UMI Count')
    ax.set_ylabel('Genes Detected')
    ax.set_title('Expression Statistics')
    
    # Add trend line
    z = np.polyfit(cell_totals, gene_detection, 1)
    p = np.poly1d(z)
    ax.plot(cell_totals, p(cell_totals), "r--", alpha=0.8)
    
    return ax


def _plot_quality_metrics(adata, perturbation_column, control_label, ax):
    """Plot basic quality metrics."""
    try:
        from ..preprocessing import analyze_perturbation_quality
        quality_report = analyze_perturbation_quality(adata, perturbation_column, control_label)
        
        if len(quality_report) > 0:
            # Plot effectiveness vs cell count
            ax.scatter(quality_report['n_cells'], quality_report['knockdown_percent'], 
                      alpha=0.7, c=quality_report['is_effective'], cmap='RdYlGn')
            ax.set_xlabel('Number of Cells')
            ax.set_ylabel('Knockdown Percentage')
            ax.set_title('Quality Metrics')
        else:
            ax.text(0.5, 0.5, "No quality data available", 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Quality Metrics')
    except Exception:
        ax.text(0.5, 0.5, "Quality analysis failed", 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Quality Metrics')
    
    return ax


def _plot_knockdown_distribution(quality_report, ax):
    """Plot knockdown effectiveness distribution."""
    ax.hist(quality_report['knockdown_percent'], bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(quality_report['knockdown_percent'].mean(), color='red', linestyle='--',
               label=f"Mean: {quality_report['knockdown_percent'].mean():.1f}%")
    ax.set_xlabel('Knockdown Percentage')
    ax.set_ylabel('Number of Perturbations')
    ax.set_title('Knockdown Effectiveness Distribution')
    ax.legend()
    return ax


def _plot_cells_vs_effectiveness(quality_report, ax):
    """Plot cell count vs effectiveness."""
    colors = ['red' if not eff else 'green' for eff in quality_report['is_effective']]
    ax.scatter(quality_report['n_cells'], quality_report['knockdown_percent'], 
               c=colors, alpha=0.7)
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Knockdown Percentage')
    ax.set_title('Cell Count vs Effectiveness')
    return ax


def _plot_expression_comparison(quality_report, ax):
    """Plot expression comparison."""
    ax.scatter(quality_report['control_mean'], quality_report['perturbed_mean'], alpha=0.7)
    ax.set_xlabel('Control Expression')
    ax.set_ylabel('Perturbed Expression')
    ax.set_title('Expression Level Comparison')
    
    # Add diagonal line
    max_val = max(quality_report['control_mean'].max(), quality_report['perturbed_mean'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    return ax


def _plot_qc_summary(quality_report, ax):
    """Plot QC summary statistics."""
    effective_count = quality_report['is_effective'].sum()
    total_count = len(quality_report)
    
    # Create bar chart
    categories = ['Effective', 'Ineffective']
    counts = [effective_count, total_count - effective_count]
    colors = ['green', 'red']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Perturbations')
    ax.set_title('QC Summary')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    return ax


def _create_static_embedding_plots(adata, embedding_keys, color_by, show_plots):
    """Create static embedding plots."""
    plots = {}
    n_embeddings = len(embedding_keys)
    
    # Determine grid size
    cols = min(3, n_embeddings)
    rows = (n_embeddings + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, emb_key in enumerate(embedding_keys):
        embedding = adata.obsm[emb_key]
        
        if embedding.shape[1] >= 2:
            if color_by and color_by in adata.obs.columns:
                scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=adata.obs[color_by].astype('category').cat.codes,
                                        alpha=0.6, s=1)
                plt.colorbar(scatter, ax=axes[i])
            else:
                axes[i].scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1)
            
            axes[i].set_xlabel(f'{emb_key} 1')
            axes[i].set_ylabel(f'{emb_key} 2')
            axes[i].set_title(f'{emb_key} Embedding')
        else:
            axes[i].text(0.5, 0.5, f'{emb_key}\n1D embedding', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{emb_key} (1D)')
        
        plots[emb_key] = axes[i]
    
    # Hide extra subplots
    for i in range(n_embeddings, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plots


def _create_interactive_embedding_plots(adata, embedding_keys, color_by):
    """Create interactive embedding plots using plotly."""
    plots = {}
    
    for emb_key in embedding_keys:
        embedding = adata.obsm[emb_key]
        
        if embedding.shape[1] >= 2:
            if color_by and color_by in adata.obs.columns:
                color_data = adata.obs[color_by]
            else:
                color_data = None
            
            fig = px.scatter(
                x=embedding[:, 0], 
                y=embedding[:, 1],
                color=color_data,
                title=f'{emb_key} Embedding',
                labels={'x': f'{emb_key} 1', 'y': f'{emb_key} 2'}
            )
            fig.update_traces(marker=dict(size=3, opacity=0.7))
            plots[emb_key] = fig
    
    return plots


def _plot_differential_expression(adata, perturbation, perturbation_column, control_label, top_genes, ax):
    """Plot differential expression for a specific perturbation."""
    # Get perturbation and control cells
    pert_mask = adata.obs[perturbation_column] == perturbation
    ctrl_mask = adata.obs[perturbation_column] == control_label
    
    if not pert_mask.any() or not ctrl_mask.any():
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center')
        ax.set_title(f'{perturbation} - DE Analysis')
        return ax
    
    # Calculate fold changes
    if hasattr(adata.X, 'toarray'):
        pert_expr = adata.X[pert_mask].toarray().mean(axis=0)
        ctrl_expr = adata.X[ctrl_mask].toarray().mean(axis=0)
    else:
        pert_expr = adata.X[pert_mask].mean(axis=0)
        ctrl_expr = adata.X[ctrl_mask].mean(axis=0)
    
    fold_changes = pert_expr - ctrl_expr
    
    # Get top changed genes
    top_indices = np.argsort(np.abs(fold_changes))[-top_genes:]
    top_fc = fold_changes[top_indices]
    
    # Use gene names if available
    if hasattr(adata.var, 'gene_name') and 'gene_name' in adata.var.columns:
        gene_names = adata.var['gene_name'].iloc[top_indices].values
    else:
        gene_names = adata.var_names[top_indices].values
    
    # Create horizontal bar plot
    colors = ['red' if fc < 0 else 'green' for fc in top_fc]
    y_pos = np.arange(len(gene_names))
    
    ax.barh(y_pos, top_fc, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_names)
    ax.set_xlabel('Expression Change')
    ax.set_title(f'{perturbation} - Top DE Genes')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    return ax


def _plot_perturbation_expression_comparison(adata, perturbation, perturbation_column, control_label, ax):
    """Plot expression comparison between perturbation and control."""
    pert_mask = adata.obs[perturbation_column] == perturbation
    ctrl_mask = adata.obs[perturbation_column] == control_label
    
    if not pert_mask.any() or not ctrl_mask.any():
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center')
        ax.set_title(f'{perturbation} - Expression Comparison')
        return ax
    
    # Get target gene expression if perturbation name matches gene
    target_gene_idx = None
    if perturbation in adata.var_names:
        target_gene_idx = adata.var_names.get_loc(perturbation)
    
    if target_gene_idx is not None:
        # Plot target gene expression
        if hasattr(adata.X, 'toarray'):
            pert_target = adata.X[pert_mask, target_gene_idx].toarray().flatten()
            ctrl_target = adata.X[ctrl_mask, target_gene_idx].toarray().flatten()
        else:
            pert_target = adata.X[pert_mask, target_gene_idx].flatten()
            ctrl_target = adata.X[ctrl_mask, target_gene_idx].flatten()
        
        # Create violin plot
        data_to_plot = [ctrl_target, pert_target]
        parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', perturbation])
        ax.set_ylabel('Expression Level')
        ax.set_title(f'{perturbation} Target Gene Expression')
        
        # Color the violins
        parts['bodies'][0].set_facecolor('blue')
        parts['bodies'][1].set_facecolor('red')
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_alpha(0.7)
    else:
        ax.text(0.5, 0.5, f"Target gene '{perturbation}'\nnot found", 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'{perturbation} - Target Gene')
    
    return ax 