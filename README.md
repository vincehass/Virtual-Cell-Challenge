# Virtual Cell Challenge: Single-Cell Data Loading & Analysis

A comprehensive repository for understanding and reproducing the single-cell perturbation data loading pipeline from the Arc Institute's **Virtual Cell Challenge**. This repository documents the complete ecosystem including the **cell-load** library, **STATE** model, evaluation frameworks, and data analysis workflows.

## 🎯 Project Overview

This repository provides a deep dive into the Arc Institute's Virtual Cell Challenge, focusing on:

1. **Data Loading Pipeline**: Complete reproduction of the cell-load library functionality
2. **Virtual Cell Challenge**: Understanding the STATE model and evaluation frameworks
3. **Data Analysis**: Comprehensive analysis of single-cell perturbation datasets
4. **Evaluation Metrics**: Implementation of Cell_Eval and other assessment frameworks
5. **Preprocessing Workflows**: Quality control and data filtering utilities

## 📚 Table of Contents

- [Virtual Cell Challenge Overview](#virtual-cell-challenge-overview)
- [Cell-Load Data Loading Pipeline](#cell-load-data-loading-pipeline)
- [Data Types and Formats](#data-types-and-formats)
- [Tasks and Evaluation](#tasks-and-evaluation)
- [Preprocessing and Quality Control](#preprocessing-and-quality-control)
- [STATE Model Integration](#state-model-integration)
- [Installation and Setup](#installation-and-setup)
- [Usage Examples](#usage-examples)
- [Analysis Notebooks](#analysis-notebooks)
- [Contributing](#contributing)

## 🔬 Virtual Cell Challenge Overview

### What is the Virtual Cell Challenge?

The Arc Institute's Virtual Cell Challenge represents an ambitious effort to build AI-powered models that can predict how cells respond to various perturbations. The challenge consists of several key components:

#### 1. **STATE Model** (State Transition and Embedding)

- **State Embedding (SE)**: Converts transcriptome data into smooth multidimensional vector spaces
- **State Transition (ST)**: Predicts how cells transition between different states in response to perturbations
- **Architecture**: Bidirectional transformer with self-attention over sets of cells
- **Training Data**: 100+ million cells from perturbation experiments

#### 2. **Cell-Load Library**

- PyTorch-based data loading framework
- Handles massive single-cell perturbation datasets
- Supports multiple cell types, datasets, and experimental conditions
- Provides zero-shot and few-shot learning capabilities

#### 3. **Cell_Eval Framework**

- Comprehensive evaluation metrics beyond simple expression counts
- Biologically relevant assessments focused on differential expression
- Perturbation strength estimation
- Transparent assessment of virtual cell model performance

### Key Datasets

#### **Tahoe-100M**

- 100 million cells from ~60,000 drug perturbation experiments
- 50 cancer models tested against 1,100+ drug treatments
- World's largest single-cell perturbation dataset
- Generated using Mosaic platform with Parse Biosciences and Ultima Genomics

#### **scBaseCount**

- 230+ million cells spanning 21 organisms and 72 tissues
- AI-driven hierarchical agent workflow for data curation
- Automated discovery and preprocessing of SRA data
- Continuously updated repository

#### **Replogle Dataset**

- High-quality perturbation data with genetic knockdowns
- Multiple cell types including Jurkat, RPE1, and others
- Comprehensive gene perturbation screens

## 🔧 Cell-Load Data Loading Pipeline

### Core Architecture

The cell-load library implements a sophisticated data loading pipeline designed for large-scale single-cell perturbation experiments:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   TOML Config   │    │   H5/AnnData     │    │  PyTorch        │
│   - Datasets    │───▶│   - Perturbations│───▶│  DataLoader     │
│   - Splits      │    │   - Cell Types   │    │  - Batched      │
│   - Tasks       │    │   - Embeddings   │    │  - Mapped       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

#### 1. **PerturbationDataModule**

Main data loading interface that:

- Manages train/validation/test splits across multiple datasets
- Handles zero-shot and few-shot learning scenarios
- Implements control cell mapping strategies
- Provides standardized batch formatting

#### 2. **PerturbationDataset**

Dataset class that:

- Loads perturbation data from H5 files (AnnData format)
- Supports multiple cell types per dataset
- Handles sparse and dense expression matrices
- Provides cell barcode tracking (optional)

#### 3. **Mapping Strategies**

- **Random Mapping**: Maps perturbed cells to random control cells
- **Batch Mapping**: Maps within the same experimental batch
- Ensures proper control-treatment pairing

### Data Format Requirements

#### **H5/AnnData Structure**

```
file.h5ad
├── X                    # Gene expression matrix (cells × genes)
├── obs/                 # Cell metadata
│   ├── gene            # Perturbation column (e.g., "AARS", "non-targeting")
│   ├── cell_type       # Cell type (e.g., "jurkat", "rpe1")
│   ├── gem_group       # Batch information
│   └── ...
├── var/                 # Gene metadata
│   ├── gene_name       # Gene symbols
│   └── ...
└── obsm/               # Embeddings
    ├── X_hvg           # Highly variable genes embedding
    ├── X_state         # STATE model embedding
    └── ...
```

#### **TOML Configuration**

```toml
# Dataset paths
[datasets]
replogle = "/path/to/replogle_dataset/"
jurkat = "/path/to/jurkat_dataset/"

# Training specifications
[training]
replogle = "train"
jurkat = "train"

# Zero-shot learning (entire cell types held out)
[zeroshot]
"replogle.jurkat" = "test"

# Few-shot learning (specific perturbations held out)
[fewshot]
[fewshot."replogle.rpe1"]
val = ["AARS"]
test = ["AARS", "NUP107", "RPUSD4"]
```

## 📊 Data Types and Formats

### Input Data Types

#### 1. **Perturbation Data**

- **Genetic Perturbations**: CRISPR knockouts, knockdowns, overexpression
- **Chemical Perturbations**: Small molecule treatments, drug compounds
- **Control Cells**: Non-targeting controls, DMSO controls
- **Combinatorial**: Multiple simultaneous perturbations

#### 2. **Expression Data**

- **Raw Counts**: UMI counts from scRNA-seq
- **Normalized**: Log-transformed, scaled expressions
- **Embeddings**: Pre-computed dimensionality reductions (PCA, UMAP, STATE)

#### 3. **Metadata**

- **Cell Type**: Jurkat, RPE1, K562, etc.
- **Batch Information**: Experimental plates, time points
- **Perturbation Details**: Target genes, concentrations, durations
- **Cell Barcodes**: Unique cell identifiers

### Output Data Format

Each data batch contains:

```python
batch = {
    'pert_cell_emb': torch.Tensor,      # Perturbed cell embeddings
    'ctrl_cell_emb': torch.Tensor,      # Control cell embeddings
    'pert_emb': torch.Tensor,           # Perturbation one-hot/embeddings
    'pert_name': List[str],             # Perturbation names
    'cell_type': List[str],             # Cell types
    'batch': torch.Tensor,              # Batch information
    'pert_cell_barcode': List[str],     # Cell barcodes (optional)
    'ctrl_cell_barcode': List[str],     # Control cell barcodes (optional)
}
```

## 🎯 Tasks and Evaluation

### Learning Tasks

#### 1. **Zero-Shot Learning**

- **Objective**: Predict responses in entirely unseen cell types
- **Setup**: Hold out entire cell types for testing
- **Challenge**: Generalize across different cellular contexts
- **Example**: Train on RPE1 cells, test on Jurkat cells

#### 2. **Few-Shot Learning**

- **Objective**: Predict responses to novel perturbations with limited data
- **Setup**: Hold out specific perturbations within cell types
- **Challenge**: Learn from few examples of new perturbations
- **Example**: Train on 90% of gene knockouts, test on remaining 10%

#### 3. **Cross-Modal Prediction**

- **Objective**: Predict one data type from another
- **Examples**:
  - Gene expression from morphology
  - Proteomics from transcriptomics
  - Time-series from single time points

### Evaluation Metrics

#### 1. **Expression-Based Metrics**

- **Pearson Correlation**: Gene-wise correlation between predicted and actual
- **Mean Squared Error (MSE)**: L2 distance in expression space
- **Cosine Similarity**: Directional similarity of expression vectors

#### 2. **Biological Metrics (Cell_Eval)**

- **Differential Expression Accuracy**: Correctly identifying DE genes
- **Perturbation Strength**: Magnitude of predicted vs. actual changes
- **Pathway Enrichment**: Biological pathway activation patterns
- **Cell State Transitions**: Accuracy of predicted state changes

#### 3. **Practical Metrics**

- **Top-K Gene Recovery**: Fraction of top differentially expressed genes recovered
- **Direction Accuracy**: Percentage of genes with correct up/down regulation
- **Effect Size Correlation**: Correlation of fold-change magnitudes

### Benchmarking Framework

#### **Cell_Eval Components**

1. **Perturbation Effect Detection**: Can the model distinguish treated from control?
2. **Gene Ranking**: How well does the model rank genes by perturbation effect?
3. **Dose-Response**: Does the model capture concentration-dependent effects?
4. **Time-Course**: Can the model predict temporal dynamics?

## 🔬 Preprocessing and Quality Control

### Quality Control Pipeline

#### 1. **On-Target Knockdown Filtering**

The library provides sophisticated QC to ensure perturbations worked:

```python
from cell_load.utils.data_utils import filter_on_target_knockdown

filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,        # 70% knockdown required
    cell_residual_expression=0.50,   # 50% knockdown per cell
    min_cells=30,                    # Minimum cells per perturbation
)
```

**Three-Stage Filtering Process:**

1. **Perturbation-level**: Keep only perturbations with average knockdown ≥ 70%
2. **Cell-level**: Within good perturbations, keep cells with ≥ 50% knockdown
3. **Minimum count**: Discard perturbations with < 30 cells after filtering

#### 2. **Data Type Detection**

```python
from cell_load.utils.data_utils import suspected_discrete_torch, suspected_log_torch

# Check if data is raw counts vs. normalized
is_raw_counts = suspected_discrete_torch(expression_data)
is_log_transformed = suspected_log_torch(expression_data)
```

#### 3. **Individual Perturbation Assessment**

```python
from cell_load.utils.data_utils import is_on_target_knockdown

# Check specific perturbation effectiveness
is_effective = is_on_target_knockdown(
    adata=adata,
    target_gene="GENE1",
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30
)
```

### Preprocessing Workflow

```python
import anndata
from cell_load.utils.data_utils import filter_on_target_knockdown, set_var_index_to_col

# Complete preprocessing pipeline
def preprocess_perturbation_data(adata_path):
# 1. Load data
    adata = anndata.read_h5ad(adata_path)

    # 2. Set gene names as index
adata = set_var_index_to_col(adata, col="gene_name")

    # 3. Apply quality control
filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30
)

# 4. Save filtered data
filtered_adata.write_h5ad("filtered_data.h5ad")

    return filtered_adata
```

## 🧠 STATE Model Integration

### Model Architecture

#### **State Embedding (SE) Model**

- Converts raw transcriptomes into smooth vector representations
- Handles technical noise and batch effects
- Creates consistent embeddings across datasets
- Similar cell types cluster together in embedding space

#### **State Transition (ST) Model**

- Bidirectional transformer architecture
- Self-attention over sets of cells
- Predicts state transitions in response to perturbations
- Captures biological and technical heterogeneity

### Integration with Cell-Load

```python
from cell_load.data_modules import PerturbationDataModule

# Configure for STATE model training
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    embed_key="X_state",              # Use STATE embeddings
    output_space="gene",               # Predict gene expression
    batch_size=128,
    num_workers=24,
)

# Setup data loading
dm.setup()
train_loader = dm.train_dataloader()
```

### Performance Benchmarks

**STATE vs. Baseline Models (Tahoe-100M):**

- **50% improvement** in distinguishing perturbation effects
- **2x accuracy** in identifying true differentially expressed genes
- **First model** to consistently beat simple linear baselines

## 🛠 Installation and Setup

### Requirements

```bash
# Core dependencies
torch>=1.13.0
anndata>=0.9.1
lightning>=2.0.0
toml>=0.10.2

# Development dependencies
pytest>=8.3.5
ruff>=0.11.8
```

### Installation

```bash
# Install from PyPI
pip install cell-load

# Or install from source
git clone https://github.com/ArcInstitute/cell-load.git
cd cell-load
pip install -e .
```

### Configuration

Create a TOML configuration file:

```toml
# config.toml
[datasets]
replogle = "/path/to/replogle_dataset/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"

[fewshot]
[fewshot."replogle.rpe1"]
val = ["AARS"]
test = ["AARS", "NUP107", "RPUSD4"]
```

## 💻 Usage Examples

### Basic Data Loading

```python
from cell_load.data_modules import PerturbationDataModule

# Initialize data module
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    embed_key="X_hvg",
    num_workers=24,
    batch_col="gem_group",
    pert_col="gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
)

# Setup and load data
dm.setup()
train_loader = dm.train_dataloader()

# Iterate through batches
for batch in train_loader:
    pert_cells = batch['pert_cell_emb']     # Shape: (batch_size, embedding_dim)
    ctrl_cells = batch['ctrl_cell_emb']     # Shape: (batch_size, embedding_dim)
    perturbations = batch['pert_emb']       # Shape: (batch_size, n_perturbations)
    pert_names = batch['pert_name']         # List of perturbation names
    cell_types = batch['cell_type']         # List of cell types
```

### Advanced Configuration

```python
# Advanced data loading with custom options
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    embed_key="X_state",                    # Use STATE embeddings
    output_space="gene",                    # Predict full gene expression
    basal_mapping_strategy="batch",         # Map controls within batch
    n_basal_samples=3,                      # Use 3 control cells per perturbation
    should_yield_control_cells=True,        # Include control cells in output
    barcode=True,                           # Include cell barcodes
    perturbation_features_file="gene_embeddings.pt",  # Use gene embeddings
    batch_size=256,
    num_workers=32,
)
```

### Quality Control Analysis

```python
import anndata
import matplotlib.pyplot as plt
from cell_load.utils.data_utils import filter_on_target_knockdown

# Load and analyze data quality
adata = anndata.read_h5ad("raw_data.h5ad")

print(f"Original data: {adata.n_obs} cells, {adata.n_vars} genes")
print(f"Perturbations: {adata.obs['gene'].unique()}")

# Apply quality control
filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30,
)

print(f"Filtered data: {filtered_adata.n_obs} cells")
print(f"Removed {adata.n_obs - filtered_adata.n_obs} cells ({100*(adata.n_obs - filtered_adata.n_obs)/adata.n_obs:.1f}%)")

# Analyze perturbation effectiveness
perturbations = adata.obs['gene'].unique()
effective_perts = []

for pert in perturbations:
    if pert != "non-targeting":
        is_effective = is_on_target_knockdown(
            adata, pert, "gene", "non-targeting", 0.30
        )
        effective_perts.append((pert, is_effective))

print(f"Effective perturbations: {sum(eff for _, eff in effective_perts)}/{len(effective_perts)}")
```

## 📊 Analysis Notebooks

This repository includes comprehensive Jupyter notebooks for:

### 1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)

- Dataset statistics and overview
- Cell type and perturbation distributions
- Quality metrics visualization
- Batch effect analysis

### 2. **Preprocessing Pipeline** (`notebooks/02_preprocessing.ipynb`)

- Complete preprocessing workflow
- Quality control implementation
- Before/after filtering comparisons
- Perturbation effectiveness analysis

### 3. **Data Loading Demo** (`notebooks/03_data_loading.ipynb`)

- Cell-load library usage examples
- Configuration setup
- Batch inspection and validation
- Control-treatment mapping visualization

### 4. **Evaluation Metrics** (`notebooks/04_evaluation.ipynb`)

- Implementation of Cell_Eval metrics
- Benchmark comparisons
- Performance visualization
- Statistical analysis

### 5. **STATE Model Analysis** (`notebooks/05_state_integration.ipynb`)

- STATE model embedding analysis
- Integration with cell-load
- Perturbation prediction examples
- Model interpretation

## 🏗 Repository Structure

```
single-cell-challenge/
├── README.md                          # This comprehensive documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── config/                           # Configuration files
│   ├── example_config.toml           # Example TOML configuration
│   └── datasets/                     # Dataset-specific configs
├── src/                              # Source code
│   ├── data_loading/                 # Data loading implementations
│   ├── preprocessing/                # Quality control utilities
│   ├── evaluation/                   # Evaluation metrics
│   └── analysis/                     # Analysis utilities
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset exploration
│   ├── 02_preprocessing.ipynb        # Preprocessing pipeline
│   ├── 03_data_loading.ipynb         # Data loading demo
│   ├── 04_evaluation.ipynb           # Evaluation metrics
│   └── 05_state_integration.ipynb    # STATE model analysis
├── tests/                            # Unit tests
│   ├── test_data_loading.py          # Data loading tests
│   ├── test_preprocessing.py         # Preprocessing tests
│   └── test_evaluation.py            # Evaluation tests
├── data/                             # Data directory
│   ├── sample/                       # Sample datasets
│   ├── processed/                    # Processed data
│   └── results/                      # Analysis results
└── docs/                             # Additional documentation
    ├── api_reference.md               # API documentation
    ├── tutorials/                     # Step-by-step tutorials
    └── papers/                        # Relevant papers and references
```

## 🔍 Key Features Documented

### Data Loading Features

- ✅ **Multi-dataset support**: Handle multiple datasets simultaneously
- ✅ **Zero-shot learning**: Entire cell types held out for testing
- ✅ **Few-shot learning**: Specific perturbations held out
- ✅ **Control mapping**: Random and batch-based control selection
- ✅ **Cell barcode tracking**: Optional cell identification
- ✅ **Sparse matrix support**: Efficient handling of sparse expression data
- ✅ **Batch processing**: Efficient mini-batch creation
- ✅ **Configuration flexibility**: TOML-based experiment setup

### Preprocessing Features

- ✅ **Quality control**: On-target knockdown filtering
- ✅ **Data validation**: Type detection and consistency checks
- ✅ **Gene filtering**: Effective perturbation identification
- ✅ **Cell filtering**: Remove low-quality or ineffective cells
- ✅ **Normalization**: Log transformation and scaling options
- ✅ **Metadata handling**: Gene names and annotations

### Evaluation Features

- ✅ **Cell_Eval metrics**: Biological relevance assessment
- ✅ **Cross-validation**: Robust performance estimation
- ✅ **Differential expression**: Gene-level effect detection
- ✅ **Perturbation strength**: Effect magnitude quantification
- ✅ **Statistical testing**: Significance assessment
- ✅ **Visualization**: Comprehensive plotting utilities

## 📖 References and Resources

### Key Papers

1. **STATE Model**: "State Embedding (SE) and State Transition (ST) Models for Cell Biology" (Arc Institute, 2024)
2. **Cell_Eval**: "A comprehensive evaluation framework for virtual cell modeling" (Arc Institute, 2024)
3. **Virtual Cell**: "Arc Institute's first virtual cell model: State" (Arc Institute News, 2024)
4. **Benchmarking**: "Benchmarking and Evaluation of AI Models in Biology" (CZI Workshop, 2024)

### Datasets

- **Tahoe-100M**: World's largest single-cell perturbation dataset
- **scBaseCount**: 230M+ cells across organisms and tissues
- **Replogle**: High-quality genetic perturbation screens
- **Virtual Cell Atlas**: Curated multi-modal cell data

### Related Tools

- **STATE Repository**: https://github.com/ArcInstitute/state
- **Cell-Load Repository**: https://github.com/ArcInstitute/cell-load
- **Virtual Cell Atlas**: https://arcinstitute.org/tools/virtualcellatlas
- **scBaseCount**: AI-curated single-cell database

## 🤝 Contributing

We welcome contributions to improve this documentation and analysis pipeline:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** improvements or fixes
4. **Submit** a pull request

### Areas for Contribution

- Additional preprocessing utilities
- New evaluation metrics
- Performance optimizations
- Documentation improvements
- Tutorial notebooks
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Arc Institute** for developing the STATE model and cell-load library
- **Virtual Cell Challenge** community for pioneering this field
- **Contributors** to the cell-load, STATE, and Cell_Eval projects
- **Open science** initiatives making this research possible

---

**Note**: This repository serves as an educational resource and documentation hub. For the official cell-load library and STATE model, please refer to the original Arc Institute repositories.

For questions or feedback, please open an issue or contact the maintainers.
