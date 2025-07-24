# Virtual Cell Challenge - Usage Guide

This guide provides practical examples for using the Virtual Cell Challenge toolkit to reproduce and analyze single-cell perturbation data.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/virtual-cell-challenge.git
cd virtual-cell-challenge

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install virtual-cell-challenge
```

### Command Line Interface

```bash
# Display toolkit information
virtual-cell info

# Validate data format
virtual-cell validate data.h5ad

# Preprocess data with quality control
virtual-cell preprocess data.h5ad -o filtered_data.h5ad

# Validate TOML configuration
virtual-cell config config.toml
```

## 📊 Data Loading Examples

### Basic Data Loading

```python
from virtual_cell_challenge.data_loading import PerturbationDataModule

# Initialize data module
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    embed_key="X_hvg",
    batch_size=128,
    num_workers=8
)

# Setup and load data
dm.setup()
train_loader = dm.train_dataloader()

# Iterate through batches
for batch in train_loader:
    pert_cells = batch['pert_cell_emb']     # Perturbed cell embeddings
    ctrl_cells = batch['ctrl_cell_emb']     # Control cell embeddings
    perturbations = batch['pert_emb']       # Perturbation encodings
    # ... process batch
```

### Advanced Configuration

```python
# Advanced data loading with STATE embeddings
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    embed_key="X_state",                    # Use STATE model embeddings
    output_space="gene",                    # Predict gene expression
    basal_mapping_strategy="batch",         # Map controls within batch
    n_basal_samples=3,                      # Use 3 control cells per perturbation
    barcode=True,                           # Include cell barcodes
    batch_size=256,
    num_workers=16
)
```

## 🔬 Preprocessing Pipeline

### Quality Control

```python
from virtual_cell_challenge.preprocessing import filter_on_target_knockdown
import anndata

# Load data
adata = anndata.read_h5ad("raw_data.h5ad")

# Apply quality control filtering
filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,        # 70% knockdown required
    cell_residual_expression=0.50,   # 50% knockdown per cell
    min_cells=30                     # Minimum cells per perturbation
)

print(f"Removed {adata.n_obs - filtered_adata.n_obs} low-quality cells")
```

### Complete Preprocessing Pipeline

```python
from virtual_cell_challenge.preprocessing import preprocess_perturbation_data

# Run complete preprocessing pipeline
results = preprocess_perturbation_data(
    adata_path="raw_data.h5ad",
    output_path="processed_data.h5ad",
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30
)

# Access results
print(f"Cell retention rate: {results['filtering_efficiency']['cell_retention_rate']:.1%}")
print(f"Quality report available: {len(results['quality_report'])} perturbations analyzed")
```

## ⚙️ Configuration Files

### Basic TOML Configuration

```toml
# config.toml
[datasets]
replogle = "/path/to/replogle_dataset/"
jurkat = "/path/to/jurkat_dataset/"

[training]
replogle = "train"
jurkat = "train"

[zeroshot]
"replogle.jurkat" = "test"

[fewshot]
[fewshot."replogle.rpe1"]
val = ["AARS", "ABCB1"]
test = ["NUP107", "RPUSD4"]
```

### Dataset-Specific Configurations

```toml
# For Tahoe-100M dataset
[datasets]
tahoe = "/data/tahoe-100m/"

[training]
tahoe = "train"

[zeroshot]
"tahoe.k562" = "test"
"tahoe.hela" = "val"

[fewshot]
[fewshot."tahoe.mcf7"]
val = ["TP53", "MYC", "EGFR"]
test = ["KRAS", "PIK3CA", "AKT1"]
```

## 🎯 Learning Task Examples

### Zero-Shot Learning Setup

```python
# Configuration for zero-shot learning
config = """
[zeroshot]
"dataset.jurkat" = "test"     # Hold out all Jurkat cells
"dataset.rpe1" = "val"        # Hold out all RPE1 cells
"""

# This setup trains on other cell types and tests generalization
# to completely unseen cell types
```

### Few-Shot Learning Setup

```python
# Configuration for few-shot learning
config = """
[fewshot."dataset.k562"]
val = ["GENE1", "GENE2", "GENE3"]           # 3 perturbations for validation
test = ["GENE4", "GENE5", "GENE6", "GENE7"] # 4 perturbations for testing
# All other perturbations in k562 go to training
"""

# This setup holds out specific perturbations within a cell type
```

## 📏 Evaluation and Analysis

### Perturbation Quality Analysis

```python
from virtual_cell_challenge.preprocessing import analyze_perturbation_quality

# Analyze all perturbations in dataset
quality_report = analyze_perturbation_quality(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30
)

# Display results
effective_perts = quality_report['is_effective'].sum()
total_perts = len(quality_report)
print(f"Effective perturbations: {effective_perts}/{total_perts}")

# Show top performers
top_performers = quality_report.nlargest(5, 'knockdown_percent')
print(top_performers[['perturbation', 'knockdown_percent', 'n_cells']])
```

### Data Validation

```python
from virtual_cell_challenge.preprocessing import validate_data_format

# Validate data format
validation = validate_data_format(adata)

if validation['is_valid']:
    print("✅ Data format is valid for cell-load pipeline")
else:
    print("❌ Data format issues found:")
    for error in validation['errors']:
        print(f"  • {error}")
```

## 🔧 Common Workflows

### 1. New Dataset Integration

```python
# Step 1: Validate format
validation = validate_data_format(adata)

# Step 2: Apply preprocessing
results = preprocess_perturbation_data(
    adata_path="new_dataset.h5ad",
    output_path="new_dataset_filtered.h5ad"
)

# Step 3: Create configuration
config = """
[datasets]
new_dataset = "/path/to/new_dataset_filtered.h5ad"

[training]
new_dataset = "train"
"""

# Step 4: Set up data loading
dm = PerturbationDataModule(toml_config_path="new_config.toml")
```

### 2. Cross-Dataset Analysis

```python
# Compare multiple datasets
config = """
[datasets]
replogle = "/data/replogle/"
tahoe = "/data/tahoe/"
jurkat = "/data/jurkat/"

[training]
replogle = "train"
tahoe = "train"

[zeroshot]
"jurkat.all" = "test"  # Use Jurkat as independent test set
"""

# This enables training on multiple datasets and testing generalization
```

### 3. Custom Mapping Strategies

```python
from virtual_cell_challenge.data_loading import RandomMappingStrategy, BatchMappingStrategy

# Random control mapping (default)
random_strategy = RandomMappingStrategy(
    random_state=42,
    n_basal_samples=1
)

# Batch-based control mapping
batch_strategy = BatchMappingStrategy(
    random_state=42,
    n_basal_samples=3  # Use 3 control cells per perturbation
)

# Use in data module
dm = PerturbationDataModule(
    toml_config_path="config.toml",
    basal_mapping_strategy="batch",  # or "random"
    n_basal_samples=3
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Missing required obs column: gene"

```python
# Solution: Check column names and rename if needed
adata.obs = adata.obs.rename(columns={'perturbation': 'gene'})
```

**Issue**: "No effective perturbations found"

```python
# Solution: Adjust quality control thresholds
filtered_adata = filter_on_target_knockdown(
    adata,
    residual_expression=0.50,  # More lenient threshold
    min_cells=10               # Lower minimum cell count
)
```

**Issue**: "Dataset path does not exist"

```python
# Solution: Use absolute paths in TOML config
[datasets]
dataset = "/absolute/path/to/dataset/"  # Not relative paths
```

### Performance Tips

1. **Use appropriate batch sizes**: Start with 128, increase if memory allows
2. **Optimize workers**: Set `num_workers` to number of CPU cores
3. **Use STATE embeddings**: More efficient than raw gene expression
4. **Filter low-quality data**: Apply preprocessing before training
5. **Cache processed data**: Save filtered datasets to avoid reprocessing

## 📚 Additional Resources

- **Notebooks**: See `notebooks/` for detailed examples
- **Configuration**: Check `config/` for template files
- **Documentation**: Refer to docstrings in source code
- **Tests**: Run `pytest tests/` to verify installation

## 🤝 Contributing

To contribute to the Virtual Cell Challenge toolkit:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

For questions or issues, please open a GitHub issue or contact the maintainers.
