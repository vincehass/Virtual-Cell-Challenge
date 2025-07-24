# 🚀 Authentic STATE Implementation Guide

Complete production-ready implementation with virtual environment management, comprehensive ablation studies, and real data analysis.

## 🎯 Quick Start

### Option 1: Full Analysis (Recommended)

```bash
# Run complete analysis with all ablation studies
./run_authentic_complete_analysis.sh
```

### Option 2: Quick Analysis

```bash
# Run quick version for testing/development
./run_authentic_complete_analysis.sh --quick
```

### Option 3: Custom Configuration

```bash
# Custom cell and gene limits
./run_authentic_complete_analysis.sh --max-cells 10000 --max-genes 2000

# Disable W&B logging
./run_authentic_complete_analysis.sh --no-wandb

# CPU-only mode
./run_authentic_complete_analysis.sh --no-gpu
```

## 📋 Command Line Options

| Option          | Description                                    | Default |
| --------------- | ---------------------------------------------- | ------- |
| `--quick`       | Run quick version with smaller data and models | false   |
| `--no-wandb`    | Disable Weights & Biases logging               | false   |
| `--no-gpu`      | Disable GPU usage                              | false   |
| `--max-cells N` | Maximum number of cells to use                 | 25000   |
| `--max-genes N` | Maximum number of genes to use                 | 4000    |
| `--help`        | Show help message                              | -       |

## 🏗️ What the Script Does

### 1. Environment Setup

- ✅ Creates virtual environment (`venv_authentic_state`)
- ✅ Installs all required packages from `requirements_authentic.txt`
- ✅ Sets necessary environment variables
- ✅ Validates system requirements

### 2. Data Preparation

- ✅ Creates necessary directories
- ✅ Checks for real single-cell data
- ✅ Falls back to synthetic data if needed
- ✅ Implements proper batch stratification

### 3. Configuration Generation

- ✅ Base configuration (full performance)
- ✅ CPU configuration (optimized for CPU)
- ✅ Quick configuration (fast testing)

### 4. Comprehensive Ablation Studies

#### Hyperparameter Ablation

- **Learning Rate**: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
- **Architecture**: Embedding dimensions, layers, attention heads
- **Regularization**: Dropout, weight decay

#### Data Ablation

- **Sample Size**: 5K, 10K, 15K, 20K, 25K cells
- **Gene Count**: 1K, 2K, 3K, 4K, 5K genes
- **Normalization**: log1p, z-score, robust_scale, min_max

#### Training Ablation

- **Optimizers**: Adam, AdamW, SGD, RMSprop
- **Batch Sizes**: 16, 32, 64, 128
- **Schedulers**: Cosine, Step, Exponential, Plateau

### 5. Complete Pipeline Execution

- ✅ Authentic STATE model training (SE + ST)
- ✅ CPU and GPU optimized versions
- ✅ Real evaluation with density metrics
- ✅ Statistical analysis with confidence intervals

### 6. Result Analysis and Reporting

- ✅ Comprehensive analysis reports
- ✅ Interactive dashboards
- ✅ Density visualizations
- ✅ Best configuration recommendations

## 📁 Output Structure

After running the script, you'll find results in:

```
data/results/
├── authentic_state_YYYYMMDD_HHMMSS/           # Timestamp-based directory
├── hyperparameter_ablation_YYYYMMDD_HHMMSS/   # Hyperparameter studies
├── data_ablation_YYYYMMDD_HHMMSS/             # Data ablation studies
├── training_ablation_YYYYMMDD_HHMMSS/         # Training ablation studies
├── complete_pipeline_YYYYMMDD_HHMMSS/         # Complete pipeline results
├── final_analysis_YYYYMMDD_HHMMSS/            # Final analysis and reports
├── dashboard_YYYYMMDD_HHMMSS.html             # Interactive dashboard
└── run_summary_YYYYMMDD_HHMMSS.txt            # Run summary

logs/                                           # All log files
├── lr_ablation_*.log                          # Learning rate ablation logs
├── embed_ablation_*.log                       # Architecture ablation logs
├── complete_pipeline_*.log                    # Pipeline execution logs
└── logs_archive_YYYYMMDD_HHMMSS.tar.gz       # Archived logs
```

## 🔬 Individual Script Usage

### Hyperparameter Ablation

```bash
# Test specific learning rate
python scripts/hyperparameter_ablation.py \
    --config configs/base_config.json \
    --learning_rate 1e-4 \
    --output_dir results/lr_test \
    --wandb_project "authentic-state-vcc"

# Test specific architecture
python scripts/hyperparameter_ablation.py \
    --config configs/base_config.json \
    --embed_dim 512 \
    --n_heads 16 \
    --n_layers 12 \
    --output_dir results/arch_test
```

### Data Ablation

```bash
python scripts/data_ablation.py \
    --config configs/base_config.json \
    --max_cells 10000 \
    --normalization "log1p" \
    --output_dir results/data_test
```

### Training Ablation

```bash
python scripts/training_ablation.py \
    --config configs/base_config.json \
    --optimizer "adamw" \
    --batch_size 64 \
    --output_dir results/training_test
```

## 🧬 Architecture Details

### State Embedding (SE) Model

- **Input**: Gene expression vectors
- **Architecture**: Multi-layer bidirectional transformer
- **Features**:
  - Gene positional embeddings
  - Multi-head self-attention
  - Layer normalization
  - Residual connections
- **Output**: Cell state embeddings

### State Transition (ST) Model

- **Input**: Baseline state + perturbation vector
- **Architecture**: Transformer for state transitions
- **Features**:
  - Perturbation encoding
  - Temporal modeling
  - Uncertainty estimation
  - Residual connections
- **Output**: Predicted perturbed state

### Complete STATE Model

- **SE**: Converts gene expression to state embeddings
- **ST**: Predicts state transitions from perturbations
- **Integration**: SE → ST → Expression prediction
- **Training**: End-to-end with real data

## 📊 Evaluation Metrics

### Perturbation Discrimination (PDisc)

- **Method**: Manhattan distance ranking
- **Bootstrap**: 1000 samples for confidence intervals
- **Per-perturbation**: Individual analysis
- **Density**: Complete distribution visualization

### Differential Expression (DE)

- **Statistical Tests**: Mann-Whitney U
- **Effect Sizes**: Cohen's d
- **Correlations**: Pearson + Spearman
- **Performance**: Precision, Recall, F1

### Expression Heterogeneity

- **Coefficient of Variation**: Gene-level variability
- **Cell-to-Cell Distances**: Population structure
- **Statistical Tests**: Kolmogorov-Smirnov
- **Density Analysis**: Complete distributions

## 🛠️ Requirements

### System Requirements

- **Python**: 3.8+ (3.9 recommended)
- **Memory**: 8GB+ RAM (16GB+ for full analysis)
- **Storage**: 5GB+ free space
- **OS**: macOS, Linux (Windows with WSL)

### Optional

- **GPU**: NVIDIA GPU with CUDA for accelerated training
- **W&B Account**: For experiment tracking and visualization

## 🚨 Troubleshooting

### Common Issues

#### Virtual Environment Issues

```bash
# If venv creation fails
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv_authentic_state
```

#### OpenMP Library Conflicts

```bash
# Already handled in script, but if you see errors:
export KMP_DUPLICATE_LIB_OK=TRUE
```

#### Memory Issues

```bash
# For limited memory systems
./run_authentic_complete_analysis.sh --max-cells 5000 --max-genes 1000 --quick
```

#### W&B Authentication

```bash
# Login to W&B (optional)
wandb login
# Or disable W&B
./run_authentic_complete_analysis.sh --no-wandb
```

### Data Issues

#### No Real Data Found

- ✅ Script automatically creates synthetic data
- ✅ Realistic single-cell properties
- ✅ Meaningful perturbation effects

#### Out of Memory

- ✅ Reduce `--max-cells` and `--max-genes`
- ✅ Use `--quick` flag
- ✅ Close other applications

## 📈 Performance Expectations

### Quick Run (`--quick`)

- **Time**: 15-30 minutes
- **Data**: 5K cells × 1K genes
- **Models**: Smaller architectures
- **Ablations**: Skipped

### Full Analysis

- **Time**: 4-8 hours
- **Data**: 25K cells × 4K genes
- **Models**: Full architectures
- **Ablations**: Complete

### Expected Results

- **PDisc Score**: 0.3-0.8 (depending on data quality)
- **DE Correlation**: 0.2-0.7 (realistic for challenging task)
- **Training Loss**: Convergence within 100-200 epochs

## 🎯 Best Practices

### For Research

1. **Use real data** when available
2. **Run full analysis** for publication-quality results
3. **Enable W&B logging** for experiment tracking
4. **Check confidence intervals** for statistical significance

### For Development

1. **Use `--quick` flag** for fast iteration
2. **Start with small data** (`--max-cells 5000`)
3. **Disable W&B** (`--no-wandb`) for faster runs
4. **Monitor logs** for debugging

### For Production

1. **Use virtual environment** (automatic)
2. **Archive logs** (automatic)
3. **Save configurations** (automatic)
4. **Reproducible runs** (timestamped)

## 🏆 Success Criteria

### Technical Success

- ✅ Models train without errors
- ✅ Evaluation metrics computed
- ✅ Reports generated successfully
- ✅ All logs archived

### Scientific Success

- ✅ Realistic performance ranges
- ✅ Meaningful confidence intervals
- ✅ Density distributions reveal patterns
- ✅ Ablation studies identify optimal configs

### Reproduction Success

- ✅ Virtual environment isolates dependencies
- ✅ Configuration files enable reproduction
- ✅ Logs provide complete audit trail
- ✅ Results are timestamped and archived

---

## 🚀 Ready to Start?

```bash
# Full authentic STATE analysis
./run_authentic_complete_analysis.sh

# Quick test run
./run_authentic_complete_analysis.sh --quick

# Check help
./run_authentic_complete_analysis.sh --help
```

**Note**: First run will create virtual environment and install packages (~5-10 minutes). Subsequent runs will be faster.

For questions or issues, check the logs in `logs/` directory or refer to the troubleshooting section above.
