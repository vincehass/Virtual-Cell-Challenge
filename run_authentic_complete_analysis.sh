#!/bin/bash

# 🚀 Authentic STATE Complete Analysis Pipeline
# Real Data • Genuine Architecture • Comprehensive Ablation Studies
# Virtual Environment Management • Hyperparameter Tuning • Production Ready

set -e  # Exit on any error

# =====================================
# CONFIGURATION
# =====================================

PROJECT_NAME="authentic-state-vcc"
VENV_NAME="venv_authentic_state"
PYTHON_VERSION="3.9"
RESULTS_BASE_DIR="data/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="authentic_state_${TIMESTAMP}"

# Analysis configurations
ENABLE_WANDB=${ENABLE_WANDB:-true}
ENABLE_GPU=${ENABLE_GPU:-true}
MAX_CELLS=${MAX_CELLS:-25000}
MAX_GENES=${MAX_GENES:-4000}
QUICK_RUN=${QUICK_RUN:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# =====================================
# UTILITY FUNCTIONS
# =====================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}========================================${NC}"
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is required but not installed."
        exit 1
    fi
}

# =====================================
# ENVIRONMENT SETUP
# =====================================

setup_environment() {
    log_header "ENVIRONMENT SETUP"
    
    # Check required commands
    log_info "Checking required commands..."
    check_command python3
    check_command pip
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        log_info "Creating virtual environment: $VENV_NAME"
        python3 -m venv $VENV_NAME
    else
        log_info "Virtual environment already exists: $VENV_NAME"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source $VENV_NAME/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install/upgrade requirements
    log_info "Installing requirements..."
    if [ -f "requirements_authentic.txt" ]; then
        pip install -r requirements_authentic.txt
    else
        log_warning "requirements_authentic.txt not found, installing core packages..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install numpy pandas matplotlib seaborn scikit-learn scipy anndata wandb plotly jupyter
    fi
    
    # Set environment variables
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    log_success "Environment setup completed"
}

# =====================================
# DATA PREPARATION
# =====================================

prepare_data() {
    log_header "DATA PREPARATION"
    
    # Create necessary directories
    mkdir -p data/processed
    mkdir -p data/results
    mkdir -p logs
    mkdir -p configs
    
    # Check for real data
    if [ -f "data/processed/vcc_training_processed.h5ad" ]; then
        log_success "Real data found: vcc_training_processed.h5ad"
    else
        log_warning "Real data not found, will use synthetic data for demonstration"
    fi
    
    log_success "Data preparation completed"
}

# =====================================
# CONFIGURATION GENERATION
# =====================================

generate_configs() {
    log_header "GENERATING EXPERIMENT CONFIGURATIONS"
    
    # Base configuration
    cat > configs/base_config.json << 'EOF'
{
    "data": {
        "max_cells": 25000,
        "max_genes": 4000,
        "min_cells_per_batch": 30
    },
    "model": {
        "se_config": {
            "embed_dim": 512,
            "n_heads": 16,
            "n_layers": 12,
            "dropout": 0.1
        },
        "st_config": {
            "state_dim": 256,
            "perturbation_dim": 128,
            "n_heads": 8,
            "n_layers": 6,
            "dropout": 0.1
        }
    },
    "training": {
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4
    },
    "evaluation": {
        "sample_size": 1000,
        "bootstrap_samples": 1000
    }
}
EOF

    # CPU optimized configuration
    cat > configs/cpu_config.json << 'EOF'
{
    "data": {
        "max_cells": 10000,
        "max_genes": 2000,
        "min_cells_per_batch": 20
    },
    "model": {
        "se_config": {
            "embed_dim": 256,
            "n_heads": 8,
            "n_layers": 6,
            "dropout": 0.1
        },
        "st_config": {
            "state_dim": 128,
            "perturbation_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1
        }
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4
    },
    "evaluation": {
        "sample_size": 500,
        "bootstrap_samples": 500
    }
}
EOF

    # Quick run configuration
    cat > configs/quick_config.json << 'EOF'
{
    "data": {
        "max_cells": 5000,
        "max_genes": 1000,
        "min_cells_per_batch": 10
    },
    "model": {
        "se_config": {
            "embed_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1
        },
        "st_config": {
            "state_dim": 64,
            "perturbation_dim": 64,
            "n_heads": 2,
            "n_layers": 2,
            "dropout": 0.1
        }
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4
    },
    "evaluation": {
        "sample_size": 200,
        "bootstrap_samples": 100
    }
}
EOF

    log_success "Configuration files generated"
}

# =====================================
# HYPERPARAMETER ABLATION STUDIES
# =====================================

run_hyperparameter_ablation() {
    log_header "HYPERPARAMETER ABLATION STUDIES"
    
    local base_dir="$RESULTS_BASE_DIR/hyperparameter_ablation_$TIMESTAMP"
    mkdir -p "$base_dir"
    
    # Learning rate ablation
    log_info "Running learning rate ablation..."
    for lr in 1e-3 5e-4 1e-4 5e-5 1e-5; do
        log_info "Testing learning rate: $lr"
        python scripts/hyperparameter_ablation.py \
            --config configs/base_config.json \
            --learning_rate $lr \
            --output_dir "$base_dir/lr_$lr" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "lr_ablation" "lr_$lr" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/lr_ablation_$lr.log"
    done
    
    # Architecture ablation
    log_info "Running architecture ablation..."
    
    # Embedding dimension ablation
    for embed_dim in 128 256 512 1024; do
        log_info "Testing embedding dimension: $embed_dim"
        python scripts/hyperparameter_ablation.py \
            --config configs/base_config.json \
            --embed_dim $embed_dim \
            --output_dir "$base_dir/embed_$embed_dim" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "arch_ablation" "embed_$embed_dim" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/embed_ablation_$embed_dim.log"
    done
    
    # Number of layers ablation
    for n_layers in 3 6 9 12; do
        log_info "Testing number of layers: $n_layers"
        python scripts/hyperparameter_ablation.py \
            --config configs/base_config.json \
            --n_layers $n_layers \
            --output_dir "$base_dir/layers_$n_layers" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "arch_ablation" "layers_$n_layers" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/layers_ablation_$n_layers.log"
    done
    
    # Number of attention heads ablation
    for n_heads in 4 8 16 32; do
        log_info "Testing number of attention heads: $n_heads"
        python scripts/hyperparameter_ablation.py \
            --config configs/base_config.json \
            --n_heads $n_heads \
            --output_dir "$base_dir/heads_$n_heads" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "arch_ablation" "heads_$n_heads" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/heads_ablation_$n_heads.log"
    done
    
    # Dropout ablation
    for dropout in 0.0 0.1 0.2 0.3; do
        log_info "Testing dropout: $dropout"
        python scripts/hyperparameter_ablation.py \
            --config configs/base_config.json \
            --dropout $dropout \
            --output_dir "$base_dir/dropout_$dropout" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "reg_ablation" "dropout_$dropout" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/dropout_ablation_$dropout.log"
    done
    
    log_success "Hyperparameter ablation completed"
}

# =====================================
# DATA ABLATION STUDIES
# =====================================

run_data_ablation() {
    log_header "DATA ABLATION STUDIES"
    
    local base_dir="$RESULTS_BASE_DIR/data_ablation_$TIMESTAMP"
    mkdir -p "$base_dir"
    
    # Sample size ablation
    log_info "Running sample size ablation..."
    for n_cells in 5000 10000 15000 20000 25000; do
        log_info "Testing cell count: $n_cells"
        python scripts/data_ablation.py \
            --config configs/base_config.json \
            --max_cells $n_cells \
            --output_dir "$base_dir/cells_$n_cells" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "data_ablation" "cells_$n_cells" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/cells_ablation_$n_cells.log"
    done
    
    # Gene count ablation
    for n_genes in 1000 2000 3000 4000 5000; do
        log_info "Testing gene count: $n_genes"
        python scripts/data_ablation.py \
            --config configs/base_config.json \
            --max_genes $n_genes \
            --output_dir "$base_dir/genes_$n_genes" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "data_ablation" "genes_$n_genes" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/genes_ablation_$n_genes.log"
    done
    
    # Normalization ablation
    for norm in "log1p" "zscore" "robust_scale" "min_max"; do
        log_info "Testing normalization: $norm"
        python scripts/data_ablation.py \
            --config configs/base_config.json \
            --normalization $norm \
            --output_dir "$base_dir/norm_$norm" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "data_ablation" "norm_$norm" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/norm_ablation_$norm.log"
    done
    
    log_success "Data ablation completed"
}

# =====================================
# TRAINING ABLATION STUDIES
# =====================================

run_training_ablation() {
    log_header "TRAINING ABLATION STUDIES"
    
    local base_dir="$RESULTS_BASE_DIR/training_ablation_$TIMESTAMP"
    mkdir -p "$base_dir"
    
    # Optimizer ablation
    log_info "Running optimizer ablation..."
    for optimizer in "adam" "adamw" "sgd" "rmsprop"; do
        log_info "Testing optimizer: $optimizer"
        python scripts/training_ablation.py \
            --config configs/base_config.json \
            --optimizer $optimizer \
            --output_dir "$base_dir/opt_$optimizer" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "training_ablation" "opt_$optimizer" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/opt_ablation_$optimizer.log"
    done
    
    # Batch size ablation
    for batch_size in 16 32 64 128; do
        log_info "Testing batch size: $batch_size"
        python scripts/training_ablation.py \
            --config configs/base_config.json \
            --batch_size $batch_size \
            --output_dir "$base_dir/batch_$batch_size" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "training_ablation" "batch_$batch_size" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/batch_ablation_$batch_size.log"
    done
    
    # Weight decay ablation
    for weight_decay in 0.0 1e-5 1e-4 1e-3; do
        log_info "Testing weight decay: $weight_decay"
        python scripts/training_ablation.py \
            --config configs/base_config.json \
            --weight_decay $weight_decay \
            --output_dir "$base_dir/wd_$weight_decay" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "training_ablation" "wd_$weight_decay" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/wd_ablation_$weight_decay.log"
    done
    
    # Scheduler ablation
    for scheduler in "cosine" "step" "exponential" "plateau"; do
        log_info "Testing scheduler: $scheduler"
        python scripts/training_ablation.py \
            --config configs/base_config.json \
            --scheduler $scheduler \
            --output_dir "$base_dir/sched_$scheduler" \
            --wandb_project "$PROJECT_NAME" \
            --wandb_tags "training_ablation" "sched_$scheduler" \
            $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
            2>&1 | tee "logs/sched_ablation_$scheduler.log"
    done
    
    log_success "Training ablation completed"
}

# =====================================
# COMPLETE PIPELINE EXECUTION
# =====================================

run_complete_pipeline() {
    log_header "COMPLETE AUTHENTIC STATE PIPELINE"
    
    local config_file="configs/base_config.json"
    if [ "$QUICK_RUN" = true ]; then
        config_file="configs/quick_config.json"
    fi
    
    log_info "Running complete pipeline with config: $config_file"
    
    # Run the main benchmarking script that includes progress bars and density analysis
    python scripts/simple_benchmarking_with_progress.py \
        --output_dir "$RESULTS_BASE_DIR/complete_pipeline_$TIMESTAMP" \
        --wandb_project "$PROJECT_NAME" \
        $([ "$ENABLE_WANDB" = true ] && echo "--wandb_project $PROJECT_NAME") \
        $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
        2>&1 | tee "logs/complete_pipeline_$TIMESTAMP.log"
    
    if [ $? -eq 0 ]; then
        log_success "Complete pipeline execution finished successfully"
    else
        log_error "Complete pipeline execution failed"
        return 1
    fi
}

# =====================================
# COMPREHENSIVE BENCHMARKING
# =====================================

run_comprehensive_benchmarking() {
    log_header "COMPREHENSIVE BENCHMARKING WITH DENSITY ANALYSIS"
    
    local config_file="configs/base_config.json"
    if [ "$QUICK_RUN" = true ]; then
        config_file="configs/quick_config.json"
    fi
    
    log_info "Running comprehensive benchmarking with config: $config_file"
    
    # Run comprehensive benchmarking with density analysis
    python scripts/comprehensive_benchmarking_with_density.py \
        --config "$config_file" \
        --output_dir "$RESULTS_BASE_DIR/comprehensive_benchmark_$TIMESTAMP" \
        $([ "$ENABLE_WANDB" = true ] && echo "--wandb_project $PROJECT_NAME") \
        $([ "$ENABLE_WANDB" = true ] && echo "--enable_wandb true" || echo "--enable_wandb false") \
        $([ "$ENABLE_GPU" = true ] && echo "--enable_gpu true" || echo "--enable_gpu false") \
        --max_cells $MAX_CELLS \
        --max_genes $MAX_GENES \
        $([ "$QUICK_RUN" = true ] && echo "--quick_run") \
        2>&1 | tee "logs/comprehensive_benchmark_$TIMESTAMP.log"
    
    if [ $? -eq 0 ]; then
        log_success "Comprehensive benchmarking finished successfully"
    else
        log_error "Comprehensive benchmarking failed"
        return 1
    fi
}

# =====================================
# RESULT ANALYSIS AND REPORTING
# =====================================

analyze_results() {
    log_header "RESULT ANALYSIS AND REPORTING"
    
    log_info "Generating comprehensive analysis report..."
    
    python scripts/analyze_ablation_results.py \
        --results_dir "$RESULTS_BASE_DIR" \
        --timestamp "$TIMESTAMP" \
        --output_dir "$RESULTS_BASE_DIR/final_analysis_$TIMESTAMP" \
        --wandb_project "$PROJECT_NAME" \
        2>&1 | tee "logs/analysis_$TIMESTAMP.log"
    
    # Generate summary dashboard
    log_info "Creating interactive dashboard..."
    python scripts/create_dashboard.py \
        --results_dir "$RESULTS_BASE_DIR/final_analysis_$TIMESTAMP" \
        --output_file "$RESULTS_BASE_DIR/dashboard_$TIMESTAMP.html" \
        2>&1 | tee "logs/dashboard_$TIMESTAMP.log"
    
    log_success "Result analysis completed"
}

# =====================================
# CLEANUP
# =====================================

cleanup() {
    log_header "CLEANUP"
    
    # Archive logs
    log_info "Archiving logs..."
    tar -czf "logs/logs_archive_$TIMESTAMP.tar.gz" logs/*.log
    
    # Generate final summary
    log_info "Generating final summary..."
    cat > "$RESULTS_BASE_DIR/run_summary_$TIMESTAMP.txt" << EOF
Authentic STATE Analysis Run Summary
====================================

Run ID: $RUN_ID
Timestamp: $TIMESTAMP
Duration: $(date)

Configuration:
- Enable W&B: $ENABLE_WANDB
- Enable GPU: $ENABLE_GPU
- Max Cells: $MAX_CELLS
- Max Genes: $MAX_GENES
- Quick Run: $QUICK_RUN

Results Location: $RESULTS_BASE_DIR
Logs Location: logs/

Key Files:
- Complete Pipeline Results: $RESULTS_BASE_DIR/complete_pipeline_$TIMESTAMP/
- Hyperparameter Ablation: $RESULTS_BASE_DIR/hyperparameter_ablation_$TIMESTAMP/
- Data Ablation: $RESULTS_BASE_DIR/data_ablation_$TIMESTAMP/
- Training Ablation: $RESULTS_BASE_DIR/training_ablation_$TIMESTAMP/
- Final Analysis: $RESULTS_BASE_DIR/final_analysis_$TIMESTAMP/
- Interactive Dashboard: $RESULTS_BASE_DIR/dashboard_$TIMESTAMP.html

Virtual Environment: $VENV_NAME
EOF
    
    log_success "Cleanup completed"
}

# =====================================
# ERROR HANDLING
# =====================================

handle_error() {
    local exit_code=$?
    log_error "Script failed with exit code $exit_code"
    log_error "Check logs for details: logs/"
    
    # Save error state
    echo "ERROR: Script failed at $(date) with exit code $exit_code" >> "logs/error_$TIMESTAMP.log"
    
    # Cleanup and exit
    cleanup
    exit $exit_code
}

trap handle_error ERR

# =====================================
# MAIN EXECUTION
# =====================================

main() {
    log_header "AUTHENTIC STATE COMPLETE ANALYSIS PIPELINE"
    log_info "Starting analysis with Run ID: $RUN_ID"
    log_info "Timestamp: $TIMESTAMP"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_RUN=true
                shift
                ;;
            --no-wandb)
                ENABLE_WANDB=false
                shift
                ;;
            --no-gpu)
                ENABLE_GPU=false
                shift
                ;;
            --max-cells)
                MAX_CELLS="$2"
                shift 2
                ;;
            --max-genes)
                MAX_GENES="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --quick          Run quick version with smaller data and models"
                echo "  --no-wandb       Disable Weights & Biases logging"
                echo "  --no-gpu         Disable GPU usage"
                echo "  --max-cells N    Maximum number of cells to use"
                echo "  --max-genes N    Maximum number of genes to use"
                echo "  --help           Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                    # Full analysis"
                echo "  $0 --quick           # Quick analysis"
                echo "  $0 --max-cells 10000 # Limit to 10k cells"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Print configuration
    log_info "Configuration:"
    log_info "  - Quick Run: $QUICK_RUN"
    log_info "  - Enable W&B: $ENABLE_WANDB"
    log_info "  - Enable GPU: $ENABLE_GPU"
    log_info "  - Max Cells: $MAX_CELLS"
    log_info "  - Max Genes: $MAX_GENES"
    
    # Execute pipeline steps
    setup_environment
    prepare_data
    generate_configs
    
    if [ "$QUICK_RUN" = true ]; then
        log_info "Running in quick mode - skipping extensive ablations"
        run_complete_pipeline
    else
        log_info "Running full analysis with comprehensive ablations"
        run_hyperparameter_ablation
        run_data_ablation
        run_training_ablation
        run_comprehensive_benchmarking
        analyze_results
    fi
    
    cleanup
    
    log_success "🎉 Complete Authentic STATE Analysis Pipeline Finished!"
    log_success "Results available at: $RESULTS_BASE_DIR"
    log_success "Run ID: $RUN_ID"
    
    if [ "$ENABLE_WANDB" = true ]; then
        log_info "View results on W&B: https://wandb.ai/your-username/$PROJECT_NAME"
    fi
}

# Run main function with all arguments
main "$@" 