# SensorLLM Training Pipeline - Technical Reference for AI Systems

## Executive Summary for LLM Understanding
**PROJECT TYPE**: Multimodal Machine Learning - Time Series + Large Language Model  
**TASK**: Human Activity Recognition from Wearable Sensor Data  
**ARCHITECTURE**: Frozen LLM + Frozen Time-Series Encoder + Trainable Projection + Trainable Classification Head  
**TRAINING STAGE**: Stage 2 (Task-specific fine-tuning, Stage 1 is pre-alignment)  
**DATASET**: Capture24 (accelerometer data from wrist-worn devices)  

## Critical Implementation Context

### Data Processing Chain
```
Raw Sensor Data (100Hz, 3-axis accelerometer) 
→ Downsampling (99x = ~1Hz effective rate)
→ Windowing (300 second segments = 5 minutes)
→ Label Coverage Filtering (≥50% single activity)
→ Chronos Tokenization (uniform bins)
→ LLM Token Embedding
→ Feature Fusion via Projection Layer
→ Classification (10 activity classes)
```

### Key Technical Constraints
- **Memory Optimization Required**: Uses FlashAttention monkey-patching
- **Distributed Training**: 2 GPUs with gradient accumulation (effective batch size 64)
- **Parameter Freezing Strategy**: Only projection layer + classification head trainable
- **Class Imbalance Handling**: Weighted loss function with 5% balance ratio
- **Evaluation Metric**: Macro F1-score (due to class imbalance)

## Architecture Overview

### Core Components
1. **Base LLM**: Llama-3.2-1B-Instruct model (FROZEN)
2. **Time-Series Encoder**: Chronos-T5-Large pretrained backbone (FROZEN)
3. **Projection Layer**: Multi-modal adapter (TRAINABLE) - ts_proj
4. **Classification Head**: Sequence classification layer (TRAINABLE) - score layer

### Training Stages Context
- **Stage 1**: Time-series ↔ Text modality alignment (NOT in this pipeline)
- **Stage 2**: Task-specific classification fine-tuning (THIS pipeline)
- **Key Flag**: `--only_stage2 True` means no Stage 1 pre-training available

## Implementation Analysis

### 1. Entry Point: Shell Script (`train_capture24_stage2_custom.sh`)

#### Dynamic Configuration System
```bash
# Extracts config to build consistent directory naming
CONFIG_PATH="$(dirname "$0")/config_stage2.yaml"
WINDOW_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['window_size_seconds'])")
DOWNSAMPLE_FACTOR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['downsample_factor'])")
OUTPUT_TAG="${WINDOW_SIZE}seconds_${DOWNSAMPLE_FACTOR}DS"  # Results in: "300seconds_99DS"
```

#### Critical Configuration Values (config_stage2.yaml)
```yaml
original_sampling_rate: 100     # Input data frequency
downsample_factor: 99          # 100/99 ≈ 1.01 Hz effective rate  
window_size_seconds: 300       # 5-minute activity windows
balance_ratio: 0.05           # Cap classes at 5% of max class size
min_label_fraction: 0.5       # Require 50%+ single activity per window
num_participants: 2           # Limit dataset size for testing
train_test_split: 50          # 50-50 train/test split
```

#### Infrastructure Setup
```bash
export CUDA_VISIBLE_DEVICES=0,1                    # Multi-GPU setup
master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))  # Avoid port conflicts
export PYTHONPATH="/path/to/sensorllm:${PYTHONPATH}"   # Module path
```

#### Training Execution
```bash
torchrun --nproc_per_node=2 --master_port=$master_port ../train/train_mem.py \
  # ... (extensive argument list follows)
```

### 2. Core Training Arguments (Critical for LLM Understanding)

#### Model Architecture Configuration
```bash
--model_type "SequenceClassification"          # vs "CasualLM" for text generation
--num_labels 10                               # Capture24 activity classes
--tokenize_method 'StanNormalizeUniformBins'  # Chronos tokenization approach
--pt_encoder_backbone_ckpt './chronos/chronos-t5-large'  # Pre-trained TS encoder
```

#### Data Pipeline Configuration  
```bash
--dataset "capture24"                         # Dataset identifier for internal routing
--preprocess_type "smry+trend+corr+Q"        # Feature engineering: Summary + Trend + Correlation + Q&A format
--data_path "${DATA_ROOT}/${OUTPUT_TAG}/train/capture24_train_data_stage2_${OUTPUT_TAG}.pkl"
--qa_path "${DATA_ROOT}/${OUTPUT_TAG}/train/capture24_train_qa_stage2_cls.json"
```

#### Training Hyperparameters (Optimized for this task)
```bash
--num_train_epochs 8                         # Training duration
--per_device_train_batch_size 4              # Per-GPU batch size
--gradient_accumulation_steps 8              # Effective batch: 4×2×8=64
--learning_rate 2e-3                         # High LR for projection layer training
--warmup_ratio 0.03                          # 3% warmup steps
--lr_scheduler_type cosine                    # Learning rate schedule
```

#### Parameter Freezing Strategy (Critical Implementation Detail)
```bash
--fix_llm True                               # Freeze all LLM parameters
--fix_ts_encoder True                        # Freeze time-series encoder
--fix_cls_head False                         # Train classification head
--tune_mm_mlp_adapter True                   # Train projection layer
```

#### Memory + Performance Optimizations
```bash
--gradient_checkpointing True                # Trade compute for memory
--bf16 True                                  # Mixed precision training
--use_cache False                           # Disable attention caching
--ddp_find_unused_parameters False           # DDP optimization
```

#### Evaluation + Saving Strategy
```bash
--eval_strategy 'steps'                      # Evaluate every N steps
--save_strategy 'steps'                      # Save every N steps
--eval_steps 50                             # Evaluation frequency
--save_steps 50                             # Checkpoint frequency
--metric_for_best_model f1_macro             # Optimize for macro F1
--load_best_model_at_end True               # Load best checkpoint
--save_total_limit 1                        # Keep only 1 checkpoint
```

### 3. Python Training Implementation (`train_mem.py` → `train.py`)

#### Memory Optimization (Critical for Inference)
```python
# FlashAttention monkey-patch MUST happen before model imports
from sensorllm.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
```

#### Model Loading Logic Flow
```python
# 1. Load base classification model
if model_args.model_type == "SequenceClassification":
    model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,  # Llama-3.2-1B-Instruct path
        num_labels=training_args.num_labels,  # 10 classes
        id2label=id2label,              # Class name mapping
        label2id=label2id,              # Reverse mapping
    )

# 2. Load time-series encoder
model.get_model().load_pt_encoder_backbone_checkpoint(
    model_args.pt_encoder_backbone_ckpt,  # './chronos/chronos-t5-large'
    tc=model_args.tokenize_method         # 'StanNormalizeUniformBins'
)

# 3. Configure parameter freezing
if training_args.fix_llm:
    model.requires_grad_(False)                              # Freeze all
    model.get_model().ts_proj.requires_grad_(True)          # Unfreeze projection
    if not training_args.fix_ts_encoder:
        model.get_model().pt_encoder_backbone.requires_grad_(True)  # Conditional unfreeze

if not training_args.fix_cls_head:
    model.score.requires_grad_(True)                         # Unfreeze classifier
```

#### Data Module Selection (Critical for Data Loading)
```python
# Stage 2 classification data module
data_module = make_ts_classification_data_module_stage2(
    tokenizer=tokenizer,                # LLM tokenizer
    chronos_tokenizer=chronos_tokenizer, # Time-series tokenizer  
    label2id=label2id,                  # Class mapping
    data_args=data_args                 # Data configuration
)
```

#### Trainer Selection (Handles Class Imbalance)
```python
if training_args.use_weighted_loss:
    trainer = SensorLLMWeightedCELossTrainer(  # Custom trainer for imbalanced data
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,        # F1, precision, recall per class
        **data_module                          # Includes class_weights
    )
```

## Data Architecture (Critical for Understanding)

### Input Data Structure
```
${DATA_ROOT}/stage_2_compare/${OUTPUT_TAG}/
├── train/
│   ├── capture24_train_data_stage2_${OUTPUT_TAG}.pkl      # Time-series segments
│   ├── capture24_train_labels_stage2_${OUTPUT_TAG}.pkl    # Activity labels  
│   └── capture24_train_qa_stage2_cls.json                 # Q&A format data
└── test/
    ├── capture24_test_data_stage2_${OUTPUT_TAG}.pkl
    ├── capture24_test_labels_stage2_${OUTPUT_TAG}.pkl
    └── capture24_test_qa_stage2_cls.json
```

### Data Processing Pipeline Detail
```python
# 1. Time-series tokenization (Chronos)
raw_sensor_data -> chronos_tokenizer -> uniform_bins

# 2. Text tokenization (LLM)  
activity_descriptions -> llm_tokenizer -> text_embeddings

# 3. Feature fusion (Projection layer)
time_series_features + text_embeddings -> ts_proj -> fused_representation

# 4. Classification (Classification head)
fused_representation -> score_layer -> activity_logits
```

### Preprocessing Components (`--preprocess_type "smry+trend+corr+Q"`)
- **smry**: Statistical summaries (mean, std, min, max, etc.)
- **trend**: Temporal trend analysis (slope, direction)
- **corr**: Cross-axis correlation features
- **Q**: Question-answering format conversion for LLM training

## Critical Technical Decisions (For LLM Architecture Understanding)

### 1. Why Multi-Stage Training?
- **Stage 1**: Learns time-series ↔ text alignment (expensive, one-time)
- **Stage 2**: Task-specific adaptation (efficient, task-specific)
- **Benefit**: Reuse Stage 1 alignment for multiple downstream tasks

### 2. Why Freeze Most Parameters?
- **LLM (Frozen)**: Preserve language understanding capabilities
- **TS Encoder (Frozen)**: Preserve temporal pattern recognition
- **Projection (Trainable)**: Learn modality alignment for this task
- **Classifier (Trainable)**: Learn task-specific decision boundaries

### 3. Why Weighted Loss?
- **Problem**: Activity class imbalance (sleep >> sports)
- **Solution**: Weight classes inversely proportional to frequency
- **Implementation**: Custom `SensorLLMWeightedCELossTrainer`

### 4. Why High Learning Rate (2e-3)?
- Most parameters frozen → small effective parameter space
- Trainable layers need strong updates for rapid adaptation
- Projection layer requires learning new time-series → text mappings

## Output Structure and Monitoring

### Training Outputs
```
outputs/SensorLLM_train_stage2/{FINETUNE_NAME}/
├── checkpoint-{step}/                    # Model checkpoints
│   ├── config.json                      # Model configuration
│   ├── model.safetensors                # Model weights
│   ├── optimizer.pt                     # Optimizer state
│   ├── scheduler.pt                     # Scheduler state
│   └── trainer_state.json               # Training state
├── logs/
│   ├── training_{TIMESTAMP}.log         # Training progress
│   ├── error_{TIMESTAMP}.log            # Error tracking
│   └── events.out.tfevents.*           # TensorBoard logs
└── runs/                                # TensorBoard visualization
```

### Key Metrics Monitored
```python
# Primary metrics (macro-averaged for imbalanced data)
- f1_macro: Macro-averaged F1 score (optimization target)
- precision_macro: Macro-averaged precision  
- recall_macro: Macro-averaged recall
- accuracy: Overall classification accuracy

# Per-class metrics (10 classes)
- f1_class_{i}: F1 for each activity class
- precision_class_{i}: Precision for each activity class  
- recall_class_{i}: Recall for each activity class
```

## Execution Flow for LLM Understanding

### 1. Prerequisites
```bash
# Environment setup
conda activate sensorllm-flash-attn
export PYTHONPATH="/path/to/sensorllm:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1

# Required model checkpoints
- Llama-3.2-1B-Instruct model weights
- Chronos-T5-Large time-series encoder
- Processed Capture24 dataset (pickle + JSON files)
```

### 2. Execution Command
```bash
bash train_capture24_stage2_custom.sh [OPTIONAL_FINETUNE_NAME]
```

### 3. Training Process
1. **Config Loading**: Extract parameters from YAML
2. **Model Initialization**: Load LLM + TS encoder + setup freezing
3. **Data Loading**: Create data modules with class weights
4. **Trainer Setup**: Configure weighted loss trainer
5. **Training Loop**: 8 epochs with step-based evaluation/saving
6. **Best Model Loading**: Load checkpoint with highest macro F1
7. **Final Evaluation**: Comprehensive metrics on test set

### 4. Expected Runtime
- **Setup**: ~2-3 minutes (model loading)
- **Training**: ~2-3 hours (8 epochs, 2 participants, 2 GPUs)
- **Memory Usage**: ~20-24GB VRAM total (dual GPU)

## Critical Implementation Notes for Other LLMs

### 1. Model Architecture Dependencies
- **SensorLLMStage2LlamaForSequenceClassification**: Custom model class
- **Chronos tokenizer**: Specific time-series tokenization approach
- **Custom trainers**: Handle weighted loss and multimodal data

### 2. Data Format Requirements
- **Time-series**: Pickle files with numpy arrays (segments × timesteps × features)
- **Labels**: Dictionary format with activity names and metadata
- **Q&A**: JSON format for LLM training compatibility

### 3. Configuration Coupling
- Output paths depend on config parameters: `{window_size}seconds_{downsample}DS`
- Model selection depends on dataset: Uses capture24-specific id2label mapping
- Training arguments are task-optimized: Not general-purpose hyperparameters

### 4. Extension Points for Other LLMs
- **Dataset**: Change `--dataset` and provide corresponding id2label mapping
- **Model Size**: Modify model_name_or_path for different LLM sizes
- **TS Encoder**: Change pt_encoder_backbone_ckpt for different time-series encoders
- **Tokenization**: Modify tokenize_method for different approaches

This pipeline represents a production-ready multimodal learning system specifically optimized for human activity recognition from wearable sensor data, with careful attention to class imbalance, memory efficiency, and distributed training. 