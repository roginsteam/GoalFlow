# Trajectory-Supervised Goal Learning System

This implementation replaces the reliance on fixed dictionaries with a dynamic query-based goal learning system that integrates seamlessly with flow matching-based trajectory generation.

## Overview

The system consists of three main components:

### 1. Goal Decoder (`goal_decoder.py`)
**Dynamic Query-Based Goal Learning**

- **Learnable Queries**: Replaces fixed vocabulary with 128 learnable goal queries
- **Multi-Head Prediction**:
  - Position head: Predicts (x, y, heading) for each query
  - Confidence head: Predicts query confidence scores
  - Imitation head: Learns from ground truth trajectories
  - DAC head: Predicts drivable area compliance

**Key Features:**
```python
from navsim.agents.goalflow.goal_decoder import GoalDecoder

decoder = GoalDecoder(
    d_model=256,
    num_queries=128,  # Learnable goal queries
    num_decoder_layers=3,
)

# Forward pass
goal_outputs = decoder(scene_features, status_encoding)
# Returns: goal_positions, goal_confidences, imitation_scores, dac_scores, goal_features

# Top-K goal selection
top_k_goals, top_k_features, indices = decoder.select_top_k_goals(
    goal_outputs, 
    k=8,
    im_weight=0.1,
    dac_weight=3.0
)
```

**Loss Functions:**
- `goal_imitation_loss`: Supervises position prediction using soft targets based on distance to GT
- `goal_dac_loss`: Binary classification for drivable area compliance
- `goal_confidence_loss`: Supervises confidence scores for queries near GT

### 2. Multi-Modal DiT Decoder (`multimodal_dit_decoder.py`)
**Advanced Trajectory Generation with CFG Support**

- **Joint Attention**: Attends across agents, lanes, trajectories, and goals simultaneously
- **CFG (Classifier-Free Guidance)**: Enables controllable trajectory generation
- **Rotary Position Encoding**: Temporal awareness for trajectory tokens
- **Type Embeddings**: Distinguishes between different modalities

**Key Features:**
```python
from navsim.agents.goalflow.multimodal_dit_decoder import MultiModalDiTDecoder

decoder = MultiModalDiTDecoder(
    d_model=256,
    num_layers=8,
    trajectory_len=12,
    cfg_dropout_prob=0.1,  # CFG training
)

# Conditional generation
output = decoder(
    noisy_trajectory=noisy_traj,
    timestep=t,
    scene_features=scene_features,
    goal_features=goal_features,
)

# With CFG (inference)
output_guided = decoder.forward_with_cfg(
    noisy_trajectory=noisy_traj,
    timestep=t,
    scene_features=scene_features,
    goal_features=goal_features,
    guidance_scale=1.5,  # >1.0 for stronger goal conditioning
)
```

**CFG Training Strategy:**
During training, randomly applies three modes:
1. Unconditional (force_dropout=True): 33%
2. Conditional (with goals): 33%
3. Goal-dropout (without goals, for CFG): 33%

### 3. Unified Model (`unified_goalflow_model.py`)
**End-to-End Integration**

Integrates all components into a unified pipeline:
1. **Scene Encoding**: V99 backbone + BEV features
2. **Goal Learning**: Dynamic query-based goal prediction
3. **Trajectory Generation**: Flow matching with goal conditioning

**Key Features:**
```python
from navsim.agents.goalflow.unified_goalflow_model import UnifiedGoalFlowModel
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig

config = GoalFlowConfig()
config.num_goal_queries = 128
config.dit_num_layers = 8
config.cfg_scale = 1.5

model = UnifiedGoalFlowModel(config)

# Training
config.training = True
outputs = model(features, targets)
# Returns: predictions with goal_positions, goal_features, trajectory, etc.

# Inference
config.training = False
outputs = model(features, targets)
# Returns: predicted trajectory with top-K goal selection and CFG
```

**Training Modes:**
- **Perception Only** (`only_perception=True`): Train scene encoder and agent detector
- **Goal Learning**: Train goal decoder with trajectory supervision
- **Trajectory Generation**: Train flow matching with goal conditioning
- **End-to-End**: Joint training of all components

**Inference Pipeline:**
1. Encode scene with BEV features
2. Predict goal positions and scores with goal decoder
3. Select top-K goals using combined score
4. Generate trajectories with goal conditioning and CFG
5. Select best trajectory (mean or closest to goal)

### 4. Unified Agent (`unified_goalflow_agent.py`)
**PyTorch Lightning Agent**

Wraps the unified model for training and evaluation:

```python
from navsim.agents.goalflow.unified_goalflow_agent import UnifiedGoalFlowAgent

agent = UnifiedGoalFlowAgent(
    config=config,
    lr=1e-4,
    step_size=20,
    gamma=0.8,
    checkpoint_path="path/to/checkpoint.ckpt",
)
```

### 5. Unified Loss (`unified_goalflow_loss.py`)
**Multi-Objective Loss Function**

Combines losses for all components:
- BEV semantic segmentation
- Agent detection (Hungarian matching)
- Goal learning (imitation + DAC + confidence)
- Trajectory flow matching (L1)

**Weighting:**
```python
# In config
config.bev_semantic_weight = 10.0
config.agent_class_weight = 10.0
config.agent_box_weight = 1.0
config.trajectory_weight = 10.0
config.goal_weight = 1.0
config.goal_imitation_weight = 10.0
config.goal_dac_weight = 10.0
config.goal_confidence_weight = 1.0
```

## Configuration

New parameters added to `GoalFlowConfig`:

```python
# Goal decoder architecture
num_goal_queries: int = 128  # Number of learnable queries
goal_decoder_layers: int = 3  # Transformer decoder layers

# Multi-modal DiT
dit_num_layers: int = 8  # Attention layers
cfg_dropout_prob: float = 0.1  # CFG dropout during training
cfg_scale: float = 1.0  # CFG guidance scale (inference)

# Loss weights
goal_weight: float = 1.0
goal_imitation_weight: float = 10.0
goal_dac_weight: float = 10.0
goal_confidence_weight: float = 1.0

# Goal selection
im_weight: float = 0.1  # Imitation score weight
dac_weight: float = 3.0  # DAC score weight
distance_weight: float = 0.0  # Distance score weight

# Goal supervision
goal_timestep: int = 7  # GT goal timestep (0-indexed)
goal_confidence_threshold: float = 5.0  # Confidence threshold (m)

# Activation
use_unified_model: bool = False  # Use unified model
```

## Usage

### Training

**Step 1: Train Perception Module** (as before)
```bash
sh scripts/training/run_goalflow_training_perception.sh
```

**Step 2: Train with Unified Goal Learning**
```bash
# Update config
ONLY_PERCEPTION=False
USE_UNIFIED_MODEL=True
FREEZE_PERCEPTION=True  # Optional
NUM_GOAL_QUERIES=128
DIT_NUM_LAYERS=8
CFG_SCALE=1.5

# Run training
sh scripts/training/run_goalflow_training_unified.sh
```

### Inference

```python
from navsim.agents.goalflow.unified_goalflow_agent import UnifiedGoalFlowAgent
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig

# Load config
config = GoalFlowConfig()
config.training = False
config.use_unified_model = True
config.cfg_scale = 1.5
config.topk = 8

# Initialize agent
agent = UnifiedGoalFlowAgent(
    config=config,
    checkpoint_path="path/to/checkpoint.ckpt"
)
agent.initialize()

# Run inference
predictions = agent(features, targets)
trajectory = predictions['trajectory']  # [batch, 8, 3]
```

## Key Innovations

### 1. Dynamic Query Learning
- **Before**: Fixed vocabulary of 8192 pre-computed goal points
- **After**: 128 learnable queries that dynamically adapt to data
- **Benefits**: More flexible, end-to-end trainable, better generalization

### 2. Multi-Modal Attention
- **Joint attention** across agents, lanes, and trajectory tokens
- **Type embeddings** distinguish modalities
- **Rotary position encoding** for temporal awareness
- Enables richer context for trajectory generation

### 3. Classifier-Free Guidance (CFG)
- **Training**: Random dropout of goal conditioning
- **Inference**: Interpolate between conditional and unconditional predictions
- **Benefits**: Controllable generation, better goal adherence, improved quality

### 4. Unified Pipeline
- **Seamless integration** from scene encoding → goal learning → trajectory generation
- **Flexible training** modes (perception, goal, trajectory, end-to-end)
- **Scalable architecture** with clear module boundaries

## Architecture Diagram

```
Input (Camera + LiDAR)
    ↓
[V99 Backbone]
    ↓
BEV Features (512 channels, 8×8)
    ↓
[Scene Encoding] → Status Encoding
    ↓                      ↓
    └──────────┬───────────┘
               ↓
    [Goal Decoder (128 queries)]
    ├─ Position Head → goal_positions
    ├─ Confidence Head → goal_confidences
    ├─ Imitation Head → imitation_scores
    └─ DAC Head → dac_scores
               ↓
    [Top-K Selection (k=8)]
               ↓
    Selected Goal Features
               ↓
    ┌──────────┴──────────┐
    ↓                      ↓
Noisy Traj (z_t)    Goal Features
    └──────────┬───────────┘
               ↓
    [Multi-Modal DiT Decoder]
    ├─ Joint Attention (agents, lanes, goals, traj)
    ├─ Temporal Encoding (rotary PE)
    ├─ Type Embeddings
    └─ CFG Support
               ↓
    Velocity Field (v_t)
               ↓
    [ODE Solver with Curved Sampling]
               ↓
    Predicted Trajectory (8 steps × 3D pose)
```

## Testing

```bash
# Run smoke tests (requires torch)
cd /home/runner/work/GoalFlow/GoalFlow
python navsim/agents/goalflow/test_goal_learning.py
```

## Compatibility

The implementation maintains backward compatibility:
- Original models (`GoalFlowTrajModel`, `GoalFlowNaviModel`) remain unchanged
- New unified model is opt-in via `use_unified_model=True`
- Can load pretrained perception weights
- Gradual migration path from vocabulary-based to query-based

## Future Extensions

1. **Hierarchical Goal Learning**: Multi-scale goals (short, medium, long-term)
2. **Uncertainty Estimation**: Aleatoric and epistemic uncertainty for goals
3. **Interactive Goal Editing**: User-specified waypoints during inference
4. **Multi-Agent Goal Coordination**: Joint goal prediction for multiple agents

## References

- Flow Matching: [Lipman et al., 2023]
- Classifier-Free Guidance: [Ho & Salimans, 2022]
- Diffusion-ES: [Yang et al., 2023]
- GoalFlow (original): [Xing et al., 2025]
