# QueryFlow Trajectory Generation

This directory contains the script for generating trajectories using the unified query-based GoalFlow model.

## Overview

`run_generate_trajs_queryflow.py` is adapted from `run_generate_trajs.py` for the trajectory-supervised goal learning system with:

- **Dynamic Learnable Queries**: 128 learnable queries instead of 8,192 fixed vocabulary points
- **Multi-Modal DiT Decoder**: Joint attention across agents, lanes, trajectories, and goals
- **CFG Support**: Classifier-Free Guidance for controllable trajectory generation
- **End-to-End**: Unified pipeline from scene encoding to trajectory prediction

## Key Differences from Original

| Feature | Original (`run_generate_trajs.py`) | QueryFlow (`run_generate_trajs_queryflow.py`) |
|---------|-----------------------------------|-----------------------------------------------|
| Goal Representation | 8,192 fixed points from disk | 128 learnable queries |
| Model | `GoalFlowTrajAgent` | `UnifiedGoalFlowAgent` |
| Config | `goalflow_training.yaml` | `queryflow_training.yaml` |
| Agent Config | `goalflow_agent_traj.yaml` | `unified_goalflow_agent.yaml` |
| Disk I/O | Requires loading score files | Pure memory, no I/O |
| CFG | Not supported | Supported with `cfg_scale` |

## Usage

### Basic Generation

```bash
python navsim/planning/script/run_generate_trajs_queryflow.py \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    output_dir=/path/to/output
```

### With Custom Configuration

```bash
python navsim/planning/script/run_generate_trajs_queryflow.py \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    agent.config.cfg_scale=1.5 \
    agent.config.topk=8 \
    agent.config.anchor_size=10 \
    output_dir=/path/to/output
```

### Using Cached Data

```bash
python navsim/planning/script/run_generate_trajs_queryflow.py \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    use_cache_without_dataset=true \
    cache_path=/path/to/cache \
    output_dir=/path/to/output
```

## Configuration Parameters

### Goal Learning Parameters

- `num_goal_queries`: Number of learnable goal queries (default: 128)
- `goal_decoder_layers`: Transformer decoder layers for goals (default: 3)
- `topk`: Number of top goals to select (default: 8)
- `im_weight`: Weight for imitation score in goal selection (default: 0.1)
- `dac_weight`: Weight for DAC score in goal selection (default: 3.0)

### Multi-Modal DiT Parameters

- `dit_num_layers`: Number of attention layers (default: 8)
- `cfg_scale`: CFG guidance scale (1.0 = no guidance, >1.0 = stronger goal conditioning)
- `cfg_dropout_prob`: CFG dropout probability during training (default: 0.1)

### Trajectory Generation Parameters

- `infer_steps`: Number of ODE solver steps (default: 100)
- `cur_sampling`: Use curved timestep schedule (default: true)
- `alpha`: Curve parameter for sampling (default: 3.0)
- `anchor_size`: Number of trajectory samples per goal (default: 10)
- `use_nearest`: Select trajectory closest to goal (default: false)

## Example Configurations

### Standard Inference (Recommended)

```yaml
agent:
  config:
    use_unified_model: true
    cfg_scale: 1.5
    topk: 8
    anchor_size: 10
    cur_sampling: true
    training: false
```

### High-Quality Generation (Slower)

```yaml
agent:
  config:
    use_unified_model: true
    cfg_scale: 2.0
    topk: 16
    anchor_size: 20
    infer_steps: 200
    cur_sampling: true
    training: false
```

### Fast Inference (Lower Quality)

```yaml
agent:
  config:
    use_unified_model: true
    cfg_scale: 1.0
    topk: 4
    anchor_size: 5
    infer_steps: 50
    cur_sampling: false
    training: false
```

## Output

The script generates trajectories and saves them according to the standard NAVSIM format. Results are stored in the specified `output_dir`.

## Requirements

- Pretrained checkpoint with unified model weights
- NAVSIM dataset (or cached features)
- GPU with at least 16GB memory (for batch_size=4)

## Troubleshooting

### Out of Memory

Reduce batch size or anchor_size:
```bash
python navsim/planning/script/run_generate_trajs_queryflow.py \
    dataloader.params.batch_size=2 \
    agent.config.anchor_size=5 \
    ...
```

### Checkpoint Loading Error

Ensure checkpoint was trained with `use_unified_model=true`:
```bash
python navsim/planning/script/run_generate_trajs_queryflow.py \
    agent.checkpoint_path=/path/to/unified_checkpoint.ckpt \
    agent.config.use_unified_model=true \
    ...
```

## See Also

- [Goal Learning System Documentation](../../../docs/goal_learning_system.md)
- [Unified GoalFlow Model](../../../navsim/agents/goalflow/unified_goalflow_model.py)
- [Training Guide](../../../docs/train.md)
