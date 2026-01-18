# Unified QueryFlow Network Architecture

## Complete Architecture Diagram with Detailed Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT PROCESSING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Camera Images [B, 6, 3, 256, 1024]                                         │
│  LiDAR BEV     [B, 1, 256, 256]                                             │
│  Status        [B, 8]  (x, y, vx, vy, ax, ay, heading, steering)           │
│                                                                               │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    V99 BACKBONE (Scene Encoding)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Camera Branch:                                                               │
│    ├─ ResNet34 Encoder                                                       │
│    │    Input:  [B, 6, 3, 256, 1024]                                        │
│    │    Output: [B, 512, 8, 32]  (per view)                                 │
│    │    Params: ~21M                                                         │
│    │                                                                          │
│    └─ Feature Aggregation                                                    │
│         Output: [B, 512, 8, 32]  (fused 6 views)                            │
│                                                                               │
│  LiDAR Branch:                                                                │
│    ├─ ResNet34 Encoder                                                       │
│    │    Input:  [B, 5, 256, 256]  (5 histogram channels)                   │
│    │    Output: [B, 512, 8, 8]                                              │
│    │    Params: ~21M                                                         │
│    │                                                                          │
│    └─ Transformer Fusion (2 layers)                                          │
│         d_model: 512                                                          │
│         n_heads: 4                                                            │
│         d_ffn:   2048                                                         │
│         Params:  ~13M                                                         │
│                                                                               │
│  Output:                                                                      │
│    ├─ bev_feature_upscale: [B, 64, 64, 64]   (for semantic head)           │
│    └─ bev_feature:         [B, 512, 8, 8]    (for downstream)              │
│                                                                               │
│  Total Params: ~55M                                                           │
│                                                                               │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ↓                   ↓                   ↓
┌──────────────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│  BEV Semantic Head       │  │  BEV Downscale   │  │  Status Encoding        │
├──────────────────────────┤  ├──────────────────┤  ├─────────────────────────┤
│                          │  │                  │  │                         │
│  Input:  [B, 64, 64, 64] │  │  Conv2d          │  │  Linear                 │
│                          │  │  512 → 256       │  │  8 → 256                │
│  Conv2d: 64 → 64, 3×3   │  │  kernel: 1×1     │  │  Params: 2,304          │
│  ReLU                    │  │                  │  │                         │
│  Conv2d: 64 → 7, 1×1    │  │  Output:         │  │  Output: [B, 256]       │
│  Upsample: 64×64 → 128×256│ │  [B, 256, 8, 8]  │  │                         │
│                          │  │  → flatten →     │  └─────────────────────────┘
│  Output: [B, 7, 128, 256]│  │  [B, 64, 256]    │
│  Params: 28,615          │  │  Params: 131,072 │
│                          │  │                  │
└──────────────────────────┘  └────────┬─────────┘
                                       │
                                       ↓
                              ┌─────────────────────────┐
                              │  KeyVal Embedding       │
                              ├─────────────────────────┤
                              │  Embedding(65, 256)     │
                              │  [64 grid + 1 status]   │
                              │  Params: 16,640         │
                              │                         │
                              │  Output: [B, 65, 256]   │
                              └────────┬────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
                    ↓                                      ↓
┌──────────────────────────────────────┐  ┌─────────────────────────────────────┐
│     AGENT DETECTION BRANCH           │  │     GOAL LEARNING BRANCH            │
├──────────────────────────────────────┤  ├─────────────────────────────────────┤
│                                      │  │                                     │
│  Query Embedding                     │  │  Goal Queries (Learnable)           │
│    Embedding(31, 256)                │  │    Embedding(128, 256)              │
│    [1 traj + 30 agents]              │  │    Init: N(0, 0.02)                 │
│    Params: 7,936                     │  │    Params: 32,768                   │
│    Output: [B, 31, 256]              │  │    Output: [B, 128, 256]            │
│                                      │  │                                     │
│  Transformer Decoder (3 layers)      │  │  Transformer Decoder (3 layers)     │
│    d_model: 256                      │  │    d_model: 256                     │
│    n_heads: 8                        │  │    n_heads: 8                       │
│    d_ffn:   1024                     │  │    d_ffn:   1024                    │
│    dropout: 0.0                      │  │    dropout: 0.1                     │
│    Params:  ~7.1M per layer          │  │    Params:  ~7.1M per layer         │
│    Total:   ~21.3M                   │  │    Total:   ~21.3M                  │
│    Output:  [B, 31, 256]             │  │    Output:  [B, 128, 256]           │
│                                      │  │                                     │
│  Agent Head:                         │  │  Multi-Head Prediction:             │
│    ├─ States MLP                     │  │    ├─ Position Head                 │
│    │    Linear: 256 → 1024           │  │    │    Linear: 256 → 1024         │
│    │    ReLU                          │  │    │    ReLU                        │
│    │    Linear: 1024 → 4             │  │    │    Linear: 1024 → 3           │
│    │    Params: 266,240              │  │    │    Params: 265,219             │
│    │    Output: [B, 30, 4]           │  │    │    Output: [B, 128, 3]        │
│    │    (x, y, heading, class)       │  │    │    (x, y, heading)             │
│    │                                  │  │    │    with tanh scaling           │
│    └─ Labels MLP                     │  │    │                                │
│         Linear: 256 → 1              │  │    ├─ Confidence Head               │
│         Params: 257                  │  │    │    Linear: 256 → 512           │
│         Output: [B, 30, 1]           │  │    │    ReLU                        │
│                                      │  │    │    Linear: 512 → 1             │
│  Total Agent Head: 266,497           │  │    │    Params: 132,097             │
│                                      │  │    │    Output: [B, 128, 1]         │
│                                      │  │    │                                │
│                                      │  │    ├─ Imitation Head                │
│                                      │  │    │    Linear: 256 → 512           │
│                                      │  │    │    ReLU                        │
│                                      │  │    │    Linear: 512 → 1             │
│                                      │  │    │    Params: 132,097             │
│                                      │  │    │    Output: [B, 128, 1]         │
│                                      │  │    │                                │
│                                      │  │    └─ DAC Head                      │
│                                      │  │         Linear: 256 → 512           │
│                                      │  │         ReLU                        │
│                                      │  │         Linear: 512 → 1             │
│                                      │  │         Params: 132,097             │
│                                      │  │         Output: [B, 128, 1]         │
│                                      │  │                                     │
│                                      │  │  Top-K Selection (Inference):       │
│                                      │  │    Combined Score = im_weight * log_softmax(im)
│                                      │  │                   + dac_weight * log_sigmoid(dac)
│                                      │  │                   + dist_weight * norm_dist
│                                      │  │    Select top-8 goals                │
│                                      │  │    Output: [B, 8, 256]              │
│                                      │  │                                     │
│                                      │  │  Total Goal Decoder: ~22.2M         │
│                                      │  │                                     │
└──────────────────────────────────────┘  └────────────┬────────────────────────┘
                                                       │
                                                       ↓
                                          ┌──────────────────────────────────────┐
                                          │  TRAJECTORY GENERATION BRANCH        │
                                          │  (Multi-Modal DiT Decoder)           │
                                          ├──────────────────────────────────────┤
                                          │                                      │
                                          │  Input Preparation:                  │
                                          │  ┌────────────────────────────────┐ │
                                          │  │ Noisy Trajectory [B, 12, 30]   │ │
                                          │  │ Timestep t [B]                 │ │
                                          │  │ Scene Features [B, 65, 256]    │ │
                                          │  │ Goal Features  [B, 8, 256]     │ │
                                          │  └────────────────────────────────┘ │
                                          │                                      │
                                          │  Trajectory Encoder:                 │
                                          │    Linear: 30 → 256                  │
                                          │    Params: 7,936                     │
                                          │    Output: [B, 12, 256]              │
                                          │                                      │
                                          │  Timestep Encoder:                   │
                                          │    ├─ SinusoidalPosEmb(256)         │
                                          │    ├─ Linear: 256 → 256             │
                                          │    ├─ ReLU                           │
                                          │    └─ Linear: 256 → 256             │
                                          │    Params: 131,584                   │
                                          │    Output: [B, 256]                  │
                                          │                                      │
                                          │  Sigma Projection:                   │
                                          │    Linear: 512 → 256                 │
                                          │    Params: 131,328                   │
                                          │    Broadcast to all tokens           │
                                          │                                      │
                                          │  Type Embeddings:                    │
                                          │    Embedding(10, 256)                │
                                          │    0: scene, 1: traj, 2: goal,      │
                                          │    3: agent, 4: lane                 │
                                          │    Params: 2,560                     │
                                          │                                      │
                                          │  Rotary Position Encoding:           │
                                          │    RotaryPE(256)                     │
                                          │    For temporal attention            │
                                          │                                      │
                                          │  Multi-Modal Attention (8 layers):   │
                                          │  ┌────────────────────────────────┐ │
                                          │  │ Per Layer:                     │ │
                                          │  │   ParallelAttentionLayer       │ │
                                          │  │     d_model: 256               │ │
                                          │  │     n_heads: 8                 │ │
                                          │  │     d_ffn:   1024              │ │
                                          │  │     dropout: 0.1               │ │
                                          │  │                                │ │
                                          │  │   Components:                  │ │
                                          │  │     ├─ Self-Attention          │ │
                                          │  │     │    Q,K,V: 256 → 256      │ │
                                          │  │     │    Params: ~197k         │ │
                                          │  │     │                          │ │
                                          │  │     ├─ LayerNorm: 256          │ │
                                          │  │     │    Params: 512           │ │
                                          │  │     │                          │ │
                                          │  │     ├─ FFN                     │ │
                                          │  │     │    Linear: 256 → 1024    │ │
                                          │  │     │    GELU                  │ │
                                          │  │     │    Dropout(0.1)          │ │
                                          │  │     │    Linear: 1024 → 256    │ │
                                          │  │     │    Params: ~524k         │ │
                                          │  │     │                          │ │
                                          │  │     └─ LayerNorm: 256          │ │
                                          │  │          Params: 512           │ │
                                          │  │                                │ │
                                          │  │   Params per layer: ~722k      │ │
                                          │  │   Total 8 layers: ~5.8M        │ │
                                          │  │                                │ │
                                          │  │   Input:  [B, 85, 256]         │ │
                                          │  │   (12 traj + 65 scene + 8 goal)│ │
                                          │  │   Output: [B, 85, 256]         │ │
                                          │  └────────────────────────────────┘ │
                                          │                                      │
                                          │  Extract Trajectory Tokens:          │
                                          │    Output: [B, 12, 256]              │
                                          │                                      │
                                          │  Decoder MLP:                        │
                                          │    Linear: 256 → 256                 │
                                          │    ReLU                              │
                                          │    Linear: 256 → 30                  │
                                          │    Params: 73,470                    │
                                          │    Output: [B, 12, 30]               │
                                          │    (velocity field)                  │
                                          │                                      │
                                          │  Total DiT Decoder: ~6.3M            │
                                          │                                      │
                                          │  CFG Support:                        │
                                          │    Training: Random dropout          │
                                          │      - 33% unconditional             │
                                          │      - 33% conditional               │
                                          │      - 33% goal-dropout              │
                                          │    Inference:                        │
                                          │      guided = uncond + scale * (cond - uncond)
                                          │      scale: 1.5 (configurable)       │
                                          │                                      │
                                          └──────────────┬───────────────────────┘
                                                         │
                                                         ↓
                                          ┌──────────────────────────────────────┐
                                          │  ODE SOLVER (Inference Only)         │
                                          ├──────────────────────────────────────┤
                                          │                                      │
                                          │  Initialize:                         │
                                          │    z_0 ~ N(0, test_scale²)          │
                                          │    [B*anchor_size, 12, 30]          │
                                          │    anchor_size: 10 (default)        │
                                          │                                      │
                                          │  Curved Sampling Schedule:           │
                                          │    t_shifted = 1 - (α*t)/(1+(α-1)*t) │
                                          │    α = 3.0 (configurable)            │
                                          │    steps = 100 (infer_steps)         │
                                          │                                      │
                                          │  Integration Loop:                   │
                                          │    for t_curr, t_prev in schedule:   │
                                          │      v_t = DiT(z_t, t_curr, cond)   │
                                          │      z_{t+1} = z_t + v_t * dt       │
                                          │                                      │
                                          │  Denormalization:                    │
                                          │    x = x * 60.0                      │
                                          │    y = y * 15.0                      │
                                          │    θ = tanh(θ) * π                   │
                                          │                                      │
                                          │  Selection:                          │
                                          │    if use_nearest:                   │
                                          │      closest to mean goal            │
                                          │    else:                             │
                                          │      mean trajectory                 │
                                          │                                      │
                                          │  Output: [B, 8, 3]                   │
                                          │    (8 timesteps, x,y,heading)        │
                                          │                                      │
                                          └──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOTAL PARAMETER COUNT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  V99 Backbone:              ~55.0M                                           │
│  BEV Processing:            ~0.2M                                            │
│  Agent Detection:           ~21.6M                                           │
│  Goal Decoder:              ~22.2M                                           │
│  DiT Decoder:               ~6.3M                                            │
│  ───────────────────────────────────                                         │
│  TOTAL:                     ~105.3M parameters                               │
│                                                                               │
│  Memory (FP16):             ~211 MB (model weights)                          │
│  Training Memory:           ~8-12 GB (batch_size=4, with gradients)         │
│  Inference Memory:          ~4-6 GB (batch_size=4, no gradients)            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Layer-by-Layer Breakdown

### 1. Input Layer
```
Camera:  [B, 6, 3, 256, 1024]    6 views, RGB, H×W
LiDAR:   [B, 1, 256, 256]        BEV histogram
Status:  [B, 8]                   ego state vector
```

### 2. V99 Backbone (Scene Encoder)

#### Camera Branch
```
Input:  [B, 6, 3, 256, 1024]
├─ ResNet34 (per view)
│  ├─ Conv1: 3 → 64, 7×7, stride=2       Params: 9,408
│  ├─ Layer1: 64 → 64, 3 blocks         Params: ~148k
│  ├─ Layer2: 64 → 128, 4 blocks        Params: ~526k
│  ├─ Layer3: 128 → 256, 6 blocks       Params: ~2.1M
│  ├─ Layer4: 256 → 512, 3 blocks       Params: ~8.4M
│  └─ Output: [B, 6, 512, 8, 32]
│
├─ View Aggregation (Attention/Pooling)
│  └─ Output: [B, 512, 8, 32]
│
Total Params: ~21M
```

#### LiDAR Branch
```
Input:  [B, 5, 256, 256]         5 histogram bins
├─ ResNet34
│  ├─ Conv1: 5 → 64, 7×7, stride=2       Params: 15,680
│  ├─ Layer1-4: (same structure as camera)
│  └─ Output: [B, 512, 8, 8]
│
Total Params: ~21M
```

#### Fusion Module
```
Input:  Camera [B, 512, 8, 32] + LiDAR [B, 512, 8, 8]
├─ Flatten: → [B, 320, 512]       (256 cam + 64 lidar tokens)
├─ Transformer (2 layers)
│  ├─ Layer 1:
│  │  ├─ MultiHeadAttention(512, heads=4)
│  │  │  Params: 1,049,088
│  │  ├─ FFN(512 → 2048 → 512)
│  │  │  Params: 2,099,200
│  │  └─ LayerNorms
│  │     Params: 2,048
│  │
│  └─ Layer 2: (same structure)
│
├─ Reshape: → [B, 512, 8, 8]
│
Output: 
  bev_feature_upscale: [B, 64, 64, 64]    (for semantic head)
  bev_feature:         [B, 512, 8, 8]     (for downstream tasks)

Total Params: ~13M
```

### 3. BEV Processing

#### BEV Downscale
```
Input:  [B, 512, 8, 8]
Conv2d: 512 → 256, 1×1
Params: 131,072
Output: [B, 256, 8, 8] → flatten → [B, 64, 256]
```

#### Status Encoding
```
Input:  [B, 8]
Linear: 8 → 256
Params: 2,304
Output: [B, 256]
```

#### KeyVal Embedding
```
Embedding(65, 256)      65 = 64 grid + 1 status
Params: 16,640
Applied to: [B, 65, 256]
```

### 4. Goal Decoder (Query-Based)

#### Goal Queries
```
Embedding(128, 256)
Init: Normal(0, 0.02)
Params: 32,768
Output: [B, 128, 256]
```

#### Transformer Decoder (3 layers)
```
Per Layer:
├─ MultiHeadAttention(256, heads=8)
│  ├─ Q,K,V projections: 256 → 256 × 3
│  │  Params: 197,632
│  ├─ Output projection: 256 → 256
│  │  Params: 65,792
│  └─ Total: 263,424
│
├─ FFN
│  ├─ Linear: 256 → 1024
│  │  Params: 263,168
│  ├─ Linear: 1024 → 256
│  │  Params: 262,400
│  └─ Total: 525,568
│
├─ LayerNorms (×2)
│  Params: 1,024
│
Layer Total: 790,016
3 Layers: 2,370,048

Output: [B, 128, 256]
```

#### Prediction Heads
```
Position Head:
├─ Linear: 256 → 1024       Params: 263,168
├─ ReLU
├─ Linear: 1024 → 3         Params: 3,075
└─ Tanh scaling
   Total: 266,243

Confidence Head:
├─ Linear: 256 → 512        Params: 131,584
├─ ReLU
├─ Linear: 512 → 1          Params: 513
└─ Total: 132,097

Imitation Head:
├─ Linear: 256 → 512        Params: 131,584
├─ ReLU
├─ Linear: 512 → 1          Params: 513
└─ Total: 132,097

DAC Head:
├─ Linear: 256 → 512        Params: 131,584
├─ ReLU
├─ Linear: 512 → 1          Params: 513
└─ Total: 132,097

Heads Total: 662,534
```

#### Top-K Selection
```
Input:  goal_positions [B, 128, 3]
        imitation_scores [B, 128, 1]
        dac_scores [B, 128, 1]

Combined Score:
  score = im_weight * log_softmax(im_scores)
        + dac_weight * log_sigmoid(dac_scores)
        + dist_weight * normalized_distance

TopK: Select top-8
Output: [B, 8, 3] positions
        [B, 8, 256] features
```

### 5. Multi-Modal DiT Decoder

#### Input Processing
```
Trajectory Encoder:
├─ Linear: 30 → 256
├─ Params: 7,936
└─ Output: [B, 12, 256]

Timestep Encoder:
├─ SinusoidalPosEmb(256)
├─ Linear: 256 → 256        Params: 65,792
├─ ReLU
├─ Linear: 256 → 256        Params: 65,792
└─ Output: [B, 256]

Sigma Projection:
├─ Linear: 512 → 256        Params: 131,328
└─ Broadcast to all tokens

Type Embedding:
├─ Embedding(10, 256)       Params: 2,560
└─ Applied per token type
```

#### Multi-Modal Attention Layers (×8)
```
Per Layer (ParallelAttentionLayer):

Self-Attention:
├─ Q,K,V: 256 → 256 × 3     Params: 197,632
├─ Output: 256 → 256         Params: 65,792
├─ Dropout(0.1)
├─ LayerNorm(256)            Params: 512
└─ Subtotal: 263,936

FFN:
├─ Linear: 256 → 1024        Params: 263,168
├─ GELU
├─ Dropout(0.1)
├─ Linear: 1024 → 256        Params: 262,400
├─ Dropout(0.1)
├─ LayerNorm(256)            Params: 512
└─ Subtotal: 526,080

Layer Total: 790,016
8 Layers: 6,320,128

Input Tokens: 85 (12 traj + 65 scene + 8 goal)
Output: [B, 85, 256]
```

#### Output Projection
```
Decoder MLP:
├─ Linear: 256 → 256         Params: 65,792
├─ ReLU
├─ Linear: 256 → 30          Params: 7,710
└─ Total: 73,502

Extract trajectory tokens: [B, 12, 256]
Output: [B, 12, 30] (velocity field)
```

### 6. ODE Solver (Inference)

#### Initialization
```
Noise: z_0 ~ N(0, test_scale²)
Shape: [B*anchor_size, 12, 30]
anchor_size: 10 (default)
```

#### Curved Sampling
```
Schedule: t_shifted = 1 - (α*t) / (1 + (α-1)*t)
Alpha: 3.0
Steps: 100

For each step:
  v_t = DiT(z_t, t, scene, goal, cfg_scale)
  z_{t+1} = z_t + v_t * dt
```

#### Denormalization
```
x *= 60.0        (meters)
y *= 15.0        (meters)
θ = tanh(θ) * π  (radians)
```

## Training Configuration

```yaml
Default Hyperparameters:
  batch_size: 4
  learning_rate: 1e-4
  scheduler: StepLR(step=20, gamma=0.8)
  
  d_model: 256
  n_heads: 8
  d_ffn: 1024
  dropout: 0.1
  
  num_goal_queries: 128
  goal_decoder_layers: 3
  dit_num_layers: 8
  
  cfg_dropout_prob: 0.1
  cfg_scale: 1.5 (inference)
  
  topk: 8
  anchor_size: 10
  infer_steps: 100
  
Loss Weights:
  bev_semantic_weight: 10.0
  agent_class_weight: 10.0
  agent_box_weight: 1.0
  trajectory_weight: 10.0
  goal_imitation_weight: 10.0
  goal_dac_weight: 10.0
  goal_confidence_weight: 1.0
```

## Memory Footprint

```
Model Weights (FP16):        ~211 MB
Activations (batch=4):       ~2-3 GB
Gradients (training):        ~211 MB
Optimizer State (Adam):      ~422 MB
Cache & Temp:                ~1-2 GB
───────────────────────────────────
Training Total:              ~4-6 GB per GPU
Inference Total:             ~2-3 GB per GPU

With Data Parallel (4 GPUs):
  Per GPU: ~4-6 GB
  Total:   ~16-24 GB
```
