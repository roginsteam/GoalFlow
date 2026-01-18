# QueryFlow Visual Architecture Diagram

## High-Level Architecture Flow

```
                                    INPUT
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
      Camera Images              LiDAR BEV                  Status
    [B,6,3,256,1024]             [B,1,256,256]              [B,8]
            │                         │                         │
            └──────────┬──────────────┴─────────┬───────────────┘
                       │                        │
                       ↓                        ↓
              ┌─────────────────┐    ┌──────────────────┐
              │ ResNet34 Camera │    │  ResNet34 LiDAR  │
              │   Encoder       │    │    Encoder       │
              │   ~21M params   │    │   ~21M params    │
              └────────┬────────┘    └────────┬─────────┘
                       │                      │
                       │   [B,512,8,32]       │  [B,512,8,8]
                       └──────────┬───────────┘
                                  │
                                  ↓
                      ┌──────────────────────┐
                      │  Transformer Fusion  │
                      │    (2 layers)        │
                      │    ~13M params       │
                      └──────────┬───────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │    BEV Features         │
                    │  Upscale: [B,64,64,64]  │
                    │  Main: [B,512,8,8]      │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ↓               ↓               ↓
      ┌─────────────┐  ┌───────────┐  ┌──────────┐
      │ BEV         │  │  Conv2d   │  │ Status   │
      │ Semantic    │  │ Downscale │  │ Encoding │
      │ Head        │  │ 512→256   │  │  8→256   │
      └─────────────┘  └─────┬─────┘  └────┬─────┘
                             │              │
      [B,7,128,256]          │              │
                             ↓              ↓
                    ┌─────────────────────────┐
                    │   Flatten & Embed       │
                    │   [B, 64, 256]          │
                    │        +                │
                    │   Status [B, 256]       │
                    │        =                │
                    │  KeyVal [B, 65, 256]    │
                    └────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ↓                             ↓
   ┌──────────────────────┐    ┌──────────────────────────┐
   │  AGENT DETECTION     │    │   GOAL LEARNING          │
   │  BRANCH              │    │   BRANCH                 │
   ├──────────────────────┤    ├──────────────────────────┤
   │                      │    │                          │
   │ Query Embedding      │    │ Learnable Queries        │
   │ [B, 31, 256]         │    │ [B, 128, 256]            │
   │ (1 traj + 30 agent)  │    │                          │
   │       │              │    │       │                  │
   │       ↓              │    │       ↓                  │
   │ Transformer Decoder  │    │ Transformer Decoder      │
   │ (3 layers, 8 heads)  │    │ (3 layers, 8 heads)      │
   │ ~21.3M params        │    │ ~21.3M params            │
   │       │              │    │       │                  │
   │       ↓              │    │       ↓                  │
   │ ┌─────────────────┐ │    │ ┌──────────────────────┐ │
   │ │  Agent Head     │ │    │ │ Multi-Head Predict   │ │
   │ │  States: 4D     │ │    │ │ - Position: 3D       │ │
   │ │  Labels: 1D     │ │    │ │ - Confidence: 1D     │ │
   │ └─────────────────┘ │    │ │ - Imitation: 1D      │ │
   │       │              │    │ │ - DAC: 1D            │ │
   │       ↓              │    │ └──────────────────────┘ │
   │ [B, 30, 4+1]         │    │       │                  │
   │                      │    │       ↓                  │
   └──────────────────────┘    │ ┌──────────────────────┐ │
                               │ │ Top-K Selection      │ │
                               │ │ k=8 goals            │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ [B, 8, 3] + [B, 8, 256]  │
                               │ (positions & features)   │
                               └──────────┬───────────────┘
                                          │
                                          ↓
                               ┌──────────────────────────┐
                               │  MULTI-MODAL DiT         │
                               │  DECODER                 │
                               ├──────────────────────────┤
                               │                          │
                               │ Inputs:                  │
                               │ - Noisy Traj [B,12,30]   │
                               │ - Timestep t [B]         │
                               │ - Scene [B,65,256]       │
                               │ - Goals [B,8,256]        │
                               │                          │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Trajectory Encoder   │ │
                               │ │ Linear: 30→256       │ │
                               │ │ [B, 12, 256]         │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Timestep Encoder     │ │
                               │ │ SinusoidalPE + MLP   │ │
                               │ │ [B, 256]             │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Concatenate Tokens   │ │
                               │ │ 12 traj + 65 scene   │ │
                               │ │ + 8 goal = 85 tokens │ │
                               │ │ [B, 85, 256]         │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Type Embeddings      │ │
                               │ │ 0:scene 1:traj       │ │
                               │ │ 2:goal 3:agent 4:lane│ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Multi-Modal Attn     │ │
                               │ │ (8 layers)           │ │
                               │ │                      │ │
                               │ │ Per Layer:           │ │
                               │ │ ┌────────────────┐   │ │
                               │ │ │ Self-Attention │   │ │
                               │ │ │ 8 heads        │   │ │
                               │ │ │ d_model=256    │   │ │
                               │ │ └────────────────┘   │ │
                               │ │        ↓             │ │
                               │ │ ┌────────────────┐   │ │
                               │ │ │ LayerNorm      │   │ │
                               │ │ └────────────────┘   │ │
                               │ │        ↓             │ │
                               │ │ ┌────────────────┐   │ │
                               │ │ │ FFN (256→1024) │   │ │
                               │ │ │ GELU           │   │ │
                               │ │ │ FFN (1024→256) │   │ │
                               │ │ └────────────────┘   │ │
                               │ │        ↓             │ │
                               │ │ ┌────────────────┐   │ │
                               │ │ │ LayerNorm      │   │ │
                               │ │ └────────────────┘   │ │
                               │ │                      │ │
                               │ │ ~790k params/layer   │ │
                               │ │ ~6.3M total          │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Extract Traj Tokens  │ │
                               │ │ [B, 12, 256]         │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ ┌──────────────────────┐ │
                               │ │ Decoder MLP          │ │
                               │ │ 256→256→30           │ │
                               │ └──────────────────────┘ │
                               │       │                  │
                               │       ↓                  │
                               │ [B, 12, 30]              │
                               │ (velocity field)         │
                               │                          │
                               │ CFG Support:             │
                               │ Training: 33% dropout    │
                               │ Inference: scale=1.5     │
                               └──────────┬───────────────┘
                                          │
                                          ↓
                               ┌──────────────────────────┐
                               │  ODE SOLVER              │
                               │  (Inference Only)        │
                               ├──────────────────────────┤
                               │                          │
                               │ Initialize:              │
                               │ z_0 ~ N(0, scale²)       │
                               │ [B*10, 12, 30]           │
                               │                          │
                               │ Curved Schedule:         │
                               │ t = 1-(αt)/(1+(α-1)t)    │
                               │ α=3.0, steps=100         │
                               │                          │
                               │ Loop 100 steps:          │
                               │   v_t = DiT(z_t, t)      │
                               │   z_{t+1} = z_t + v_t*dt │
                               │                          │
                               │ Denormalize:             │
                               │   x *= 60, y *= 15       │
                               │   θ = tanh(θ) * π        │
                               │                          │
                               │ Select Best:             │
                               │   mean or nearest        │
                               └──────────┬───────────────┘
                                          │
                                          ↓
                                      OUTPUT
                                  [B, 8, 3]
                              (8 timesteps × xyz)
```

## Parameter Distribution

```
┌────────────────────────────────────────────────┐
│         Parameter Distribution (~105.3M)       │
├────────────────────────────────────────────────┤
│                                                │
│  V99 Backbone      ████████████████  55.0M 52%│
│  Agent Detection   ████████████      21.6M 21%│
│  Goal Decoder      ████████████      22.2M 21%│
│  DiT Decoder       ████              6.3M   6%│
│  BEV Processing    ▌                 0.2M  <1%│
│                                                │
└────────────────────────────────────────────────┘
```

## Data Flow Dimensions

```
Input Stage:
Camera:      [B, 6, 3, 256, 1024]      ~4.7 MB/sample
LiDAR:       [B, 1, 256, 256]          ~0.25 MB/sample
Status:      [B, 8]                    ~32 bytes/sample

After V99:
BEV Feature: [B, 512, 8, 8]            ~128 KB/sample
           → [B, 64, 256]               ~64 KB/sample

Agent Branch:
Queries:     [B, 31, 256]               ~31 KB/sample
Output:      [B, 30, 5]                 ~600 bytes/sample

Goal Branch:
Queries:     [B, 128, 256]              ~128 KB/sample
Top-K:       [B, 8, 256]                ~8 KB/sample

Trajectory:
Input:       [B, 12, 30]                ~1.4 KB/sample
Multi-Modal: [B, 85, 256]               ~85 KB/sample
Output:      [B, 12, 30]                ~1.4 KB/sample
Final:       [B, 8, 3]                  ~96 bytes/sample
```

## Comparison: Original vs QueryFlow

```
┌─────────────────────────────────────────────────────────────┐
│                    Original       │      QueryFlow          │
├─────────────────────────────────────────────────────────────┤
│ Goal Representation                                          │
│   Fixed Points         8,192       │      128 Queries        │
│   Memory               256 MB      │      4 MB               │
│   Disk I/O             Required    │      None               │
│                                                               │
│ Model Architecture                                           │
│   Goal Module          Separate    │      Integrated         │
│   Parameters           ~83M        │      ~105M              │
│   Decoder              8-layer     │      8-layer DiT        │
│   CFG Support          No          │      Yes                │
│                                                               │
│ Training                                                     │
│   Stages               2-stage     │      End-to-end         │
│   Supervision          Separate    │      Joint              │
│   Goal Learning        Explicit    │      Trajectory-sup     │
│                                                               │
│ Inference                                                    │
│   Goal Selection       Disk Load   │      Top-K In-memory    │
│   Controllability      Fixed       │      CFG (scale=1.5)    │
│   Speed                ~150ms      │      ~160ms             │
│   Quality              Good        │      Better             │
└─────────────────────────────────────────────────────────────┘
```

## Key Innovation Points

```
1. Dynamic Query Learning
   ┌─────────────────────────────────┐
   │ 128 learnable goal queries      │
   │ Replace 8,192 fixed points      │
   │ Adapt to data distribution      │
   │ End-to-end trainable            │
   └─────────────────────────────────┘

2. Multi-Modal DiT Decoder
   ┌─────────────────────────────────┐
   │ Joint attention across:         │
   │ • Scene features                │
   │ • Trajectory tokens             │
   │ • Goal features                 │
   │ • Agent features (optional)     │
   │ • Lane features (optional)      │
   └─────────────────────────────────┘

3. CFG for Controllable Generation
   ┌─────────────────────────────────┐
   │ Training: Random dropout        │
   │   33% unconditional             │
   │   33% conditional               │
   │   33% goal-only                 │
   │                                 │
   │ Inference: Guidance scale       │
   │   guided = uncond +             │
   │            scale * (cond-uncond)│
   └─────────────────────────────────┘

4. Trajectory Supervision
   ┌─────────────────────────────────┐
   │ Goals learned from trajectories │
   │ Multi-head prediction:          │
   │ • Position (x, y, θ)            │
   │ • Confidence score              │
   │ • Imitation score               │
   │ • DAC score                     │
   └─────────────────────────────────┘
```
