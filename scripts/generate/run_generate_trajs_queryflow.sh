#!/bin/bash
# QueryFlow Trajectory Generation Script
# This script uses the unified query-based goal learning model
# instead of the fixed vocabulary approach

# LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export CUDA_VISIBLE_DEVICES=0,5 # 1,2,3,4
# export HYDRA_FULL_ERROR=1

# ============================================================
# Configuration Paths
# ============================================================
FEATURE_CACHE='' # set your feature_cache path
CHECKPOINT_PATH=$NAVSIM_DEVKIT_ROOT/data/unified_goalflow_checkpoint.ckpt

# Note: No need for VOC_PATH and GOAL_POINT_SCORES with query-based approach
# The model learns 128 queries dynamically instead of using 8,192 fixed points

# ============================================================
# Run QueryFlow Trajectory Generation
# ============================================================
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs_queryflow.py \
agent=unified_goalflow_agent \
experiment_name=queryflow_test \
scene_filter=navtest \
split=test \
cache_path=$FEATURE_CACHE \
scene_filter.num_future_frames=10 \
dataloader.params.batch_size=4 \
use_cache_without_dataset=True \
agent.config.use_unified_model=True \
agent.config.training=False \
agent.config.generate='trajectory' \
agent.config.num_goal_queries=128 \
agent.config.goal_decoder_layers=3 \
agent.config.dit_num_layers=8 \
agent.config.cfg_scale=1.5 \
agent.config.topk=8 \
agent.config.im_weight=0.1 \
agent.config.dac_weight=3.0 \
agent.config.distance_weight=0.0 \
agent.config.use_nearest=False \
agent.config.cur_sampling=True \
agent.config.alpha=3.0 \
agent.config.train_scale=0.1 \
agent.config.test_scale=0.1 \
agent.config.start=True \
agent.config.infer_steps=100 \
agent.config.anchor_size=10 \
agent.config.tf_d_model=256 \
agent.checkpoint_path=$CHECKPOINT_PATH

# ============================================================
# Key Differences from Original Script:
# ============================================================
# 1. Script: run_generate_trajs_queryflow.py (instead of run_generate_trajs.py)
# 2. Agent: unified_goalflow_agent (instead of goalflow_agent_traj)
# 3. No VOC_PATH needed (no fixed vocabulary)
# 4. No GOAL_POINT_SCORES needed (no pre-computed scores)
# 5. New parameters:
#    - use_unified_model=True (enable query-based approach)
#    - num_goal_queries=128 (learnable queries)
#    - cfg_scale=1.5 (CFG guidance for controllable generation)
#    - dit_num_layers=8 (multi-modal DiT decoder)
# 6. Removed legacy parameters:
#    - has_navi, has_student_navi (replaced by unified goal learning)
#    - fusion, beta, cond_threshold (legacy trajectory fusion)
#    - ep_score_weight, ep_point_weight (fixed point scoring)
#    - theta (legacy scoring parameter)
# 7. Optimized defaults:
#    - anchor_size=10 (down from 384, more efficient)
#    - infer_steps=100 (up from 5, higher quality)
#    - topk=8 (down from 15, using learnable queries)
