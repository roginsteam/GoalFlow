"""
Unified GoalFlow Model with Trajectory-Supervised Goal Learning

Integrates:
1. Scene encoding (BEV features from backbone)
2. Dynamic query-based goal learning
3. Flow matching-based trajectory decoding with multi-modal DiT
4. CFG support for goal-conditioned planning
"""

import math
from typing import Dict, Optional
import torch
import torch.nn as nn

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.v99_backbone import V299Backbone
from navsim.agents.goalflow.goal_decoder import (
    GoalDecoder,
    goal_imitation_loss,
    goal_dac_loss,
    goal_confidence_loss,
)
from navsim.agents.goalflow.multimodal_dit_decoder import MultiModalDiTDecoder
from navsim.agents.goalflow.goalflow_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
import numpy as np


class UnifiedGoalFlowModel(nn.Module):
    """
    Unified model integrating trajectory-supervised goal learning with flow matching.
    
    Key Features:
    - Dynamic query-based goal learning (replaces fixed dictionaries)
    - Multi-modal DiT decoder with joint attention
    - CFG support for controllable trajectory generation
    - End-to-end trainable pipeline
    """
    
    def __init__(self, config: GoalFlowConfig):
        super().__init__()
        
        self._config = config
        self._query_splits = [1, config.num_bounding_boxes]
        
        # ============ Scene Encoding ============
        self._backbone = V299Backbone(config)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )
        
        # Scene feature embeddings
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        
        # Perception decoder for agents
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        
        # Agent detection head
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        
        # ============ Goal Learning Module ============
        self.goal_decoder = GoalDecoder(
            d_model=config.tf_d_model,
            num_queries=getattr(config, 'num_goal_queries', 128),
            num_decoder_layers=getattr(config, 'goal_decoder_layers', 3),
            num_heads=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
        )
        
        # ============ Trajectory Generation Module ============
        if not config.only_perception:
            self.trajectory_decoder = MultiModalDiTDecoder(
                d_model=config.tf_d_model,
                num_layers=getattr(config, 'dit_num_layers', 8),
                num_heads=config.tf_num_head,
                trajectory_len=12,
                trajectory_dim=30,
                dropout=config.tf_dropout,
                cfg_dropout_prob=getattr(config, 'cfg_dropout_prob', 0.1),
            )
        
        # Freeze perception if specified
        if config.freeze_perception:
            self._freeze_perception()
    
    def _freeze_perception(self):
        """Freeze perception module parameters."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        for param in self._bev_downscale.parameters():
            param.requires_grad = False
        for param in self._bev_semantic_head.parameters():
            param.requires_grad = False
        for param in self._keyval_embedding.parameters():
            param.requires_grad = False
        for param in self._query_embedding.parameters():
            param.requires_grad = False
        for param in self._status_encoding.parameters():
            param.requires_grad = False
        for param in self._tf_decoder.parameters():
            param.requires_grad = False
        for param in self._agent_head.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of unified model.
        
        Args:
            features: Input features dictionary
            targets: Target dictionary for supervision
            
        Returns:
            Dictionary of predictions and intermediate outputs
        """
        # Extract inputs
        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]
        gt_trajs = features['gt_trajs'].to(status_feature)
        
        batch_size = status_feature.shape[0]
        device = status_feature.device
        dtype = status_feature.dtype
        
        # ============ Scene Encoding ============
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)  # [batch, 64, d_model]
        
        # Status encoding
        if self._config.has_history:
            used_dims = [0, 1, 2, 3, 7, 8, 9, 10]
            status_encoding = self._status_encoding(status_feature[:, -1, used_dims])
        else:
            status_encoding = self._status_encoding(status_feature)
        
        # Scene context
        keyval = torch.cat([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]
        
        # Agent perception
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        agents = self._agent_head(agents_query)
        
        # BEV semantic map
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        
        # ============ Goal Learning ============
        goal_outputs = self.goal_decoder(
            scene_features=keyval,
            status_encoding=status_encoding,
        )
        
        output = {
            "bev_semantic_map": bev_semantic_map,
            **agents,
            **goal_outputs,
        }
        
        # Early return if only perception
        if self._config.only_perception:
            output['trajectory'] = gt_trajs  # Placeholder
            return output
        
        # ============ Trajectory Generation ============
        if self._config.training:
            trajectory_output = self._forward_training(
                gt_trajs=gt_trajs,
                scene_features=keyval,
                goal_features=goal_outputs['goal_features'],
                trajectory_query=trajectory_query,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            trajectory_output = self._forward_inference(
                gt_trajs=gt_trajs,
                scene_features=keyval,
                goal_outputs=goal_outputs,
                trajectory_query=trajectory_query,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        
        output.update(trajectory_output)
        return output
    
    def _forward_training(
        self,
        gt_trajs: torch.Tensor,
        scene_features: torch.Tensor,
        goal_features: torch.Tensor,
        trajectory_query: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with flow matching."""
        # Prepare trajectories
        if gt_trajs.shape[-2] == 12:
            gt_trajs = gt_trajs[..., :11, :]
        
        if self._config.start:
            start_point = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
            gt_trajs = torch.cat([start_point, gt_trajs], dim=1)
        
        # Normalize trajectories
        normal_trajs = self._normalize_trajectory(gt_trajs)
        
        # Generate noise
        noise = torch.randn(
            size=(batch_size, 12, 30),
            device=device,
            dtype=dtype
        ) * self._config.train_scale
        
        if self._config.start:
            noise[:, [0], :] = normal_trajs[:, [0], :]
        
        # Sample timestep and interpolate
        t = torch.rand(batch_size, 1, 1, device=device)
        z_t = t * normal_trajs + (1. - t) * noise
        target = normal_trajs - noise
        
        timesteps = t.squeeze(-1) * self._config.infer_steps
        
        # Random CFG dropout during training
        import random
        dropout_flag = random.random()
        
        if dropout_flag < 0.33:  # Unconditional
            pred = self.trajectory_decoder(
                noisy_trajectory=z_t,
                timestep=timesteps,
                scene_features=scene_features,
                goal_features=goal_features,
                force_dropout=True,
            )
        elif dropout_flag < 0.67:  # Conditional
            pred = self.trajectory_decoder(
                noisy_trajectory=z_t,
                timestep=timesteps,
                scene_features=scene_features,
                goal_features=goal_features,
                force_dropout=False,
                goal_dropout=False,
            )
        else:  # Goal dropout (for CFG)
            pred = self.trajectory_decoder(
                noisy_trajectory=z_t,
                timestep=timesteps,
                scene_features=scene_features,
                goal_features=goal_features,
                force_dropout=False,
                goal_dropout=True,
            )
        
        return {
            'trajectory': pred.reshape(batch_size, -1, 30),
            'target': target,
        }
    
    def _forward_inference(
        self,
        gt_trajs: torch.Tensor,
        scene_features: torch.Tensor,
        goal_outputs: Dict[str, torch.Tensor],
        trajectory_query: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Inference with ODE-based trajectory solving and CFG."""
        # Select top-K goals
        k = getattr(self._config, 'topk', 8)
        top_k_goals, top_k_goal_features, _ = self.goal_decoder.select_top_k_goals(
            goal_outputs,
            k=k,
            im_weight=getattr(self._config, 'im_weight', 0.1),
            dac_weight=getattr(self._config, 'dac_weight', 3.0),
            distance_weight=getattr(self._config, 'distance_weight', 0.0),
        )
        
        # Prepare trajectory normalization reference
        if gt_trajs.shape[-2] == 12:
            gt_trajs = gt_trajs[..., :11, :]
        
        if self._config.start:
            start_point = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
            gt_trajs_ref = torch.cat([start_point, gt_trajs], dim=1)
        else:
            gt_trajs_ref = gt_trajs
        
        normal_ref = self._normalize_trajectory(gt_trajs_ref)
        
        # Initialize noise for multiple trajectories per goal
        anchor_size = getattr(self._config, 'anchor_size', 10)
        noise = torch.randn(
            size=(batch_size * anchor_size, 12, 30),
            device=device,
            dtype=dtype
        ) * self._config.test_scale
        
        if self._config.start:
            noise[:, [0], :] = normal_ref[:1, [0], :].repeat(batch_size * anchor_size, 1, 1)
        
        # Expand features for multiple samples
        scene_features_expanded = scene_features.unsqueeze(1).repeat(
            1, anchor_size, 1, 1
        ).view(-1, scene_features.shape[1], scene_features.shape[2])
        
        # Use mean of top-K goals
        mean_goal_features = top_k_goal_features.mean(dim=1, keepdim=True)
        goal_features_expanded = mean_goal_features.unsqueeze(1).repeat(
            1, anchor_size, 1, 1
        ).view(-1, mean_goal_features.shape[1], mean_goal_features.shape[2])
        
        # ODE solving with curved sampling
        trajs = noise
        infer_steps = self._config.infer_steps
        
        if getattr(self._config, 'cur_sampling', False):
            # Curved timestep schedule
            alpha = getattr(self._config, 'alpha', 3.0)
            timesteps = torch.linspace(0, 1, infer_steps + 1, device=device)
            t_shifted = 1 - (alpha * timesteps) / (1 + (alpha - 1) * timesteps)
            t_shifted = t_shifted.flip(0) * infer_steps
            
            for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                step = t_prev - t_curr
                
                # Use CFG if enabled
                cfg_scale = getattr(self._config, 'cfg_scale', 1.0)
                if cfg_scale > 1.0:
                    net_output = self.trajectory_decoder.forward_with_cfg(
                        noisy_trajectory=trajs,
                        timestep=t_curr.unsqueeze(0).repeat(trajs.shape[0]),
                        scene_features=scene_features_expanded,
                        goal_features=goal_features_expanded,
                        guidance_scale=cfg_scale,
                    )
                else:
                    net_output = self.trajectory_decoder(
                        noisy_trajectory=trajs,
                        timestep=t_curr.unsqueeze(0).repeat(trajs.shape[0]),
                        scene_features=scene_features_expanded,
                        goal_features=goal_features_expanded,
                    )
                
                trajs = trajs.detach().clone() + net_output * (step / infer_steps)
        else:
            # Linear timestep schedule
            timesteps = torch.linspace(infer_steps, infer_steps, 1, device=device)
            for t in timesteps:
                net_output = self.trajectory_decoder(
                    noisy_trajectory=trajs,
                    timestep=t.unsqueeze(0).repeat(trajs.shape[0]),
                    scene_features=scene_features_expanded,
                    goal_features=goal_features_expanded,
                )
                trajs = trajs.detach().clone() + net_output * (1 / infer_steps)
        
        # Denormalize trajectories
        pred_trajs = self._denormalize_trajectory(trajs)
        pred_trajs = pred_trajs.reshape(batch_size, anchor_size, -1, 3)
        
        # Select best trajectory (mean or closest to goal)
        if getattr(self._config, 'use_nearest', False):
            # Select trajectory closest to mean goal
            distances = torch.norm(
                pred_trajs[:, :, 8, :2] - top_k_goals[:, :, :2].mean(dim=1, keepdim=True),
                dim=-1
            )
            min_idx = torch.argmin(distances, dim=1)
            pred_trajs = pred_trajs[torch.arange(batch_size), min_idx].unsqueeze(1)
        else:
            # Take mean
            pred_trajs = pred_trajs.mean(dim=1, keepdim=True)
        
        # Extract final trajectory
        if self._config.start:
            pred = pred_trajs[:, :, 1:9, :].mean(1)
        else:
            pred = pred_trajs[:, :, :8, :].mean(1)
        
        return {
            'trajectory': pred,
            'target': torch.zeros_like(pred),  # Placeholder
            'selected_goals': top_k_goals,
        }
    
    def _normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalize trajectory with rotation augmentation."""
        N = trajectory.shape[1]
        downsample_trajectory = trajectory[:, :N, :].clone()
        
        x_scale = 60.0
        y_scale = 15.0
        heading_scale = math.pi
        
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:, :, 2] /= heading_scale
        downsample_trajectory[:, :, 2] = downsample_trajectory[:, :, 2].atanh()
        
        # Apply 10 rotations
        rotated_trajectories = []
        for i in range(10):
            theta = 2 * math.pi * i / 10
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ], device=trajectory.device, dtype=trajectory.dtype)
            
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(
                downsample_trajectory.size(0), -1, -1
            )
            
            rotated_xy = torch.einsum(
                'bij,bkj->bik',
                rotation_matrix,
                downsample_trajectory[:, :, :2]
            )
            rotated_traj = torch.cat(
                [rotated_xy, downsample_trajectory[:, :, -1:].permute(0, 2, 1)],
                dim=1
            )
            rotated_trajectories.append(rotated_traj)
        
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0, 2, 1)
        return trajectory
    
    def _denormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalize trajectory (inverse of normalize)."""
        inverse_rotated_trajectories = []
        
        for i in range(10):
            theta = 2 * math.pi * i / 10
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            inverse_rotation_matrix = torch.tensor([
                [cos_theta, sin_theta],
                [-sin_theta, cos_theta]
            ], device=trajectory.device, dtype=trajectory.dtype)
            
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(
                trajectory.size(0), -1, -1
            )
            
            inverse_rotated_xy = torch.einsum(
                'bij,bkj->bik',
                inverse_rotation_matrix,
                trajectory[:, :, 3*i:3*i+2]
            )
            inverse_rotated_traj = torch.cat(
                [inverse_rotated_xy, trajectory[:, :, 3*i+2:3*i+3].permute(0, 2, 1)],
                dim=1
            )
            inverse_rotated_trajectories.append(inverse_rotated_traj)
        
        final_trajectory = torch.cat(inverse_rotated_trajectories, 1).permute(0, 2, 1)
        final_trajectory = final_trajectory[:, :, :3]
        
        final_trajectory[:, :, 0] *= 60.0
        final_trajectory[:, :, 1] *= 15.0
        final_trajectory[:, :, 2] = final_trajectory[:, :, 2].tanh() * math.pi
        
        return final_trajectory


class AgentHead(nn.Module):
    """Agent detection head."""
    
    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super().__init__()
        
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        
        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )
        
        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )
    
    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )
        
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        
        return {"agent_states": agent_states, "agent_labels": agent_labels}
