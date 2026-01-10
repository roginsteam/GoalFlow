"""
Multi-Modal DiT (Diffusion Transformer) Decoder

Extends Flow-Planner architecture with:
- Joint attention across agents, lanes, and implicit trajectory tokens
- CFG (Classifier-Free Guidance) support for goal-conditioned trajectory planning
- Integration with trajectory-supervised goal learning
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import math

from navsim.agents.goalflow.diffusion_es import (
    SinusoidalPosEmb,
    RotaryPositionEncoding,
    ParallelAttentionLayer,
)


class MultiModalDiTDecoder(nn.Module):
    """
    Multi-modal DiT decoder for trajectory generation with goal conditioning.
    
    Features:
    - Joint attention across multiple modalities (agents, lanes, trajectories, goals)
    - CFG support for controllable generation
    - Seamless integration with flow matching
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        trajectory_len: int = 12,
        trajectory_dim: int = 30,  # 10 rotations * 3 (x, y, heading)
        dropout: float = 0.1,
        cfg_dropout_prob: float = 0.1,
    ):
        """
        Initialize Multi-Modal DiT Decoder.
        
        Args:
            d_model: Feature dimension
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            trajectory_len: Length of trajectory sequence
            trajectory_dim: Dimension of trajectory features (flattened)
            dropout: Dropout rate
            cfg_dropout_prob: Probability of dropping conditions for CFG training
        """
        super().__init__()
        
        self.d_model = d_model
        self.trajectory_len = trajectory_len
        self.trajectory_dim = trajectory_dim
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # Timestep (sigma) encoding for diffusion/flow matching
        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.sigma_proj = nn.Linear(d_model * 2, d_model)
        
        # Trajectory encoding
        self.trajectory_encoder = nn.Linear(trajectory_dim, d_model)
        
        # Type embeddings for different modalities
        self.type_embedding = nn.Embedding(10, d_model)
        # 0: scene features
        # 1: trajectory tokens
        # 2: goal features
        # 3: agent features
        # 4: lane features
        
        # Rotary position encoding for temporal information
        self.trajectory_time_embeddings = RotaryPositionEncoding(d_model)
        
        # Multi-modal attention layers with joint attention
        self.attention_layers = nn.ModuleList([
            MultiModalAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, trajectory_dim)
        )
        
    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        scene_features: torch.Tensor,
        goal_features: Optional[torch.Tensor] = None,
        agent_features: Optional[torch.Tensor] = None,
        lane_features: Optional[torch.Tensor] = None,
        force_dropout: bool = False,
        goal_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with multi-modal conditioning.
        
        Args:
            noisy_trajectory: Noisy trajectory [batch, trajectory_len, trajectory_dim]
            timestep: Diffusion timestep [batch] or [batch, 1]
            scene_features: Scene encoding features [batch, seq_len, d_model]
            goal_features: Optional goal features [batch, num_goals, d_model]
            agent_features: Optional agent features [batch, num_agents, d_model]
            lane_features: Optional lane features [batch, num_lanes, d_model]
            force_dropout: Force drop all conditions (for unconditional training)
            goal_dropout: Drop only goal condition (for CFG)
            
        Returns:
            Predicted velocity/flow field [batch, trajectory_len, trajectory_dim]
        """
        batch_size = noisy_trajectory.shape[0]
        device = noisy_trajectory.device
        
        # Encode trajectory
        trajectory_features = self.trajectory_encoder(noisy_trajectory)
        # [batch, trajectory_len, d_model]
        
        # Prepare type embeddings
        traj_type_embed = self.type_embedding(
            torch.tensor([1], device=device)
        ).unsqueeze(0).repeat(batch_size, self.trajectory_len, 1)
        
        scene_type_embed = self.type_embedding(
            torch.tensor([0], device=device)
        ).unsqueeze(0).repeat(batch_size, scene_features.shape[1], 1)
        
        # Collect all features and type embeddings
        all_features = [trajectory_features]
        all_type_embeddings = [traj_type_embed]
        
        # Add scene features
        if not force_dropout:
            all_features.append(scene_features)
            all_type_embeddings.append(scene_type_embed)
        
        # Add goal features with CFG dropout
        if goal_features is not None and not force_dropout and not goal_dropout:
            goal_type_embed = self.type_embedding(
                torch.tensor([2], device=device)
            ).unsqueeze(0).repeat(batch_size, goal_features.shape[1], 1)
            all_features.append(goal_features)
            all_type_embeddings.append(goal_type_embed)
        
        # Add agent features
        if agent_features is not None and not force_dropout:
            agent_type_embed = self.type_embedding(
                torch.tensor([3], device=device)
            ).unsqueeze(0).repeat(batch_size, agent_features.shape[1], 1)
            all_features.append(agent_features)
            all_type_embeddings.append(agent_type_embed)
        
        # Add lane features
        if lane_features is not None and not force_dropout:
            lane_type_embed = self.type_embedding(
                torch.tensor([4], device=device)
            ).unsqueeze(0).repeat(batch_size, lane_features.shape[1], 1)
            all_features.append(lane_features)
            all_type_embeddings.append(lane_type_embed)
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=1)
        all_type_embeddings = torch.cat(all_type_embeddings, dim=1)
        # [batch, total_seq_len, d_model]
        
        # Encode timestep
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)
        timestep = timestep.float() / 100.0  # Normalize by max steps
        sigma_embeddings = self.sigma_encoder(timestep)
        # [batch, d_model]
        
        # Add sigma to all features
        sigma_embeddings_expanded = sigma_embeddings.unsqueeze(1).repeat(1, all_features.shape[1], 1)
        all_features_with_sigma = torch.cat([all_features, sigma_embeddings_expanded], dim=-1)
        all_features = self.sigma_proj(all_features_with_sigma)
        
        # Generate temporal position embeddings
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=device)
        temporal_embedding = self.trajectory_time_embeddings(
            indices.unsqueeze(0).repeat(batch_size, 1)
        )
        
        # Generate attention mask (allow trajectory to attend to everything)
        # But prevent non-trajectory tokens from attending to each other
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        traj_len = self.trajectory_len
        # Allow trajectory tokens to attend to each other within local window
        for i in range(traj_len):
            for j in range(traj_len):
                if abs(i - j) > 1:
                    attn_mask[i, j] = True
        
        # Apply multi-modal attention layers
        for layer in self.attention_layers:
            all_features = layer(
                all_features,
                temporal_pos=temporal_embedding,
                semantic_pos=all_type_embeddings,
                attn_mask=attn_mask,
            )
        
        # Extract trajectory features and decode
        trajectory_output = all_features[:, :self.trajectory_len]
        output = self.decoder_mlp(trajectory_output)
        
        return output
    
    def forward_with_cfg(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        scene_features: torch.Tensor,
        goal_features: torch.Tensor,
        guidance_scale: float = 1.5,
        agent_features: Optional[torch.Tensor] = None,
        lane_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance.
        
        Args:
            noisy_trajectory: Noisy trajectory
            timestep: Diffusion timestep
            scene_features: Scene features
            goal_features: Goal features
            guidance_scale: CFG guidance scale (1.0 = no guidance)
            agent_features: Optional agent features
            lane_features: Optional lane features
            
        Returns:
            Guided prediction
        """
        # Conditional prediction (with goal)
        cond_output = self.forward(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
            agent_features=agent_features,
            lane_features=lane_features,
            goal_dropout=False,
        )
        
        # Unconditional prediction (without goal)
        uncond_output = self.forward(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
            agent_features=agent_features,
            lane_features=lane_features,
            goal_dropout=True,
        )
        
        # Apply CFG
        guided_output = uncond_output + guidance_scale * (cond_output - uncond_output)
        
        return guided_output


class MultiModalAttentionLayer(nn.Module):
    """
    Multi-modal attention layer with joint attention across modalities.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = ParallelAttentionLayer(
            d_model=d_model,
            n_heads=num_heads,
            dropout=dropout,
            self_attention1=True,
            self_attention2=False,
            cross_attention1=False,
            cross_attention2=False,
            rotary_pe=True,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        temporal_pos: torch.Tensor,
        semantic_pos: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            temporal_pos: Temporal position encodings
            semantic_pos: Semantic type embeddings
            attn_mask: Attention mask
            
        Returns:
            Output features [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(
            x, None, None, None,
            seq1_pos=temporal_pos,
            seq1_sem_pos=semantic_pos,
            attn_mask_11=attn_mask,
        )
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
