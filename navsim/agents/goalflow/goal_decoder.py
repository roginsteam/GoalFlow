"""
Trajectory-Supervised Goal Decoder Module

This module implements dynamic query-based goal learning that replaces
fixed vocabulary dictionaries with learnable queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GoalDecoder(nn.Module):
    """
    Dynamic query-based goal decoder that learns potential goals from trajectory data.
    
    Features:
    - Learnable goal queries instead of fixed dictionaries
    - Position and confidence prediction for each query
    - Supervised learning from ground truth trajectories
    - Top-K goal selection for inference
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 128,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize Goal Decoder.
        
        Args:
            d_model: Feature dimension
            num_queries: Number of learnable goal queries
            num_decoder_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learnable goal queries
        self.goal_queries = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.goal_queries.weight, mean=0.0, std=0.02)
        
        # Transformer decoder for query refinement
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Goal position prediction head (x, y, heading)
        self.position_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 3)  # x, y, heading
        )
        
        # Goal confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Imitation score head (for trajectory supervision)
        self.imitation_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # DAC (Drivable Area Compliance) score head
        self.dac_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
    def forward(
        self,
        scene_features: torch.Tensor,
        status_encoding: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of goal decoder.
        
        Args:
            scene_features: BEV scene features [batch, seq_len, d_model]
            status_encoding: Optional ego status encoding [batch, d_model]
            
        Returns:
            Dictionary containing:
                - goal_positions: Predicted goal positions [batch, num_queries, 3]
                - goal_confidences: Confidence scores [batch, num_queries, 1]
                - imitation_scores: Trajectory imitation scores [batch, num_queries, 1]
                - dac_scores: Drivable area compliance scores [batch, num_queries, 1]
                - goal_features: Refined goal query features [batch, num_queries, d_model]
        """
        batch_size = scene_features.shape[0]
        
        # Get learnable queries
        queries = self.goal_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # [batch, num_queries, d_model]
        
        # Add status encoding if provided
        if status_encoding is not None:
            queries = queries + status_encoding.unsqueeze(1)
        
        # Refine queries with scene context via transformer decoder
        goal_features = self.transformer_decoder(queries, scene_features)
        # [batch, num_queries, d_model]
        
        # Predict goal positions (x, y, heading)
        goal_positions = self.position_head(goal_features)
        # Scale positions appropriately
        goal_positions[..., :2] = torch.tanh(goal_positions[..., :2]) * 60.0  # x, y in meters
        goal_positions[..., 2] = torch.tanh(goal_positions[..., 2]) * 3.14159  # heading in radians
        
        # Predict confidence scores
        goal_confidences = self.confidence_head(goal_features)
        
        # Predict imitation scores (for supervision)
        imitation_scores = self.imitation_head(goal_features)
        
        # Predict DAC scores
        dac_scores = self.dac_head(goal_features)
        
        return {
            'goal_positions': goal_positions,
            'goal_confidences': goal_confidences,
            'imitation_scores': imitation_scores,
            'dac_scores': dac_scores,
            'goal_features': goal_features,
        }
    
    def select_top_k_goals(
        self,
        goal_outputs: Dict[str, torch.Tensor],
        k: int = 8,
        im_weight: float = 0.1,
        dac_weight: float = 3.0,
        distance_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-K goals based on combined scores.
        
        Args:
            goal_outputs: Dictionary from forward pass
            k: Number of top goals to select
            im_weight: Weight for imitation score
            dac_weight: Weight for DAC score
            distance_weight: Weight for distance score
            
        Returns:
            Tuple of:
                - top_k_positions: Selected goal positions [batch, k, 3]
                - top_k_features: Selected goal features [batch, k, d_model]
                - top_k_indices: Indices of selected goals [batch, k]
        """
        batch_size = goal_outputs['goal_positions'].shape[0]
        
        # Extract scores
        im_scores = goal_outputs['imitation_scores'].squeeze(-1)  # [batch, num_queries]
        dac_scores = goal_outputs['dac_scores'].squeeze(-1)
        positions = goal_outputs['goal_positions']
        features = goal_outputs['goal_features']
        
        # Compute combined scores
        # Imitation score (log softmax for stability)
        im_score_term = im_weight * torch.log_softmax(im_scores, dim=-1)
        
        # DAC score (log sigmoid)
        dac_score_term = dac_weight * torch.log(torch.sigmoid(dac_scores) + 1e-7)
        
        # Optional: distance score (prefer farther goals for progress)
        if distance_weight > 0.0:
            distances = torch.norm(positions[..., :2], dim=-1)
            # Normalize distances
            dist_min = distances.min(dim=-1, keepdim=True)[0]
            dist_max = distances.max(dim=-1, keepdim=True)[0]
            distances_norm = (distances - dist_min) / (dist_max - dist_min + 1e-7)
            distance_score_term = distance_weight * distances_norm
        else:
            distance_score_term = 0.0
        
        # Combined score
        final_scores = im_score_term + dac_score_term + distance_score_term
        
        # Select top-K
        top_k_values, top_k_indices = torch.topk(final_scores, k, dim=-1)
        
        # Gather top-K positions and features
        batch_indices = torch.arange(batch_size, device=positions.device)[:, None].expand(-1, k)
        
        top_k_positions = positions[batch_indices, top_k_indices]
        top_k_features = features[batch_indices, top_k_indices]
        
        return top_k_positions, top_k_features, top_k_indices


def goal_imitation_loss(
    predicted_scores: torch.Tensor,
    predicted_positions: torch.Tensor,
    gt_goal: torch.Tensor,
) -> torch.Tensor:
    """
    Imitation loss for goal learning.
    
    Supervises goal queries to predict positions close to ground truth goal.
    Uses soft target distribution based on distance to GT.
    
    Args:
        predicted_scores: Predicted imitation scores [batch, num_queries, 1]
        predicted_positions: Predicted goal positions [batch, num_queries, 3]
        gt_goal: Ground truth goal position [batch, 1, 3]
        
    Returns:
        Loss value
    """
    # Compute distances to ground truth
    distances = torch.sum((predicted_positions[..., :2] - gt_goal[..., :2]) ** 2, dim=-1)
    # [batch, num_queries]
    
    # Create soft targets (inverse distance weighting)
    y_i = F.softmax(-distances, dim=-1)
    
    # Predicted distribution
    pred_probs = F.softmax(predicted_scores.squeeze(-1), dim=-1)
    
    # Cross-entropy loss
    loss = -torch.sum(y_i * torch.log(pred_probs + 1e-6), dim=-1)
    
    return loss.mean()


def goal_dac_loss(
    predicted_dac_scores: torch.Tensor,
    gt_dac_labels: torch.Tensor,
) -> torch.Tensor:
    """
    DAC (Drivable Area Compliance) loss for goal learning.
    
    Binary classification loss for whether goal is in drivable area.
    
    Args:
        predicted_dac_scores: Predicted DAC scores [batch, num_queries, 1]
        gt_dac_labels: Ground truth DAC labels [batch, num_queries]
        
    Returns:
        Loss value
    """
    pred_probs = torch.sigmoid(predicted_dac_scores.squeeze(-1))
    
    # Binary cross-entropy
    loss = -(
        gt_dac_labels * torch.log(pred_probs + 1e-6) +
        (1 - gt_dac_labels) * torch.log(1 - pred_probs + 1e-6)
    )
    
    return loss.mean()


def goal_confidence_loss(
    predicted_confidences: torch.Tensor,
    predicted_positions: torch.Tensor,
    gt_goal: torch.Tensor,
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Confidence loss for goal learning.
    
    Supervises confidence to be high for queries close to GT goal.
    
    Args:
        predicted_confidences: Predicted confidence scores [batch, num_queries, 1]
        predicted_positions: Predicted goal positions [batch, num_queries, 3]
        gt_goal: Ground truth goal position [batch, 1, 3]
        threshold: Distance threshold for positive samples (meters)
        
    Returns:
        Loss value
    """
    # Compute distances to ground truth
    distances = torch.norm(predicted_positions[..., :2] - gt_goal[..., :2], dim=-1)
    
    # Binary labels: 1 if close to GT, 0 otherwise
    gt_labels = (distances < threshold).float()
    
    # BCE with logits
    loss = F.binary_cross_entropy_with_logits(
        predicted_confidences.squeeze(-1),
        gt_labels,
        reduction='mean'
    )
    
    return loss
