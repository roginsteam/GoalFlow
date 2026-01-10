"""
Unified loss function for trajectory-supervised goal learning.

Extends the original GoalFlow loss to support:
- Dynamic query-based goal learning
- Combined imitation and DAC losses for goal supervision
- Trajectory flow matching losses
"""

from typing import Dict
import torch
import torch.nn.functional as F

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_loss import _agent_loss
from navsim.agents.goalflow.goal_decoder import (
    goal_imitation_loss,
    goal_dac_loss,
    goal_confidence_loss,
)


def unified_goalflow_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: GoalFlowConfig
) -> Dict[str, torch.Tensor]:
    """
    Unified loss function for trajectory-supervised goal learning.
    
    Args:
        targets: Dictionary of ground truth targets
        predictions: Dictionary of model predictions
        config: GoalFlow configuration
        
    Returns:
        Dictionary of loss components and total loss
    """
    loss_dict = {
        'bev_semantic_loss': None,
        'agent_class_loss': None,
        'agent_box_loss': None,
        'trajectory_loss': None,
        'goal_imitation_loss': None,
        'goal_dac_loss': None,
        'goal_confidence_loss': None,
        'loss': None,
    }
    
    # BEV semantic segmentation loss (always computed)
    loss_dict['bev_semantic_loss'] = F.cross_entropy(
        predictions["bev_semantic_map"],
        targets["bev_semantic_map"].long()
    )
    
    # ============ Perception Only Mode ============
    if config.only_perception:
        loss_dict['agent_class_loss'], loss_dict['agent_box_loss'] = _agent_loss(
            targets, predictions, config
        )
        loss_dict['loss'] = (
            config.agent_class_weight * loss_dict['agent_class_loss']
            + config.agent_box_weight * loss_dict['agent_box_loss']
            + config.bev_semantic_weight * loss_dict['bev_semantic_loss']
        )
        return loss_dict
    
    # ============ Goal Learning Mode ============
    # Check if we're training goal decoder (has goal-specific outputs)
    has_goal_outputs = (
        'goal_positions' in predictions and
        'imitation_scores' in predictions and
        'dac_scores' in predictions
    )
    
    if has_goal_outputs and config.training:
        # Extract ground truth goal (typically the end point of trajectory)
        gt_trajs = targets.get('trajectory', predictions.get('gt_trajectory'))
        if gt_trajs is not None:
            # Use the goal point (e.g., 8th timestep or last point)
            goal_timestep = getattr(config, 'goal_timestep', 7)  # 0-indexed, so 7 = 8th point
            gt_goal = gt_trajs[:, goal_timestep:goal_timestep+1, :]  # [batch, 1, 3]
            
            # Goal imitation loss (supervise position prediction)
            loss_dict['goal_imitation_loss'] = goal_imitation_loss(
                predicted_scores=predictions['imitation_scores'],
                predicted_positions=predictions['goal_positions'],
                gt_goal=gt_goal,
            )
            
            # Goal confidence loss (supervise confidence prediction)
            loss_dict['goal_confidence_loss'] = goal_confidence_loss(
                predicted_confidences=predictions['goal_confidences'],
                predicted_positions=predictions['goal_positions'],
                gt_goal=gt_goal,
                threshold=getattr(config, 'goal_confidence_threshold', 5.0),
            )
        
        # Goal DAC loss (supervise drivable area compliance)
        if 'dac_labels' in targets:
            loss_dict['goal_dac_loss'] = goal_dac_loss(
                predicted_dac_scores=predictions['dac_scores'],
                gt_dac_labels=targets['dac_labels'],
            )
        elif 'dac_score_feature' in targets:
            # Fallback to feature-based DAC (for backward compatibility)
            loss_dict['goal_dac_loss'] = goal_dac_loss(
                predicted_dac_scores=predictions['dac_scores'],
                gt_dac_labels=targets['dac_score_feature'],
            )
        else:
            # No DAC supervision available
            loss_dict['goal_dac_loss'] = torch.tensor(0.0, device=predictions['dac_scores'].device)
    
    # ============ Trajectory Generation Mode ============
    if 'trajectory' in predictions and 'target' in predictions:
        # Flow matching loss (L1 between predicted and target velocity)
        loss_dict['trajectory_loss'] = F.l1_loss(
            predictions["trajectory"],
            predictions['target']
        )
        
        # Optional agent detection loss
        if getattr(config, 'agent_loss', True):
            loss_dict['agent_class_loss'], loss_dict['agent_box_loss'] = _agent_loss(
                targets, predictions, config
            )
        else:
            loss_dict['agent_class_loss'] = torch.tensor(0.0)
            loss_dict['agent_box_loss'] = torch.tensor(0.0)
    
    # ============ Compute Total Loss ============
    total_loss = torch.tensor(0.0, device=predictions['bev_semantic_map'].device)
    
    # BEV semantic
    total_loss += config.bev_semantic_weight * loss_dict['bev_semantic_loss']
    
    # Agent detection
    if loss_dict['agent_class_loss'] is not None:
        total_loss += config.agent_class_weight * loss_dict['agent_class_loss']
    if loss_dict['agent_box_loss'] is not None:
        total_loss += config.agent_box_weight * loss_dict['agent_box_loss']
    
    # Trajectory generation
    if loss_dict['trajectory_loss'] is not None:
        total_loss += config.trajectory_weight * loss_dict['trajectory_loss']
    
    # Goal learning
    goal_weight = getattr(config, 'goal_weight', 1.0)
    if loss_dict['goal_imitation_loss'] is not None:
        im_weight = getattr(config, 'goal_imitation_weight', 10.0)
        total_loss += goal_weight * im_weight * loss_dict['goal_imitation_loss']
    if loss_dict['goal_dac_loss'] is not None:
        dac_weight = getattr(config, 'goal_dac_weight', 10.0)
        total_loss += goal_weight * dac_weight * loss_dict['goal_dac_loss']
    if loss_dict['goal_confidence_loss'] is not None:
        conf_weight = getattr(config, 'goal_confidence_weight', 1.0)
        total_loss += goal_weight * conf_weight * loss_dict['goal_confidence_loss']
    
    loss_dict['loss'] = total_loss
    
    return loss_dict
