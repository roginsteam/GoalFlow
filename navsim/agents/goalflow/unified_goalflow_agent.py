"""
Unified GoalFlow Agent with Trajectory-Supervised Goal Learning

This agent integrates:
- Dynamic query-based goal learning
- Multi-modal DiT decoder with CFG support
- Seamless flow matching-based trajectory generation
"""

from typing import Any, List, Dict, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.unified_goalflow_model import UnifiedGoalFlowModel
from navsim.agents.goalflow.unified_goalflow_loss import unified_goalflow_loss
from navsim.agents.goalflow.goalflow_callback import GoalFlowCallback
from navsim.agents.goalflow.goalflow_features import (
    GoalFlowFeatureBuilder,
    GoalFlowTargetBuilder,
)


class UnifiedGoalFlowAgent(AbstractAgent):
    """
    Unified GoalFlow agent with trajectory-supervised goal learning.
    
    Key Features:
    - Replaces fixed vocabulary with learnable goal queries
    - End-to-end goal and trajectory learning
    - CFG support for controllable generation
    - Scalable and maintainable architecture
    """
    
    def __init__(
        self,
        config: GoalFlowConfig,
        lr: float = 1e-4,
        step_size: int = 20,
        gamma: float = 0.8,
        checkpoint_path: str = None,
    ):
        """
        Initialize Unified GoalFlow Agent.
        
        Args:
            config: GoalFlow configuration
            lr: Learning rate
            step_size: Step size for learning rate scheduler
            gamma: Gamma for learning rate scheduler
            checkpoint_path: Path to checkpoint for initialization
        """
        super().__init__()
        
        self._config = config
        self._lr = lr
        self._step_size = step_size
        self._gamma = gamma
        self._checkpoint_path = checkpoint_path
        
        # Initialize unified model
        self._goalflow_model = UnifiedGoalFlowModel(config)
    
    def name(self) -> str:
        """Return agent name."""
        return self.__class__.__name__
    
    def initialize(self) -> None:
        """Initialize agent from checkpoint if provided."""
        if self._checkpoint_path is None:
            return
        
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path,
                map_location=torch.device("cpu")
            )["state_dict"]
        
        # Remove 'agent.' prefix from keys
        cleaned_state_dict = {
            k.replace("agent.", ""): v for k, v in state_dict.items()
        }
        
        # Load state dict with strict=False to allow partial loading
        # (e.g., when adding new goal decoder components)
        self.load_state_dict(cleaned_state_dict, strict=False)
    
    def get_sensor_config(self) -> SensorConfig:
        """Return sensor configuration."""
        return SensorConfig.build_all_sensors(include=[3])
    
    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Return target builders."""
        return [GoalFlowTargetBuilder(config=self._config)]
    
    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Return feature builders."""
        return [GoalFlowFeatureBuilder(config=self._config)]
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features
            targets: Ground truth targets
            
        Returns:
            Model predictions
        """
        return self._goalflow_model(features, targets)
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            features: Input features
            targets: Ground truth targets
            predictions: Model predictions
            
        Returns:
            Loss dictionary
        """
        return unified_goalflow_loss(targets, predictions, self._config)
    
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Get optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._goalflow_model.parameters()),
            lr=self._lr
        )
        
        scheduler = StepLR(optimizer, step_size=self._step_size, gamma=self._gamma)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Get training callbacks.
        
        Returns:
            List of callbacks
        """
        return [
            GoalFlowCallback(self._config),
            pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1)
        ]
