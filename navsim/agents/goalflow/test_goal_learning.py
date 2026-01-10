"""
Tests for trajectory-supervised goal learning components.
"""

import torch
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goal_decoder import (
    GoalDecoder,
    goal_imitation_loss,
    goal_dac_loss,
    goal_confidence_loss,
)
from navsim.agents.goalflow.multimodal_dit_decoder import MultiModalDiTDecoder
from navsim.agents.goalflow.unified_goalflow_model import UnifiedGoalFlowModel


class TestGoalDecoder:
    """Tests for GoalDecoder module."""
    
    def test_goal_decoder_initialization(self):
        """Test goal decoder initializes correctly."""
        decoder = GoalDecoder(
            d_model=256,
            num_queries=128,
            num_decoder_layers=3,
        )
        assert decoder.num_queries == 128
        assert decoder.d_model == 256
    
    def test_goal_decoder_forward(self):
        """Test goal decoder forward pass."""
        batch_size = 2
        seq_len = 64
        d_model = 256
        
        decoder = GoalDecoder(d_model=d_model, num_queries=128)
        
        scene_features = torch.randn(batch_size, seq_len, d_model)
        status_encoding = torch.randn(batch_size, d_model)
        
        outputs = decoder(scene_features, status_encoding)
        
        # Check output shapes
        assert outputs['goal_positions'].shape == (batch_size, 128, 3)
        assert outputs['goal_confidences'].shape == (batch_size, 128, 1)
        assert outputs['imitation_scores'].shape == (batch_size, 128, 1)
        assert outputs['dac_scores'].shape == (batch_size, 128, 1)
        assert outputs['goal_features'].shape == (batch_size, 128, d_model)
        
        # Check position ranges
        positions = outputs['goal_positions']
        assert positions[..., 0].abs().max() <= 60.0  # x in [-60, 60]
        assert positions[..., 1].abs().max() <= 60.0  # y in [-60, 60]
        assert positions[..., 2].abs().max() <= 3.15  # heading in [-pi, pi]
    
    def test_goal_decoder_top_k_selection(self):
        """Test top-K goal selection."""
        batch_size = 2
        decoder = GoalDecoder(d_model=256, num_queries=128)
        
        # Create mock outputs
        goal_outputs = {
            'goal_positions': torch.randn(batch_size, 128, 3),
            'goal_features': torch.randn(batch_size, 128, 256),
            'imitation_scores': torch.randn(batch_size, 128, 1),
            'dac_scores': torch.randn(batch_size, 128, 1),
        }
        
        k = 8
        top_k_pos, top_k_feat, top_k_idx = decoder.select_top_k_goals(
            goal_outputs, k=k
        )
        
        # Check shapes
        assert top_k_pos.shape == (batch_size, k, 3)
        assert top_k_feat.shape == (batch_size, k, 256)
        assert top_k_idx.shape == (batch_size, k)
        
        # Check indices are in valid range
        assert (top_k_idx >= 0).all()
        assert (top_k_idx < 128).all()


class TestGoalLosses:
    """Tests for goal learning loss functions."""
    
    def test_goal_imitation_loss(self):
        """Test goal imitation loss computation."""
        batch_size = 2
        num_queries = 128
        
        predicted_scores = torch.randn(batch_size, num_queries, 1)
        predicted_positions = torch.randn(batch_size, num_queries, 3)
        gt_goal = torch.randn(batch_size, 1, 3)
        
        loss = goal_imitation_loss(predicted_scores, predicted_positions, gt_goal)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Non-negative
    
    def test_goal_dac_loss(self):
        """Test DAC loss computation."""
        batch_size = 2
        num_queries = 128
        
        predicted_dac_scores = torch.randn(batch_size, num_queries, 1)
        gt_dac_labels = torch.randint(0, 2, (batch_size, num_queries)).float()
        
        loss = goal_dac_loss(predicted_dac_scores, gt_dac_labels)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Non-negative
    
    def test_goal_confidence_loss(self):
        """Test confidence loss computation."""
        batch_size = 2
        num_queries = 128
        
        predicted_confidences = torch.randn(batch_size, num_queries, 1)
        predicted_positions = torch.randn(batch_size, num_queries, 3)
        gt_goal = torch.randn(batch_size, 1, 3)
        
        loss = goal_confidence_loss(
            predicted_confidences, predicted_positions, gt_goal, threshold=5.0
        )
        
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Non-negative


class TestMultiModalDiTDecoder:
    """Tests for Multi-Modal DiT Decoder."""
    
    def test_dit_decoder_initialization(self):
        """Test DiT decoder initializes correctly."""
        decoder = MultiModalDiTDecoder(
            d_model=256,
            num_layers=8,
            trajectory_len=12,
        )
        assert decoder.d_model == 256
        assert decoder.trajectory_len == 12
    
    def test_dit_decoder_forward(self):
        """Test DiT decoder forward pass."""
        batch_size = 2
        trajectory_len = 12
        trajectory_dim = 30
        d_model = 256
        
        decoder = MultiModalDiTDecoder(
            d_model=d_model,
            trajectory_len=trajectory_len,
            trajectory_dim=trajectory_dim,
        )
        
        noisy_trajectory = torch.randn(batch_size, trajectory_len, trajectory_dim)
        timestep = torch.tensor([50, 50])
        scene_features = torch.randn(batch_size, 64, d_model)
        goal_features = torch.randn(batch_size, 8, d_model)
        
        output = decoder(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
        )
        
        # Check output shape
        assert output.shape == (batch_size, trajectory_len, trajectory_dim)
    
    def test_dit_decoder_cfg(self):
        """Test Classifier-Free Guidance."""
        batch_size = 2
        trajectory_len = 12
        trajectory_dim = 30
        d_model = 256
        
        decoder = MultiModalDiTDecoder(
            d_model=d_model,
            trajectory_len=trajectory_len,
            trajectory_dim=trajectory_dim,
        )
        
        noisy_trajectory = torch.randn(batch_size, trajectory_len, trajectory_dim)
        timestep = torch.tensor([50, 50])
        scene_features = torch.randn(batch_size, 64, d_model)
        goal_features = torch.randn(batch_size, 8, d_model)
        
        output = decoder.forward_with_cfg(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
            guidance_scale=1.5,
        )
        
        # Check output shape
        assert output.shape == (batch_size, trajectory_len, trajectory_dim)
    
    def test_dit_decoder_dropout_modes(self):
        """Test different dropout modes for CFG training."""
        batch_size = 2
        trajectory_len = 12
        trajectory_dim = 30
        d_model = 256
        
        decoder = MultiModalDiTDecoder(
            d_model=d_model,
            trajectory_len=trajectory_len,
            trajectory_dim=trajectory_dim,
        )
        
        noisy_trajectory = torch.randn(batch_size, trajectory_len, trajectory_dim)
        timestep = torch.tensor([50, 50])
        scene_features = torch.randn(batch_size, 64, d_model)
        goal_features = torch.randn(batch_size, 8, d_model)
        
        # Test unconditional (force_dropout=True)
        output_uncond = decoder(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
            force_dropout=True,
        )
        assert output_uncond.shape == (batch_size, trajectory_len, trajectory_dim)
        
        # Test goal dropout (goal_dropout=True)
        output_goal_drop = decoder(
            noisy_trajectory=noisy_trajectory,
            timestep=timestep,
            scene_features=scene_features,
            goal_features=goal_features,
            goal_dropout=True,
        )
        assert output_goal_drop.shape == (batch_size, trajectory_len, trajectory_dim)


class TestUnifiedGoalFlowModel:
    """Tests for Unified GoalFlow Model."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = GoalFlowConfig()
        config.num_goal_queries = 32  # Smaller for faster tests
        config.dit_num_layers = 2  # Fewer layers for tests
        config.only_perception = False
        config.freeze_perception = False
        config.training = True
        config.use_unified_model = True
        return config
    
    def test_model_initialization(self, config):
        """Test unified model initializes correctly."""
        model = UnifiedGoalFlowModel(config)
        
        # Check components exist
        assert hasattr(model, '_backbone')
        assert hasattr(model, 'goal_decoder')
        assert hasattr(model, 'trajectory_decoder')
    
    @pytest.mark.skip(reason="Requires full feature setup")
    def test_model_forward_training(self, config):
        """Test model forward pass in training mode."""
        # This test is skipped as it requires complex feature setup
        # In practice, this would be tested through integration tests
        pass
    
    @pytest.mark.skip(reason="Requires full feature setup")
    def test_model_forward_inference(self, config):
        """Test model forward pass in inference mode."""
        # This test is skipped as it requires complex feature setup
        # In practice, this would be tested through integration tests
        pass


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_tests()
