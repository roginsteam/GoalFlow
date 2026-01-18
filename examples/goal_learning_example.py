"""
Example usage of the trajectory-supervised goal learning system.

This script demonstrates configuration and usage patterns.
Run with: python examples/goal_learning_example.py
"""

import sys
sys.path.insert(0, '.')

try:
    import torch
    from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
    from navsim.agents.goalflow.unified_goalflow_loss import unified_goalflow_loss
except ImportError as e:
    print(f"Import error: {e}")
    print("This script requires torch and the navsim package.")
    sys.exit(1)


def example_configuration():
    """Example: Configure the unified model."""
    config = GoalFlowConfig()
    config.use_unified_model = True
    config.num_goal_queries = 128
    config.dit_num_layers = 8
    config.cfg_scale = 1.5
    return config


def main():
    print("=" * 60)
    print("TRAJECTORY-SUPERVISED GOAL LEARNING SYSTEM")
    print("Configuration Example")
    print("=" * 60)
    
    config = example_configuration()
    print(f"\n✓ Configuration created:")
    print(f"  - Goal queries: {config.num_goal_queries}")
    print(f"  - DiT layers: {config.dit_num_layers}")
    print(f"  - CFG scale: {config.cfg_scale}")
    print("\nSee docs/goal_learning_system.md for full documentation.")


if __name__ == "__main__":
    main()
