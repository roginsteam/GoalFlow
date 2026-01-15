"""
Generate trajectories using the unified query-based GoalFlow model.

This script is adapted for the trajectory-supervised goal learning system with:
- Dynamic learnable queries (128) instead of fixed vocabulary (8,192 points)
- Multi-modal DiT decoder with CFG support
- End-to-end goal and trajectory generation
"""

from typing import Tuple
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "queryflow_training"

# use to limit len gt_trajs, can ignore or delete it.
def limit_len_collate_fn(batch):
    features = {}
    targets = {}
    for feature_name in batch[0][0].keys():
        if feature_name == 'gt_trajs':
            features.update({feature_name: torch.stack([item[0][feature_name][..., :10, :] for item in batch], dim=0)})
        elif feature_name == 'token':
            features.update({feature_name: [item[0][feature_name] for item in batch]})
        else:
            features.update({feature_name: torch.stack([item[0][feature_name] for item in batch], dim=0)})
    for key in batch[0][1].keys():
        if key == 'trajectory':
            targets.update({key: torch.stack([item[1][key][..., :10, :] for item in batch], dim=0)})
        else:
            targets.update({key: torch.stack([item[1][key] for item in batch], dim=0)})
    
    return (features, targets)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    print()
    logger.info("=" * 80)
    logger.info("QueryFlow Trajectory Generation with Dynamic Goal Learning")
    logger.info("=" * 80)
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Unified QueryFlow Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    
    # Log model configuration
    if hasattr(agent, '_config'):
        config = agent._config
        logger.info("QueryFlow Configuration:")
        logger.info(f"  - Use Unified Model: {getattr(config, 'use_unified_model', False)}")
        logger.info(f"  - Goal Queries: {getattr(config, 'num_goal_queries', 128)}")
        logger.info(f"  - DiT Layers: {getattr(config, 'dit_num_layers', 8)}")
        logger.info(f"  - CFG Scale: {getattr(config, 'cfg_scale', 1.0)}")
        logger.info(f"  - Top-K Goals: {getattr(config, 'topk', 8)}")

    logger.info("Building Lightning Module")
    if cfg.agent.checkpoint_path:
        logger.info(f"Loading checkpoint from: {cfg.agent.checkpoint_path}")
        lightning_module = AgentLightningModule.load_from_checkpoint(
            agent=agent,
            checkpoint_path=cfg.agent.checkpoint_path,
            strict=False
        )
    else:
        logger.info("No checkpoint provided, using random initialization")
        lightning_module = AgentLightningModule(
            agent=agent,
        )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation == False, \
            "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, \
            "cache_path must be provided when using cached data without building SceneLoader"
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.test_logs,
        )
    else:
        logger.info("Building Datasets")
        _, val_data = build_datasets(cfg, agent)

    logger.info("Building Dataloader")
    val_dataloader = DataLoader(
        val_data,
        **cfg.dataloader.params,
        shuffle=False,
        collate_fn=limit_len_collate_fn
    )
    logger.info("Num validation samples: %d", len(val_data))
    
    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    logger.info("=" * 80)
    logger.info("Starting Trajectory Generation")
    logger.info("=" * 80)

    trainer.test(
        model=lightning_module,
        dataloaders=val_dataloader
    )
    
    logger.info("=" * 80)
    logger.info("Trajectory Generation Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
