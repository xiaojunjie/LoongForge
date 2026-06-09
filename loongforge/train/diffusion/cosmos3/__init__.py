# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Cosmos3 Video Foundation Model Training

基于 Cosmos Framework 的视频生成模型训练框架。

特性:
- FSDP 数据并行
- Context Parallel (CP)
- LoRA 微调
- EMA 权重管理
"""

from .trainer import Cosmos3Trainer, register_cosmos3_trainer
from .model import cosmos3_model_provider
from .data import cosmos3_dataloader_provider
from .config import get_cosmos3_config, Cosmos3Config

__all__ = [
    # 训练器
    "Cosmos3Trainer",
    "register_cosmos3_trainer",
    
    # 模型和数据
    "cosmos3_model_provider",
    "cosmos3_dataloader_provider",
    
    # 配置
    "get_cosmos3_config",
    "Cosmos3Config",
]

# 自动注册训练器
register_cosmos3_trainer()
