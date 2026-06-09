# Simplified VLMConfig - stripped of MoT/processor dependencies
import os
from typing import Any

import attrs
import torch.distributed as dist

from loongforge.utils.cosmos.flags import INTERNAL
from loongforge.utils.cosmos.lazy_config import LazyCall as L, LazyDict, instantiate as lazy_instantiate
from loongforge.utils.cosmos import log
from loongforge.utils.cosmos.config_helper import ConfigStore
from loongforge.utils.cosmos.easy_io import easy_io


def create_vlm_config(base_config: LazyDict, **overrides):
    vlm_config = lazy_instantiate(base_config)
    for key, value in overrides.items():
        setattr(vlm_config, key, value)
    return vlm_config


def get_rank_safe() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def download_tokenizer_files(model_name: str, config_variant: str) -> str:
    _sanitized = model_name.replace("/", "_").replace("-", "_")
    override = (
        os.environ.get(f"COSMOS_TOKENIZER_PATH_{_sanitized}")
        or os.environ.get("COSMOS_TOKENIZER_PATH")
    )
    if override:
        log.info(f"Using local tokenizer override for {model_name}: {override}")
        return override
    if config_variant == "hf":
        return model_name
    raise ValueError(f"Remote download not supported in LoongForge standalone mode. Set COSMOS_TOKENIZER_PATH env var.")


@attrs.define(slots=False)
class PretrainedWeightsConfig:
    enabled: bool = True
    backbone_path: str = ""
    credentials_path: str = ""
    enable_gcs_patch_in_boto3: bool = False
    checkpoint_format: str | None = None


@attrs.define(slots=False)
class VLMConfig:
    """VLM backbone identity."""
    model_name: str = ""
    safetensors_path: str = ""
    pretrained_weights: PretrainedWeightsConfig = PretrainedWeightsConfig()
    model_instance: Any | None = None
    tokenizer: Any | None = None
    layer_module: str | None = None
    qk_norm: bool = False
    tie_word_embeddings: bool = True
    qk_norm_for_text: bool = False
    freeze_und: bool = False
