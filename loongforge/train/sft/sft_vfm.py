"""
Cosmos3 VLMModel standalone training script for LoongForge.
Uses DataPackerDataLoader + BridgeData2DataPacker for proper input_ids/labels format.

Usage:
    torchrun --nproc_per_node=8 --master_port=50095 \
        examples/cosmos3/train_vlm.py \
        --checkpoint-path examples/checkpoints/Cosmos3-Nano \
        --tokenizer-path /path/to/Qwen3-VL-8B-Instruct \
        --dataset-path /path/to/sft_dataset_bridge/train/video_dataset_file.jsonl
"""
import argparse
import os
import sys
import types as _types_module
import pathlib

# === Namespace shims: bypass megatron-dependent __init__.py ===
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

for _pkg_name, _pkg_rel in [
    ("loongforge", "loongforge"),
    ("loongforge.utils", "loongforge/utils"),
    ("loongforge.data", "loongforge/data"),
    ("loongforge.data.cosmos3", "loongforge/data/cosmos3"),
    ("loongforge.train", "loongforge/train"),
    ("loongforge.train.diffusion", "loongforge/train/diffusion"),
    ("loongforge.train.diffusion.cosmos3", "loongforge/train/diffusion/cosmos3"),
    ("loongforge.models", "loongforge/models"),
    ("loongforge.models.diffusion", "loongforge/models/diffusion"),
    ("loongforge.models.diffusion.cosmos3", "loongforge/models/diffusion/cosmos3"),
]:
    if _pkg_name not in sys.modules:
        _m = _types_module.ModuleType(_pkg_name)
        _m.__path__ = [os.path.join(_ROOT, _pkg_rel)]
        _m.__package__ = _pkg_name
        sys.modules[_pkg_name] = _m

# Fix pathlib._local for Python 3.12 loading 3.13+ DCP metadata pickles
_local_mod = _types_module.ModuleType("pathlib._local")
for _attr in dir(pathlib):
    if not _attr.startswith("_"):
        setattr(_local_mod, _attr, getattr(pathlib, _attr))
sys.modules["pathlib._local"] = _local_mod

# === Now safe to import loongforge modules ===
import torch
import torch.distributed as dist
from types import SimpleNamespace

from loongforge.train.diffusion.cosmos3.imaginaire_trainer import ImaginaireTrainer
from loongforge.models.diffusion.cosmos3.vlm_model import VLMModel, VLMModelConfig
from loongforge.train.diffusion.cosmos3.configs_pkg.base.vlm.defaults.policy_config import PolicyConfig
from loongforge.train.diffusion.cosmos3.configs_pkg.base.defaults.parallelism import ParallelismConfig
from loongforge.train.diffusion.cosmos3.configs_pkg.base.defaults.activation_checkpointing import (
    ActivationCheckpointingConfig,
)
from loongforge.train.diffusion.cosmos3.configs_pkg.base.defaults.vlm import VLMConfig
from loongforge.train.diffusion.cosmos3.callbacks_grad_clip import GradClip
from loongforge.train.diffusion.cosmos3.checkpointing_dcp import DistributedCheckpointer
from loongforge.utils.cosmos.lazy_config import LazyCall as L, LazyConfig, PLACEHOLDER
from loongforge.utils.cosmos.optimizer import build_optimizer, build_lr_scheduler
from loongforge.utils.cosmos import log

from loongforge.data.cosmos3.data_packer_dataloader import DataPackerDataLoader
from loongforge.data.cosmos3.bridge_data_packer import BridgeData2Dataset, BridgeData2DataPacker
from loongforge.data.cosmos3.qwen3vl_processor import Qwen3VLProcessor

# Monkey-patch save_yaml/save_pkl to handle SimpleNamespace config
_orig_save_yaml = LazyConfig.save_yaml
_orig_save_pkl = LazyConfig.save_pkl


@staticmethod
def _patched_save_yaml(cfg, filename):
    try:
        return _orig_save_yaml(cfg, filename)
    except Exception:
        pass
    return filename


@staticmethod
def _patched_save_pkl(cfg, filename):
    try:
        return _orig_save_pkl(cfg, filename)
    except Exception:
        pass
    return filename


LazyConfig.save_yaml = _patched_save_yaml
LazyConfig.save_pkl = _patched_save_pkl


class SyntheticPackedDataloader:
    """Yields synthetic packed token batches for VLMModel.training_step (fallback)."""

    def __init__(self, vocab_size=151643, seq_len=2048, num_batches=200):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self._iter_count = 0

    def __iter__(self):
        self._iter_count = 0
        return self

    def __next__(self):
        if self._iter_count >= self.num_batches:
            raise StopIteration
        self._iter_count += 1
        input_ids = torch.randint(0, self.vocab_size, (1, self.seq_len), dtype=torch.long)
        labels = torch.randint(0, self.vocab_size, (1, self.seq_len), dtype=torch.long)
        mask = torch.rand(1, self.seq_len) < 0.3
        labels[mask] = -100
        attention_mask = torch.ones(1, self.seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_real_dataloader(args):
    """Build DataPackerDataLoader with BridgeData2 dataset (produces input_ids/labels)."""
    processor = Qwen3VLProcessor(name=args.tokenizer_path)
    dataset = BridgeData2Dataset(
        jsonl_path=args.dataset_path,
        shuffle=True,
        seed=args.seed,
    )
    data_packer = BridgeData2DataPacker(processor=processor, max_seq_len=args.max_tokens)
    return DataPackerDataLoader(
        data_source=dataset,
        data_packer=data_packer,
        max_tokens=args.max_tokens,
        max_batch_size=1,
        pool_size=8,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos3 VLM Standalone Training (LoongForge)")
    parser.add_argument("--checkpoint-path", default=os.environ.get("BASE_CHECKPOINT_PATH", ""))
    parser.add_argument("--tokenizer-path", default=os.environ.get("COSMOS_TOKENIZER_PATH", ""))
    parser.add_argument("--output-dir", default="./outputs/train")
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH", ""),
                        help="Path to video_dataset_file.jsonl")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16000)
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    if local_rank == 0:
        log.info(
            f"Cosmos3 VLM Training: world_size={world_size}, "
            f"tokenizer={args.tokenizer_path}, checkpoint={args.checkpoint_path}"
        )

    model = VLMModel(
        config=VLMModelConfig(
            precision="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=-1,
                data_parallel_replicate_degree=1,
                context_parallel_shard_degree=1,
                cfg_parallel_shard_degree=1,
                fsdp_master_dtype="float32",
            ),
            activation_checkpointing=ActivationCheckpointingConfig(mode="full"),
            policy=PolicyConfig(
                backbone=VLMConfig(model_name=args.tokenizer_path),
                attn_implementation="flash_attention_2",
            ),
        ),
        checkpoint=SimpleNamespace(
            load_path=args.checkpoint_path,
            load_from_object_store=SimpleNamespace(credentials="", bucket="", enabled=False),
        ),
    )

    # Build dataloader
    if args.use_synthetic or not args.dataset_path:
        log.info("Using synthetic data")
        dataloader_train = SyntheticPackedDataloader(seq_len=args.seq_len, num_batches=args.num_batches)
    else:
        log.info(f"Building DataPackerDataLoader from {args.dataset_path}")
        dataloader_train = build_real_dataloader(args)

    os.makedirs(args.output_dir, exist_ok=True)

    optimizer_config = L(build_optimizer)(
        model=PLACEHOLDER,
        optimizer_type="AdamW",
        lr=args.lr, betas=[0.9, 0.95], eps=1e-6, fused=True,
        weight_decay=0, keys_to_select=[],
        lr_multipliers={},
    )
    scheduler_config = L(build_lr_scheduler)(
        optimizer=PLACEHOLDER,
        lr_scheduler_type="LambdaCosine",
        cycle_lengths=[1000], f_max=[1.0], f_min=[0.0], f_start=[0.0],
        verbosity_interval=0, warm_up_steps=[50],
    )

    config = SimpleNamespace(
        model=SimpleNamespace(),
        job=SimpleNamespace(
            project="cosmos3", group="sft", name="cosmos3_vlm_sft",
            wandb_mode="disabled", path_local=args.output_dir, path=args.output_dir,
        ),
        model_parallel=SimpleNamespace(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, sequence_parallel=False,
        ),
        trainer=SimpleNamespace(
            distributed_parallelism="fsdp",
            grad_accum_iter=args.grad_accum,
            logging_iter=1,
            max_iter=args.train_iters,
            max_val_iter=None,
            memory_format=torch.preserve_format,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=args.seed,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=SimpleNamespace(recompile_limit=8, use_duck_shape=False),
            cudnn=SimpleNamespace(benchmark=True, deterministic=False),
            ddp=SimpleNamespace(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args={"enabled": False},
            straggler_detection=SimpleNamespace(
                enabled=False, report_freq=100, profile_freq=100,
                max_diff=0.1, raise_error=False, save_s3=False,
                analyze_dataloading=False, analyze_forward=False,
                analyze_backward=False, analyze_optimizer=False,
            ),
            profiling=SimpleNamespace(
                enabled=False, profile_start_step=0, profile_end_step=0,
                enable_profiling=False, enable_memory_snapshot=False,
                enable_nsys=False, profile_freq=10, profile_warmup=1,
                profile_memory=False, record_shape=False,
                with_stack=False, with_modules=False,
                target_ranks=[0], save_s3=False,
                memory_snapshot=SimpleNamespace(enabled=False),
                nsys=SimpleNamespace(enabled=False),
            ),
            callbacks={"grad_clip": L(GradClip)(clip_norm=0.1, force_finite=True)},
        ),
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        checkpoint=SimpleNamespace(
            type=L(DistributedCheckpointer)(),
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False, enable_gcs_patch_in_boto3=False,
            keys_not_to_resume=[], keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False, load_path=args.checkpoint_path,
            load_training_state=False, only_load_scheduler_state=False,
            save_iter=args.save_iter, strict_resume=False, verbose=True,
            save_to_object_store=SimpleNamespace(bucket="", credentials="", enabled=False),
            load_from_object_store=SimpleNamespace(bucket="", credentials="", enabled=False),
            hf_export=SimpleNamespace(enabled=False),
            jit=SimpleNamespace(enabled=False),
        ),
    )

    trainer = ImaginaireTrainer(config)
    trainer.train(model=model, dataloader_train=dataloader_train, dataloader_val=None)


if __name__ == "__main__":
    main()
