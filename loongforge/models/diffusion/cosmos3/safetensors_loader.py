# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Distributed safetensors loading and HF→Cosmos3 weight conversion.

Layered API
-----------

Three layers of functionality, lowest first:

1. **Multi-rank checkpoint I/O** — :class:`MultiRankCheckpointLoader` distributes
   safetensors file reads across the FSDP ``dp_shard`` ranks and then
   broadcasts each tensor to every rank.  It is checkpoint-format-agnostic:
   it just yields ``(name, tensor)`` pairs from the raw HF state dict.

2. **Name / weight conversion** — Per-family converters translate raw HF
   parameter names (and optionally shard the tensor along FSDP / EP axes)
   into the Cosmos3 VFM layout:

   - :func:`convert_weight_from_qwen3_hf`         — Qwen3 VL / LLM (dense + MoE).
   - :func:`convert_weight_from_nemotron_vl_hf`   — Nemotron-3 Dense VL (hybrid 56-block layout).
   - :func:`convert_weight_from_nemotron_llm_hf`  — Nemotron-3 pure LLM.

   For the generic VLM path, :func:`_make_name_converter` consumes the model's
   ``_checkpoint_conversion_mapping`` (transformers v4) or falls back to
   suffix-lookup against the model's own state dict (transformers v5).

3. **High-level loaders** — Composing the above:

   - :func:`load_language_model` — loads HF text-tower weights into the MoT
     language model.  Auto-detects the checkpoint format
     (:func:`detect_vlm_checkpoint_format`).
   - :func:`load_vlm_model` — generic loader for HF VLM checkpoints into an
     FSDP-wrapped ``HFModel``; honors a skip-pattern overlay.

Borrowed from cosmos_rl's ``MultiRankWeightLoader`` (renamed to
``MultiRankCheckpointLoader`` here) with modifications for loading from
S3 / GCS and support for Cosmos3 VFM models.
https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/multi_rank_weight_loader.py
"""

import os
import re
import time
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
from safetensors.torch import load as load_safetensors
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from loongforge.utils.cosmos.flags import INTERNAL
from loongforge.utils.cosmos import log
from loongforge.utils.cosmos.easy_io import easy_io
from loongforge.utils.cosmos.parallelism import ParallelDims

# Prefixes stripped when matching checkpoint keys to model state-dict keys.
# Order matters: longest first.  For each model key, the longest matching
# prefix is stripped (yielding the most specific tail) before we record it
# in the lookup table.  The trailing empty string acts as a default that
# leaves keys without any known prefix unchanged.
# Ref: cosmos-rl cosmos_rl/policy/model/hf_models/__init__.py:465-472.
_VLM_KEY_PREFIXES: tuple[str, ...] = (
    "model.language_model.model.",
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
    "",
)

_HF_URI_PREFIX = "hf://"


def _looks_like_hf_repo_id(checkpoint_path: str) -> bool:
    """Return True for unambiguous bare Hugging Face repo IDs.

    Explicit ``hf://`` paths are handled separately.  For bare paths, require the
    common ``namespace/repo`` shape so local relative paths such as ``ckpt`` are
    not silently treated as Hub repos.
    """
    if os.path.exists(os.path.expanduser(checkpoint_path)):
        return False
    if checkpoint_path.startswith(("/", "./", "../", "~")):
        return False
    if "://" in checkpoint_path:
        return False
    return re.fullmatch(r"[\w.-]+/[\w.-]+", checkpoint_path) is not None


def _download_hf_checkpoint(checkpoint_path: str) -> str:
    """Download safetensors from Hugging Face Hub and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    repo_id = checkpoint_path.removeprefix(_HF_URI_PREFIX)
    hf_home = os.environ.get("HF_HOME")
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    token = os.environ.get("HF_TOKEN")
    log.info(f"Resolving Hugging Face checkpoint: {repo_id}", rank0_only=False)
    local_path = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    log.info(f"Resolved Hugging Face checkpoint {repo_id} to {local_path}", rank0_only=False)
    return local_path


def _is_hf_checkpoint_candidate(checkpoint_path: str) -> bool:
    return checkpoint_path.startswith(_HF_URI_PREFIX) or _looks_like_hf_repo_id(checkpoint_path)


def _make_backend_args(checkpoint_path: str, credential_path: str | None) -> dict[str, str | None] | None:
    if checkpoint_path.startswith("s3://"):
        return {
            "backend": "s3",
            "s3_credential_path": credential_path,
        }
    return None


def _list_safetensors_files(
    checkpoint_path: str,
    backend_args: dict[str, str | None] | None,
) -> list[str]:
    return list(
        easy_io.list_dir_or_file(
            checkpoint_path,
            list_dir=False,
            list_file=True,
            suffix="safetensors",
            recursive=False,
            backend_args=backend_args,
        )
    )


def _get_local_rank_and_size(device_mesh: DeviceMesh) -> tuple[int, int]:
    """Get the local rank and size of a device mesh.

    Args:
        device_mesh: The device mesh to get the attributes from.

    Returns:
        A tuple of (local rank, size).
    """
    return device_mesh.get_local_rank(), device_mesh.size()


def _shard_tensor_on_fsdp_mesh(
    tensor: torch.Tensor,
    parallel_dims: ParallelDims | None,
) -> torch.Tensor:
    """Slice ``tensor`` along dim 0 according to the FSDP ``dp_shard`` mesh.

    Returns the rank-local shard when ``dp_shard`` is enabled, otherwise the
    full tensor (made contiguous).  Requires that ``tensor.shape[0]`` is
    divisible by ``dp_shard_size`` — this is a hard requirement of the even-
    split semantics; uneven splits should go through :func:`_shard_first_dim`.

    Args:
        tensor: The tensor to shard.
        parallel_dims: Parallel dims object, or None for single-rank.

    Returns:
        Contiguous rank-local shard (or full tensor if dp_shard is disabled).
    """
    if parallel_dims is None or not parallel_dims.dp_shard_enabled:
        return tensor.contiguous()

    fsdp_rank, fsdp_size = _get_local_rank_and_size(parallel_dims.dp_shard_mesh)
    if tensor.shape[0] % fsdp_size != 0:
        raise ValueError(f"Shard shape {tensor.shape} is not divisible by dp_shard_size {fsdp_size} on dim 0")
    shard = tensor.chunk(chunks=fsdp_size, dim=0)[fsdp_rank]
    return shard.contiguous()


def _get_dp_shard_mesh(parallel_dims: ParallelDims | None) -> DeviceMesh | None:
    """Get the dp_shard mesh from the parallel dimensions.

    Args:
        parallel_dims: The parallel dimensions to use for the conversion.

    Returns:
        The dp_shard mesh, or None if dp_shard is not enabled.
    """
    if parallel_dims is not None and parallel_dims.dp_shard_enabled:
        return parallel_dims.dp_shard_mesh
    else:
        return None


def _build_model_key_by_tail(state_dict: dict) -> dict[str, str]:
    """Build a ``tail → model_key`` lookup table for suffix-based key matching.

    For each model key, strip the longest matching prefix in
    ``_VLM_KEY_PREFIXES`` and record ``tail -> model_key``.  The longest
    prefix yields the shortest, most specific tail.  The trailing empty
    prefix in ``_VLM_KEY_PREFIXES`` ensures keys with no known prefix map
    to themselves as their own tail.
    """
    table: dict[str, str] = {}
    for model_key in state_dict:
        for pfx in _VLM_KEY_PREFIXES:
            if model_key.startswith(pfx):
                tail = model_key[len(pfx) :]
                if tail and tail not in table:
                    table[tail] = model_key
                    break
    return table


def _is_moe_vlm(model: torch.nn.Module) -> bool:
    """Detect whether an HF VLM is a Mixture-of-Experts model.

    MoE VLMs (Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B) need replicated-gate +
    FSDP-fused-expert shard rules that load_vlm_model does NOT yet implement.
    Callers use this to raise NotImplementedError before sharding.

    Detection sources (any one is sufficient):
    - ``model.config.text_config.num_experts`` (if present and non-None)
    - ``model.config.text_config.num_local_experts`` (if present and non-None)
    - Same attributes on ``model.config`` directly (text-only fallback)
    - Any state-dict key containing ``.mlp.experts.``
    """
    text_cfg = getattr(model.config, "text_config", None) or model.config
    for attr in ("num_experts", "num_local_experts"):
        value = getattr(text_cfg, attr, None)
        if value is not None and value != 0:
            return True
    for name in model.state_dict().keys():
        if ".mlp.experts." in name:
            return True
    return False


def _make_name_converter(
    state_dict: dict,
    hf_conv_map: dict[str, str] | None,
) -> Callable[[str], str]:
    """Return a callable that maps checkpoint keys to model keys.

    Two strategies, matching cosmos-rl's flow:
    1. If ``hf_conv_map`` is non-empty (transformers v4 pre-computed pattern
       mapping), apply each pattern/replacement as a regex substitution and
       return on the first match (no further fallback).
    2. Otherwise (transformers v5 or no map), use a direct-match against the
       model's state dict, then a longest-prefix-stripped suffix lookup
       through ``_VLM_KEY_PREFIXES``.  Names that match nothing are returned
       unchanged (the caller is responsible for filtering / raising).
    """
    model_key_by_tail = _build_model_key_by_tail(state_dict)

    def convert(name: str) -> str:
        if hf_conv_map:
            for pattern, replacement in hf_conv_map.items():
                if re.search(pattern, name):
                    return re.sub(pattern, replacement, name)
            return name
        if name in state_dict:
            return name
        for pfx in _VLM_KEY_PREFIXES:
            if name.startswith(pfx):
                tail = name[len(pfx) :]
                if tail and tail in model_key_by_tail:
                    return model_key_by_tail[tail]
        return name

    return convert


class MultiRankCheckpointLoader:
    """Multi-rank loader for model weights stored as safetensors files.

    Files in the checkpoint directory are statically partitioned across the
    ranks of the ``dp_shard`` sub-mesh by ``file_idx % world_size``.  Each
    rank reads its assigned files locally and the per-tensor data is later
    broadcast (via :meth:`broadcast_tensor`) so every rank ends up with the
    full tensor before sharding.

    When constructed with ``dp_shard_mesh=None`` the loader degrades to a
    single-rank fallback: ``world_size = 1``, every rank reads every file,
    and broadcasts are no-ops.

    Renamed from cosmos-rl's ``MultiRankWeightLoader`` and extended to load
    from S3 / GCS via easy_io and to support Cosmos3 VFM models.
    https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/multi_rank_weight_loader.py
    """

    # Mapping from dtype to integer for broadcasting
    DTYPE_TO_INT = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
        torch.int8: 5,
        torch.uint8: 6,
        torch.float8_e4m3fn: 7,
        torch.float8_e5m2: 8,
    }
    # Mapping from integer to dtype for broadcasting
    INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}

    def __init__(self, dp_shard_mesh: DeviceMesh | None):
        """Initialize the multi-rank weight loader.

        Args:
            dp_shard_mesh: 1-D ``dp_shard`` mesh, or None if dp_shard is not
                enabled.  Callers should obtain this via
                :func:`_get_dp_shard_mesh` so the ``parallel_dims is None`` and
                ``dp_shard <= 1`` cases collapse to the single-rank fallback.
        """
        if dp_shard_mesh is None:
            self.group = None
            self.rank = 0
            self.world_size = 1
        else:
            self.group = dp_shard_mesh.get_group()
            self.rank = dp_shard_mesh.get_local_rank()
            self.world_size = dp_shard_mesh.size()

def _shard_first_dim(tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Slice a tensor along dim 0 for FSDP sharding.

    Matches cosmos-rl weight_converter.py:71-79 semantics: even splits use
    tensor_split; uneven splits use ceil-divide with the last rank getting
    the remainder (may be smaller than average).  This layout must match
    FSDP2's local_view shape per rank — caller asserts shape equality.
    """
    tensor = tensor.contiguous()
    row_size = tensor.shape[0]
    if world_size == 1:
        return tensor
    if row_size % world_size == 0:
        return tensor.tensor_split(world_size, dim=0)[rank].contiguous()
    avg = (row_size + world_size - 1) // world_size
    start = rank * avg
    end = min(start + avg, row_size)
    return tensor[start:end].contiguous()


def load_language_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    parallel_dims: ParallelDims | None,
    checkpoint_format: str | None = None,
) -> set[str]:
    """
    Universal language model loading function using SafeTensors (.safetensors) format.
    Handles key remapping for "model.language_model." -> "model." by default.

    Args:
        model: The language model to load weights into.
        checkpoint_path: Path to checkpoint containing .safetensors files. Local
            paths and S3 URIs are tried first; if no safetensors are found,
            explicit ``hf://org/model`` Hub URIs and bare ``org/model`` repo IDs
            fall back to Hugging Face.
        credential_path: Path to S3 credentials, or None for local/HF.
        parallel_dims: ParallelDims object to use for parallel loading.
            If None, the loading is done in a single rank.
        checkpoint_format: ``"qwen3"``, ``"nemotron_3_dense_vl"``, ``"nemotron_3_llm"``, or None to auto-detect.

    Returns:
        Set of model state-dict keys successfully loaded from the checkpoint.
    """
    if not INTERNAL:
        from loongforge.utils.cosmos.checkpoint_db import download_checkpoint, sanitize_uri

        checkpoint_path = download_checkpoint(sanitize_uri(checkpoint_path))

    start_time = time.time()
    log.info(f"load_language_model: loading weights from {checkpoint_path}")

    lm_state_dict = {}
    for name, tensor in model.state_dict().items():
        # Remove the original module (torch compiled module) and checkpoint wrapped module prefixes.
        final_name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        lm_state_dict[final_name] = tensor

    # Initialize multi-rank weight loader
    loader = MultiRankCheckpointLoader(_get_dp_shard_mesh(parallel_dims))

    # Step 1: Load files in parallel
    rank_tensors, rank_tensor_metadata, weights_of_ckpt_names = loader.load_files_parallel(
        checkpoint_path=checkpoint_path,
        credential_path=credential_path,
        loading_device="cpu",
    )

    # Step 2: Gather tensor names and build mapping
    all_tensor_names, tensor_to_rank_map = loader.gather_tensor_names_and_build_mapping(
        weights_of_ckpt_names, rank_tensors
    )

    resolved_format = checkpoint_format or detect_vlm_checkpoint_format(all_tensor_names)
    log.info(f"Language model checkpoint format: {resolved_format}", rank0_only=False)

    # Step 3: Process each tensor
    keys_loaded = set()
    for name, tensor in loader.iterate_tensors(
        all_tensor_names,
        tensor_to_rank_map,
        rank_tensors,
        rank_tensor_metadata,
        device="cuda",
    ):
        if resolved_format == "nemotron_3_dense_vl":
            dest_name, dest_weight = convert_weight_from_nemotron_vl_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        elif resolved_format == "nemotron_3_llm":
            dest_name, dest_weight = convert_weight_from_nemotron_llm_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        elif resolved_format == "qwen3":
            dest_name, dest_weight = convert_weight_from_qwen3_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        else:
            raise ValueError(f"Unexpected checkpoint format: {resolved_format}")

        if dest_name is None:
            # This is due to the visual weights of VLM models.
            continue

        # If the weight is not found in the language model's state dict, then the weight is
        # unexpected. The unexpected weights should be from the visual part of the VLM (already
        # handled by the previous check). All weights in the language part should be used by
        # the Cosmos3 VFM.
        if dest_name not in lm_state_dict:
            raise ValueError(
                f"Unexpected weight found in checkpoint: {name}, "
                f"language model's corresponding weight {dest_name} not found."
            )

        target_tensor = lm_state_dict[dest_name]
        is_dist_tensor = isinstance(target_tensor, DTensor)
        local_view = target_tensor.to_local() if is_dist_tensor else target_tensor

        if dest_weight.device != local_view.device:
            dest_weight = dest_weight.to(local_view.device)

        assert local_view.shape == dest_weight.shape, (
            f"Shape mismatch: {local_view.shape} != {dest_weight.shape} "
            f"for {dest_name} with original shape {target_tensor.shape}"
        )
        with torch.no_grad():
            local_view.data.copy_(dest_weight)

        keys_loaded.add(dest_name)

    keys_missing = set(lm_state_dict.keys()) - keys_loaded

    # Tied-embedding fix-up.  HF Qwen3-VL 2B/4B safetensors set
    # `tie_word_embeddings=True` and omit `lm_head.weight` from the
    # checkpoint (it's redundant with `embed_tokens.weight`).  In Cosmos3,
    # the language model is constructed on the meta device — where HF's
    # `post_init()` ties `lm_head.weight` to `embed_tokens.weight` — but
    # `to_empty(device='cuda')` then allocates fresh CUDA tensors for
    # every parameter, breaking that sharing.  `init_weights()` randomly
    # inits both independently.  Without a fix-up, this loader would then
    # populate `embed_tokens.weight` from disk while leaving
    # `lm_head.weight` at its random init, so any downstream consumer of
    # `lm_head` (text-token CE loss during training, the reasoner AR
    # loop in `OmniMoTModel.generate_reasoner_text`) would see pure-noise
    # logits.  We therefore copy `embed_tokens.weight` -> `lm_head.weight`
    # whenever (a) the config flags tied embeddings AND (b) the
    # checkpoint did not contain `lm_head.weight`.  Note this is a
    # one-shot data copy (not Parameter-level tying); callers that need
    # continued tying through training must additionally re-tie at the
    # Parameter level, which is fragile under FSDP and outside this
    # loader's scope.

    tie_embeddings = getattr(model.config, "tie_word_embeddings", False)
    if tie_embeddings:
        assert "lm_head.weight" in keys_missing, (
            f"lm_head.weight is found in the checkpoint but tie_word_embeddings is True"
        )
    else:
        assert "lm_head.weight" not in keys_missing, (
            f"lm_head.weight is not found in the checkpoint but tie_word_embeddings is False"
        )

    if tie_embeddings:
        # The `*ForCausalLM` classes in
        # `projects/cosmos3/vfm/models/mot/unified_mot.py` override
        # `get_input_embeddings` (canonical HF idiom) to return the inner
        # `model.embed_tokens`, so this call returns a real `nn.Embedding`
        # rather than raising `NotImplementedError`.
        embed = model.get_input_embeddings()
        head = model.lm_head
        if embed is None or head is None:
            raise ValueError(
                "Tied-embedding fix-up: could not locate input embeddings or lm_head; "
                "lm_head.weight may remain at random init and downstream text logits "
                "will be garbage."
            )
        with torch.no_grad():
            head.weight.data.copy_(embed.weight.data)
        log.info(
            "Copied embed_tokens.weight -> lm_head.weight "
            "(tie_word_embeddings=True; lm_head.weight missing from checkpoint)."
        )
        keys_missing.remove("lm_head.weight")

    # Perform more error checking to ensure the checkpoint is valid. If the keys are missing,
    # then the missing keys should be from the generation pathway. All keys from the
    # understanding pathway must be present in the checkpoint. Additionally, for 2B and 4B
    # dense Qwen VLMs, the `lm_head.weight` key is not present in the checkpoint. For these
    # models, the input embedding and generation layer share the same params due to
    # `tie_word_embeddings` being set to True in the configs. For the 0.6B LLM, 8B and 32B dense
    # VLMs, and the 30B and 235B MoE VLMs, the `lm_head.weight` key is present in the
    # checkpoint.
    real_keys_missing = {k for k in keys_missing if "_moe_gen" not in k}
    if real_keys_missing:
        raise ValueError(
            f"load_language_model: {len(real_keys_missing)} required model "
            f"parameter(s) not found in checkpoint '{checkpoint_path}'. "
            f"First up to 10: {sorted(real_keys_missing)[:10]}"
        )

    log.info(
        f"load_language_model: successfully loaded {len(keys_loaded)} tensors "
        f"from {checkpoint_path} in {time.time() - start_time:.1f}s"
    )
    return keys_loaded


def load_vlm_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    parallel_dims: ParallelDims | None,
    skip_patterns: list[str] | None = None,
) -> set[str]:
    """Load a HF VLM checkpoint (safetensors) into an FSDP-wrapped HFModel.

    Local paths and S3 URIs are tried first; if no safetensors are found,
    explicit ``hf://org/model`` Hub URIs and bare ``org/model`` repo IDs fall
    back to Hugging Face.

    Both ``tensor_names_to_skip`` and ``extra_skip_patterns`` are lists of
    regex patterns applied to the RESOLVED model key (post-name_converter).
    Phase-5 skips any model key matched by either list; Phase-6's
    completeness check tolerates missing model keys matched by either
    list.  The two kwargs are semantically identical — separate names let
    call sites distinguish "model-type fixed skips" (from
    ``_tensor_names_to_skip_for``) from "overlay-specific skips" (from
    ``VLMModel._init_vlm`` for the pretrained_weights.backbone_path overlay).

    Cosmos-rl-style universal loader — no per-family hand-coded key mapping.
    Resolves the FSDP shard sub-group via :func:`_get_dp_shard_mesh`, which
    reads ``parallel_dims.dp_shard_mesh`` (the 1-D ``dp_shard`` sub-mesh
    populated by ``ParallelDims.build_meshes()``).  ``cp`` and ``cfgp`` live
    in their own overlay meshes and do NOT participate in checkpoint sharding.

    Preconditions:
    - ``parallelize()`` has been called on the HFModel (parameters are DTensors).
    - ``HFModel.tie_embeddings()`` has been called before this function so that
      tied ``lm_head.weight`` / ``embed_tokens.weight`` share DTensor storage.
    - When ``parallel_dims`` is provided AND ``parallel_dims.dp_shard > 1``,
      ``parallel_dims.build_meshes()`` MUST have been called by the caller.
      Otherwise ``dp_shard_mesh`` returns None and the loader silently falls
      back to single-rank loading — every rank reads every file and slices
      locally, which is correct for ``dp_shard <= 1`` but a silent perf /
      correctness regression for FSDP runs.  Pass ``parallel_dims=None``
      explicitly for the single-process / unit-test fallback.

    Raises:
        NotImplementedError: for MoE VLMs (not yet supported — see spec §2.2).
        ValueError: when the checkpoint is missing a required model parameter.

    Returns:
        Set of model state-dict keys successfully loaded from the checkpoint.
    """
    start_time = time.time()
    log.info(f"Loading VLM weights in safetensors format from: {checkpoint_path}")

    # Phase 1: canonical model state dict with compile/FSDP wrapper prefixes stripped.
    vlm_state_dict = {
        name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", ""): tensor
        for name, tensor in model.state_dict().items()
    }

    # Phase 2+3: suffix-lookup table + name converter.
    hf_conv_map = getattr(model, "_checkpoint_conversion_mapping", None)
    name_converter = _make_name_converter(
        vlm_state_dict,
        hf_conv_map=hf_conv_map if hf_conv_map else None,
    )

    # Phase 4: MoE precheck — fail early rather than silently mis-shard.
    if _is_moe_vlm(model):
        raise NotImplementedError(
            "load_vlm_model does not yet support MoE VLMs "
            "(e.g. Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B). Expected follow-up MR "
            "ports cosmos-rl's is_moe_mlp_fused_into_dp_shard / replicated-gate "
            "handling. Use a dense VLM checkpoint (2B, 4B, 8B, 32B) until then."
        )

    # FUTURE: to re-enable FSDP-2 CPU offload, detect CPU local_views via
    # ``sample.device.type == "cpu"``, force the loader to single-rank (None
    # instead of _get_dp_shard_mesh), and pin ``target_device`` to ``"cpu"``.
    loader = MultiRankCheckpointLoader(_get_dp_shard_mesh(parallel_dims))
    rank_tensors, rank_tensor_meta, ckpt_names = loader.load_files_parallel(
        checkpoint_path=checkpoint_path,
        credential_path=credential_path if credential_path else "",
        loading_device="cpu",
    )
    all_tensor_names, tensor_to_rank = loader.gather_tensor_names_and_build_mapping(
        ckpt_names,
        rank_tensors,
    )

    # Phase 5: per-tensor copy.  Skip patterns match the MODEL key (post-
    # name_converter), not the raw ckpt key — this matches cosmos-rl's
    # semantics and avoids fragility with prefix variations.  The same
    # compiled list drives Phase-5 skip and Phase-6 tolerance.
    compiled_skip_patterns = [re.compile(p) for p in (skip_patterns or [])]
    keys_loaded: set[str] = set()
    skipped_model_keys: set[str] = set()

    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the FSDP shard axis.
    dp_shard_mesh = _get_dp_shard_mesh(parallel_dims)
    if dp_shard_mesh is not None:
        shard_rank = dp_shard_mesh.get_local_rank()
        shard_size = dp_shard_mesh.size()
    else:
        shard_rank = 0
        shard_size = 1

    for ckpt_name, tensor in loader.iterate_tensors(
        all_tensor_names,
        tensor_to_rank,
        rank_tensors,
        rank_tensor_meta,
        device=target_device,
    ):
        dest_name = name_converter(ckpt_name)

        if any(p.fullmatch(dest_name) for p in compiled_skip_patterns):
            skipped_model_keys.add(dest_name)
            continue

        if dest_name not in vlm_state_dict:
            continue  # extra checkpoint key — ignore

        target = vlm_state_dict[dest_name]
        is_dtensor = isinstance(target, DTensor)
        local_view = target.to_local() if is_dtensor else target

        # Slice with the FSDP (shard_rank, shard_size), not loader.rank/world_size.
        shard = _shard_first_dim(tensor, shard_size, shard_rank)
        if shard.device != local_view.device:
            shard = shard.to(local_view.device)

        if shard.shape != local_view.shape:
            raise ValueError(
                f"Shape mismatch for {dest_name}: local_view={tuple(local_view.shape)}, shard={tuple(shard.shape)}"
            )
        with torch.no_grad():
            local_view.data.copy_(shard)
        keys_loaded.add(dest_name)

    # Phase 6: completeness check with tied-embedding AND skip-list tolerance.
    missing = set(vlm_state_dict) - keys_loaded - skipped_model_keys

    # Also tolerate missing model keys that match a skip pattern directly —
    # handles the case where the ckpt doesn't contain the key at all, so the
    # Phase 5 loop never saw it and skipped_model_keys didn't accumulate it.
    missing = {k for k in missing if not any(p.fullmatch(k) for p in compiled_skip_patterns)}
    tie = getattr(model.config, "tie_word_embeddings", False)
    real_missing = {k for k in missing if not (tie and "lm_head.weight" in k)}
    if real_missing:
        raise ValueError(
            f"load_vlm_model: {len(real_missing)} required model parameter(s) "
            f"not found in checkpoint '{checkpoint_path}'. First up to 10: "
            f"{sorted(real_missing)[:10]}"
        )
    log.info(
        f"load_vlm_model: loaded {len(keys_loaded)} tensors from {checkpoint_path} in {time.time() - start_time:.1f}s"
    )
    return keys_loaded


