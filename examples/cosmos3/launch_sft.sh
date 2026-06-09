#!/usr/bin/env bash
# Cosmos3 VLM SFT training launcher (8-GPU FSDP2) with DataPackerDataLoader.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

: "${BASE_CHECKPOINT_PATH:=/mnt/cluster/xiaojunjie/code/cosmos-framework/examples/checkpoints/Cosmos3-Nano}"
: "${COSMOS_TOKENIZER_PATH:=/mnt/cluster/models/Qwen/Qwen3-VL-8B-Instruct}"
: "${DATASET_PATH:=/mnt/cluster/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge/train/video_dataset_file.jsonl}"
: "${OUTPUT_ROOT:=outputs/train}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_PORT:=50095}"

[[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$WORKDIR/$BASE_CHECKPOINT_PATH"
[[ "$COSMOS_TOKENIZER_PATH" = /* ]] || COSMOS_TOKENIZER_PATH="$WORKDIR/$COSMOS_TOKENIZER_PATH"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$WORKDIR/$OUTPUT_ROOT"

export BASE_CHECKPOINT_PATH COSMOS_TOKENIZER_PATH DATASET_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> $(date +%H:%M:%S) Cosmos3 VLM SFT Training (DataPackerDataLoader)"
echo ">>> checkpoint:  $BASE_CHECKPOINT_PATH"
echo ">>> tokenizer:   $COSMOS_TOKENIZER_PATH"
echo ">>> dataset:     $DATASET_PATH"
echo ">>> output:      $OUTPUT_ROOT"
echo ">>> GPUs:        $NPROC_PER_NODE"

[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -d "$COSMOS_TOKENIZER_PATH" ]] || { echo "ERROR: tokenizer not found: $COSMOS_TOKENIZER_PATH" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/vlm_sft.log"

PYTHONPATH=. torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    loongforge/train/sft/sft_vfm.py \
    --checkpoint-path "$BASE_CHECKPOINT_PATH" \
    --tokenizer-path "$COSMOS_TOKENIZER_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_ROOT" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date +%H:%M:%S) Done (exit $EXIT_CODE)"
exit $EXIT_CODE
