"""BridgeData2 DataPacker: transforms JSONL+video into input_ids/labels for VLMModel."""
from __future__ import annotations
import json
import os
import torch
import torch.utils.data
from loongforge.data.cosmos3.data_packer import DataPacker
from loongforge.data.cosmos3.qwen3vl_processor import Qwen3VLProcessor, IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD

_MAX_VIDEO_FRAMES = 32
_TARGET_VIDEO_FPS = 2.0


def _decode_video_to_pil_frames(video_path: str):
    """Decode video to PIL frames using torchvision."""
    import torchvision.io
    from PIL import Image
    import numpy as np
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
    # frames: [T, H, W, 3] uint8
    total_frames = frames.shape[0]
    source_fps = float(info.get("video_fps", 5.0))
    if total_frames <= 0:
        raise ValueError(f"video has zero frames: {video_path}")
    stride = max(1, int(round(source_fps / _TARGET_VIDEO_FPS)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > _MAX_VIDEO_FRAMES:
        step = len(indices) / _MAX_VIDEO_FRAMES
        indices = [indices[int(i * step)] for i in range(_MAX_VIDEO_FRAMES)]
    pil_frames = [Image.fromarray(frames[i].numpy()) for i in indices]
    effective_fps = source_fps / stride if stride > 0 else source_fps
    return pil_frames, float(effective_fps)


class BridgeData2Dataset(torch.utils.data.IterableDataset):
    """Reads BridgeData2 JSONL + local video files (infinite)."""
    def __init__(self, jsonl_path: str, data_root: str = "", shuffle=True, seed=42):
        self.jsonl_path = jsonl_path
        self.data_root = data_root or os.path.dirname(jsonl_path)
        self.shuffle = shuffle
        self.seed = seed
        self._metadata = None

    def _load_metadata(self):
        if self._metadata is not None:
            return self._metadata
        metadata = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
        self._metadata = metadata
        return metadata

    def __iter__(self):
        import random as _random
        metadata = self._load_metadata()
        indices = list(range(len(metadata)))
        _seed = self.seed
        if self.shuffle:
            rng = _random.Random(_seed)
            rng.shuffle(indices)
        while True:
            for idx in indices:
                item = metadata[idx]
                windows = item.get("t2w_windows", [])
                if windows:
                    w = _random.Random(_seed + idx).choice(windows)
                    caption = w.get("caption", "")
                else:
                    caption = ""
                vp = item["vision_path"]
                if not os.path.isabs(vp):
                    vp = os.path.join(self.data_root, vp)
                yield {"vision_path": vp, "caption": caption, "uuid": item.get("uuid", "")}
            _seed += 1
            if self.shuffle:
                rng = _random.Random(_seed)
                rng.shuffle(indices)


class BridgeData2DataPacker(DataPacker):
    """Transforms BridgeData2 items into tokenized VLM training batches."""
    def __init__(self, processor: Qwen3VLProcessor, max_seq_len: int = 16000):
        self._processor = processor
        self._max_seq_len = max_seq_len

    def sft_process_sample(self, item: dict) -> dict:
        caption = item.get("caption", "") or item.get("ai_caption", "")
        video_path = item["vision_path"]
        frames, fps = _decode_video_to_pil_frames(video_path)
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames, "fps": fps},
                {"type": "text", "text": "Describe this video in detail."},
            ]},
            {"role": "assistant", "content": caption},
        ]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = IGNORE_INDEX
        if input_ids.shape[0] > self._max_seq_len:
            input_ids = input_ids[:self._max_seq_len]
            labels = labels[:self._max_seq_len]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:self._max_seq_len]
        result = {"input_ids": input_ids, "labels": labels}
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None and key != "input_ids":
                result[key] = inputs[key]
        return result

    def compute_num_tokens(self, sample: dict) -> int:
        return int(sample["input_ids"].shape[0])

    def sft_collate_fn(self, samples, max_len, ignore_label_id=IGNORE_INDEX):
        assert len(samples) == 1
        s = samples[0]
        batch = {"input_ids": s["input_ids"].unsqueeze(0), "labels": s["labels"].unsqueeze(0)}
        if "attention_mask" in s and s["attention_mask"] is not None:
            batch["attention_mask"] = s["attention_mask"].unsqueeze(0)
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if key in s and s[key] is not None:
                batch[key] = s[key]
        return batch
