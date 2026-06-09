"""Qwen3-VL processor wrapper for VLM SFT data tokenization."""
import os
from typing import Dict, List
import numpy as np
import torch
from transformers.models.auto.processing_auto import AutoProcessor

IGNORE_INDEX = -100

PROCESSOR_KEYS_TO_ADD = [
    "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
    "image_grid_thw", "video_grid_thw", "second_per_grid_ts",
]


def convert_string_content_to_list_content(messages):
    for i, msg in enumerate(messages):
        if isinstance(msg["content"], str):
            messages[i]["content"] = [{"type": "text", "text": msg["content"]}]
    return messages


def maybe_parse_video_content(messages):
    num_video, video_fps, video_total_num_frames, video_frames_indices = 0, [], [], []
    for msg in messages:
        if isinstance(msg["content"], list):
            for sub in msg["content"]:
                if sub.get("type") == "video" and isinstance(sub["video"], list):
                    num_video += 1
                    video_fps.append(sub.get("fps"))
                    video_total_num_frames.append(len(sub["video"]))
                    video_frames_indices.append(list(range(video_total_num_frames[-1])))
    return num_video, video_fps, video_total_num_frames, video_frames_indices


class Qwen3VLProcessor:
    def __init__(self, name="Qwen/Qwen3-VL-8B-Instruct"):
        path = name if os.path.isdir(name) else name
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.image_token_id = (self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
                               if hasattr(self.processor, "image_token") else None)
        self.video_token_id = (self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
                               if hasattr(self.processor, "video_token") else None)
        self.eos_id = self.processor.tokenizer.eos_token_id
        self.pad_id = self.processor.tokenizer.pad_token_id

    def apply_chat_template(self, messages, add_generation_prompt=False, return_tensors="pt", tokenize=True, **kwargs):
        messages = convert_string_content_to_list_content(messages)
        proc_kwargs = {}
        num_video, video_fps, total_frames, frames_indices = maybe_parse_video_content(messages)
        if num_video > 0:
            proc_kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": dict(fps=video_fps[0], total_num_frames=total_frames[0],
                                       frames_indices=frames_indices[0]),
            }
        inputs = self.processor.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            return_dict=True, return_tensors=return_tensors, **proc_kwargs)
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]
        return inputs

    def add_assistant_tokens_mask(self, tokens):
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1
        bos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        role_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")
        start_indices = np.where(np_tokens == bos_id)[0]
        end_indices = np.where(np_tokens == eos_id)[0]
        masks = np.zeros_like(np_tokens, dtype=bool)
        for start, end in zip(start_indices, end_indices):
            if np_tokens[start + 1] == role_id:
                masks[start + 3: end + 1] = True
        return torch.from_numpy(masks) if isinstance(tokens, torch.Tensor) else masks.tolist()
