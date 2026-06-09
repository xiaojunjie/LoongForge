"""Abstract DataPacker protocol for VLM training batches."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class DataPacker(ABC):
    @abstractmethod
    def sft_process_sample(self, item: Any) -> dict:
        """Convert one raw dataset item into a training-ready sample dict."""

    @abstractmethod
    def compute_num_tokens(self, sample: dict) -> int:
        """Return the token cost of one sample for the packing budget."""

    @abstractmethod
    def sft_collate_fn(self, samples: list[dict], max_len: int, ignore_label_id: int = -100) -> dict:
        """Collate a list of packed samples into one training batch."""
