"""Pool-based token-budget bin-packing IterableDataset."""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Union
import torch


class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PackingIterableDataset(torch.utils.data.IterableDataset, ABC):
    def __init__(self, datasets_cfg, max_tokens, pool_size, max_batch_size,
                 long_threshold, batching_strategy, apply_long_sample_halving=True):
        super().__init__()
        assert batching_strategy in ("prefer_first", "prefer_closest")
        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.long_threshold = long_threshold
        self.max_batch_size = max_batch_size
        self.batching_strategy = batching_strategy
        self.apply_long_sample_halving = apply_long_sample_halving
        self._pool: deque[dict] = deque()
        self._dataset_names: list[str] = []
        self._ratios: list[float] = []
        self._datasets: list[torch.utils.data.IterableDataset] = []
        for name, cfg in datasets_cfg.items():
            ratio = cfg["ratio"]
            if ratio == 0:
                continue
            ds = cfg["dataset"]
            if isinstance(ds, torch.utils.data.DataLoader):
                ds = ds.dataset
            assert isinstance(ds, torch.utils.data.IterableDataset)
            self._dataset_names.append(name)
            self._ratios.append(float(ratio))
            self._datasets.append(ds)
        assert len(self._datasets) > 0
        self._data_len = 10**12
        self.iterators = [iter(ds) for ds in self._datasets]

    @abstractmethod
    def compute_sample_tokens(self, sample: dict) -> int:
        pass

    def collate_batch(self, samples: list[dict]) -> Any:
        return samples

    def __len__(self):
        return self._data_len

    def __iter__(self):
        while True:
            batch = self._best_fit_batch()
            yield self.collate_batch(batch)

    def _max_tokens(self, cur_max):
        if not self.apply_long_sample_halving:
            return self.max_tokens
        return self.max_tokens if cur_max < 1000 else self.max_tokens // 2

    def _get_next_sample(self):
        index_id = random.choices(range(len(self.iterators)), weights=self._ratios, k=1)[0]
        try:
            return next(self.iterators[index_id])
        except StopIteration:
            self.iterators[index_id] = iter(self._datasets[index_id])
            return next(self.iterators[index_id])

    def _fill_pool(self):
        while len(self._pool) < self.pool_size:
            self._pool.append(self._get_next_sample())

    def _get_modality(self, sample):
        if "pixel_values" in sample:
            return Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return Modality.VIDEO
        return Modality.TEXT

    def _best_fit_batch(self):
        self._fill_pool()
        seed = self._pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.compute_sample_tokens(seed)
        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]
        chosen = [seed]
        cur_max = L0
        while self._pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best(cur_max, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove(best_idx)
            chosen.append(cand)
            cur_max = max(cur_max, self.compute_sample_tokens(cand))
        return chosen

    def _find_best(self, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_cost = None
        best_diff = None
        for idx, cand in enumerate(self._pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.compute_sample_tokens(cand)
            new_max = max(cur_max, L)
            cost = new_max * (num_chosen + 1)
            if cost <= self._max_tokens(cur_max):
                diff = abs(L - cur_max)
                if best_cost is None or cost < best_cost or (cost == best_cost and diff < best_diff):
                    best_cost = cost
                    best_idx = idx
                    best_diff = diff
        return best_idx

    def _remove(self, idx):
        if idx == 0:
            return self._pool.popleft()
        elif idx == len(self._pool) - 1:
            return self._pool.pop()
        else:
            self._pool.rotate(-idx)
            item = self._pool.popleft()
            self._pool.rotate(idx)
            return item
