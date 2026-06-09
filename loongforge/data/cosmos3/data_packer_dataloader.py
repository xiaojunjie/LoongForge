"""DataPackerDataLoader: wraps any iterable + DataPacker into a training DataLoader."""
from __future__ import annotations
from typing import Any
import torch
import torch.utils.data
from loongforge.data.cosmos3.data_packer import DataPacker
from loongforge.data.cosmos3.packing_iterable_dataset import PackingIterableDataset


class _IterableWrapper(torch.utils.data.IterableDataset):
    def __init__(self, iterable, dp_rank=0, dp_world_size=1):
        super().__init__()
        self._iterable = iterable
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        total = self._dp_world_size * num_workers
        my_stream = self._dp_rank * num_workers + worker_id
        for i, item in enumerate(self._iterable):
            if i % total == my_stream:
                yield item


class _DataPackerIterableDataset(PackingIterableDataset):
    def __init__(self, data_source, data_packer, max_tokens, pool_size,
                 max_batch_size, long_threshold, batching_strategy, dp_rank=0, dp_world_size=1):
        data_source = _IterableWrapper(data_source, dp_rank=dp_rank, dp_world_size=dp_world_size)
        datasets_cfg = {"default": {"dataset": data_source, "ratio": 1.0}}
        super().__init__(datasets_cfg=datasets_cfg, max_tokens=max_tokens, pool_size=pool_size,
                         max_batch_size=max_batch_size, long_threshold=long_threshold,
                         batching_strategy=batching_strategy)
        self._data_packer = data_packer

    def _get_next_sample(self):
        raw_item = super()._get_next_sample()
        return self._data_packer.sft_process_sample(raw_item)

    def compute_sample_tokens(self, sample):
        return self._data_packer.compute_num_tokens(sample)

    def collate_batch(self, samples):
        max_len = max(self.compute_sample_tokens(s) for s in samples)
        return self._data_packer.sft_collate_fn(samples, max_len)


class DataPackerDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_source, data_packer: DataPacker, max_tokens: int,
                 pool_size=16, max_batch_size=1, long_threshold=6400,
                 batching_strategy="prefer_closest", num_workers=0,
                 prefetch_factor=None, persistent_workers=False, pin_memory=False):
        if torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()
        else:
            dp_rank, dp_world_size = 0, 1
        dataset = _DataPackerIterableDataset(
            data_source=data_source, data_packer=data_packer,
            max_tokens=max_tokens, pool_size=pool_size,
            max_batch_size=max_batch_size, long_threshold=long_threshold,
            batching_strategy=batching_strategy, dp_rank=dp_rank, dp_world_size=dp_world_size)
        loader_kwargs = dict(num_workers=num_workers,
                             persistent_workers=persistent_workers and num_workers > 0,
                             pin_memory=pin_memory)
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(dataset, batch_size=None, **loader_kwargs)
