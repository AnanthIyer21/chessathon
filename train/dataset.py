import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import encoding  # noqa: E402


class ShardDataset(IterableDataset):
    """Streams (planes, move_idx, outcome) from npz shards, one shard in
    memory at a time, shuffled within shard and across shard order."""

    def __init__(self, shard_paths, shuffle=True, seed=0):
        self.shard_paths = list(shard_paths)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        paths = self.shard_paths
        if info is not None:
            paths = paths[info.id::info.num_workers]
        rng = random.Random(self.seed + (info.id if info else 0))
        if self.shuffle:
            paths = paths[:]
            rng.shuffle(paths)
        for path in paths:
            with np.load(path) as z:
                boards, metas = z["boards"], z["metas"]
                moves, outcomes = z["moves"], z["outcomes"]
            order = np.arange(len(moves))
            if self.shuffle:
                np.random.default_rng(rng.randrange(1 << 30)).shuffle(order)
            for i in order:
                planes = encoding.array_to_planes(boards[i], metas[i])
                yield (torch.from_numpy(planes),
                       torch.tensor(int(moves[i]), dtype=torch.long),
                       torch.tensor(float(outcomes[i])))
