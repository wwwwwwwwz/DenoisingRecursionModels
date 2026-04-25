import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    max_tasks_per_set: Optional[int] = None
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_mask_token_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_mask_token_id = current_metadata.mask_token_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_mask_token_id == current_metadata.mask_token_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            if self.config.max_tasks_per_set is None:
                current_total_puzzles = current_metadata.total_puzzles
                current_total_groups = current_metadata.total_groups
                current_total_examples = current_metadata.mean_puzzle_examples * current_metadata.total_puzzles
            else:
                current_total_puzzles = 0
                current_total_groups = 0
                current_total_examples = 0.0
                for set_name in current_metadata.sets:
                    truncated_puzzles, truncated_groups, truncated_examples = self._load_truncated_counts(dataset_path, set_name)
                    current_total_puzzles += truncated_puzzles
                    current_total_groups += truncated_groups
                    current_total_examples += truncated_examples

            mean_puzzle_examples += current_total_examples
            total_puzzles += current_total_puzzles
            total_groups += current_total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles if total_puzzles > 0 else 0.0

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            mask_token_id=prev_mask_token_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )
        if self.config.rank == 0 and self.config.max_tasks_per_set is not None:
            print(
                "[data] Restricting each loaded set to the first "
                f"{self.config.max_tasks_per_set} puzzle(s)/task(s) for fast debugging."
            )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _load_truncated_counts(self, dataset_path: str, set_name: str) -> Tuple[int, int, float]:
        puzzle_indices = np.load(os.path.join(dataset_path, self.split, f"{set_name}__puzzle_indices.npy"), mmap_mode="r")
        group_indices = np.load(os.path.join(dataset_path, self.split, f"{set_name}__group_indices.npy"), mmap_mode="r")
        num_puzzles = min(self.config.max_tasks_per_set or 0, len(puzzle_indices) - 1)
        if num_puzzles <= 0:
            return 0, 0, 0.0

        truncated_groups = int(np.searchsorted(group_indices, num_puzzles, side="right") - 1)
        if truncated_groups < len(group_indices) - 1 and group_indices[truncated_groups] != num_puzzles:
            truncated_groups += 1
        truncated_examples = float(puzzle_indices[num_puzzles])
        return num_puzzles, truncated_groups, truncated_examples

    def _truncate_loaded_dataset(self, dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.config.max_tasks_per_set is None:
            return dataset

        num_puzzles = min(self.config.max_tasks_per_set, len(dataset["puzzle_indices"]) - 1)
        if num_puzzles <= 0:
            return {
                "inputs": dataset["inputs"][:0],
                "labels": dataset["labels"][:0],
                "puzzle_identifiers": dataset["puzzle_identifiers"][:0],
                "puzzle_indices": dataset["puzzle_indices"][:1],
                "group_indices": dataset["group_indices"][:1],
            }

        max_examples = int(dataset["puzzle_indices"][num_puzzles])
        group_indices = dataset["group_indices"]
        truncated_group_count = int(np.searchsorted(group_indices, num_puzzles, side="right") - 1)
        truncated_group_indices = np.asarray(group_indices[:truncated_group_count + 1], dtype=np.int32)
        if truncated_group_indices[-1] != num_puzzles:
            truncated_group_indices = np.concatenate(
                [truncated_group_indices, np.array([num_puzzles], dtype=np.int32)]
            )

        return {
            "inputs": dataset["inputs"][:max_examples],
            "labels": dataset["labels"][:max_examples],
            "puzzle_identifiers": np.asarray(dataset["puzzle_identifiers"][:num_puzzles], dtype=np.int32),
            "puzzle_indices": np.asarray(dataset["puzzle_indices"][:num_puzzles + 1], dtype=np.int32),
            "group_indices": truncated_group_indices,
        }

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                loaded = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }
                self._data[set_name_] = self._truncate_loaded_dataset(loaded)


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    # def _collate_batch(self, batch):
    #     # Convert dtype
    #     batch = {k: v.astype(np.int32) for k, v in batch.items()}

    #     # 🔹 Keep a copy of the unmasked labels (0–11 intact)
    #     batch["labels_raw"] = batch["labels"].copy()

    #     # Convert ignore label IDs (mask PAD=0 -> -100)
    #     if self.metadata.ignore_label_id is not None:
    #         batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

    #     # Pad
    #     if batch["puzzle_identifiers"].size < self.local_batch_size:
    #         pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
    #         pad_values = {
    #             "inputs": self.metadata.pad_id,
    #             "labels": IGNORE_LABEL_ID,
    #             "labels_raw": self.metadata.pad_id,  # 🔹 keep raw padded as 0, not -100
    #             "puzzle_identifiers": self.metadata.blank_identifier_id
    #         }
    #         batch = {
    #             k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k])
    #             for k, v in batch.items()
    #         }

    #     # To tensor
    #     return {k: torch.from_numpy(v) for k, v in batch.items()} 
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
