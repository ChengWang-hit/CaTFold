"""Datasets and batching for CaTFold pretraining and fine-tuning."""

import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .sequence import (
    absolute_position_embedding,
    dotbracket_to_pairs,
    sequence_onehot,
)


class StructureDataset(Dataset):
    def __init__(self, path, mask_dir, embedding_dim):
        self.path = Path(path)
        self.mask_dir = Path(mask_dir)
        self.embedding_dim = embedding_dim
        with self.path.open("rb") as handle:
            values = pickle.load(handle)
        self.sequences = [sequence.upper() for sequence in values["seq"]]
        self.structures = values["ss"]
        self.mask_indices = values.get("mask_matrix_idx")
        if self.mask_indices is None:
            raise ValueError(
                f"{self.path} has no mask_matrix_idx; run generate_masks.py first"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        length = len(sequence)
        mask_path = self.mask_dir / "mask_matrix" / f"{self.mask_indices[index]}.pickle"
        with mask_path.open("rb") as handle:
            pairs = np.asarray(pickle.load(handle), dtype=np.int64)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        pair_indices = torch.from_numpy(pairs.T.copy())
        contact = np.zeros((length, length), dtype=np.int8)
        true_pairs = np.asarray(self.structures[index])
        if true_pairs.size:
            contact[true_pairs[:, 0], true_pairs[:, 1]] = 1
        common = (
            torch.tensor(sequence_onehot(sequence), dtype=torch.float32),
            absolute_position_embedding(length, self.embedding_dim),
            pair_indices,
            length,
        )
        labels = contact[pairs[:, 0], pairs[:, 1]] if pairs.size else np.empty(0)
        return common + (torch.tensor(labels, dtype=torch.float32),)


class PretrainDataset(Dataset):
    def __init__(self, path, embedding_dim, show_progress=False):
        self.path = Path(path)
        self.embedding_dim = embedding_dim
        self.sequences = []
        self.structures = []
        with h5py.File(self.path, "r") as handle:
            sequence_data = handle["seq"]
            structure_data = handle["ss"]
            if len(sequence_data) != len(structure_data):
                raise ValueError(
                    f"Sequence and structure counts differ in {self.path}"
                )
            total = len(sequence_data)
            chunk_size = 100_000
            progress = tqdm(
                total=total,
                desc=f"Loading {self.path.stem}",
                unit="sequence",
                disable=not show_progress,
            )
            with progress:
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    self.sequences.extend(
                        self._decode(value) for value in sequence_data[start:end]
                    )
                    self.structures.extend(
                        self._decode(value) for value in structure_data[start:end]
                    )
                    progress.update(end - start)

    @staticmethod
    def _decode(value):
        return value if isinstance(value, str) else bytes(value).decode("utf-8")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index].upper()
        length = len(sequence)
        contact = np.zeros((length, length), dtype=np.int8)
        pairs = dotbracket_to_pairs(self.structures[index])
        if pairs:
            rows, columns = np.asarray(pairs).T
            contact[rows, columns] = 1
        candidates = torch.triu_indices(length, length, offset=1)
        labels = contact[candidates[0], candidates[1]]
        return (
            torch.tensor(sequence_onehot(sequence), dtype=torch.float32),
            absolute_position_embedding(length, self.embedding_dim),
            candidates,
            length,
            torch.tensor(labels, dtype=torch.float32),
        )


def collate_train(samples):
    lengths = [sample[3] for sample in samples]
    max_length = max(lengths)
    onehot, positions, pairs, masks, labels = [], [], [], [], []
    offset = 0
    for sample, length in zip(samples, lengths):
        node = torch.zeros(max_length, sample[0].size(-1))
        node[:length] = sample[0]
        onehot.append(node)
        position = torch.zeros(max_length, sample[1].size(-1))
        position[:length] = sample[1]
        positions.append(position)
        pairs.append(sample[2] + offset)
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[length:] = True
        masks.append(mask)
        labels.append(sample[4])
        offset += length
    return (
        torch.stack(onehot),
        torch.stack(positions),
        torch.cat(pairs, dim=1),
        torch.stack(masks),
        torch.tensor(lengths, dtype=torch.long),
        torch.cat(labels),
    )
