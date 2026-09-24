"""RNA sequence normalization and tensor features."""

import numpy as np
import torch


ONEHOT = {
    "A": [1, 0, 0, 0],
    "U": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1],
}


def normalize_sequence(sequence):
    return "".join(sequence.split()).upper().replace("T", "U")


def sequence_onehot(sequence):
    return np.asarray([ONEHOT.get(base, [0, 0, 0, 0]) for base in sequence])


def absolute_position_embedding(length, embedding_dim):
    embedding = torch.zeros(length, embedding_dim, dtype=torch.float32)
    position = torch.arange(length).unsqueeze(1).float()
    divisor = (
        10000 ** ((2 * torch.arange(0, embedding_dim / 2)) / embedding_dim)
    ).unsqueeze(1).T
    embedding[:, 0::2] = torch.sin(position @ (1 / divisor))
    embedding[:, 1::2] = torch.cos(position @ (1 / divisor))
    return embedding


def dotbracket_to_pairs(structure):
    opening = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {value: key for key, value in opening.items()}
    stacks = {token: [] for token in opening}
    pairs = []
    for index, token in enumerate(structure):
        if token in opening:
            stacks[token].append(index)
        elif token in closing:
            start_token = closing[token]
            if not stacks[start_token]:
                raise ValueError(f"Unmatched closing bracket at position {index}")
            start = stacks[start_token].pop()
            pairs.extend(((start, index), (index, start)))
    if any(stacks.values()):
        raise ValueError("Unmatched opening bracket in structure")
    return pairs
