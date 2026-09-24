"""Generate valid candidate base-pair coordinates for an RNA sequence."""

import numpy as np


def create_candidate_pairs(sequence, min_loop_length=3):
    sequence = sequence.upper().replace("T", "U")
    length = len(sequence)
    if length == 0:
        return np.empty((0, 2), dtype=np.int32)
    bases = np.asarray(list(sequence))
    row = bases.reshape(length, 1)
    column = bases.reshape(1, length)
    directed = (
        ((row == "A") & (column == "U"))
        | ((row == "G") & (column == "C"))
        | ((row == "G") & (column == "U"))
    )
    allowed = directed | directed.T
    indices = np.arange(length)
    distance = indices.reshape(1, length) - indices.reshape(length, 1)
    rows, columns = np.where(allowed & (distance > min_loop_length))
    return np.stack((rows, columns), axis=1).astype(np.int32, copy=False)
