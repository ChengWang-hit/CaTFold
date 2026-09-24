#!/usr/bin/env python3
"""Generate candidate-pair masks and matching dataset indices."""

import argparse
import json
import pickle
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from catfold.masking import create_candidate_pairs
from catfold.sequence import normalize_sequence


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def atomic_pickle(path, value):
    # Replace the destination only after the new pickle is fully written.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def process_namespace(root, specification, min_loop_length):
    mask_dir = root / specification["mask_dir"] / "mask_matrix"
    mask_dir.mkdir(parents=True, exist_ok=True)
    sequence_to_index = {}
    next_index = 0

    # Primary files define the shared mask index space for this namespace.
    for relative in specification["primary_files"]:
        path = root / relative
        data = load_pickle(path)
        indices = []
        for sequence in tqdm(
            data["seq"],
            desc=f"Generating masks: {relative}",
            unit="sequence",
        ):
            normalized = normalize_sequence(sequence)
            index = next_index
            next_index += 1
            indices.append(index)
            sequence_to_index.setdefault(normalized, index)
            pairs = create_candidate_pairs(normalized, min_loop_length)
            atomic_pickle(mask_dir / f"{index}.pickle", pairs.tolist())
        data["mask_matrix_idx"] = indices
        atomic_pickle(path, data)

    # Reuse existing masks for dataset splits drawn from the primary files.
    for relative in specification.get("reuse_files", []):
        path = root / relative
        data = load_pickle(path)
        indices = []
        for sequence in tqdm(
            data["seq"],
            desc=f"Assigning indices: {relative}",
            unit="sequence",
        ):
            normalized = normalize_sequence(sequence)
            if normalized not in sequence_to_index:
                raise ValueError(f"Sequence from {path} is absent from primary data")
            indices.append(sequence_to_index[normalized])
        data["mask_matrix_idx"] = indices
        atomic_pickle(path, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    # Resolve configured data paths from the project root.
    root = (PROJECT_ROOT / config.get("data_root", "data")).resolve()
    for specification in config["namespaces"]:
        process_namespace(root, specification, int(config["min_loop_length"]))


if __name__ == "__main__":
    main()
