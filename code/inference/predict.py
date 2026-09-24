#!/usr/bin/env python3
"""Predict RNA structures from a multi-record FASTA file."""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from catfold.checkpoint import load_weights
from catfold.decoding import decode
from catfold.masking import create_candidate_pairs
from catfold.model import FineTuneNet
from catfold.sequence import (
    absolute_position_embedding,
    normalize_sequence,
    sequence_onehot,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def read_fasta(path):
    records = []
    header = None
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, normalize_sequence("".join(lines))))
            header, lines = line[1:].strip(), []
        else:
            if header is None:
                raise ValueError("FASTA sequence appears before the first header")
            lines.append(line)
    if header is not None:
        records.append((header, normalize_sequence("".join(lines))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def build_model(config):
    return FineTuneNet(
        config["embedding_dim"],
        config["layer_num"],
        config["nhead"],
        adapter_hidden_dim=config["adapter_hidden_dim"],
        adapter_dropout=config["adapter_dropout"],
        train_pair_out=config["train_pair_out"],
    )


def infer(model, sequence, config, device, pairs=None):
    if pairs is None:
        pairs = create_candidate_pairs(sequence, config["min_loop_length"])
    length = len(sequence)
    logits_matrix = np.full((length, length), np.nan, dtype=np.float32)
    probabilities = np.zeros((length, length), dtype=np.float32)
    if pairs.size:
        onehot = torch.tensor(
            sequence_onehot(sequence), dtype=torch.float32, device=device
        ).unsqueeze(0)
        positions = absolute_position_embedding(length, config["embedding_dim"])
        positions = positions.to(device).unsqueeze(0)
        pair_tensor = torch.from_numpy(pairs.T.astype(np.int64)).to(device)
        pad_mask = torch.zeros((1, length), dtype=torch.bool, device=device)
        lengths = torch.tensor([length], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits_tensor = model.inference(
                (onehot, positions, pair_tensor, pad_mask, lengths)
            )
            logits = logits_tensor.float().cpu().numpy()
            pair_probabilities = torch.sigmoid(logits_tensor).float().cpu().numpy()
        rows, columns = pairs.T
        logits_matrix[rows, columns] = logits
        logits_matrix[columns, rows] = logits
        probabilities[rows, columns] = pair_probabilities
        probabilities[columns, rows] = pair_probabilities
    contact = decode(
        torch.from_numpy(probabilities),
        config["threshold"],
    ).numpy().astype(np.uint8)
    return logits_matrix, contact


def contact_to_partners(contact):
    if not np.array_equal(contact, contact.T) or np.any(contact.sum(axis=1) > 1):
        raise ValueError("Decoded structure cannot be represented as BPSEQ")
    partners = np.zeros(contact.shape[0], dtype=np.int64)
    rows, columns = np.where(np.triu(contact, k=1) > 0)
    partners[rows] = columns + 1
    partners[columns] = rows + 1
    return partners


def write_bpseq(path, sequence, partners):
    lines = [
        f"{index} {base} {int(partner)}"
        for index, (base, partner) in enumerate(zip(sequence, partners), start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_structure(bpseq_path, output_path):
    import forgi
    import forgi.visual.mplotlib as forgi_plot

    rna = forgi.load_rna(str(bpseq_path), allow_many=False)
    if isinstance(rna, (list, tuple)):
        if len(rna) != 1:
            raise ValueError("Expected one RNA structure in BPSEQ")
        rna = rna[0]
    figure, axis = plt.subplots(figsize=(8, 8))
    forgi_plot.plot_rna(rna, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def safe_name(header, index, used):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", header.split()[0]).strip("._-")
    name = name or f"sequence_{index}"
    candidate, suffix = name, 2
    while candidate in used:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = read_fasta(args.fasta)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_root = args.output.resolve()
    print(f"Input FASTA: {args.fasta.resolve()}")
    print(f"Sequences: {len(records)}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Device: {device}")
    print(f"Output directory: {output_root}")
    print("Loading model...")
    model = build_model(config)
    load_weights(model, args.checkpoint)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    used = set()
    predictions = []
    failures = []
    progress = tqdm(records, desc="Predicting", unit="sequence")
    for index, (header, sequence) in enumerate(progress, start=1):
        progress.set_postfix(id=header.split()[0], length=len(sequence))
        output_dir = args.output / safe_name(header, index, used)
        try:
            if len(sequence) > config["validated_max_length"]:
                warnings.warn(
                    f"{header}: length {len(sequence)} exceeds the validated length "
                    f"of {config['validated_max_length']}; inference will continue "
                    "and may require substantially more memory",
                    RuntimeWarning,
                    stacklevel=2,
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            logits, contact = infer(model, sequence, config, device)
            partners = contact_to_partners(contact)
            np.save(output_dir / "logits.npy", logits)
            bpseq = output_dir / "prediction.bpseq"
            write_bpseq(bpseq, sequence, partners)
            draw_structure(bpseq, output_dir / "secondary_structure.png")
        except Exception as error:
            failures.append(
                {
                    "id": header,
                    "length": len(sequence),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            tqdm.write(
                f"Failed: {header} ({type(error).__name__}: {error})",
                file=sys.stderr,
            )
            continue
        predictions.append(
            {
                "id": header,
                "length": len(sequence),
                "pairs": int(contact.sum() // 2),
                "output_directory": output_dir.relative_to(args.output).as_posix(),
            }
        )
    summary_path = args.output / "summary.json"
    summary = {
        "total": len(records),
        "successful": len(predictions),
        "failed": len(failures),
        "output_directory": ".",
        "predictions": predictions,
        "failures": failures,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nInference summary")
    print(f"Total sequences: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Results: {output_root}")
    print(f"Summary: {summary_path.resolve()}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
