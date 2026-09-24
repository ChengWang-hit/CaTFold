#!/usr/bin/env python3
"""Evaluate all released CaTFold checkpoints on five benchmarks."""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from catfold.checkpoint import load_weights
from catfold.familyfold import FamilyFoldFusionNet
from catfold.metrics import precision_recall_f1
from catfold.sequence import normalize_sequence
from inference.predict import build_model, infer


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows):
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("precision", "recall", "f1")
    }


def format_summary_table(rows):
    name_width = max(len("Dataset / Family"), *(len(row["name"]) for row in rows))
    header = (
        f"{'Dataset / Family':<{name_width}}  "
        f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<{name_width}}  "
            f"{row['precision']:>9.6f}  "
            f"{row['recall']:>9.6f}  "
            f"{row['f1']:>9.6f}"
        )
    return "\n".join(lines)


def evaluate_split(
    model,
    model_config,
    device,
    data_path,
    mask_dir,
    benchmark,
    split,
    source_file,
    checkpoint,
):
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    indices = data.get("mask_matrix_idx")
    if indices is None:
        raise ValueError(f"{data_path} has no mask_matrix_idx; generate masks first")
    count = len(data["seq"])
    if len(data["ss"]) != count or len(indices) != count:
        raise ValueError(f"Sequence, structure, and mask counts differ in {data_path}")
    rows = []
    for record_index, (sequence, structure, mask_index) in enumerate(
        tqdm(
            zip(data["seq"], data["ss"], indices),
            total=count,
            desc=f"{benchmark} / {split}",
            unit="seq",
        )
    ):
        sequence = normalize_sequence(sequence)
        with (mask_dir / "mask_matrix" / f"{mask_index}.pickle").open("rb") as handle:
            pairs = np.asarray(pickle.load(handle), dtype=np.int32)
        _, prediction = infer(model, sequence, model_config, device, pairs=pairs)
        target = np.zeros_like(prediction)
        true_pairs = np.asarray(structure, dtype=np.int64).reshape(-1, 2)
        if true_pairs.size:
            target[true_pairs[:, 0], true_pairs[:, 1]] = 1
        precision, recall, f1 = precision_recall_f1(prediction, target)
        ss_true = sorted(
            {
                (min(int(i), int(j)), max(int(i), int(j)))
                for i, j in true_pairs
                if i != j
            }
        )
        pred_rows, pred_columns = np.where(np.triu(prediction, k=1) > 0)
        ss_pred = list(zip(pred_rows.tolist(), pred_columns.tolist()))
        rows.append(
            {
                "benchmark": benchmark,
                "source_split": split,
                "source_file": source_file,
                "checkpoint": checkpoint,
                "record_index": record_index,
                "sequence": sequence,
                "length": len(sequence),
                "ss_true": ss_true,
                "ss_pred": ss_pred,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def load_released_model(checkpoint, model_config, device):
    model = build_model(model_config)
    load_weights(model, checkpoint)
    return model.to(device).eval()


def load_familyfold_model(
    backbone_checkpoint,
    adapter_checkpoint,
    model_config,
    familyfold_model_config,
    device,
):
    model = FamilyFoldFusionNet(
        model_config["embedding_dim"],
        model_config["layer_num"],
        model_config["nhead"],
        hidden_dim=familyfold_model_config["hidden_dim"],
        num_blocks=familyfold_model_config["resnet_blocks"],
        kernel_size=familyfold_model_config["kernel_size"],
    )
    load_weights(model, backbone_checkpoint, allow_missing_adapter=True)
    load_weights(model.adapter, adapter_checkpoint)
    model.freeze_pretrained()
    model.requires_grad_(False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    model_config.update(config["decoding"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    data_root = PROJECT_ROOT / "data"
    checkpoint_root = PROJECT_ROOT / "checkpoints"
    output_root = PROJECT_ROOT / config["output_dir"]
    per_sequence = []
    conventional = []
    display_rows = []

    for benchmark in config["conventional"]:
        checkpoint_name = benchmark["checkpoint"]
        tqdm.write(f"Loading {benchmark['name']}: {checkpoint_name}")
        model = load_released_model(
            checkpoint_root / checkpoint_name, model_config, device
        )
        benchmark_rows = []
        split_summaries = []
        for split in benchmark["splits"]:
            split_rows = evaluate_split(
                model,
                model_config,
                device,
                data_root / split["file"],
                data_root / split["mask_dir"],
                benchmark["name"],
                split["name"],
                split["file"],
                checkpoint_name,
            )
            benchmark_rows.extend(split_rows)
            if len(benchmark["splits"]) > 1:
                split_summaries.append(
                    {"name": f"{benchmark['name']} / {split['name']}", **mean_metrics(split_rows)}
                )
        per_sequence.extend(benchmark_rows)
        metrics = mean_metrics(benchmark_rows)
        conventional.append({"benchmark": benchmark["name"], **metrics})
        display_rows.append({"name": benchmark["name"], **metrics})
        display_rows.extend(split_summaries)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    family_rows = []
    familyfold_model_config = config["familyfold_model"]
    familyfold_backbone = checkpoint_root / familyfold_model_config["backbone_checkpoint"]
    for family in config["familyfold"]:
        checkpoint_name = family["checkpoint"]
        tqdm.write(f"Loading FamilyFold / {family['name']}: {checkpoint_name}")
        model = load_familyfold_model(
            familyfold_backbone,
            checkpoint_root / checkpoint_name,
            model_config,
            familyfold_model_config,
            device,
        )
        rows = evaluate_split(
            model,
            model_config,
            device,
            data_root / family["file"],
            data_root / "ArchiveII",
            "FamilyFold",
            family["name"],
            family["file"],
            checkpoint_name,
        )
        per_sequence.extend(rows)
        metrics = mean_metrics(rows)
        family_rows.append({"family": family["name"], **metrics})
        display_rows.append({"name": f"FamilyFold / {family['name']}", **metrics})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    family_mean = {
        key: sum(row[key] for row in family_rows) / len(family_rows)
        for key in ("precision", "recall", "f1")
    }
    all_benchmarks = conventional + [{"benchmark": "FamilyFold", **family_mean}]
    display_rows.append({"name": "FamilyFold (mean)", **family_mean})
    per_sequence_dir = output_root / "per_sequence"
    metric_fields = [
        "benchmark",
        "source_split",
        "record_index",
        "length",
        "precision",
        "recall",
        "f1",
    ]
    write_csv(
        per_sequence_dir / "all.csv",
        [{field: row[field] for field in metric_fields} for row in per_sequence],
        metric_fields,
    )
    with (per_sequence_dir / "all.pickle").open("wb") as handle:
        pickle.dump(per_sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_csv(
        output_root / "conventional_benchmarks.csv",
        conventional,
        ["benchmark", "precision", "recall", "f1"],
    )
    write_csv(
        output_root / "familyfold_per_family.csv",
        family_rows,
        ["family", "precision", "recall", "f1"],
    )
    write_csv(
        output_root / "all_benchmarks.csv",
        all_benchmarks,
        ["benchmark", "precision", "recall", "f1"],
    )
    print("\nEvaluation results")
    print(format_summary_table(display_rows))
    print(
        "Conventional and PDB split rows are sequence means; "
        "FamilyFold (mean) weights each of the nine families equally."
    )
    print(f"Per-sequence metrics: {per_sequence_dir / 'all.csv'}")
    print(f"Per-sequence structures: {per_sequence_dir / 'all.pickle'}")


if __name__ == "__main__":
    main()
