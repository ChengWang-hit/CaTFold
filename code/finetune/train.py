#!/usr/bin/env python3
"""Fine-tune CaTFold on a configured benchmark split."""

import argparse
import json
import sys
import warnings
from bisect import bisect_right
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from catfold.checkpoint import load_weights
from catfold.data import StructureDataset, collate_train
from catfold.decoding import decode
from catfold.familyfold import FamilyFoldFusionNet
from catfold.metrics import precision_recall_f1
from catfold.model import FineTuneNet


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    base = config.pop("base", None)
    if base:
        inherited = load_config((path.parent / base).resolve())
        inherited.update(config)
        config = inherited
    held_out = config.pop("held_out_family", None)
    if held_out:
        families = [
            "5s",
            "16s",
            "23s",
            "RNaseP",
            "grp1",
            "srp",
            "tRNA",
            "telomerase",
            "tmRNA",
        ]
        if held_out not in families:
            raise ValueError(f"Unknown FamilyFold family: {held_out}")
        config["train"] = [
            {
                "file": f"data/ArchiveII/family_fold/{family}.pickle",
                "mask_dir": "data/ArchiveII",
            }
            for family in families
            if family != held_out
        ]
        config["evaluation"] = {
            "name": f"FamilyFold / {held_out}",
            "file": f"data/ArchiveII/family_fold/{held_out}.pickle",
            "mask_dir": "data/ArchiveII",
        }
    return config


def dataset_from_entry(entry, config):
    return StructureDataset(
        PROJECT_ROOT / entry["file"],
        PROJECT_ROOT / entry["mask_dir"],
        config["embedding_dim"],
    )


def dataset_from_group(group, config):
    datasets = [dataset_from_entry(entry, config) for entry in group["datasets"]]
    if not datasets:
        raise ValueError(f"Dataset group is empty: {group['name']}")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def structure_at(dataset, index):
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect_right(dataset.cumulative_sizes, index)
        previous_size = (
            0
            if dataset_index == 0
            else dataset.cumulative_sizes[dataset_index - 1]
        )
        return dataset.datasets[dataset_index].structures[index - previous_size]
    return dataset.structures[index]


def count_trainable_parameters(modules):
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def configured_training_stages(config):
    stages = config.get("stages")
    if stages is None:
        stages = [
            {
                "name": "Fine-tuning",
                "epochs": config["epochs"],
                "train": config["train"],
            }
        ]
    if not stages:
        raise ValueError("At least one training stage is required")
    for stage in stages:
        if not stage.get("train"):
            raise ValueError(f"Training data is empty for stage: {stage['name']}")
        if int(stage["epochs"]) < 1:
            raise ValueError(f"Epochs must be positive for stage: {stage['name']}")
    return stages


def periodic_checkpoint_epochs(epochs, checkpoint_count):
    epochs = int(epochs)
    checkpoint_count = min(int(checkpoint_count), epochs)
    if checkpoint_count < 1:
        raise ValueError("Periodic checkpoint count must be positive")
    return tuple(
        (index * epochs + checkpoint_count - 1) // checkpoint_count
        for index in range(1, checkpoint_count + 1)
    )


def training_loader(config, entries):
    datasets = []
    dataset_summaries = []
    for entry in entries:
        dataset = dataset_from_entry(entry, config)
        datasets.append(dataset)
        dataset_summaries.append(
            {
                "path": PROJECT_ROOT / entry["file"],
                "records": len(dataset),
            }
        )
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    generator = None
    if config.get("model_variant") == "familyfold_fusion":
        generator = torch.Generator().manual_seed(config["seed"])
    return (
        DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=collate_train,
            generator=generator,
        ),
        dataset_summaries,
    )


def build_model(config):
    if config.get("model_variant") == "familyfold_fusion":
        return FamilyFoldFusionNet(
            config["embedding_dim"],
            config["layer_num"],
            config["nhead"],
            hidden_dim=config["adapter_hidden_dim"],
            num_blocks=config["resnet_blocks"],
            kernel_size=config["kernel_size"],
        )
    return FineTuneNet(
        config["embedding_dim"],
        config["layer_num"],
        config["nhead"],
        adapter_hidden_dim=config["adapter_hidden_dim"],
        adapter_dropout=config["adapter_dropout"],
        train_pair_out=config["train_pair_out"],
    )


def format_file_size(path):
    size = path.stat().st_size
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    return f"{size / 1024**2:.2f} MiB"


def print_startup_summary(
    accelerator,
    config,
    config_path,
    model,
    training_stages,
    evaluation_dataset,
    test_dataset,
):
    sequence_parameters = count_trainable_parameters(
        (model.node_embedding, model.position_embedding, model.encoder)
    )
    prediction_head_parameters = count_trainable_parameters((model.predictor_ss,))
    pair_encoder_parameters = count_trainable_parameters((model.adapter,))
    total_parameters = count_trainable_parameters((model,))
    grouped_parameters = (
        sequence_parameters
        + prediction_head_parameters
        + pair_encoder_parameters
    )
    if grouped_parameters != total_parameters:
        raise RuntimeError(
            "Model parameter groups do not cover all trainable parameters"
        )

    per_gpu_batch = int(config["batch_size"])
    global_batch = per_gpu_batch * accelerator.num_processes
    gpu_count = accelerator.num_processes if accelerator.device.type == "cuda" else 0
    checkpoint_path = PROJECT_ROOT / config["init_checkpoint"]
    output_dir = PROJECT_ROOT / config["output_dir"]
    soft_f1_weight = float(config["soft_f1_weight"])
    objective = (
        "weighted two-class cross-entropy on [-logit, logit]"
        if config.get("model_variant") == "familyfold_fusion"
        else "weighted binary cross-entropy"
    )
    if soft_f1_weight:
        objective += f" + {soft_f1_weight:g} * soft-F1 loss"

    summary = [
        "Fine-tuning setup",
        f"  Config: {config_path}",
        f"  Initial checkpoint: {checkpoint_path}",
        f"  Fine-tuning mode: {config['pretrain_finetune_mode']}",
        f"  Training stages: {len(training_stages)}",
        "  Reload validation-best weights between stages: "
        f"{config.get('reload_best_between_stages', False)}",
    ]
    for stage_index, stage in enumerate(training_stages, start=1):
        checkpoint_epochs = periodic_checkpoint_epochs(
            stage["epochs"], config["periodic_checkpoint_count"]
        )
        summary.append(
            f"  Stage {stage_index}: {stage['name']} "
            f"(epochs={stage['epochs']}, "
            f"records={sum(item['records'] for item in stage['dataset_summaries']):,})"
        )
        summary.append(
            "    Periodic checkpoints: "
            + ", ".join(str(epoch) for epoch in checkpoint_epochs)
        )
        summary.append(
            f"    Validation-best checkpoint: "
            f"{'enabled' if config['save_best'] else 'disabled'}"
        )
        for item in stage["dataset_summaries"]:
            summary.append(
                f"    Training data: {item['path']} "
                f"(records={item['records']:,}, "
                f"size={format_file_size(item['path'])})"
            )
    evaluation_entry = config["evaluation"]
    evaluation_path = PROJECT_ROOT / evaluation_entry["file"]
    summary.extend(
        [
            f"  Per-epoch evaluation: {evaluation_entry['name']}",
            f"  Evaluation data: {evaluation_path} "
            f"(records={len(evaluation_dataset):,}, "
            f"size={format_file_size(evaluation_path)})",
            f"  Evaluation post-processing: MWM (threshold={config['threshold']})",
        ]
    )
    test_entry = config.get("test")
    if test_entry:
        summary.append(f"  Per-epoch test: {test_entry['name']}")
        summary.append(f"  Total test records: {len(test_dataset):,}")
        for entry in test_entry["datasets"]:
            test_path = PROJECT_ROOT / entry["file"]
            summary.append(
                f"  Test data: {test_path} (size={format_file_size(test_path)})"
            )
        summary.append(
            f"  Test post-processing: MWM (threshold={config['threshold']})"
        )
    summary.extend(
        [
            "  Total trainable parameters: "
            f"{total_parameters:,} ({total_parameters / 1e6:.3f} M)",
            "  Trainable Sequence Encoder parameters: "
            f"{sequence_parameters:,} ({sequence_parameters / 1e6:.3f} M)",
            "  Trainable pretraining prediction head parameters: "
            f"{prediction_head_parameters:,} "
            f"({prediction_head_parameters / 1e6:.3f} M)",
            "  Trainable Pair Encoder (adapter) parameters: "
            f"{pair_encoder_parameters:,} "
            f"({pair_encoder_parameters / 1e6:.3f} M)",
            f"  Architecture: embedding_dim={config['embedding_dim']}, "
            f"layers={config['layer_num']}, heads={config['nhead']}, "
            f"adapter_hidden_dim={config['adapter_hidden_dim']}, "
            f"adapter_dropout={config['adapter_dropout']}",
            f"  Objective: {objective}",
            "  Optimizer: AdamW",
            f"  Seed: {config['seed']}",
            f"  Sequence Encoder learning rate: {config['pretrain_learning_rate']}",
            "  Pretraining prediction head learning rate: "
            f"{config['pretrain_learning_rate']}",
            "  Pair Encoder (adapter) learning rate: "
            f"{config['adapter_learning_rate']}",
            f"  Layer-wise learning-rate decay: "
            f"{config.get('pretrain_layer_decay', 1.0)}",
            f"  Weight decay: {config['weight_decay']}",
            f"  Positive-class weight: {config['pos_weight']}",
            f"  Gradient-clipping threshold: {config['max_grad_norm']}",
            f"  Batch size per process/GPU: {per_gpu_batch}",
            f"  Training processes: {accelerator.num_processes}",
            f"  GPUs: {gpu_count}",
            f"  Device: {accelerator.device}",
            f"  Global batch size: {global_batch}",
            f"  Mixed precision: {accelerator.mixed_precision}",
            f"  DataLoader workers per process: {config['num_workers']}",
            f"  Output directory: {output_dir}",
        ]
    )
    accelerator.print("\n".join(summary))


def evaluate_epoch(
    accelerator,
    model,
    dataset,
    config,
    stage_label,
    epoch,
    epochs,
    name,
    phase,
):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    local_totals = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    indices = range(accelerator.process_index, len(dataset), accelerator.num_processes)
    progress = tqdm(
        indices,
        desc=f"{stage_label} {phase} epoch {epoch}/{epochs}: {name}",
        unit="sequence",
        disable=not accelerator.is_main_process,
        leave=False,
    )
    with torch.inference_mode():
        for index in progress:
            sample = dataset[index]
            batch = tuple(
                value.to(accelerator.device) for value in collate_train([sample])
            )
            onehot, positions, pairs, pad_mask, lengths, _ = batch
            logits = unwrapped_model(
                (onehot, positions, pairs, pad_mask, lengths)
            )
            length = int(lengths[0].item())
            probabilities = torch.zeros((length, length), dtype=torch.float32)
            pair_indices = pairs.cpu()
            pair_probabilities = torch.sigmoid(logits).float().cpu()
            probabilities[pair_indices[0], pair_indices[1]] = pair_probabilities
            probabilities[pair_indices[1], pair_indices[0]] = pair_probabilities
            prediction = decode(probabilities, config["threshold"])

            target = torch.zeros_like(prediction)
            true_pairs = torch.as_tensor(structure_at(dataset, index), dtype=torch.long)
            if true_pairs.numel():
                true_pairs = true_pairs.reshape(-1, 2)
                target[true_pairs[:, 0], true_pairs[:, 1]] = 1
            precision, recall, f1 = precision_recall_f1(prediction, target)
            local_totals += torch.tensor(
                (precision, recall, f1, 1),
                dtype=torch.float64,
                device=accelerator.device,
            )
    progress.close()
    totals = accelerator.reduce(local_totals, reduction="sum")
    metrics = totals[:3] / totals[3]
    accelerator.print(
        f"stage={stage_label} epoch={epoch}/{epochs} {phase}={name} "
        f"precision={metrics[0].item():.6f} "
        f"recall={metrics[1].item():.6f} "
        f"f1={metrics[2].item():.6f}"
    )
    model.train()
    return metrics[2].item()


def checkpoint_name(stage_index, stage_count, label, epoch=None, run_prefix=""):
    prefix = run_prefix if stage_count == 1 else f"{run_prefix}stage_{stage_index}_"
    if label == "epoch":
        return f"{prefix}epoch_{epoch:04d}.pt"
    return f"{prefix}best.pt"


def save_model_weights(accelerator, model, paths, adapter_only=False):
    if not paths:
        return
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = (
            unwrapped_model.adapter.state_dict()
            if adapter_only
            else unwrapped_model.state_dict()
        )
        for path in paths:
            accelerator.save(state_dict, path)
            accelerator.print(f"Saved model weights: {path}")


def build_optimizer(model, config):
    if config.get("model_variant") == "familyfold_fusion":
        return torch.optim.AdamW(
            model.adapter.parameters(),
            lr=config["adapter_learning_rate"],
            weight_decay=config["weight_decay"],
        )
    pretrained, pair_out, adapter = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("adapter."):
            adapter.append(parameter)
        elif name.startswith("predictor_ss.pair_out."):
            pair_out.append(parameter)
        else:
            pretrained.append((name, parameter))
    groups = []
    if pretrained:
        layer_decay = config.get("pretrain_layer_decay", 1.0)
        by_depth = {}
        for name, parameter in pretrained:
            depth = 0
            prefix = "encoder.ct_block."
            if name.startswith(prefix):
                block = int(name[len(prefix) :].split(".", 1)[0])
                depth = config["layer_num"] - 1 - block
            elif name.startswith(("node_embedding.", "position_embedding.")):
                depth = config["layer_num"]
            by_depth.setdefault(depth, []).append(parameter)
        for depth, parameters in by_depth.items():
            rate = config["pretrain_learning_rate"] * layer_decay**depth
            groups.append({"params": parameters, "lr": rate})
    if pair_out:
        groups.append(
            {"params": pair_out, "lr": config["pretrain_learning_rate"]}
        )
    if adapter:
        groups.append(
            {
                "params": adapter,
                "lr": config["adapter_learning_rate"],
            }
        )
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def soft_f1_loss(logits, labels, pairs, lengths, eps=1e-6):
    probabilities = torch.sigmoid(logits)
    losses = []
    pair_offset = 0
    node_offset = 0
    for length_tensor in lengths:
        length = int(length_tensor)
        count = int(
            ((pairs[0] >= node_offset) & (pairs[0] < node_offset + length)).sum()
        )
        if count:
            probability = probabilities[pair_offset : pair_offset + count]
            label = labels[pair_offset : pair_offset + count]
            true_positive = (probability * label).sum()
            losses.append(
                1 - (2 * true_positive + eps) / (
                    probability.sum() + label.sum() + eps
                )
            )
        pair_offset += count
        node_offset += length
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def familyfold_comparison_loss(logits, labels, config):
    two_class_logits = torch.stack((-logits, logits), dim=1)
    class_weights = torch.tensor(
        [config["negative_class_weight"], config["positive_class_weight"]],
        dtype=logits.dtype,
        device=logits.device,
    )
    return torch.nn.functional.cross_entropy(
        two_class_logits,
        labels.long(),
        weight=class_weights,
    )


def train_stage(
    accelerator,
    model,
    optimizer,
    train_data,
    evaluation_dataset,
    test_dataset,
    loss_function,
    config,
    stage,
    stage_index,
    stage_count,
    output_dir,
    checkpoint_prefix,
):
    stage_label = f"Stage {stage_index}/{stage_count}: {stage['name']}"
    epochs = int(stage["epochs"])
    periodic_epochs = set(
        periodic_checkpoint_epochs(epochs, config["periodic_checkpoint_count"])
    )
    best_evaluation_f1 = float("-inf")
    accelerator.print(f"\nStarting {stage_label}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_data,
            desc=f"{stage_label} epoch {epoch}/{epochs}",
            unit="batch",
            disable=not accelerator.is_main_process,
            leave=False,
        )
        for onehot, positions, pairs, pad_mask, lengths, labels in progress:
            logits = model((onehot, positions, pairs, pad_mask, lengths))
            loss = loss_function(logits, labels).mean()
            if config["soft_f1_weight"]:
                loss = loss + config["soft_f1_weight"] * soft_f1_loss(
                    logits, labels, pairs, lengths
                )
            accelerator.backward(loss)
            if config["max_grad_norm"] > 0:
                accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
        accelerator.print(
            f"stage={stage_label} epoch={epoch}/{epochs} "
            f"mean_batch_loss={total_loss / max(len(train_data), 1):.8f}"
        )
        evaluation_f1 = evaluate_epoch(
            accelerator,
            model,
            evaluation_dataset,
            config,
            stage_label,
            epoch,
            epochs,
            config["evaluation"]["name"],
            "evaluation",
        )
        checkpoint_paths = []
        if epoch in periodic_epochs:
            checkpoint_paths.append(
                output_dir
                / checkpoint_name(
                    stage_index,
                    stage_count,
                    "epoch",
                    epoch,
                    checkpoint_prefix,
                )
            )
        if config["save_best"] and evaluation_f1 > best_evaluation_f1:
            best_evaluation_f1 = evaluation_f1
            checkpoint_paths.append(
                output_dir
                / checkpoint_name(
                    stage_index,
                    stage_count,
                    "best",
                    run_prefix=checkpoint_prefix,
                )
            )
        save_model_weights(
            accelerator,
            model,
            checkpoint_paths,
            adapter_only=config.get("model_variant") == "familyfold_fusion",
        )
        if test_dataset is not None:
            evaluate_epoch(
                accelerator,
                model,
                test_dataset,
                config,
                stage_label,
                epoch,
                epochs,
                config["test"]["name"],
                "test",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    set_seed(config["seed"] + accelerator.process_index)
    model = build_model(config)
    load_weights(
        model,
        PROJECT_ROOT / config["init_checkpoint"],
        allow_missing_adapter=True,
    )
    if config["pretrain_finetune_mode"] == "frozen":
        model.freeze_pretrained()
    training_stages = []
    for stage in configured_training_stages(config):
        train_data, dataset_summaries = training_loader(config, stage["train"])
        training_stages.append(
            {
                **stage,
                "train_data": train_data,
                "dataset_summaries": dataset_summaries,
            }
        )
    evaluation_dataset = dataset_from_entry(config["evaluation"], config)
    test_dataset = (
        dataset_from_group(config["test"], config) if config.get("test") else None
    )
    print_startup_summary(
        accelerator,
        config,
        args.config.resolve(),
        model,
        training_stages,
        evaluation_dataset,
        test_dataset,
    )
    if config.get("model_variant") == "familyfold_fusion":
        loss_function = lambda logits, labels: familyfold_comparison_loss(
            logits, labels, config
        )
    else:
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config["pos_weight"]], device=accelerator.device),
            reduction="none",
        )
    model = accelerator.prepare(model)
    output_dir = PROJECT_ROOT / config["output_dir"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    stage_count = len(training_stages)
    checkpoint_prefix = "" if config["save_best"] else f"{args.config.stem}_"
    if config.get("reload_best_between_stages", False) and not config["save_best"]:
        raise ValueError(
            "reload_best_between_stages requires validation-best checkpoint saving"
        )
    for stage_index, stage in enumerate(training_stages, start=1):
        optimizer = build_optimizer(accelerator.unwrap_model(model), config)
        optimizer, train_data = accelerator.prepare(
            optimizer,
            stage["train_data"],
        )
        train_stage(
            accelerator,
            model,
            optimizer,
            train_data,
            evaluation_dataset,
            test_dataset,
            loss_function,
            config,
            stage,
            stage_index,
            stage_count,
            output_dir,
            checkpoint_prefix,
        )
        if (
            config.get("reload_best_between_stages", False)
            and stage_index < stage_count
        ):
            best_path = output_dir / checkpoint_name(
                stage_index,
                stage_count,
                "best",
                run_prefix=checkpoint_prefix,
            )
            accelerator.wait_for_everyone()
            load_weights(accelerator.unwrap_model(model), best_path)
            accelerator.wait_for_everyone()
            accelerator.print(
                f"Reloaded validation-best weights before Stage {stage_index + 1}: "
                f"{best_path}"
            )


if __name__ == "__main__":
    main()
