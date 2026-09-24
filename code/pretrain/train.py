#!/usr/bin/env python3
"""Pretrain the CaTFold sequence encoder."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from catfold.data import PretrainDataset, collate_train
from catfold.model import PretrainNet


def distributed_mean(accelerator, value_sum, count):
    statistics = torch.stack((value_sum, value_sum.new_tensor(count)))
    statistics = accelerator.reduce(statistics, reduction="sum")
    return (statistics[0] / statistics[1]).item()


def count_trainable_parameters(modules):
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def print_startup_summary(accelerator, config, config_path, data_path, model):
    with h5py.File(data_path, "r") as handle:
        sequence_count = len(handle["seq"])
        structure_count = len(handle["ss"])
    if sequence_count != structure_count:
        raise ValueError(f"Sequence and structure counts differ in {data_path}")

    sequence_parameters = count_trainable_parameters(
        (model.node_embedding, model.position_embedding, model.encoder)
    )
    prediction_head_parameters = count_trainable_parameters((model.predictor_ss,))
    total_parameters = count_trainable_parameters((model,))
    if sequence_parameters + prediction_head_parameters != total_parameters:
        raise RuntimeError("Model parameter groups do not cover the complete model")

    per_gpu_batch = int(config["batch_size"])
    global_batch = per_gpu_batch * accelerator.num_processes
    gpu_count = accelerator.num_processes if accelerator.device.type == "cuda" else 0
    summary = [
        "Pretraining setup",
        f"  Config: {config_path}",
        f"  Dataset: {data_path}",
        f"  Dataset records: {sequence_count:,}",
        f"  Dataset file size: {data_path.stat().st_size / 1024**3:.2f} GiB",
        "  Total trainable parameters: "
        f"{total_parameters:,} ({total_parameters / 1e6:.3f} M)",
        "  Trainable Sequence Encoder parameters: "
        f"{sequence_parameters:,} ({sequence_parameters / 1e6:.3f} M)",
        "  Trainable pretraining prediction head parameters: "
        f"{prediction_head_parameters:,} "
        f"({prediction_head_parameters / 1e6:.3f} M)",
        f"  Architecture: embedding_dim={config['embedding_dim']}, "
        f"layers={config['layer_num']}, heads={config['nhead']}",
        "  Objective: weighted binary cross-entropy",
        "  Optimizer: AdamW",
        f"  Learning rate: {config['learning_rate']}",
        f"  Weight decay: {config['weight_decay']}",
        f"  Positive-class weight: {config['pos_weight']}",
        f"  Gradient-clipping threshold: {config['max_grad_norm']}",
        f"  Epochs: {config['epochs']}",
        f"  Batch size per process/GPU: {per_gpu_batch}",
        f"  Training processes: {accelerator.num_processes}",
        f"  GPUs: {gpu_count}",
        f"  Device: {accelerator.device}",
        f"  Global batch size: {global_batch}",
        f"  Mixed precision: {accelerator.mixed_precision}",
        f"  DataLoader workers per process: {config['num_workers']}",
        f"  Loss log interval: {config['loss_log_interval']} batches",
        f"  Checkpoint interval: {config['save_every']} epoch(s)",
        f"  Output directory: {PROJECT_ROOT / config['output_dir']}",
    ]
    accelerator.print("\n".join(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    loss_log_interval = int(config["loss_log_interval"])
    if loss_log_interval <= 0:
        raise ValueError("loss_log_interval must be a positive integer")
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    set_seed(config["seed"] + accelerator.process_index)
    data_path = PROJECT_ROOT / config["training_data"]
    model = PretrainNet(config["embedding_dim"], config["layer_num"], config["nhead"])
    print_startup_summary(
        accelerator,
        config,
        args.config.resolve(),
        data_path,
        model,
    )
    accelerator.print(f"Loading pretraining data: {data_path}")
    dataset = PretrainDataset(
        data_path,
        config["embedding_dim"],
        show_progress=accelerator.is_main_process,
    )
    accelerator.print(f"Loaded pretraining sequences: {len(dataset):,}")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_train,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config["pos_weight"]], device=accelerator.device)
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    output_dir = PROJECT_ROOT / config["output_dir"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = torch.zeros((), device=accelerator.device)
        interval_loss = torch.zeros((), device=accelerator.device)
        interval_batches = 0
        progress = tqdm(
            loader,
            desc=f"Pretrain epoch {epoch}/{config['epochs']}",
            unit="batch",
            disable=not accelerator.is_main_process,
            leave=False,
        )
        for batch_index, (onehot, positions, pairs, pad_mask, _, labels) in enumerate(
            progress, start=1
        ):
            logits = model((onehot, positions, pairs, pad_mask))
            loss = loss_function(logits, labels)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = loss.detach().float()
            total_loss += loss_value
            interval_loss += loss_value
            interval_batches += 1
            if batch_index % loss_log_interval == 0:
                mean_loss = distributed_mean(
                    accelerator, interval_loss, interval_batches
                )
                if accelerator.is_main_process:
                    tqdm.write(
                        f"epoch={epoch} batch={batch_index}/{len(loader)} "
                        f"interval_mean_loss={mean_loss:.8f}"
                    )
                interval_loss.zero_()
                interval_batches = 0
        progress.close()
        mean_loss = distributed_mean(accelerator, total_loss, len(loader))
        accelerator.print(
            f"epoch={epoch} mean_batch_loss={mean_loss:.8f}"
        )
        if epoch % config["save_every"] == 0 and accelerator.is_main_process:
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                output_dir / f"pretrain_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
