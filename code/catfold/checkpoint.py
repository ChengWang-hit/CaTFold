"""Strict loading for released CaTFold model weights."""

import torch


def state_dict_from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError(f"Expected a model state_dict in {path}")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_weights(model, path, allow_missing_adapter=False):
    incompatible = model.load_state_dict(state_dict_from_checkpoint(path), strict=False)
    missing = list(incompatible.missing_keys)
    if allow_missing_adapter:
        missing = [key for key in missing if not key.startswith("adapter.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return model
