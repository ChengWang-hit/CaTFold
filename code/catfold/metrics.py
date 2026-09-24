"""Per-sequence base-pair metrics."""

import torch


def precision_recall_f1(prediction, target, eps=1e-11):
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    true_positive = torch.sign(prediction * target).sum()
    predicted_positive = torch.sign(prediction).sum()
    target_positive = target.sum()
    false_positive = predicted_positive - true_positive
    false_negative = target_positive - true_positive
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    f1 = (2 * true_positive + eps) / (
        2 * true_positive + false_positive + false_negative + eps
    )
    return tuple(float(value.cpu()) for value in (precision, recall, f1))
