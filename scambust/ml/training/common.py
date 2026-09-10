"""Shared utilities for fine-tuning MuRIL classifiers."""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from transformers import TrainerCallback

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 256
NUM_ENCODER_LAYERS = 12  # MuRIL-base


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


class WeightedLossTrainerMixin:
    """Mixin that applies class-weighted CrossEntropyLoss. Use with transformers.Trainer."""
    class_weights: torch.Tensor = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def set_trainable_layers(model, min_trainable_layer: int):
    """Freezes embeddings + encoder layers [0, min_trainable_layer), leaves
    encoder layers [min_trainable_layer, NUM_ENCODER_LAYERS), pooler, and the
    classification head trainable. min_trainable_layer=0 -> everything but
    embeddings trains (the original near-full-finetune setup).
    min_trainable_layer=NUM_ENCODER_LAYERS -> only pooler + head train."""
    base = model.base_model
    for param in base.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(base.encoder.layer):
        trainable = i >= min_trainable_layer
        for param in layer.parameters():
            param.requires_grad = trainable
    if hasattr(base, "pooler") and base.pooler is not None:
        for param in base.pooler.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class GradualUnfreezeCallback(TrainerCallback):
    """ULMFiT-style gradual unfreezing: starts with only the head (+pooler)
    trainable, then unfreezes encoder layers top-down at each epoch boundary
    according to `schedule` (a list of min_trainable_layer values, one per
    epoch, e.g. [12, 10, 8, 6] unfreezes 2 more layers each epoch, ending
    with layers 6-11 trainable by the final epoch)."""

    def __init__(self, model, schedule: list[int]):
        self.model = model
        self.schedule = schedule

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        stage = min(epoch_idx, len(self.schedule) - 1)
        min_trainable = self.schedule[stage]
        set_trainable_layers(self.model, min_trainable)
        trainable, total = count_trainable_params(self.model)
        print(f"[GradualUnfreeze] epoch {epoch_idx}: layers >= {min_trainable} trainable "
              f"({trainable:,}/{total:,} params, {100*trainable/total:.1f}%)")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy, bucketed by confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter=200) -> float:
    """Fits scalar temperature T calibrating softmax(logits / T), minimising NLL
    (Guo et al. 2017).

    Two corrections over the naive version, both prompted by a real failure:
    an earlier fit returned T = 0.05, i.e. the clamp floor, which is a
    degenerate result that makes the model *more* confident rather than
    calibrated -- and silently broke the escalation threshold that depends on
    confidence meaning something.

      1. Optimise log(T) instead of T. T must stay positive; the unconstrained
         parameter could cross zero and flip the sign of every logit.
      2. Reject degenerate fits. If the calibration set is small and the model
         separates it almost perfectly, NLL is genuinely minimised as T -> 0
         (infinite confidence). That is overfitting the calibration set, not
         calibration, so we fall back to T = 1.0 (identity) and say so.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros(1))          # log(1.0) == 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / torch.exp(log_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(torch.exp(log_t.detach()).item())

    if not (0.25 <= t <= 5.0) or not np.isfinite(t):
        print(f"  ! temperature fit degenerated (T={t:.4f}); falling back to T=1.0 "
              f"(uncalibrated). Calibration set likely too small or too easy.")
        return 1.0
    return t
