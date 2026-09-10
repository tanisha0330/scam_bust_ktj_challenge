"""
Fine-tunes MuRIL-base as the Tier-1 call/voice classifier.

Given the call dataset is reasonably large (~20k+ examples once ICFD-31k is
folded in), we do a near-full fine-tune (only the embedding layer is frozen,
everything else trains) rather than a frozen-encoder linear probe.

Usage:
    python -m ml.training.train_call_classifier
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .common import (
    MAX_LENGTH,
    MODEL_NAME,
    TextClassificationDataset,
    WeightedLossTrainerMixin,
    compute_metrics,
    fit_temperature,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "models_out", "call_classifier")


class WeightedTrainer(WeightedLossTrainerMixin, Trainer):
    pass


def load_split(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"call_dataset_{name}.csv"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # Hold out a small stratified slice of train for loss monitoring / early stopping.
    fit_df, monitor_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"], random_state=42
    )
    print(f"Fit: {len(fit_df)}  Monitor: {len(monitor_df)}  Val (calib): {len(val_df)}  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Freeze only the embedding layer -- with ~20k+ examples full fine-tuning
    # of the rest is reasonable and should outperform a frozen encoder.
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    fit_ds = TextClassificationDataset(fit_df["text"], fit_df["label"], tokenizer)
    monitor_ds = TextClassificationDataset(monitor_df["text"], monitor_df["label"], tokenizer)

    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=fit_df["label"].values
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights (0,1): {class_weights}")

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * (len(fit_df) / (8 * 4)) * 4)),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=fit_ds,
        eval_dataset=monitor_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.class_weights = class_weights_t

    trainer.train()

    # --- Calibrate temperature on the held-out val split (ICFD-31k's own val set) ---
    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer)
    val_logits = trainer.predict(val_ds).predictions
    temperature = fit_temperature(val_logits, val_df["label"].values)
    print(f"Calibrated temperature: {temperature:.4f}")

    # --- Final report: real-ish (ICFD-31k) test split vs. held-out synthetic slice ---
    test_ds = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer)
    test_output = trainer.predict(test_ds)
    test_metrics = compute_metrics((test_output.predictions, test_df["label"].values))
    print("=== TEST METRICS (call classifier, ICFD-31k real-ish held-out) ===")
    print(json.dumps(test_metrics, indent=2))

    extra_metrics = {}
    for label, fname in [
        ("synthetic", "call_dataset_synthetic_eval.csv"),
        ("icfd_stress_val", "call_dataset_icfd_stress_val.csv"),
        ("icfd_stress_test", "call_dataset_icfd_stress_test.csv"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        extra_df = pd.read_csv(path)
        extra_ds = TextClassificationDataset(extra_df["text"], extra_df["label"], tokenizer)
        extra_output = trainer.predict(extra_ds)
        m = compute_metrics((extra_output.predictions, extra_df["label"].values))
        extra_metrics[label] = m
        note = " (fraud-only recall stress test, not representative)" if "stress" in label else ""
        print(f"=== TEST METRICS ({label}){note} ===")
        print(json.dumps(m, indent=2))

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({
            "temperature": temperature,
            "test_metrics_main": test_metrics,
            **{f"test_metrics_{k}": v for k, v in extra_metrics.items()},
        }, f, indent=2)

    print(f"Saved call classifier -> {OUT_DIR}")


if __name__ == "__main__":
    main()
