"""
Shared training/eval routine for call-classifier layer-freezing experiments.
Not run directly -- see train_call_classifier_partial_freeze.py and
train_call_classifier_gradual_unfreeze.py.
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
    MODEL_NAME,
    TextClassificationDataset,
    WeightedLossTrainerMixin,
    compute_metrics,
    count_trainable_params,
    fit_temperature,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


class WeightedTrainer(WeightedLossTrainerMixin, Trainer):
    pass


def load_split(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"call_dataset_{name}.csv"))


def run_experiment(variant_name: str, out_dir: str, configure_model_fn, extra_callbacks_fn=None, epochs=4,
                    resume_from_checkpoint=None):
    """configure_model_fn(model) -> None, sets initial requires_grad state.
    extra_callbacks_fn(model) -> list[TrainerCallback], e.g. gradual unfreeze."""
    os.makedirs(out_dir, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    fit_df, monitor_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"], random_state=42
    )
    print(f"[{variant_name}] Fit: {len(fit_df)}  Monitor: {len(monitor_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    configure_model_fn(model)
    trainable, total = count_trainable_params(model)
    print(f"[{variant_name}] Initial trainable params: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    fit_ds = TextClassificationDataset(fit_df["text"], fit_df["label"], tokenizer)
    monitor_ds = TextClassificationDataset(monitor_df["text"], monitor_df["label"], tokenizer)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=fit_df["label"].values)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * (len(fit_df) / (8 * 4)) * epochs)),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    if extra_callbacks_fn is not None:
        callbacks += extra_callbacks_fn(model)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=fit_ds,
        eval_dataset=monitor_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.class_weights = class_weights_t
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer)
    val_logits = trainer.predict(val_ds).predictions
    temperature = fit_temperature(val_logits, val_df["label"].values)
    print(f"[{variant_name}] Calibrated temperature: {temperature:.4f}")

    test_ds = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer)
    test_metrics = compute_metrics((trainer.predict(test_ds).predictions, test_df["label"].values))
    print(f"[{variant_name}] === TEST METRICS (primary, in-domain) ===")
    print(json.dumps(test_metrics, indent=2))

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"variant": variant_name, "temperature": temperature, "test_metrics_main": test_metrics}, f, indent=2)

    print(f"[{variant_name}] Saved -> {out_dir}")
    return out_dir
