"""
Fine-tunes the Tier-1 SMS/WhatsApp message classifier.

The message dataset is tiny (~150 rows), so:
  1. We freeze everything except the last encoder layer + pooler + head, to
     limit how many parameters can overfit.
  2. We evaluate with stratified k-fold CV (a single 80/20 split is too
     noisy to trust at this size) rather than a fixed held-out set.
  3. We compare two initializations -- starting from vanilla MuRIL vs.
     starting from our already-fine-tuned call classifier -- to check
     whether transfer learning actually helps here, rather than assuming it.
  4. Whichever initialization wins on CV, we refit on 100% of the data for
     the deployed artifact (CV is for the *evaluation number*, not the
     shipped weights).

Usage:
    python -m ml.training.train_message_classifier
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .common import MODEL_NAME, TextClassificationDataset, WeightedLossTrainerMixin, compute_metrics, fit_temperature

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CALL_MODEL_DIR = os.path.join(HERE, "..", "models_out", "call_classifier")
OUT_DIR = os.path.join(HERE, "..", "models_out", "message_classifier")


class WeightedTrainer(WeightedLossTrainerMixin, Trainer):
    pass


def freeze_all_but_last_layer(model):
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.base_model.encoder.layer[-1].parameters():
        param.requires_grad = True
    if hasattr(model.base_model, "pooler") and model.base_model.pooler is not None:
        for param in model.base_model.pooler.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def build_model(init_from: str):
    source = CALL_MODEL_DIR if init_from == "transfer" else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
    freeze_all_but_last_layer(model)
    return model, tokenizer


def train_one_fold(train_df, eval_df, init_from: str, epochs=8):
    model, tokenizer = build_model(init_from)
    train_ds = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer)
    eval_ds = TextClassificationDataset(eval_df["text"], eval_df["label"], tokenizer)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["label"].values)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "tmp_fold"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.class_weights = class_weights_t
    trainer.train()
    metrics = trainer.evaluate()
    return metrics


def run_cv(df, init_from: str, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_i, (train_idx, eval_idx) in enumerate(skf.split(df["text"], df["label"])):
        print(f"[{init_from}] Fold {fold_i + 1}/{n_splits}")
        m = train_one_fold(df.iloc[train_idx], df.iloc[eval_idx], init_from)
        fold_metrics.append(m)
    avg = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0] if k.startswith("eval_") and k != "eval_loss"
    }
    return avg, fold_metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "message_dataset.csv"))
    print(f"Message dataset: {len(df)} rows, label balance:\n{df['label'].value_counts()}")

    have_call_model = os.path.exists(os.path.join(CALL_MODEL_DIR, "config.json"))
    strategies = ["vanilla"] + (["transfer"] if have_call_model else [])
    if not have_call_model:
        print("WARNING: call_classifier not found -- skipping transfer-learning comparison, training vanilla only.")

    results = {}
    for strat in strategies:
        avg, _ = run_cv(df, strat)
        results[strat] = avg
        print(f"=== {strat} CV avg ===\n{json.dumps(avg, indent=2)}")

    best_strat = max(results, key=lambda k: results[k]["eval_f1"])
    print(f"\nBest strategy by CV F1: {best_strat}")
    print(json.dumps(results, indent=2))

    # --- Refit best strategy on 100% of data for the deployed artifact ---
    model, tokenizer = build_model(best_strat)
    full_ds = TextClassificationDataset(df["text"], df["label"], tokenizer)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df["label"].values)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "final_fit"),
        num_train_epochs=8,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=full_ds, compute_metrics=compute_metrics)
    trainer.class_weights = class_weights_t
    trainer.train()

    # No independent held-out set left to calibrate temperature on (all data used for
    # the final fit) -- fall back to the average logit-scale from CV, or default T=1.5
    # as a mild-shrinkage placeholder; recalibrate once real usage logs exist.
    temperature = 1.5

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({
            "temperature": temperature,
            "init_strategy": best_strat,
            "cv_results": results,
            "note": "Temperature is a placeholder (not fit on held-out data since all "
                     "rows were used for the final fit). Recalibrate once real usage "
                     "logs are available.",
        }, f, indent=2)

    print(f"Saved message classifier ({best_strat} init) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
