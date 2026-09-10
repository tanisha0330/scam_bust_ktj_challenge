"""
Trains THE unified binary scam classifier (SMS + WhatsApp + calls, one model).

Training procedure follows the gradual-unfreeze schedule that scored best on
the unseen-domain test in earlier experiments (accuracy 0.473 / recall 0.459,
vs 0.444 / 0.431 for a near-full fine-tune and 0.358 / 0.332 for a static
partial freeze). Note that gap is from single runs and may be partly noise --
hence the --seed flag here, so results can be repeated and averaged.

Evaluated on four held-out sets, deliberately of increasing difficulty:
  1. unified_test                     -- in-distribution
  2. unified_eval_synthetic           -- held-out synthetic slice
  3. unified_eval_generated_holdout   -- generated scam domains never trained on
  4. ICFD-31k cross_domain            -- 1,000 real-shaped convos, 5 unseen domains
                                         (the metric that has driven every decision)

Also prints a PER-SOURCE breakdown. That matters because sources correlate with
labels (e.g. the legit banking corpus is 100% label 0): if the model scores well
on a source purely by recognising its writing style, that shows up here as
suspiciously perfect per-source numbers rather than as genuine fraud detection.

Usage:
    python -m ml.training.train_unified_classifier [--seed 42]
"""
import argparse
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
    NUM_ENCODER_LAYERS,
    GradualUnfreezeCallback,
    TextClassificationDataset,
    WeightedLossTrainerMixin,
    compute_metrics,
    count_trainable_params,
    fit_temperature,
    set_trainable_layers,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "models_out", "unified_classifier")

UNFREEZE_SCHEDULE = [NUM_ENCODER_LAYERS, 10, 8, 6]   # one entry per epoch


class WeightedTrainer(WeightedLossTrainerMixin, Trainer):
    pass


def evaluate_set(trainer, tokenizer, df, name):
    ds = TextClassificationDataset(df["text"], df["label"], tokenizer)
    logits = trainer.predict(ds).predictions
    metrics = compute_metrics((logits, df["label"].values))
    print(f"\n=== {name} ({len(df)} rows) ===")
    print(json.dumps(metrics, indent=2))

    if "source" in df.columns and df["source"].nunique() > 1:
        preds = np.argmax(logits, axis=-1)
        rows = []
        for src, grp_idx in df.groupby("source").groups.items():
            mask = df.index.isin(grp_idx)
            y, p = df.loc[mask, "label"].values, preds[mask]
            rows.append({
                "source": src, "n": int(mask.sum()),
                "scam_frac": round(float(y.mean()), 3),
                "accuracy": round(float((y == p).mean()), 3),
            })
        print(pd.DataFrame(rows).sort_values("n", ascending=False).to_string(index=False))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=len(UNFREEZE_SCHEDULE))
    ap.add_argument("--resume", type=str, default=None)
    # Which dataset build to train on: "unified" (use-everything, 31.5k rows)
    # or "balanced" (real/Indian-weighted, ~8.5k rows).
    ap.add_argument("--data-prefix", type=str, default="unified")
    ap.add_argument("--out-name", type=str, default=None)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(OUT_DIR), args.out_name) if args.out_name else OUT_DIR
    out_dir = base if args.seed == 42 else f"{base}_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    p = args.data_prefix
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"{p}_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, f"{p}_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, f"{p}_test.csv"))
    print(f"Dataset: {p}  ->  out_dir: {out_dir}")

    fit_df, monitor_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"], random_state=args.seed
    )
    print(f"Fit: {len(fit_df)}  Monitor: {len(monitor_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("Train label balance:", dict(train_df["label"].value_counts()))
    print("Train sources:", dict(train_df["source"].value_counts()))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    # Register the full eventual trainable range before the Trainer builds its
    # optimizer -- HF filters params by requires_grad exactly once, so layers
    # unfrozen later by the callback would otherwise never be updated.
    set_trainable_layers(model, min(UNFREEZE_SCHEDULE))
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    fit_ds = TextClassificationDataset(fit_df["text"], fit_df["label"], tokenizer)
    monitor_ds = TextClassificationDataset(monitor_df["text"], monitor_df["label"], tokenizer)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=fit_df["label"].values)
    print(f"Class weights (0,1): {class_weights}")

    targs = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * (len(fit_df) / 32) * args.epochs)),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=fit_ds, eval_dataset=monitor_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            GradualUnfreezeCallback(model, UNFREEZE_SCHEDULE),
        ],
    )
    trainer.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    trainer.train(resume_from_checkpoint=args.resume)

    # Calibrate on val, then evaluate everything.
    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer)
    temperature = fit_temperature(trainer.predict(val_ds).predictions, val_df["label"].values)
    print(f"\nCalibrated temperature: {temperature:.4f}")

    results = {"seed": args.seed, "temperature": temperature}
    results["test"] = evaluate_set(trainer, tokenizer, test_df, "IN-DISTRIBUTION TEST")

    for fname, label in [
        ("unified_eval_synthetic.csv", "HELD-OUT SYNTHETIC"),
        ("unified_eval_generated_holdout.csv", "GENERATED UNSEEN DOMAINS"),
        ("_icfd31k_raw_cross_domain.csv", "ICFD-31k CROSS-DOMAIN (5 unseen domains)"),
        ("indian_crossdomain_eval.csv", "INDIAN UNSEEN DOMAINS"),
        ("marketing_probe.csv", "REAL MARKETING FALSE-POSITIVE PROBE"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "text" in df and "label" in df:
                results[fname] = evaluate_set(trainer, tokenizer, df, label)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved unified classifier -> {out_dir}")


if __name__ == "__main__":
    main()
