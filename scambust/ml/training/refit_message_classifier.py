"""
Redoes just the final full-data fit + save for the message classifier, using
the CV-selected best strategy ("transfer"), without repeating the 10-fold CV
comparison (already run once; see train_message_classifier.py's CV results).
"""
import json
import os

import pandas as pd
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments

from .common import TextClassificationDataset, compute_metrics
from .train_message_classifier import DATA_DIR, OUT_DIR, WeightedTrainer, build_model

BEST_STRATEGY = "transfer"

CV_RESULTS = {
    "vanilla": {"eval_accuracy": 0.8357142857142856, "eval_precision": 0.8357142857142856,
                "eval_recall": 1.0, "eval_f1": 0.9104072398190045, "eval_roc_auc": 0.7834057971014492},
    "transfer": {"eval_accuracy": 0.9142857142857143, "eval_precision": 0.9419420289855072,
                 "eval_recall": 0.9572463768115942, "eval_f1": 0.9491057662658033, "eval_roc_auc": 0.9113224637681159},
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "message_dataset.csv"))

    model, tokenizer = build_model(BEST_STRATEGY)
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

    temperature = 1.5
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({
            "temperature": temperature,
            "init_strategy": BEST_STRATEGY,
            "cv_results": CV_RESULTS,
            "note": "Temperature is a placeholder (not fit on held-out data since all "
                     "rows were used for the final fit). Recalibrate once real usage "
                     "logs are available.",
        }, f, indent=2)

    print(f"Saved message classifier ({BEST_STRATEGY} init) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
