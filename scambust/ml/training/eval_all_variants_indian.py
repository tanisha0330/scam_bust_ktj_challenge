"""
Runs EVERY trained model variant against the Indian-context unseen-domain set.

WHY THIS IS A USEFUL CONTROL, NOT JUST A LEADERBOARD
The four `call_classifier*` variants were trained before any LLM-generated data
existed in the pipeline -- they have never seen a single Groq/Ollama-written
row. The unified model has (273 of them). So:

  * If the OLD variants also score well here, the 0.852 recall we measured for
    the unified model is unlikely to be generator-style recognition -- it would
    mean Indian-context novel scams are simply easier than the US-flavoured
    ICFD ones, which is the honest conclusion we want.
  * If ONLY the unified model scores well, the style-confound worry is real and
    that number should be discounted.

Note on temperature: softmax(logits / T) is a monotonic transform, so it does
not change argmax predictions or ranking. Accuracy/precision/recall/F1/ROC-AUC
are therefore identical regardless of each model's calibration temperature, and
we can compare variants directly.

Usage:
    python -m ml.training.eval_all_variants_indian
"""
import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODELS_ROOT = os.path.join(HERE, "..", "models_out")
MAX_LENGTH = 256
BATCH_SIZE = 32

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Ordered oldest -> newest so the table reads as a timeline.
VARIANTS = [
    ("call_classifier", "Baseline (near-full fine-tune)"),
    ("call_classifier_partial_freeze", "Static partial freeze (bottom 6)"),
    ("call_classifier_gradual_unfreeze", "Gradual unfreeze v1 (4ep / ~20k)"),
    ("call_classifier_gradual_unfreeze_v2", "Gradual unfreeze v2 (8ep / ~38k)"),
    ("message_classifier", "Message classifier (140 rows, transfer)"),
    ("unified_classifier", "Unified (production)"),
]


@torch.no_grad()
def predict_logits(texts, model, tokenizer):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [str(t) for t in texts[i:i + BATCH_SIZE]]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        out.append(model(**inputs).logits.float().cpu().numpy())
    return np.vstack(out)


def main():
    path = os.path.join(DATA_DIR, "indian_crossdomain_eval.csv")
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    y = df["label"].values
    print(f"Indian unseen-domain set: {len(df)} rows "
          f"({int((y == 1).sum())} scam / {int((y == 0).sum())} legit)\n")

    results = {}
    for dirname, label in VARIANTS:
        mdir = os.path.join(MODELS_ROOT, dirname)
        if not os.path.exists(os.path.join(mdir, "config.json")):
            print(f"  ! skipping {dirname} (no model found)")
            continue
        print(f"Evaluating {label} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(_device)
        model.eval()

        logits = predict_logits(df["text"].tolist(), model, tokenizer)
        m = compute_metrics((logits, y))
        preds = logits.argmax(axis=-1)

        # Per-domain scam recall, to see where each model breaks down.
        per_domain = {}
        for dom, idx in df.groupby("domain").groups.items():
            mask = df.index.isin(idx)
            scam = mask & (y == 1)
            if scam.any():
                per_domain[dom] = round(float((preds[scam] == 1).mean()), 3)
        results[dirname] = {"label": label, "metrics": m, "per_domain_recall": per_domain}

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("ALL VARIANTS on INDIAN-CONTEXT UNSEEN DOMAINS")
    print("=" * 96)
    print(f"{'variant':<40}{'acc':>8}{'prec':>8}{'recall':>9}{'f1':>8}{'auc':>8}")
    for k, v in results.items():
        m = v["metrics"]
        print(f"{v['label']:<40}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}{m['roc_auc']:>8.3f}")

    print(f"\n{'variant':<40}" + "".join(f"{d[:14]:>16}" for d in sorted(
        next(iter(results.values()))["per_domain_recall"])) if results else "")
    for k, v in results.items():
        row = "".join(f"{v['per_domain_recall'].get(d, float('nan')):>16.3f}"
                      for d in sorted(v["per_domain_recall"]))
        print(f"{v['label']:<40}{row}")

    out = os.path.join(MODELS_ROOT, "all_variants_indian_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
