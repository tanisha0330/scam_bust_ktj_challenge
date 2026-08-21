"""
Evaluates the unified classifier on the Indian-context unseen-domain set, and
puts the result side by side with the ICFD-31k cross_domain result.

THE QUESTION
ICFD cross_domain gave 0.427 recall, but it mixes two shifts: new scam types
AND a US context ("Bank of America", "SSN", "the IRS") that our India-centric
training never covered. This set holds the five scam types constant while
moving the context back to India. Comparing the two isolates which shift the
model is actually failing on.

Also reports per-domain accuracy, so we can see whether the difficulty is
uniform or concentrated in particular scam types.

Usage:
    python -m ml.training.eval_indian_crossdomain
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models_out", "unified_classifier")
MAX_LENGTH = 256
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = float(os.getenv("TIER1_CONFIDENCE_THRESHOLD", "0.85"))

_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(texts, model, tokenizer, temperature):
    logits_all, confs = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [str(t) for t in texts[i:i + BATCH_SIZE]]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits / temperature, dim=-1)
        logits_all.append(logits.cpu().numpy())
        confs.append(probs.max(dim=-1).values.cpu().numpy())
    return np.vstack(logits_all), np.concatenate(confs)


def main():
    path = os.path.join(DATA_DIR, "indian_crossdomain_eval.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} -- run ml.data_prep.generate_indian_crossdomain_eval first")
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    print(f"Indian cross-domain eval: {len(df)} rows | label balance {dict(df['label'].value_counts())}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    model.eval()
    temperature = json.load(open(os.path.join(MODEL_DIR, "calibration.json"))).get("temperature", 1.0)

    logits, confs = predict(df["text"].tolist(), model, tokenizer, temperature)
    preds = logits.argmax(axis=-1)
    y = df["label"].values
    metrics = compute_metrics((logits, y))

    print("\n=== INDIAN-CONTEXT UNSEEN DOMAINS ===")
    print(json.dumps(metrics, indent=2))

    print("\nPer-domain breakdown:")
    rows = []
    for dom, idx in df.groupby("domain").groups.items():
        mask = df.index.isin(idx)
        yy, pp = y[mask], preds[mask]
        scam = yy == 1
        rows.append({
            "domain": dom,
            "n": int(mask.sum()),
            "accuracy": round(float((yy == pp).mean()), 3),
            "scam_recall": round(float((pp[scam] == 1).mean()), 3) if scam.any() else float("nan"),
            "legit_acc": round(float((pp[~scam] == 0).mean()), 3) if (~scam).any() else float("nan"),
        })
    print(pd.DataFrame(rows).sort_values("scam_recall").to_string(index=False))

    # How often would the cascade escalate here?
    esc = confs < CONFIDENCE_THRESHOLD
    is_wrong = preds != y
    print(f"\nCascade behaviour at threshold {CONFIDENCE_THRESHOLD}:")
    print(f"  escalated to Tier-2      : {100 * esc.mean():.1f}%")
    print(f"  Tier-1 mistakes          : {is_wrong.sum()}")
    print(f"  mistakes that escalate   : {100 * (esc & is_wrong).sum() / max(is_wrong.sum(), 1):.1f}%")

    # Side-by-side with the US-flavoured ICFD cross_domain result.
    icfd_path = os.path.join(MODEL_DIR, "calibration.json")
    icfd = json.load(open(icfd_path)).get("_icfd31k_raw_cross_domain.csv")
    if icfd:
        print("\n=== SAME SCAM TYPES, DIFFERENT COUNTRY CONTEXT ===")
        print(f"{'metric':<12}{'ICFD (US-flavoured)':>22}{'Ours (Indian)':>16}{'delta':>10}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            a, b = icfd[k], metrics[k]
            print(f"{k:<12}{a:>22.3f}{b:>16.3f}{b - a:>+10.3f}")

    out = os.path.join(HERE, "..", "models_out", "indian_crossdomain_eval.json")
    with open(out, "w") as f:
        json.dump({"n": len(df), "metrics": metrics, "per_domain": rows,
                   "escalation_rate": float(esc.mean())}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
