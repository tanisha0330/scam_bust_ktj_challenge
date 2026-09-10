"""
Re-fits temperature calibration for an already-trained model and prints the
complete evaluation table across every held-out set.

No retraining happens here -- calibration is a post-hoc scalar fit on the
validation logits, so it can be redone in seconds. Note that temperature is a
monotonic transform of the logits: it changes CONFIDENCE values (and therefore
Tier-2 escalation behaviour) but leaves accuracy / precision / recall / F1 /
ROC-AUC completely unchanged.

Usage:
    python -m ml.training.recalibrate_and_report [model_dir] [data_prefix]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import compute_metrics, expected_calibration_error, fit_temperature

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MAX_LENGTH, BATCH = 256, 32
_device = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_SETS = [
    ("{p}_test.csv", "in-distribution test"),
    ("unified_eval_synthetic.csv", "held-out synthetic"),
    ("unified_eval_generated_holdout.csv", "generated unseen domains"),
    ("_icfd31k_raw_cross_domain.csv", "ICFD cross-domain (US-flavoured)"),
    ("indian_crossdomain_eval.csv", "INDIAN unseen domains"),
    ("marketing_probe.csv", "real marketing FP probe"),
    ("call_dataset_icfd_stress_test.csv", "ICFD fraud-only stress test"),
]


@torch.no_grad()
def logits_for(texts, model, tok):
    out = []
    for i in range(0, len(texts), BATCH):
        b = [str(t) for t in texts[i:i + BATCH]]
        x = tok(b, truncation=True, padding="max_length", max_length=MAX_LENGTH,
                return_tensors="pt").to(_device)
        out.append(model(**x).logits.float().cpu().numpy())
    return np.vstack(out)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "models_out", "balanced_classifier")
    prefix = sys.argv[2] if len(sys.argv) > 2 else "balanced"

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device)
    model.eval()

    # ---- recalibrate on the validation split -------------------------------
    val = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}_val.csv")).dropna(subset=["text", "label"])
    vl = logits_for(val["text"].tolist(), model, tok)
    yv = val["label"].values
    ece_before = expected_calibration_error(softmax(vl, 1.0), yv)
    T = fit_temperature(vl, yv)
    ece_after = expected_calibration_error(softmax(vl, T), yv)
    print(f"CALIBRATION (on {len(val)} validation rows)")
    print(f"  temperature      : {T:.4f}")
    print(f"  ECE before (T=1) : {ece_before:.4f}")
    print(f"  ECE after        : {ece_after:.4f}")
    print(f"  mean confidence  : {softmax(vl, T).max(axis=1).mean():.4f}  "
          f"(accuracy {(vl.argmax(1) == yv).mean():.4f})")

    # ---- full evaluation ---------------------------------------------------
    rows = []
    print(f"\n{'evaluation set':<36}{'n':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}{'pred_scam':>11}")
    print("-" * 92)
    for fname, label in EVAL_SETS:
        path = os.path.join(DATA_DIR, fname.format(p=prefix))
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).dropna(subset=["text", "label"])
        lg = logits_for(df["text"].tolist(), model, tok)
        y = df["label"].values
        m = compute_metrics((lg, y))
        pred_rate = float((lg.argmax(1) == 1).mean())
        auc = m["roc_auc"]
        auc_s = f"{auc:>8.3f}" if np.isfinite(auc) else f"{'n/a':>8}"
        print(f"{label:<36}{len(df):>6}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{auc_s}{pred_rate:>11.3f}")
        rows.append({"set": label, "n": len(df), "actual_scam_rate": float(y.mean()),
                     "predicted_scam_rate": pred_rate, **m})

    # ---- per-source breakdown on the in-distribution test ------------------
    test_path = os.path.join(DATA_DIR, f"{prefix}_test.csv")
    df = pd.read_csv(test_path).dropna(subset=["text", "label"])
    if "source" in df.columns:
        lg = logits_for(df["text"].tolist(), model, tok)
        pred = lg.argmax(1)
        y = df["label"].values
        print(f"\nPER-SOURCE on in-distribution test:")
        print(f"  {'source':<22}{'n':>6}{'scam_frac':>11}{'accuracy':>10}")
        out = []
        for src, idx in df.groupby("source").groups.items():
            mask = df.index.isin(idx)
            out.append((src, int(mask.sum()), float(y[mask].mean()), float((y[mask] == pred[mask]).mean())))
        for src, n, sf, acc in sorted(out, key=lambda r: -r[1]):
            print(f"  {src:<22}{n:>6}{sf:>11.3f}{acc:>10.3f}")

    calib_path = os.path.join(model_dir, "calibration.json")
    data = json.load(open(calib_path)) if os.path.exists(calib_path) else {}
    data.update({"temperature": T, "ece_before": ece_before, "ece_after": ece_after,
                 "full_report": rows})
    with open(calib_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated -> {calib_path}")


if __name__ == "__main__":
    main()
