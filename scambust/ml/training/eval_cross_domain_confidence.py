"""
Re-analyzes the cross_domain eval with confidence + the cascade's escalation
threshold factored in: how many of the misses would actually have triggered
Tier-2 LLM escalation in the live system, rather than silently returning a
wrong Tier-1-only verdict?
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CALL_MODEL_DIR = os.path.join(HERE, "..", "models_out", "call_classifier")
THRESHOLD = float(os.getenv("TIER1_CONFIDENCE_THRESHOLD", "0.85"))


def main():
    cross_df = pd.read_csv(os.path.join(DATA_DIR, "_icfd31k_raw_cross_domain.csv"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CALL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CALL_MODEL_DIR).to(device)
    model.eval()

    with open(os.path.join(CALL_MODEL_DIR, "calibration.json")) as f:
        temperature = json.load(f).get("temperature", 1.0)

    preds, confs = [], []
    with torch.no_grad():
        for text in cross_df["text"]:
            inputs = tokenizer(str(text), truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0].cpu()
            probs = torch.softmax(logits / temperature, dim=-1)
            label = int(torch.argmax(probs).item())
            preds.append(label)
            confs.append(float(probs[label].item()))

    cross_df["pred"] = preds
    cross_df["confidence"] = confs
    cross_df["would_escalate"] = cross_df["confidence"] < THRESHOLD
    cross_df["tier1_correct"] = cross_df["pred"] == cross_df["label"]

    n = len(cross_df)
    n_escalate = cross_df["would_escalate"].sum()
    print(f"Threshold: {THRESHOLD}")
    print(f"Total rows: {n}")
    print(f"Would escalate to LLM (Tier-1 confidence < threshold): {n_escalate} ({100*n_escalate/n:.1f}%)")

    trusted = cross_df[~cross_df["would_escalate"]]
    escalated = cross_df[cross_df["would_escalate"]]

    print(f"\n--- Among the {len(trusted)} rows Tier-1 was CONFIDENT about (no escalation) ---")
    print(f"Tier-1 accuracy on these: {trusted['tier1_correct'].mean():.3f}")
    print(trusted.groupby("label")["tier1_correct"].agg(["count", "mean"]))

    print(f"\n--- Among the {len(escalated)} rows that WOULD escalate to Tier-2 ---")
    print(f"Tier-1 accuracy on these (irrelevant in prod, LLM decides instead): {escalated['tier1_correct'].mean():.3f}")
    print(escalated.groupby("label")["tier1_correct"].agg(["count", "mean"]))

    # The real question: of the actual scams (label=1) Tier-1 got wrong, how many
    # would have been caught by escalation instead of silently missed?
    missed_scams = cross_df[(cross_df["label"] == 1) & (~cross_df["tier1_correct"])]
    missed_and_would_escalate = missed_scams["would_escalate"].sum()
    print(f"\n--- Missed scams (label=1, Tier-1 wrong): {len(missed_scams)} ---")
    print(f"Of those, would have escalated to LLM: {missed_and_would_escalate} ({100*missed_and_would_escalate/max(len(missed_scams),1):.1f}%)")
    print(f"Silently missed even with cascade (confident AND wrong): {len(missed_scams) - missed_and_would_escalate}")

    confidently_wrong = missed_scams[~missed_scams["would_escalate"]]
    print(f"\n--- Confidence distribution of the {len(confidently_wrong)} confidently-wrong missed scams ---")
    print(confidently_wrong["confidence"].describe())
    for thresh in [0.85, 0.90, 0.95, 0.99]:
        would_catch = (confidently_wrong["confidence"] < thresh).sum()
        print(f"  If threshold were {thresh}: {would_catch}/{len(confidently_wrong)} of these would now escalate ({100*would_catch/max(len(confidently_wrong),1):.1f}%)")


if __name__ == "__main__":
    main()
