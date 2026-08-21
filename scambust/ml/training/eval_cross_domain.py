"""
Evaluates the call classifier on ICFD-31k's 'cross_domain' split -- a ~1,000
conversation split that is NOT one of the standard train/validation/test
splits used anywhere in this project (not in training, not in val/test, not
in the stress-test sets). Loaded directly from its parquet files since it
isn't exposed via the datasets-server /splits API.

Usage:
    python -m ml.training.eval_cross_domain
"""
import json
import os

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
DEFAULT_MODEL_DIR = os.path.join(HERE, "..", "models_out", "call_classifier")

ICFD_REPO = "rishia2220/icfd-31k"
CROSS_DOMAIN_FILES = [
    "streaming_chunks/cross_domain-00000-of-00002.parquet",
    "streaming_chunks/cross_domain-00001-of-00002.parquet",
]


def load_cross_domain_final_chunks():
    cache_path = os.path.join(DATA_DIR, "_icfd31k_raw_cross_domain.csv")
    if os.path.exists(cache_path):
        print(f"Using cached {cache_path}")
        return pd.read_csv(cache_path)

    print("Downloading ICFD-31k 'cross_domain' parquet files ...")
    dfs = []
    for repo_path in CROSS_DOMAIN_FILES:
        local_path = hf_hub_download(repo_id=ICFD_REPO, filename=repo_path, repo_type="dataset")
        dfs.append(pd.read_parquet(local_path))
    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(raw_df)} raw rows from cross_domain split")

    best_by_conv = {}
    for row in raw_df.to_dict("records"):
        conv_id = row["conversation_uid"]
        ts = row["chunk_timestamp"]
        prev = best_by_conv.get(conv_id)
        if prev is None or ts > prev[0]:
            label = 1 if str(row["final_verdict"]).strip().upper() == "YES" else 0
            best_by_conv[conv_id] = (ts, row["cumulative_text"], label, row.get("case_type", ""), row.get("domain", ""))

    rows = [
        {"text": v[1], "label": v[2], "case_type": v[3], "domain": v[4]}
        for v in best_by_conv.values()
    ]
    df = pd.DataFrame(rows)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"], keep="first")
    print(f"-> {len(df)} unique final-chunk conversations")
    df.to_csv(cache_path, index=False)
    return df


def check_no_overlap(cross_df):
    """Sanity check: confirm none of these texts appear in anything we already used."""
    seen_texts = set()
    for fname in [
        "call_dataset_train.csv", "call_dataset_val.csv", "call_dataset_test.csv",
        "call_dataset_synthetic_eval.csv", "call_dataset_icfd_stress_val.csv",
        "call_dataset_icfd_stress_test.csv",
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            seen_texts.update(pd.read_csv(path)["text"].astype(str).tolist())
    overlap = cross_df["text"].isin(seen_texts).sum()
    print(f"Overlap check: {overlap} / {len(cross_df)} cross_domain rows also appear in previously-used data")
    return overlap


def main():
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    variant_name = os.path.basename(model_dir.rstrip("/\\"))
    print(f"Evaluating model at: {model_dir}")

    cross_df = load_cross_domain_final_chunks()
    print(f"\nLabel balance:\n{cross_df['label'].value_counts()}")

    check_no_overlap(cross_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    with open(os.path.join(model_dir, "calibration.json")) as f:
        temperature = json.load(f).get("temperature", 1.0)

    all_logits = []
    with torch.no_grad():
        for text in cross_df["text"]:
            inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0].cpu()
            all_logits.append(logits.tolist())

    import numpy as np
    logits_arr = np.array(all_logits)
    metrics = compute_metrics((logits_arr, cross_df["label"].values))

    print(f"\n=== RESULTS on ICFD-31k 'cross_domain' split [{variant_name}] ({len(cross_df)} never-before-used conversations) ===")
    print(json.dumps(metrics, indent=2))

    out_path = os.path.join(HERE, "..", "models_out", f"cross_domain_eval_{variant_name}.json")
    with open(out_path, "w") as f:
        json.dump({"variant": variant_name, "n_rows": len(cross_df), "metrics": metrics}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
