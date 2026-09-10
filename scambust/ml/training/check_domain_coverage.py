"""
Checks which of ICFD-31k's 10 training domains are actually represented in
our processed call-classifier training pool. The original data_prep script
discarded the `domain` column when building call_dataset_train.csv, so this
re-streams the train split (fast: parquet files are already in the local HF
cache from the first run, no re-download) and reports domain coverage among
the same final-chunk conversations that fed the training set.
"""
import os

import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
ICFD_REPO = "rishia2220/icfd-31k"


def main():
    print("Streaming ICFD-31k 'train' split (should hit local cache, no re-download) ...")
    ds = load_dataset(ICFD_REPO, split="train", streaming=True)

    best_by_conv = {}
    n_seen = 0
    for row in ds:
        n_seen += 1
        conv_id = row["conversation_uid"]
        ts = row["chunk_timestamp"]
        prev = best_by_conv.get(conv_id)
        if prev is None or ts > prev[0]:
            best_by_conv[conv_id] = (ts, row["domain"], row["case_type"], row["final_verdict"])
        if n_seen % 200000 == 0:
            print(f"  ...scanned {n_seen} chunks")

    rows = [{"domain": v[1], "case_type": v[2], "final_verdict": v[3]} for v in best_by_conv.values()]
    df = pd.DataFrame(rows)
    print(f"\nTotal final-chunk conversations in ICFD-31k train split: {len(df)}")

    print("\n=== Domain coverage (all conversations that fed our training pool) ===")
    print(df["domain"].value_counts())
    print(f"\nNumber of distinct domains: {df['domain'].nunique()}")

    print("\n=== Cross-tab: domain x final_verdict ===")
    print(pd.crosstab(df["domain"], df["final_verdict"]))

    out_path = os.path.join(DATA_DIR, "_icfd31k_train_domain_coverage.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
