"""
Builds ONE unified binary (scam / not-scam) dataset for a single classifier
covering SMS, WhatsApp and calls.

WHY UNIFIED: the previous split into two models left the message classifier
with only 140 training rows -- by far the weakest component, and the one that
produced a real false positive on an ordinary family message. SMS, WhatsApp
and call text are close enough in kind that pooling them turns 140 rows into
tens of thousands, and lets short-message classification benefit from every
call transcript we have.

SOURCES
  Synthetic scam-heavy (pre-existing):
    - scambust public_sms / public_whatsapp / public_calls / public_audio_transcripts
    - India_Cyber_Scam_Hinglish_Dataset.csv  (deduped 10k -> 743 unique templates)
    - ICFD-31k train split
  Real human text (new -- fixes thin "normal life" coverage):
    - legit_sources.csv  (UCI SMS ham+spam, CMU Hinglish chat, legit bank calls)
  Generated novel domains (new -- fixes narrow scam-type coverage):
    - generated_novel_domains.csv  (Ollama qwen2.5:7b, scam AND legit per domain)

TWO DELIBERATE CHOICES:

1. ICFD-31k reverts to ONE chunk per conversation (~21k rows), not the 6-chunk
   expansion. Measured: the 20k/1-chunk model scored 0.473 accuracy / 0.459
   recall on the unseen-domain test, while the 38k/6-chunk model dropped to
   0.431 / 0.412. More chunks of the same 10 domains actively hurt
   generalisation. Capacity is better spent on new, diverse sources.

2. Generated domains marked split_role="holdout" are excluded from training
   entirely and written to their own eval file, so we keep an honest
   generalisation test even after expanding domain coverage.

Outputs (ml/data/):
    unified_{train,val,test}.csv       -- [text, label, source]
    unified_eval_generated_holdout.csv -- unseen generated domains
    (ICFD-31k cross_domain + synthetic_eval + stress sets stay as-is)
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .prepare_call_data import (
    OUT_DIR,
    clean,
    load_audio_transcripts,
    load_calls,
    load_icfd31k_split,
    load_india_cyber_csv,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "..", "dataset")


def load_sms():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_sms.csv"))
    return pd.DataFrame({
        "text": df["message_text"], "label": df["is_scam"].astype(int), "source": "scambust_sms",
    })


def load_whatsapp():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_whatsapp.csv"))
    return pd.DataFrame({
        "text": df["conversation_text"], "label": df["is_scam"].astype(int), "source": "scambust_whatsapp",
    })


def load_optional_csv(filename, required_cols=("text", "label", "source")):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  ! {filename} not found -- skipping")
        return None
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ! {filename} missing columns {missing} -- skipping")
        return None
    return df


def main():
    # ---- scam-heavy synthetic sources (pre-existing) ----
    local_parts = [load_sms(), load_whatsapp(), load_calls(), load_audio_transcripts(), load_india_cyber_csv()]
    local_df = clean(pd.concat(local_parts, ignore_index=True))
    print(f"Synthetic local sources: {len(local_df)} rows")

    # Keep the same held-out synthetic slice as before, for continuity of the
    # "synthetic vs real" comparison across experiments.
    local_train, local_synth_eval = train_test_split(
        local_df, test_size=0.2, stratify=local_df["label"], random_state=42
    )

    # ---- ICFD-31k, one chunk per conversation (see note 1 above) ----
    icfd_train_raw = clean(load_icfd31k_split("train", chunks_per_conversation=1))
    print(f"ICFD-31k train (1 chunk/conversation): {len(icfd_train_raw)} rows")

    pool_parts = [local_train, icfd_train_raw]

    # ---- real human legitimate/spam text ----
    legit = load_optional_csv("legit_sources.csv")
    if legit is not None:
        legit = clean(legit[["text", "label", "source"]])
        print(f"Real human sources: {len(legit)} rows")
        print(legit.groupby(["source", "label"]).size())
        pool_parts.append(legit)

    # ---- generated novel domains ----
    gen_holdout = None
    gen = load_optional_csv("generated_novel_domains.csv", required_cols=("text", "label", "source", "split_role"))
    if gen is not None:
        gen = gen.dropna(subset=["text"])
        gen_train = clean(gen[gen["split_role"] == "train"][["text", "label", "source"]])
        gen_holdout = clean(gen[gen["split_role"] == "holdout"][["text", "label", "source"]])
        print(f"Generated novel domains: {len(gen_train)} train / {len(gen_holdout)} holdout")
        if len(gen_train):
            pool_parts.append(gen_train)

    pool_df = clean(pd.concat(pool_parts, ignore_index=True))
    print(f"\nUnified pool after dedup: {len(pool_df)} rows")

    train_df, val_test_df = train_test_split(
        pool_df, test_size=0.06, stratify=pool_df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.5, stratify=val_test_df["label"], random_state=42
    )

    train_df.to_csv(os.path.join(OUT_DIR, "unified_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "unified_val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "unified_test.csv"), index=False)
    local_synth_eval.to_csv(os.path.join(OUT_DIR, "unified_eval_synthetic.csv"), index=False)
    if gen_holdout is not None and len(gen_holdout):
        gen_holdout.to_csv(os.path.join(OUT_DIR, "unified_eval_generated_holdout.csv"), index=False)

    print("\n=== UNIFIED DATASET SUMMARY ===")
    for name, d in [
        ("train", train_df), ("val", val_df), ("test", test_df),
        ("eval_synthetic (held out)", local_synth_eval),
        ("eval_generated_holdout (unseen domains)", gen_holdout),
    ]:
        if d is None or not len(d):
            continue
        print(f"\n{name}: {len(d)} rows")
        print("  label:", dict(d["label"].value_counts()))
        if "source" in d:
            print("  sources:", dict(d["source"].value_counts()))


if __name__ == "__main__":
    main()
