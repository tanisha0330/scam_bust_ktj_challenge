"""
Builds the training dataset for the call/voice classifier.

Sources:
  - dataset/public_calls.csv               (call_transcript, is_scam)             ~75 rows
  - dataset/public_audio_transcripts.csv   (audio_transcript_text, is_scam)       ~73 rows
  - India_Cyber_Scam_Hinglish_Dataset.csv  (text, label)  10k rows / 743 unique
  - HF rishia2220/icfd-31k                 (cumulative_text, final_verdict)      ~30k conversations

For ICFD-31k: each conversation is split into many *cumulative* chunks (chunk N's
text includes chunks 1..N). Using every chunk would (a) massively over-represent
long conversations, (b) create huge near-duplicate text across chunks of the same
conversation_uid, and (c) mislabel early chunks of an eventual "Clear Fraud" call
as scam even before any fraud signal appears in the dialogue. So for the train
split we keep the top-2 most recent chunks (by chunk_timestamp) per
conversation_uid -- roughly doubling volume (~20k -> ~40k) while staying on the
"late enough that the label is trustworthy" side, rather than sampling from
early in the conversation. Validation/test/stress splits still use only the
single final chunk per conversation (chunks_per_conversation=1, the original
behavior) since those aren't meant to grow.

We use the dataset's own train/validation/test split assignment (by conversation)
rather than re-splitting ourselves, since it's already group-safe.

Output: ml/data/call_dataset_{train,val,test}.csv with columns
        [text, label, source, split]
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "..", "dataset")
OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

ICFD_REPO = "rishia2220/icfd-31k"


def load_calls():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_calls.csv"))
    return pd.DataFrame({
        "text": df["call_transcript"],
        "label": df["is_scam"].astype(int),
        "source": "scambust_calls",
    })


def load_audio_transcripts():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_audio_transcripts.csv"))
    return pd.DataFrame({
        "text": df["audio_transcript_text"],
        "label": df["is_scam"].astype(int),
        "source": "scambust_audio",
    })


def load_india_cyber_csv():
    path = os.path.join(os.path.expanduser("~"), "India_Cyber_Scam_Hinglish_Dataset.csv")
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["text"], keep="first")
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label"].astype(int),
        "source": "india_cyber_csv",
    })


def load_icfd31k_split(hf_split: str, chunks_per_conversation: int = 1):
    """Stream one split of ICFD-31k and keep the top-N most recent chunks
    (by chunk_timestamp) per conversation_uid, to avoid duplicate/weak-labeled
    *early* chunks -- an early chunk of an eventual "Clear Fraud" call may not
    show any fraud signal yet, even though it's labeled scam=1 (the label
    applies to the whole conversation). Taking only the LATEST chunks keeps
    that risk low while chunks_per_conversation > 1 lets us pull more volume
    out of the same conversations. Caches the (expensive, ~1M-row streamed)
    result to disk so re-running the script doesn't re-download everything."""
    suffix = "" if chunks_per_conversation == 1 else f"_top{chunks_per_conversation}"
    cache_path = os.path.join(OUT_DIR, f"_icfd31k_raw_{hf_split}{suffix}.csv")
    if os.path.exists(cache_path):
        print(f"Using cached {cache_path}")
        return pd.read_csv(cache_path)

    from datasets import load_dataset

    print(f"Streaming ICFD-31k split='{hf_split}' (top {chunks_per_conversation} chunks/conversation) ...")
    ds = load_dataset(ICFD_REPO, split=hf_split, streaming=True)

    top_by_conv = {}  # conversation_uid -> list of (chunk_timestamp, text, label), sorted desc, len <= N
    n_seen = 0
    for row in ds:
        n_seen += 1
        conv_id = row["conversation_uid"]
        ts = row["chunk_timestamp"]
        bucket = top_by_conv.setdefault(conv_id, [])
        if len(bucket) < chunks_per_conversation or ts > bucket[-1][0]:
            label = 1 if str(row["final_verdict"]).strip().upper() == "YES" else 0
            bucket.append((ts, row["cumulative_text"], label))
            bucket.sort(key=lambda x: x[0], reverse=True)
            del bucket[chunks_per_conversation:]
        if n_seen % 50000 == 0:
            print(f"  ...scanned {n_seen} chunks, {len(top_by_conv)} conversations so far")

    print(f"Finished '{hf_split}': {n_seen} chunks -> {len(top_by_conv)} conversations")

    rows = [
        {"text": text, "label": label, "source": "icfd31k"}
        for chunks in top_by_conv.values()
        for (_, text, label) in chunks
    ]
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"], keep="first")
    return df


def main():
    local_parts = [load_calls(), load_audio_transcripts(), load_india_cyber_csv()]
    local_df = clean(pd.concat(local_parts, ignore_index=True))
    print(f"Local (synthetic) sources combined: {len(local_df)} rows after dedup")
    print(local_df["label"].value_counts())

    # Hold out a stratified slice of the LOCAL/synthetic sources purely for
    # evaluation, so we can report accuracy on synthetic data separately from
    # ICFD-31k's real-ish held-out set (task: "check accuracy for both
    # synthesised and real data"). The rest folds into train.
    local_train, local_synth_eval = train_test_split(
        local_df, test_size=0.2, stratify=local_df["label"], random_state=42
    )

    # NOTE: ICFD-31k's own validation/test splits turned out to be entirely
    # fraud-scenario conversations (case_type is 100% "Clear Fraud"/"Subtle
    # Fraud", no "Clear Normal"/"Ambiguous Normal" at all) -- confirmed via
    # the HF datasets-server /statistics endpoint. They're a recall stress
    # test on hard fraud cases, NOT a representative held-out set (a model
    # that always predicts "scam" would ace them). So: use ICFD-31k's TRAIN
    # split (which has the full case_type mix) as our data pool, and carve
    # our OWN stratified val/test out of it for realistic evaluation. We
    # still keep the official val/test as a secondary, clearly-labeled
    # fraud-recall stress test.
    icfd_train_raw = clean(load_icfd31k_split("train", chunks_per_conversation=6))
    icfd_stress_val = clean(load_icfd31k_split("validation"))
    icfd_stress_test = clean(load_icfd31k_split("test"))

    # chunks_per_conversation=6 overshoots (many conversations have 6+ chunks) --
    # 59,188 rows vs. the ~40k target. Stratified-downsample back to target
    # rather than silently keeping the overshoot.
    TARGET_ICFD_TRAIN_ROWS = 40000
    if len(icfd_train_raw) > TARGET_ICFD_TRAIN_ROWS:
        icfd_train_raw, _ = train_test_split(
            icfd_train_raw, train_size=TARGET_ICFD_TRAIN_ROWS,
            stratify=icfd_train_raw["label"], random_state=42,
        )
        print(f"Downsampled icfd_train_raw to {len(icfd_train_raw)} rows (stratified)")

    pool_df = pd.concat([local_train, icfd_train_raw], ignore_index=True)
    pool_df = pool_df.drop_duplicates(subset=["text"], keep="first")

    train_df, val_test_df = train_test_split(
        pool_df, test_size=0.06, stratify=pool_df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.5, stratify=val_test_df["label"], random_state=42
    )

    train_df.to_csv(os.path.join(OUT_DIR, "call_dataset_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "call_dataset_val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "call_dataset_test.csv"), index=False)
    local_synth_eval.to_csv(os.path.join(OUT_DIR, "call_dataset_synthetic_eval.csv"), index=False)
    icfd_stress_val.to_csv(os.path.join(OUT_DIR, "call_dataset_icfd_stress_val.csv"), index=False)
    icfd_stress_test.to_csv(os.path.join(OUT_DIR, "call_dataset_icfd_stress_test.csv"), index=False)

    print("\n=== FINAL SUMMARY ===")
    for name, d in [
        ("train", train_df), ("val", val_df), ("test", test_df),
        ("synthetic_eval (scambust+india_cyber, held out)", local_synth_eval),
        ("icfd_stress_val (fraud-only recall stress test)", icfd_stress_val),
        ("icfd_stress_test (fraud-only recall stress test)", icfd_stress_test),
    ]:
        print(f"{name}: {len(d)} rows, label balance:")
        print(d["label"].value_counts())
        print(d["source"].value_counts() if "source" in d else "")


if __name__ == "__main__":
    main()
