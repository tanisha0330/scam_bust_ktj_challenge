"""
Builds the training dataset for the SMS/WhatsApp message classifier.

Sources (all local, already labeled):
  - dataset/public_sms.csv        (message_text, is_scam)
  - dataset/public_whatsapp.csv   (conversation_text, is_scam)

Output: ml/data/message_dataset.csv with columns [text, label, source, language]
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "..", "dataset")
OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def load_sms():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_sms.csv"))
    return pd.DataFrame({
        "text": df["message_text"],
        "label": df["is_scam"].astype(int),
        "source": "scambust_sms",
        "language": df.get("language", "unknown"),
    })


def load_whatsapp():
    df = pd.read_csv(os.path.join(DATASET_DIR, "public_whatsapp.csv"))
    return pd.DataFrame({
        "text": df["conversation_text"],
        "label": df["is_scam"].astype(int),
        "source": "scambust_whatsapp",
        "language": df.get("language", "unknown"),
    })


def main():
    parts = [load_sms(), load_whatsapp()]
    combined = pd.concat(parts, ignore_index=True)

    before = len(combined)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    after = len(combined)

    print(f"Loaded {before} rows -> {after} after dedup on exact text")
    print("Label balance:")
    print(combined["label"].value_counts())
    print("Source breakdown:")
    print(combined["source"].value_counts())

    out_path = os.path.join(OUT_DIR, "message_dataset.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
