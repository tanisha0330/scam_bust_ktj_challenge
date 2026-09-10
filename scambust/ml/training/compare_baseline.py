"""
Compares the OLD TF-IDF + LogisticRegression/RandomForest ensemble against
the NEW fine-tuned MuRIL call classifier, on the same held-out sets:
  - call_dataset_test.csv            (ICFD-31k real-ish held-out)
  - call_dataset_synthetic_eval.csv  (held-out scambust + India_Cyber CSV slice)

The old model was trained on a totally different (much smaller, ~296-row)
dataset and vocabulary, so this isn't a perfectly controlled A/B -- it answers
the practical question "would swapping in the new model actually help,"
not "how much did the architecture change help in isolation."

Usage:
    python -m ml.training.compare_baseline
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OLD_MODEL_DIR = os.path.join(HERE, "..", "..", "dataset")
NEW_MODEL_DIR = os.path.join(HERE, "..", "models_out", "call_classifier")


def get_text_metrics(text):
    blob = TextBlob(str(text))
    words = str(text).split()
    return [
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        len(words),
        np.mean([len(w) for w in words]) if len(words) > 0 else 0,
    ]


def load_old_models():
    with open(os.path.join(OLD_MODEL_DIR, "scam_detection_lr.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(OLD_MODEL_DIR, "scam_detection_rf.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(OLD_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    return lr, rf, tfidf


def predict_old(texts, lr, rf, tfidf):
    preds = []
    for text in texts:
        metrics = get_text_metrics(text)
        vec = tfidf.transform([text])
        features = hstack([vec, [metrics]])
        lr_prob = lr.predict_proba(features)[0][1]
        rf_prob = rf.predict_proba(features)[0][1]
        preds.append(1 if (lr_prob + rf_prob) / 2 > 0.45 else 0)
    return np.array(preds)


def predict_new(texts, model, tokenizer, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits[0]
            preds.append(int(torch.argmax(logits).item()))
    return np.array(preds)


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"  {name}: accuracy={acc:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    lr, rf, tfidf = load_old_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_DIR)
    new_model = AutoModelForSequenceClassification.from_pretrained(NEW_MODEL_DIR).to(device)

    all_results = {}
    for split_name, fname in [
        ("ICFD-31k real-ish test", "call_dataset_test.csv"),
        ("Synthetic held-out (scambust + India_Cyber)", "call_dataset_synthetic_eval.csv"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"\n=== {split_name} ({len(df)} rows) ===")

        old_preds = predict_old(df["text"].tolist(), lr, rf, tfidf)
        new_preds = predict_new(df["text"].tolist(), new_model, tokenizer, device)

        old_report = report("OLD (TF-IDF + LR/RF)", df["label"], old_preds)
        new_report = report("NEW (MuRIL fine-tuned)", df["label"], new_preds)
        all_results[split_name] = {"old": old_report, "new": new_report}

    out_path = os.path.join(HERE, "..", "models_out", "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comparison -> {out_path}")


if __name__ == "__main__":
    main()
