"""
Builds and MEASURES an out-of-distribution detector for the unified classifier.

THE PROBLEM IT TARGETS
On scam domains absent from training, the classifier is not merely wrong -- it
is *confidently* wrong. 93.7% of missed scams scored above the 0.85 escalation
threshold (mean confidence 0.983), so the LLM safety net almost never fired.
Raising the threshold does not fix this (0.95 recovers only 5.5% of them).

THE IDEA
Softmax confidence answers "how sure is the model?", which is unreliable under
distribution shift. An OOD score instead answers "is this input unlike anything
the model was trained on?" -- which is precisely the condition under which the
model should defer to Tier-2, regardless of how confident it feels.

THE EXPERIMENT
The honest test is not "does OOD detect unseen domains" (trivially yes) but
"does OOD predict TIER-1 BEING WRONG better than confidence does?" So we frame
it as a detection problem:
    target = (tier1 prediction != true label)
    score  = candidate escalation signal
and compare ROC-AUC for each signal. A signal only earns its place if it beats
the incumbent (1 - softmax confidence).

Two OOD scores are implemented and compared:
  * Mahalanobis (Lee et al. 2018): class-conditional Gaussians with shared
    covariance over [CLS] embeddings; score = min distance to any class mean.
  * kNN cosine: 1 - mean cosine similarity to the k nearest training embeddings.

Usage:
    python -m ml.training.build_ood_detector
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models_out", "unified_classifier")

MAX_LENGTH = 256
BATCH_SIZE = 32
REFERENCE_SAMPLE = 12000   # training rows used to characterise "in-distribution"
KNN_K = 10
CONFIDENCE_THRESHOLD = 0.85

_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed_and_predict(texts, model, tokenizer, temperature=1.0):
    """Returns ([CLS] embeddings, predicted labels, calibrated confidences)."""
    embs, preds, confs = [], [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [str(t) for t in texts[i:i + BATCH_SIZE]]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        out = model(**inputs, output_hidden_states=True)
        # [CLS] token of the final hidden layer -- the representation the
        # classification head actually consumes.
        cls = out.hidden_states[-1][:, 0, :].float().cpu().numpy()
        probs = torch.softmax(out.logits.float() / temperature, dim=-1).cpu().numpy()
        embs.append(cls)
        preds.append(probs.argmax(axis=-1))
        confs.append(probs.max(axis=-1))
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"    embedded {i}/{len(texts)}", flush=True)
    return np.vstack(embs), np.concatenate(preds), np.concatenate(confs)


def fit_mahalanobis(train_emb, train_labels):
    """Class-conditional means + shared covariance (Lee et al. 2018)."""
    means, centered = {}, []
    for c in np.unique(train_labels):
        cls_emb = train_emb[train_labels == c]
        mu = cls_emb.mean(axis=0)
        means[int(c)] = mu
        centered.append(cls_emb - mu)
    centered = np.vstack(centered)
    cov = np.cov(centered, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-3      # ridge for numerical stability
    precision = np.linalg.pinv(cov)
    return means, precision


def mahalanobis_score(emb, means, precision):
    """Minimum Mahalanobis distance to any class mean -- higher = more OOD."""
    dists = []
    for mu in means.values():
        delta = emb - mu
        dists.append(np.einsum("ij,jk,ik->i", delta, precision, delta))
    return np.sqrt(np.maximum(np.min(np.vstack(dists), axis=0), 0))


def knn_score(emb, train_emb_norm, k=KNN_K, chunk=256):
    """1 - mean cosine similarity to the k nearest training embeddings."""
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    out = []
    for i in range(0, len(emb_norm), chunk):
        sims = emb_norm[i:i + chunk] @ train_emb_norm.T
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        out.append(1.0 - topk.mean(axis=1))
    return np.concatenate(out)


def report_signal(name, score, is_wrong):
    """How well does this signal predict 'Tier-1 got it wrong'?"""
    if len(np.unique(is_wrong)) < 2:
        return None
    auc = roc_auc_score(is_wrong, score)
    print(f"  {name:28s} AUC(predicts tier1-wrong) = {auc:.3f}")
    return auc


def escalation_table(name, score, is_wrong, quantiles=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """At each threshold: how much traffic escalates, and how many of Tier-1's
    mistakes get caught by that escalation."""
    print(f"\n  {name} operating points:")
    print(f"  {'threshold':>10} {'escalated%':>11} {'mistakes caught%':>18} {'wasted escalations%':>21}")
    rows = []
    for q in quantiles:
        thr = np.quantile(score, q)
        esc = score >= thr
        caught = (esc & is_wrong).sum() / max(is_wrong.sum(), 1)
        wasted = (esc & ~is_wrong).sum() / max((~is_wrong).sum(), 1)
        print(f"  {q:>10.2f} {100 * esc.mean():>10.1f}% {100 * caught:>17.1f}% {100 * wasted:>20.1f}%")
        rows.append({"quantile": q, "escalated": float(esc.mean()),
                     "mistakes_caught": float(caught), "wasted": float(wasted)})
    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    model.eval()
    temperature = json.load(open(os.path.join(MODEL_DIR, "calibration.json"))).get("temperature", 1.0)
    print(f"Model loaded (temperature={temperature:.4f}, device={_device})")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "unified_train.csv"))
    if len(train_df) > REFERENCE_SAMPLE:
        train_df = train_df.sample(REFERENCE_SAMPLE, random_state=42)
    print(f"\nEmbedding {len(train_df)} reference (training) rows ...")
    train_emb, _, _ = embed_and_predict(train_df["text"].tolist(), model, tokenizer, temperature)
    train_labels = train_df["label"].values

    print("Fitting Mahalanobis (class means + shared covariance) ...")
    means, precision = fit_mahalanobis(train_emb, train_labels)
    train_emb_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9)

    results = {"temperature": temperature, "reference_rows": len(train_df)}

    for fname, label in [
        ("unified_test.csv", "IN-DISTRIBUTION TEST"),
        ("_icfd31k_raw_cross_domain.csv", "ICFD-31k CROSS-DOMAIN (5 unseen domains)"),
        ("unified_eval_generated_holdout.csv", "GENERATED UNSEEN DOMAINS"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"\n{'=' * 70}\n{label}  ({len(df)} rows)\n{'=' * 70}")
        emb, preds, confs = embed_and_predict(df["text"].tolist(), model, tokenizer, temperature)
        y = df["label"].values
        is_wrong = preds != y

        maha = mahalanobis_score(emb, means, precision)
        knn = knn_score(emb, train_emb_norm)

        print(f"  Tier-1 accuracy: {(~is_wrong).mean():.3f}  ({is_wrong.sum()} mistakes)")
        cur_esc = confs < CONFIDENCE_THRESHOLD
        print(f"  Current policy (confidence < {CONFIDENCE_THRESHOLD}): "
              f"{100 * cur_esc.mean():.1f}% escalated, "
              f"{100 * (cur_esc & is_wrong).sum() / max(is_wrong.sum(), 1):.1f}% of mistakes caught")

        entry = {
            "n": len(df),
            "tier1_accuracy": float((~is_wrong).mean()),
            "mistakes": int(is_wrong.sum()),
            "current_policy_escalated": float(cur_esc.mean()),
            "current_policy_mistakes_caught": float((cur_esc & is_wrong).sum() / max(is_wrong.sum(), 1)),
            "auc": {},
        }
        print("\n  Which signal best predicts that Tier-1 is wrong?")
        entry["auc"]["one_minus_confidence"] = report_signal("1 - softmax confidence", 1 - confs, is_wrong)
        entry["auc"]["mahalanobis"] = report_signal("Mahalanobis OOD", maha, is_wrong)
        entry["auc"]["knn_cosine"] = report_signal("kNN-cosine OOD", knn, is_wrong)

        entry["mahalanobis_operating_points"] = escalation_table("Mahalanobis", maha, is_wrong)
        results[fname] = entry

    out = os.path.join(HERE, "..", "models_out", "ood_evaluation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Persist the detector itself so the backend can use it.
    np.savez(
        os.path.join(MODEL_DIR, "ood_detector.npz"),
        class_means=np.stack(list(means.values())),
        precision=precision,
        train_emb_norm=train_emb_norm.astype(np.float32),
    )
    print(f"\nSaved evaluation -> {out}")
    print(f"Saved detector    -> {os.path.join(MODEL_DIR, 'ood_detector.npz')}")


if __name__ == "__main__":
    main()
