"""
Answers three open questions on the Indian unseen-domain set:

  1. THRESHOLD. Every metric so far used argmax (implicitly p=0.5). For a
     scam-warning app the costs are asymmetric -- a missed scam costs a senior
     money, a false alarm costs a moment of doubt -- so 0.5 is almost certainly
     the wrong operating point. Sweep it and show the trade-off.

  2. TIER-2 ALONE. We have never measured the LLM by itself. Team04's paper
     found GPT scoring 0.98 where their fine-tuned BERT got 0.91. If our Tier-2
     is similarly strong, "use the LLM more" beats "improve the classifier".

  3. THE ACTUAL CASCADE. What the user experiences is not Tier-1 or Tier-2 in
     isolation but the combination. Simulate the end-to-end system across
     escalation thresholds and report what really lands on screen, plus the
     share of traffic that leaves the device (a privacy and cost figure).

Nothing here retrains or overwrites a model.

Usage:
    python -m ml.training.eval_cascade_and_tier2
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models_out", "unified_classifier")
MAX_LENGTH, BATCH_SIZE = 256, 32
_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def tier1_scam_probs(texts, model, tokenizer, temperature):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [str(t) for t in texts[i:i + BATCH_SIZE]]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(_device)
        probs = torch.softmax(model(**inputs).logits.float() / temperature, dim=-1)
        out.append(probs[:, 1].cpu().numpy())      # P(scam)
    return np.concatenate(out)


def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return float((y == pred).mean()), p, r, f


def run_tier2(texts, client, model_name):
    """Ask the LLM for a verdict on every row. Returns array of 1/0/-1(failed)."""
    verdicts = []
    for i, t in enumerate(texts):
        prompt = (
            f"Analyze this message or call transcript: '{str(t)[:2000]}'. "
            "Decide if it is a scam/fraud attempt targeting an Indian senior citizen. "
            'Respond ONLY with JSON: {"is_scam": true/false}'
        )
        v = -1
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, response_format={"type": "json_object"},
                )
                v = 1 if json.loads(resp.choices[0].message.content).get("is_scam") else 0
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        verdicts.append(v)
        if (i + 1) % 25 == 0:
            print(f"    tier2 {i + 1}/{len(texts)}", flush=True)
    return np.array(verdicts)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "indian_crossdomain_eval.csv")).dropna(subset=["text", "label"])
    y = df["label"].values
    texts = df["text"].tolist()
    print(f"Indian unseen-domain set: {len(df)} rows ({int((y==1).sum())} scam / {int((y==0).sum())} legit)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    model.eval()
    temperature = json.load(open(os.path.join(MODEL_DIR, "calibration.json"))).get("temperature", 1.0)
    p_scam = tier1_scam_probs(texts, model, tokenizer, temperature)

    # ---- 1. Threshold sweep -------------------------------------------------
    print("\n=== 1. TIER-1 THRESHOLD SWEEP (currently argmax == 0.50) ===")
    print(f"{'thresh':>8}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}{'missed scams':>14}")
    sweep = []
    for thr in [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        pred = (p_scam >= thr).astype(int)
        a, p, r, f = prf(y, pred)
        missed = int(((y == 1) & (pred == 0)).sum())
        star = "  <- current" if abs(thr - 0.5) < 1e-9 else ""
        print(f"{thr:>8.2f}{a:>10.3f}{p:>11.3f}{r:>9.3f}{f:>8.3f}{missed:>14}{star}")
        sweep.append({"threshold": thr, "accuracy": a, "precision": p, "recall": r, "f1": f, "missed": missed})

    results = {"n": len(df), "tier1_threshold_sweep": sweep}

    # ---- 2. Tier-2 alone ----------------------------------------------------
    import dotenv
    from openai import OpenAI
    dotenv.load_dotenv(os.path.join(HERE, "..", "..", "dataset", ".env"))
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("\nNo GROQ_API_KEY -- skipping Tier-2 sections.")
    else:
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        print(f"\n=== 2. TIER-2 ALONE ({model_name}) ===")
        t0 = time.time()
        t2 = run_tier2(texts, client, model_name)
        elapsed = time.time() - t0
        ok = t2 >= 0
        print(f"  completed {ok.sum()}/{len(t2)} calls in {elapsed:.0f}s ({elapsed/max(len(t2),1):.2f}s/row)")
        if ok.sum():
            a, p, r, f = prf(y[ok], t2[ok])
            print(f"  accuracy={a:.3f} precision={p:.3f} recall={r:.3f} f1={f:.3f}")
            results["tier2_alone"] = {"accuracy": a, "precision": p, "recall": r, "f1": f,
                                      "completed": int(ok.sum()), "sec_per_row": elapsed / max(len(t2), 1)}

        # ---- 3. Full cascade simulation ------------------------------------
        print("\n=== 3. FULL CASCADE (Tier-1, escalate when unsure, Tier-2 decides) ===")
        print(f"{'escalate if conf <':>20}{'escalated%':>12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}")
        casc = []
        conf = np.maximum(p_scam, 1 - p_scam)          # calibrated confidence
        for thr in [0.0, 0.60, 0.70, 0.85, 0.95, 1.01]:
            esc = conf < thr
            final = (p_scam >= 0.5).astype(int)
            use_t2 = esc & ok
            final[use_t2] = t2[use_t2]
            a, p, r, f = prf(y, final)
            label = "never (Tier-1 only)" if thr == 0.0 else ("always (Tier-2 only)" if thr > 1 else f"{thr:.2f}")
            print(f"{label:>20}{100*esc.mean():>11.1f}%{a:>10.3f}{p:>11.3f}{r:>9.3f}{f:>8.3f}")
            casc.append({"threshold": thr, "escalated": float(esc.mean()),
                         "accuracy": a, "precision": p, "recall": r, "f1": f})
        results["cascade"] = casc

    out = os.path.join(HERE, "..", "models_out", "cascade_tier2_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
