"""
Tier-1 (local MuRIL) + Tier-2 (Groq LLM) cascade.

Tier-1 always runs first (fast, local, works without internet). If its
calibrated confidence is below CONFIDENCE_THRESHOLD, we escalate to the LLM
for a second opinion. There is no more "online/offline/auto" mode -- the
cascade decides automatically.

CONFIDENCE_THRESHOLD starts conservative (escalate often) because a missed
scam is far costlier than an extra LLM call. Retune once train_call_classifier
/ train_message_classifier report real precision/recall numbers.
"""
import json
import os
import re

import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Raised 0.85 -> 0.95 on measured evidence: on the Indian unseen-domain set,
# escalating below 0.85 sent 18% of traffic to Tier-2 for 0.934 recall, while
# 0.95 sends 34.5% for 0.951 recall. Since a missed scam costs a senior money
# and an extra LLM call costs ~2.5s, the trade favours escalating more.
CONFIDENCE_THRESHOLD = float(os.getenv("TIER1_CONFIDENCE_THRESHOLD", "0.95"))
MAX_LENGTH = 256

# Verdicts. Binary scam/safe forces a call on messages that are genuinely
# ambiguous -- a real chit-fund instalment reminder and a chit-fund scam are
# near-identical in text, and inspection showed BOTH our classifier and the
# LLM flag such messages, i.e. the label itself is arguable rather than the
# model being wrong. "suspicious" is the honest third answer for those.
VERDICT_SCAM = "scam"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_SAFE = "safe"

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))  # .../scambust/dataset/backend
_DATASET_DIR = os.path.dirname(_BACKEND_DIR)                # .../scambust/dataset
SCAMBUST_DIR = os.path.dirname(_DATASET_DIR)                 # .../scambust (ml/ lives here, not under dataset/)
# ONE model now serves both SMS/WhatsApp and calls. Previously there were two
# (call + message); the message model had only ~140 training rows and was
# measurably the weakest part of the system. Pooling every source into a single
# binary classifier gives short-message classification the benefit of the full
# call corpus. `channel` is still tracked, but only for logging and the
# keyword pre-filter -- it no longer selects a model.
_MODELS = os.path.join(SCAMBUST_DIR, "ml", "models_out")
# Tried in order; first one present wins. `balanced_classifier` is the current
# production model -- trained on the real/Indian-weighted 10k build. It is
# strictly better than `unified_classifier` on the metrics that matter:
# unseen-domain recall 0.648 vs 0.478, and -- decisively -- a 10.7% false
# positive rate on real legitimate marketing SMS versus 89.3%.
MODEL_SEARCH_PATH = [
    os.path.join(_MODELS, "balanced_classifier"),
    os.path.join(_MODELS, "unified_classifier"),
    os.path.join(_MODELS, "call_classifier"),   # oldest fallback
]

API_KEY = os.getenv("GROQ_API_KEY")
_llm_client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1") if API_KEY else None

_device = "cuda" if torch.cuda.is_available() else "cpu"


class _Tier1Model:
    """Lazily loads a fine-tuned MuRIL classifier + its calibration temperature."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.available = os.path.exists(os.path.join(model_dir, "config.json"))
        self.tokenizer = None
        self.model = None
        self.temperature = 1.0
        if self.available:
            self._load()

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(_device)
        self.model.eval()
        calib_path = os.path.join(self.model_dir, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                self.temperature = json.load(f).get("temperature", 1.0)

    @torch.no_grad()
    def predict(self, text: str):
        """Returns (is_scam: bool, confidence: float in [0.5, 1.0])."""
        if not self.available:
            return None, None
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt"
        ).to(_device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits / self.temperature, dim=-1)
        label = int(torch.argmax(probs).item())
        confidence = float(probs[label].item())
        return bool(label == 1), confidence


_unified_model = None


def _get_model():
    """Returns the single Tier-1 classifier used for every channel."""
    global _unified_model
    if _unified_model is None:
        chosen = next(
            (d for d in MODEL_SEARCH_PATH if os.path.exists(os.path.join(d, "config.json"))),
            None,
        )
        if chosen is None:
            print(f"No trained model found in any of: {MODEL_SEARCH_PATH}")
            chosen = MODEL_SEARCH_PATH[0]
        elif chosen != MODEL_SEARCH_PATH[0]:
            print(f"Preferred model missing; falling back to {os.path.basename(chosen)}")
        else:
            print(f"Loaded Tier-1 model: {os.path.basename(chosen)}")
        _unified_model = _Tier1Model(chosen)
    return _unified_model


def _extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def _call_llm(text: str, task_hint: str):
    """Calls Groq with structured JSON output. Returns (is_scam, reason) or (None, error_str)."""
    if _llm_client is None:
        return None, "LLM unavailable (no GROQ_API_KEY)"
    try:
        prompt = (
            f"Analyze this {task_hint}: '{text}'. "
            "Decide if it is a scam/fraud attempt targeting an Indian senior citizen. "
            'Respond ONLY with JSON: {"is_scam": true/false, "reason": "short Hinglish reason"}'
        )
        resp = _llm_client.chat.completions.create(
            # Groq retired the llama-3.* family; calling llama-3.3-70b-versatile
            # now 404s, which silently broke Tier-2 escalation. Override with
            # GROQ_MODEL in .env if this one is retired too.
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            # temperature 0: this is a classification decision, not generation.
            # At 0.1 the model flip-flopped between runs on genuinely borderline
            # financial messages, which made the "suspicious" verdict itself
            # non-deterministic. Sampling buys nothing here.
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = _extract_json(resp.choices[0].message.content)
        if result is None or "is_scam" not in result:
            return None, "LLM returned unparseable response"
        return bool(result["is_scam"]), result.get("reason", "")
    except Exception as e:
        return None, str(e)


def classify(text: str, channel: str) -> dict:
    """Runs the tier-1 -> tier-2 cascade. `channel` ('sms' or 'call') no longer
    selects a model -- one unified classifier handles both -- but is still used
    to phrase the Tier-2 prompt and to tag the ScanLog row.

    Returns a dict with tier1_label, tier1_confidence, escalated_to_llm,
    tier2_label, tier2_reason, final_label, verdict, reason -- callers
    (views.py) persist this into ScanLog.

    `final_label` stays a plain bool for backward compatibility; `verdict` adds
    the three-way scam/suspicious/safe answer described above.
    """
    tier1 = _get_model()
    tier1_label, tier1_confidence = tier1.predict(text)

    result = {
        "tier1_label": tier1_label,
        "tier1_confidence": tier1_confidence,
        "escalated_to_llm": False,
        "tier2_label": None,
        "tier2_reason": "",
        "final_label": tier1_label,
        "reason": f"Local MuRIL analysis (confidence: {tier1_confidence * 100:.1f}%)" if tier1_confidence else "",
    }

    # Escalate on low confidence OR on any positive scam call, even a confident
    # one. Two reasons, both measured:
    #   * Tier-1 is *confidently* wrong on ambiguous financial messages -- it
    #     rated a real chit-fund instalment reminder 0.987 scam, far above any
    #     workable threshold, so confidence alone will never catch these.
    #   * A "scam" verdict auto-sends an SMS to the user's family contact. That
    #     is the consequential branch and is worth a ~2.5s second opinion.
    # Scam predictions are the minority of traffic, so this bounds the extra cost.
    needs_escalation = (
        tier1_label is None
        or tier1_confidence < CONFIDENCE_THRESHOLD
        or tier1_label is True
    )
    if needs_escalation:
        task_hint = "phone call transcript" if channel == "call" else "SMS/WhatsApp message"
        tier2_label, tier2_reason = _call_llm(text, task_hint)
        result["escalated_to_llm"] = True
        result["tier2_label"] = tier2_label
        result["tier2_reason"] = tier2_reason
        if tier2_label is not None:
            result["final_label"] = tier2_label
            result["reason"] = tier2_reason
        elif tier1_label is None:
            # Tier-1 unavailable AND LLM failed -- nothing left to fall back to.
            result["final_label"] = None
            result["reason"] = f"Analysis unavailable: {tier2_reason}"
        # else: tier1 succeeded but LLM failed -- keep tier1's verdict as final_label (already set).

    result["verdict"] = _derive_verdict(result)
    return result


def _derive_verdict(result: dict) -> str | None:
    """Maps the cascade outcome onto scam / suspicious / safe.

    The strongest ambiguity signal available is two independent models
    disagreeing: when the local classifier and the LLM reach opposite verdicts
    on the same text, that text is genuinely borderline and the user is better
    served by "verify this independently" than by a confident yes/no.
    """
    if result["final_label"] is None:
        return None

    t1, t2 = result["tier1_label"], result["tier2_label"]
    if result["escalated_to_llm"] and t1 is not None and t2 is not None and t1 != t2:
        return VERDICT_SUSPICIOUS

    return VERDICT_SCAM if result["final_label"] else VERDICT_SAFE
