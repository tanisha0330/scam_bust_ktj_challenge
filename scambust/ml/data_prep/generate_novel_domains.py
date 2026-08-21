"""
Generates training data for scam categories the model has NEVER seen, using a
local Ollama model (qwen2.5:7b).

WHY: the measured failure is that the classifier only recognises the 10 scam
domains present in ICFD-31k (~43% recall on 5 unseen domains). All 10 known
domains are already saturated at ~2,100 conversations each, and adding more
volume within them made cross-domain generalisation *worse* (v1 20k rows beat
v2 38k rows). The bottleneck is domain coverage, not volume.

TWO DESIGN RULES, both learned from measured failures in this project:

1. Generate LEGITIMATE conversations in the same novel domains too.
   If every new domain contained only scams, the model would simply learn
   "unfamiliar topic => scam", which is a worse shortcut than the one we are
   trying to fix.

2. Maximise diversity, not volume.
   The India_Cyber CSV is 10,000 rows built from only 743 unique templates
   (some repeated 60x). That inflated its metrics and taught the model
   surface patterns. Here we vary channel, language mix, victim reaction,
   scam stage, speaker names, amounts and framing on every single sample,
   sample at high temperature, and dedupe hard afterwards. Target volume is
   deliberately modest (~2.5k) because coverage is what we're buying.

Output is appended incrementally to ml/data/generated_novel_domains.csv so a
long run can be interrupted and resumed without losing work.

NOTE ON PURPOSE: this produces labelled *training data for a defensive
classifier* that warns senior citizens about fraud. It is not a source of
deployable scam scripts.

Usage:
    python -m ml.data_prep.generate_novel_domains            # default target
    python -m ml.data_prep.generate_novel_domains 500        # custom target
"""
import json
import os
import random
import sys

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "generated_novel_domains.csv")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
GROQ_MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 6           # conversations requested per LLM call
DEFAULT_TARGET = 2500

# Backend strategy: Groq first (measured ~2s/call vs ~60s/call for local
# qwen2.5:7b on a 6GB card -- a ~30x speedup, and noticeably better Hinglish),
# falling back to local Ollama whenever Groq errors or rate-limits. That keeps
# generation running unattended without burning the whole Groq daily quota.
# NOTE: Groq retired the llama-3.* models; gpt-oss-120b is the current pick.

# Domains deliberately chosen to sit OUTSIDE ICFD-31k's 10 categories
# (bank_payment, ecommerce, insurance, personal_emergency, loan_shark,
#  tech_support, job_employment, travel_lottery, govt_impersonation, utility_bill).
# The last three are held out of training entirely, as an honest test of
# generalisation to domains unseen even after this expansion.
TRAIN_DOMAINS = {
    "matrimonial_romance": "a matrimonial/marriage website or dating contact who builds trust then asks for money",
    "crypto_investment": "a fake cryptocurrency or share-trading app promising guaranteed high returns",
    "sim_swap_telecom": "someone claiming to upgrade a SIM to 5G or port a number, seeking OTP/SIM details",
    "traffic_echallan": "a fake traffic police e-challan / vehicle fine notice demanding online payment",
    "gas_subsidy_lpg": "a fake LPG gas connection subsidy or cylinder KYC update",
}
HOLDOUT_DOMAINS = {
    "army_officer_marketplace": "someone posing as an army/CRPF officer buying second-hand goods online, using a fake payment request",
    "fake_scholarship_edu": "a fake student scholarship or exam-result fee demand",
    "rental_deposit_fraud": "a fake landlord or property agent demanding an advance rental deposit sight-unseen",
}
ALL_DOMAINS = {**TRAIN_DOMAINS, **HOLDOUT_DOMAINS}

CHANNELS = ["sms", "whatsapp", "call"]
LANGUAGES = ["Hinglish (romanised Hindi mixed with English, Latin script only)", "Indian English"]
VICTIM_REACTIONS = ["compliant and trusting", "suspicious and pushing back", "confused and asking to repeat", "busy and distracted"]
SCAM_STAGES = ["initial lure/hook", "building pressure and urgency", "the actual payment or OTP request"]
LEGIT_FLAVOURS = [
    "a genuine routine interaction that happens to involve money or account details",
    "an ordinary friendly conversation with no money involved at all",
    "a genuine service update or reminder from a real company",
    "a family member discussing everyday household matters",
]


def build_prompt(domain, domain_desc, is_scam, channel, language, n):
    shape = {
        "sms": "a single short SMS message (1-3 sentences, no speaker labels)",
        "whatsapp": "a short WhatsApp exchange of 3-6 messages. Put EACH message on its own new line, formatted exactly as 'Name: message text'",
        "call": "a phone call transcript of 4-10 turns. Put EACH turn on its own new line, formatted exactly as 'Speaker: spoken text'",
    }[channel]

    if is_scam:
        situation = (
            f"FRAUDULENT. The scenario is {domain_desc}. "
            f"Show the {random.choice(SCAM_STAGES)}. "
            f"The target is an Indian senior citizen who is {random.choice(VICTIM_REACTIONS)}."
        )
    else:
        situation = (
            f"COMPLETELY LEGITIMATE - no fraud whatsoever. It must be set in the same general "
            f"subject area ({domain.replace('_', ' ')}), but be {random.choice(LEGIT_FLAVOURS)}. "
            f"It should be the kind of message a real person or real company genuinely sends."
        )

    return f"""You are helping build a training dataset for a fraud-detection system that protects Indian senior citizens from phone and message scams. Realistic examples are needed so the detector can learn to recognise them.

Generate {n} DIFFERENT examples. Each must be {shape}.

Content: {situation}

Language: {language}. Use Indian names and places. Write money as "Rs 5000" (never use a currency symbol). Use only plain ASCII punctuation - straight quotes, no fancy dashes or ellipses. Write fluent, natural, grammatical sentences that a real person would actually send.

Vary the names, amounts, phrasing, sentence length and tone a lot between examples - they must not read like the same template refilled.

Respond ONLY with JSON in exactly this form:
{{"examples": ["first example text", "second example text", ...]}}"""


# Smart quotes / dashes that Ollama emits get mangled into U+FFFD if decoded
# with the wrong codec. We decode explicitly as UTF-8 (below) and then fold the
# remaining fancy punctuation down to ASCII, so that no purely-typographic
# artefact can become a spurious signal the classifier latches onto.
_PUNCT_FIXES = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "₹": "Rs ",
    "�": "'",
}


def _clean(text: str) -> str:
    for bad, good in _PUNCT_FIXES.items():
        text = text.replace(bad, good)
    return " ".join(text.split()) if "\n" not in text else "\n".join(
        " ".join(line.split()) for line in text.split("\n") if line.strip()
    )


def _extract_examples(parsed):
    examples = parsed.get("examples", [])
    if isinstance(examples, str):
        examples = [examples]
    return [_clean(str(e)).strip() for e in examples if str(e).strip()]


def call_ollama(prompt, timeout=180):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 1.0, "top_p": 0.95, "num_predict": 1600},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    # Decode explicitly -- requests' charset guess mangles UTF-8 here.
    return _extract_examples(json.loads(json.loads(resp.content.decode("utf-8")).get("response", "{}")))


def _groq_client():
    try:
        import dotenv
        from openai import OpenAI

        dotenv.load_dotenv(os.path.join(HERE, "..", "..", "dataset", ".env"))
        key = os.getenv("GROQ_API_KEY")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") if key else None
    except Exception:
        return None


def call_groq(client, prompt):
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        response_format={"type": "json_object"},
    )
    return _extract_examples(json.loads(resp.choices[0].message.content))


def load_existing():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        print(f"Resuming: {len(df)} examples already generated")
        return df
    return pd.DataFrame(columns=["text", "label", "source", "domain", "channel", "split_role"])


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    df = load_existing()
    seen = set(df["text"].astype(str)) if len(df) else set()

    domain_items = list(ALL_DOMAINS.items())
    fails = 0
    groq = _groq_client()
    print(f"Backends: groq={'yes' if groq else 'no'} ({GROQ_MODEL}), ollama fallback ({OLLAMA_MODEL})")

    while len(df) < target:
        domain, desc = random.choice(domain_items)
        is_scam = random.random() < 0.5
        channel = random.choice(CHANNELS)
        language = random.choice(LANGUAGES)

        prompt = build_prompt(domain, desc, is_scam, channel, language, BATCH_SIZE)
        examples, backend = None, None
        if groq is not None:
            try:
                examples, backend = call_groq(groq, prompt), "groq"
            except Exception as e:
                print(f"  ~ groq unavailable ({e.__class__.__name__}), falling back to ollama")
        if examples is None:
            try:
                examples, backend = call_ollama(prompt), "ollama"
            except Exception as e:
                fails += 1
                print(f"  ! generation failed ({e.__class__.__name__}: {e}); {fails} consecutive")
                if fails >= 5:
                    print("Too many consecutive failures on both backends. Stopping.")
                    break
                continue
        fails = 0

        new_rows = []
        for text in examples:
            if len(text) < 25 or text in seen:
                continue
            seen.add(text)
            new_rows.append({
                "text": text,
                "label": 1 if is_scam else 0,
                "source": "ollama_generated",
                "domain": domain,
                "channel": channel,
                "split_role": "holdout" if domain in HOLDOUT_DOMAINS else "train",
            })

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(OUT_PATH, index=False)   # incremental save = resumable
            print(f"[{len(df)}/{target}] +{len(new_rows)} {domain} "
                  f"({'scam' if is_scam else 'legit'}, {channel}, via {backend})")

    print("\n=== GENERATION SUMMARY ===")
    if len(df):
        print(df.groupby(["split_role", "domain", "label"]).size())
        print(f"\nUnique texts: {df['text'].nunique()} / {len(df)} rows")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
