"""
Builds a clean unseen-domain evaluation set in INDIAN context.

WHY THIS EXISTS
The ICFD-31k `cross_domain` split (our headline 0.427 recall) turned out to
conflate TWO distribution shifts at once:
  1. genuinely new scam types (crypto, charity, MLM, tax-refund), and
  2. an unintended geography shift -- samples reference "Bank of America",
     "Social Security number" and "the IRS", while all our training data is
     India-centric (UPI, Aadhaar, PAN, RBI).

So we could not tell whether the model fails because the *scam* is new or
because the *country* is. This set removes variable 2 by mirroring the exact
same five scam types in Indian context.

DELIBERATELY NOT "MADE DIFFERENT"
Team04's paper uses deliberately divergent prompts for test-set generation to
avoid leakage. We are NOT doing that here -- this reuses `build_prompt` from
generate_novel_domains.py verbatim. The question being asked is the plain one:
"if we just naturally extend our own data pipeline to new Indian scam types,
how does the model do?" That means a style-similarity confound remains (the
generator is the same), and the result should be read as an optimistic bound,
not a clean generalisation number.

None of these five domains appear in ICFD-31k's 10 training domains, nor in
the 8 domains used by generate_novel_domains.py.

Usage:
    python -m ml.data_prep.generate_indian_crossdomain_eval [N]
"""
import os
import random
import sys

import pandas as pd

from .generate_novel_domains import (
    BATCH_SIZE,
    CHANNELS,
    _groq_client,
    build_prompt,
    call_groq,
    call_ollama,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "indian_crossdomain_eval.csv")

DEFAULT_TARGET = 500

# Each entry mirrors one ICFD cross_domain category, re-set in India.
#   cross_domain_1 -> crypto/investment platform fraud
#   cross_domain_2 -> bank officer harvesting security answers
#   cross_domain_3 -> fake charity donation
#   cross_domain_4 -> pyramid / MLM scheme
#   cross_domain_5 -> tax-refund identity fraud
DOMAINS = {
    "crypto_trading_app_india": (
        "a fake cryptocurrency or share-trading app run from India, promising guaranteed "
        "high returns and asking the victim to deposit more funds via UPI"
    ),
    "bank_security_question_harvest": (
        "someone posing as a bank officer from an Indian bank (SBI, HDFC, ICICI) who does not "
        "ask for money directly, but harvests security-question answers -- mother's maiden name, "
        "date of birth, first school, spouse's name -- to take over the account later"
    ),
    "charity_donation_fraud_india": (
        "a fake Indian charity or NGO collecting urgent donations -- temple construction, flood "
        "relief, a child's medical treatment -- asking for money via UPI or bank transfer"
    ),
    "chit_fund_mlm_india": (
        "an Indian chit-fund or multi-level-marketing pyramid scheme, where the victim must buy "
        "initial stock or pay a joining amount and then recruit more members to earn returns"
    ),
    "income_tax_refund_india": (
        "a fake Income Tax Department refund notice, claiming a duplicate PAN filing or pending "
        "refund, and asking the victim to confirm PAN/Aadhaar and bank details"
    ),
}

LANGUAGES = [
    "Hinglish (romanised Hindi mixed with English, Latin script only)",
    "Indian English",
]


def load_existing():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        print(f"Resuming: {len(df)} rows already generated")
        return df
    return pd.DataFrame(columns=["text", "label", "source", "domain", "channel"])


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    df = load_existing()
    seen = set(df["text"].astype(str)) if len(df) else set()

    groq = _groq_client()
    print(f"Target {target} rows | groq={'yes' if groq else 'no'}")

    items = list(DOMAINS.items())
    fails = 0
    while len(df) < target:
        domain, desc = random.choice(items)
        is_scam = random.random() < 0.5
        channel = random.choice(CHANNELS)
        language = random.choice(LANGUAGES)

        prompt = build_prompt(domain, desc, is_scam, channel, language, BATCH_SIZE)
        examples, backend = None, None
        if groq is not None:
            try:
                examples, backend = call_groq(groq, prompt), "groq"
            except Exception as e:
                print(f"  ~ groq failed ({e.__class__.__name__}), trying ollama")
        if examples is None:
            try:
                examples, backend = call_ollama(prompt), "ollama"
            except Exception as e:
                fails += 1
                print(f"  ! generation failed ({e.__class__.__name__}); {fails} consecutive")
                if fails >= 5:
                    print("Both backends failing. Stopping.")
                    break
                continue
        fails = 0

        rows = []
        for text in examples:
            if len(text) < 25 or text in seen:
                continue
            seen.add(text)
            rows.append({
                "text": text,
                "label": 1 if is_scam else 0,
                "source": "indian_crossdomain_eval",
                "domain": domain,
                "channel": channel,
            })
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(OUT_PATH, index=False)
            print(f"[{len(df)}/{target}] +{len(rows)} {domain} "
                  f"({'scam' if is_scam else 'legit'}, {channel}, via {backend})")

    print("\n=== SUMMARY ===")
    if len(df):
        print(df.groupby(["domain", "label"]).size())
        print(f"\nUnique texts: {df['text'].nunique()} / {len(df)}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
