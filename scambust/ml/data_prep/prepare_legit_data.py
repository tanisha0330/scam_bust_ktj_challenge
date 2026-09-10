"""
Pulls REAL (non-synthetic) human conversation data to fix the project's
weakest axis: coverage of ordinary, legitimate messages and calls.

Everything else in this project is synthetic. The measured consequence was a
false positive on a perfectly normal family message about a bank statement --
the model had seen far more fraud than normal life (the India_Cyber CSV has
only ~110 distinct "safe" templates).

Sources (all real human text):
  1. ucirvine/sms_spam       -- UCI SMS Spam Collection. 5,574 real SMS.
                                We take BOTH classes, not just the ham. If we
                                took only ham, "source" would perfectly predict
                                "legit" and the model could learn writing style
                                instead of fraud semantics -- the same shortcut
                                trap as template leakage. Including its 747 real
                                spam messages breaks that correlation.
  2. festvox/cmu_hinglish_dog -- real casual Hinglish chat (movie chit-chat).
                                Individual turns are tiny, so we group by
                                conversation and stitch turns into transcript-
                                shaped text matching our call data.
  3. talkmap/banking-conversation-corpus -- legitimate agent/client BANK calls.
                                These are the hard negatives: real conversations
                                about payments, bills and account access that are
                                entirely legitimate. Exactly the shape most likely
                                to trip a fraud classifier. 5.5M turns, subsampled.

Output: ml/data/legit_sources.csv  [text, label, source]
"""
import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data")
REAL_DIR = os.path.join(OUT_DIR, "real_sources")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

# Crowdsourced real Indian SMS (1,000 ham + 1,000 spam), Hindi/English.
INDIAN_SMS_URL = (
    "https://raw.githubusercontent.com/princebari/"
    "-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages"
    "/main/indian_spam.csv"
)

# How many grouped conversations to keep from the big corpora.
N_BANKING_CONVERSATIONS = 5000
N_HINGLISH_CONVERSATIONS = 2500
MIN_TURNS = 3          # skip trivially short conversations
MAX_TURNS_KEPT = 14    # keep transcripts in the same length ballpark as our call data


def load_uci_sms():
    from datasets import load_dataset

    print("Loading ucirvine/sms_spam ...")
    ds = load_dataset("ucirvine/sms_spam", split="train")
    df = pd.DataFrame({
        "text": [r.strip() for r in ds["sms"]],
        "label": list(ds["label"]),          # 0 = ham (legit), 1 = spam
        "source": "uci_sms",
    })
    print(f"  -> {len(df)} rows ({(df.label == 0).sum()} legit / {(df.label == 1).sum()} spam)")
    return df


def load_cmu_hinglish_dog():
    from datasets import load_dataset

    print("Loading festvox/cmu_hinglish_dog ...")
    frames = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("festvox/cmu_hinglish_dog", split=split)
        # NOTE: `uid` is only the speaker ROLE ("user1"/"user2") and `docIdx` is
        # the turn-group index *within* a chat -- neither identifies a
        # conversation. A session is keyed by (uid1LogInTime, user2_id).
        frames.append(pd.DataFrame({
            "conv_id": [f"{split}-{a}-{b}" for a, b in zip(ds["uid1LogInTime"], ds["user2_id"])],
            "uid": ds["uid"],
            "ts": ds["utcTimestamp"],
            "text": [t["hi_en"] for t in ds["translation"]],
        }))
    turns = pd.concat(frames, ignore_index=True)

    # The raw rows are single utterances ("hi", "nahi aur batao"). Group them
    # into conversations and stitch into speaker-labelled transcripts so the
    # shape matches our call/WhatsApp data instead of being one-word fragments.
    turns = turns.sort_values(["conv_id", "ts"])
    convs = []
    for conv_id, grp in turns.groupby("conv_id"):
        grp = grp.head(MAX_TURNS_KEPT)
        if len(grp) < MIN_TURNS:
            continue
        speakers = {u: f"Speaker{i + 1}" for i, u in enumerate(grp["uid"].unique())}
        lines = [f"{speakers[r.uid]}: {str(r.text).strip()}" for r in grp.itertuples() if str(r.text).strip()]
        if len(lines) >= MIN_TURNS:
            convs.append("\n".join(lines))

    convs = convs[:N_HINGLISH_CONVERSATIONS]
    print(f"  -> {len(convs)} grouped Hinglish conversations")
    return pd.DataFrame({"text": convs, "label": 0, "source": "cmu_hinglish_dog"})


def load_banking_corpus():
    from datasets import load_dataset

    print("Loading talkmap/banking-conversation-corpus (streaming, subsampled) ...")
    ds = load_dataset("talkmap/banking-conversation-corpus", split="train", streaming=True)

    by_conv = {}
    completed = []
    for row in ds:
        cid = row["conversation_id"]
        bucket = by_conv.setdefault(cid, [])
        if len(bucket) < MAX_TURNS_KEPT:
            speaker = "Agent" if str(row["speaker"]).lower() == "agent" else "Customer"
            text = str(row["text"]).strip()
            if text:
                bucket.append(f"{speaker}: {text}")
        elif cid not in completed:
            completed.append(cid)
        # Stop once we have enough conversations with enough turns each.
        if len(by_conv) > N_BANKING_CONVERSATIONS * 2:
            break

    convs = [
        "\n".join(lines) for lines in by_conv.values() if len(lines) >= MIN_TURNS
    ][:N_BANKING_CONVERSATIONS]
    print(f"  -> {len(convs)} grouped legitimate banking conversations")
    return pd.DataFrame({"text": convs, "label": 0, "source": "banking_legit"})


def load_indian_real_sms():
    """Real crowdsourced Indian SMS -- the only genuinely Indian, genuinely
    human text in the whole project.

    IMPORTANT LABELLING DECISION: this corpus splits ham/spam, but its "spam"
    is *commercial marketing*, not fraud -- movie promos, Noida flat ads,
    quiz contests, coaching-institute adverts. Treating those as scams would
    teach the model that promotional urgency ("reply within 24 hrs to win")
    equals fraud, which is precisely the false-positive failure we are trying
    to fix. So only the ham enters training, and the marketing half is written
    out separately as a false-positive probe (see marketing_probe.csv).
    """
    path = os.path.join(REAL_DIR, "indian_spam.csv")
    if not os.path.exists(path):
        print("Downloading real Indian SMS corpus ...")
        r = requests.get(INDIAN_SMS_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    df = pd.read_csv(path)
    df["v2"] = df["v2"].astype(str).str.strip()
    ham = df[df["v1"] == "ham"]
    marketing = df[df["v1"] == "spam"]

    # Diagnostic set: legitimate commercial messages that superficially look
    # scam-like. Not training data, and not scored as fraud.
    marketing_out = pd.DataFrame({
        "text": marketing["v2"], "label": 0, "source": "indian_marketing_probe",
    })
    marketing_out.to_csv(os.path.join(OUT_DIR, "marketing_probe.csv"), index=False)
    print(f"  -> {len(marketing_out)} marketing messages held out as a false-positive probe")

    print(f"  -> {len(ham)} real Indian ham messages")
    return pd.DataFrame({"text": ham["v2"], "label": 0, "source": "indian_real_ham"})


def main():
    parts = [load_uci_sms(), load_cmu_hinglish_dog(), load_banking_corpus(), load_indian_real_sms()]
    df = pd.concat(parts, ignore_index=True)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"], keep="first")

    out_path = os.path.join(OUT_DIR, "legit_sources.csv")
    df.to_csv(out_path, index=False)

    print("\n=== LEGIT SOURCES SUMMARY ===")
    print(df.groupby(["source", "label"]).size())
    print(f"\nTotal: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
