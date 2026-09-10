"""
Builds a REBALANCED training set that maximises real and Indian data share,
accepting a much smaller row count than the 31.5k "use everything" build.

MOTIVATION
The previous mix was 31,499 rows but only ~21% real and ~7% genuinely Indian --
ICFD-31k (synthetic, US/generic) supplied 63% of it on its own. Meanwhile the
single biggest measured gain of the whole project came from adding just 925 real
Indian messages (cross-domain recall 0.427 -> 0.484, ~13 sigma). Signal per row
is clearly far higher for real Indian text, so let it occupy a bigger share.

HARD CONSTRAINT WORTH KNOWING
Every real Indian corpus we have is LEGITIMATE (crowdsourced ham, marketing SMS,
casual Hinglish chat). There is no real Indian *scam* text available, so the
positive class stays synthetic/generated regardless. What we can control is how
much non-Indian synthetic bulk it is diluted by.

STRATEGY (priority tiers)
  1. Take ALL real Indian data.                       (never subsampled)
  2. Take ALL Indian-context scam data.               (the only positives we have)
  3. Cap real non-Indian data.                        (keeps linguistic diversity)
  4. Fill remaining budget with ICFD, subsampled to hit the target class ratio.

The marketing SMS are included as label=0 here. They are real, Indian, and
legitimate commercial messages -- and withholding them is what produced an
87.9% false-positive rate on exactly that kind of text. A slice is still held
back as the probe so the metric stays honest.

Outputs balanced_{train,val,test}.csv plus a composition report.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .prepare_call_data import OUT_DIR, clean, load_icfd31k_split, load_india_cyber_csv
from .prepare_unified_data import (
    DATASET_DIR,
    load_audio_transcripts,
    load_calls,
    load_sms,
    load_whatsapp,
)

# (is_real, is_indian) for every source we can draw on.
SOURCE_META = {
    "icfd31k":            (False, False),
    "uci_sms":            (True,  False),
    "banking_legit":      (True,  False),
    "indian_real_ham":    (True,  True),
    "indian_marketing":   (True,  True),
    "cmu_hinglish_dog":   (True,  True),
    "india_cyber_csv":    (False, True),
    "ollama_generated":   (False, True),
    "scambust_sms":       (False, True),
    "scambust_whatsapp":  (False, True),
    "scambust_calls":     (False, True),
    "scambust_audio":     (False, True),
}

CAP_REAL_NON_INDIAN = 3000     # from UCI + banking (10k available)
TARGET_SCAM_FRACTION = 0.35    # more positive signal than the old 22%
MARKETING_HOLDOUT = 300        # kept back as the false-positive probe


def load_marketing():
    path = os.path.join(OUT_DIR, "real_sources", "indian_spam.csv")
    df = pd.read_csv(path)
    mk = df[df["v1"] == "spam"].copy()
    mk["text"] = mk["v2"].astype(str).str.strip()
    return pd.DataFrame({"text": mk["text"], "label": 0, "source": "indian_marketing"})


def load_indian_ham():
    path = os.path.join(OUT_DIR, "real_sources", "indian_spam.csv")
    df = pd.read_csv(path)
    ham = df[df["v1"] == "ham"].copy()
    return pd.DataFrame({
        "text": ham["v2"].astype(str).str.strip(), "label": 0, "source": "indian_real_ham",
    })


def report(df, title):
    df = df.copy()
    df["real"] = df["source"].map(lambda s: SOURCE_META.get(s, (False, False))[0])
    df["indian"] = df["source"].map(lambda s: SOURCE_META.get(s, (False, False))[1])
    n = len(df)
    print(f"\n=== {title}: {n} rows ===")
    print(f"  scam            : {df['label'].mean() * 100:.1f}%")
    print(f"  REAL data       : {df['real'].mean() * 100:.1f}%")
    print(f"  INDIAN context  : {df['indian'].mean() * 100:.1f}%")
    print(f"  real AND indian : {(df['real'] & df['indian']).mean() * 100:.1f}%")
    comp = df.groupby("source").agg(rows=("label", "size"), scam=("label", "sum"))
    comp["real"] = [SOURCE_META.get(s, (False, False))[0] for s in comp.index]
    comp["indian"] = [SOURCE_META.get(s, (False, False))[1] for s in comp.index]
    print(comp.sort_values("rows", ascending=False).to_string())


def main():
    legit = pd.read_csv(os.path.join(OUT_DIR, "legit_sources.csv"))

    # ---- Tier 1: all real Indian data --------------------------------------
    ham = clean(load_indian_ham())
    hinglish = clean(legit[legit["source"] == "cmu_hinglish_dog"][["text", "label", "source"]])
    marketing = clean(load_marketing())
    marketing_train, marketing_probe = train_test_split(
        marketing, test_size=MARKETING_HOLDOUT, random_state=42
    )
    tier1 = pd.concat([ham, hinglish, marketing_train], ignore_index=True)

    # ---- Tier 2: all Indian-context scam/positive data ---------------------
    scambust = clean(pd.concat(
        [load_sms(), load_whatsapp(), load_calls(), load_audio_transcripts()], ignore_index=True))
    india_cyber = clean(load_india_cyber_csv())
    gen_path = os.path.join(OUT_DIR, "generated_novel_domains.csv")
    gen = pd.read_csv(gen_path)
    gen_train = clean(gen[gen["split_role"] == "train"][["text", "label", "source"]])
    tier2 = pd.concat([scambust, india_cyber, gen_train], ignore_index=True)

    # ---- Tier 3: capped real non-Indian ------------------------------------
    real_non_indian = clean(legit[legit["source"].isin(["uci_sms", "banking_legit"])][["text", "label", "source"]])
    if len(real_non_indian) > CAP_REAL_NON_INDIAN:
        real_non_indian, _ = train_test_split(
            real_non_indian, train_size=CAP_REAL_NON_INDIAN,
            stratify=real_non_indian["label"], random_state=42)

    core = clean(pd.concat([tier1, tier2, real_non_indian], ignore_index=True))

    # ---- Tier 4: top up with ICFD to reach the target class ratio -----------
    icfd = clean(load_icfd31k_split("train", chunks_per_conversation=1))
    n_scam, n_legit = int(core["label"].sum()), int((core["label"] == 0).sum())
    # Solve for how many extra scam rows hit TARGET_SCAM_FRACTION overall.
    need_scam = max(0, int((TARGET_SCAM_FRACTION * (n_scam + n_legit) - n_scam)
                           / (1 - TARGET_SCAM_FRACTION)))
    icfd_scam = icfd[icfd["label"] == 1].sample(min(need_scam, (icfd["label"] == 1).sum()), random_state=42)
    # ICFD legit must OUT-NUMBER ICFD scam, matching its natural ~72/28 lean.
    # Measured failure from getting this wrong: a first attempt used half as
    # many legit as scam transcripts, which made conversation-length correlate
    # with the label (long transcript => scam, short SMS => safe). The model
    # duly learned that shortcut and flagged 62.6% of a 21.9%-scam test set.
    # Every other legit source here is short-form, so ICFD is the *only*
    # supplier of legitimate long conversations and cannot be starved.
    n_legit_needed = int(len(icfd_scam) * 1.5)
    icfd_legit = icfd[icfd["label"] == 0].sample(
        min(n_legit_needed, int((icfd["label"] == 0).sum())), random_state=42)
    print(f"\nTopping up with ICFD: {len(icfd_scam)} scam + {len(icfd_legit)} legit")

    pool = clean(pd.concat([core, icfd_scam, icfd_legit], ignore_index=True))
    report(pool, "BALANCED POOL")

    train_df, rest = train_test_split(pool, test_size=0.08, stratify=pool["label"], random_state=42)
    val_df, test_df = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)

    train_df.to_csv(os.path.join(OUT_DIR, "balanced_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "balanced_val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "balanced_test.csv"), index=False)
    marketing_probe.to_csv(os.path.join(OUT_DIR, "marketing_probe.csv"), index=False)

    report(train_df, "TRAIN")
    print(f"\nval={len(val_df)}  test={len(test_df)}  marketing_probe={len(marketing_probe)}")
    print("Saved balanced_{train,val,test}.csv")


if __name__ == "__main__":
    main()
