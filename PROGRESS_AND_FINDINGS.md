# Scam Shield AI — Progress, Experiments and Findings

A complete record of the work: what was broken, what was tried, what the numbers said, and what was learned — including the things that **did not** work, which are often the more useful record.

Companion documents: `TECHNICAL_DEEP_DIVE.md` (what the system is), `PROJECT_EXPLANATION.md` (architecture + operational handoff).

---

## 0. Where the project started

The original submission was a hackathon build:

- **Backend:** a single-file Django script, TF-IDF + LogisticRegression/RandomForest ensemble, `.pkl` files, a hardcoded Python list of spam numbers, bare `except:` everywhere
- **Frontend:** one 684-line `main.dart`, a user-facing "Auto / Online / Offline" dropdown
- **Data:** ~296 rows of synthetic data
- **Evaluation:** accuracy only, single 80/20 split, no held-out generalisation test

### Bugs found in the initial review

| Bug | Impact |
|---|---|
| `check_number`'s response ignored client-side | The spam database was **completely inert** — known spam numbers never triggered the red dialog |
| `bool isScam = data['is_scam'];` | Unhandled cast crash if the LLM's JSON omitted the key |
| WhatsApp alerting advertised but never implemented | Translation keys existed; only `sms:` URLs were ever launched |
| Train/test leakage in `train_scam_shield.py` | Synthetic + adversarial rows (duplicated ×5) were appended **before** `train_test_split`, so near-duplicates landed in both sides — reported accuracy was inflated |
| `ALLOWED_HOSTS=['*']`, hardcoded `SECRET_KEY`, `DEBUG=True`, CORS `*`, no auth | Fine on a LAN demo, unshippable otherwise |
| Personal LAN IP hardcoded in source | `10.145.73.107:8000` committed into `main.dart` |
| No `requirements.txt` anywhere | Dependencies had to be guessed |

---

## 1. Dataset archaeology — the numbers behind the data

Before touching the model, the data was audited. Three findings changed the plan.

### 1.1 The original dataset is tiny and synthetic

`public_unified_multimodal.csv` turned out to be the other four files concatenated: **296 unique rows**, ~81% scam. The dataset's own README states it is *"fully synthetic (fabricated) for hackathon use... contains no real incidents or personal data."*

### 1.2 The 10,000-row Hinglish CSV is 743 templates

`India_Cyber_Scam_Hinglish_Dataset.csv` presents as 10,000 rows with a clean 5,000/5,000 class balance. At the unique-text level:

- **743 unique texts**, some repeated **60×**
- **633 unique scam templates vs only 110 unique safe templates**

The 50/50 row balance was manufactured by repeating 110 safe templates far more often. Splitting on rows rather than unique text would put identical sentences in train and test.

### 1.3 ICFD-31k's own test splits are unusable as tests

`rishia2220/icfd-31k` (31,000 conversations, 1.07M streaming chunks) has train/validation/test/cross_domain splits. Querying the HF `datasets-server` statistics endpoint revealed:

| Split | Composition |
|---|---|
| train | 10 domains, ~2,100 conversations each, full `case_type` mix (Clear Normal, Ambiguous Normal, Subtle Fraud, Clear Fraud) |
| validation | **100% fraud scenarios** — 4,450 scam / 49 legit |
| test | **100% fraud scenarios** — 4,450 scam / 49 legit |

A model that always answers "scam" would score ~0.95 accuracy on the official test split. These were reclassified as *fraud-recall stress tests*, and the real train/val/test was carved from the realistically-balanced train split instead.

### 1.4 Preprocessing decision: one chunk per conversation

ICFD-31k stores *cumulative* chunks (chunk N contains chunks 1..N). Using all 1.07M would over-represent long conversations, create massive near-duplication, and mislabel early chunks of an eventual fraud call as scam before any fraud signal appears. Only the **final chunk per `conversation_uid`** was kept: 712,316 chunks → **20,996 conversations**.

---

## 2. Replacing the model: TF-IDF → MuRIL

### 2.1 The old model had collapsed

The legacy TF-IDF + LR/RF ensemble was benchmarked against the new one on identical held-out sets:

| Test set | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ICFD real-shaped (652) | **OLD** | 0.296 | 0.296 | 1.000 | 0.457 |
| ICFD real-shaped (652) | **NEW** | 0.989 | 0.974 | 0.990 | 0.982 |
| Synthetic held-out (178) | **OLD** | 0.921 | 0.915 | 1.000 | 0.956 |
| Synthetic held-out (178) | **NEW** | 1.000 | 1.000 | 1.000 | 1.000 |

**The critical detail:** the old model's precision of **0.296 exactly equals the test set's scam prevalence**, with recall 1.000. That is the signature of a **near-constant "always scam" classifier**. It looked fine on data resembling its narrow training vocabulary (0.921) and collapsed entirely outside it. It had no discriminative power at all — it had simply never been tested on anything unfamiliar.

### 2.2 Why MuRIL

Data is romanised Hinglish in Latin script — **zero Devanagari anywhere**. MuRIL's pre-training includes transliterated Indic text, matching the actual input distribution. Language distribution in ICFD-31k: ~70% English / ~30% Hinglish, all Latin script.

---

## 3. Fine-tuning experiments — four variants, one conclusion

All four evaluated on 1,000 held-out ICFD-31k conversations from **five domains absent from training**.

| Variant | In-domain F1 | Cross-domain acc | Cross-domain recall | Cross-domain F1 |
|---|---|---|---|---|
| Baseline (near-full fine-tune) | 0.982 | 0.444 | 0.431 | 0.595 |
| Static partial freeze (bottom 6) | 0.974 | **0.358** | **0.332** | 0.495 |
| Gradual unfreeze v1 (4 ep / ~20k) | 0.941 | **0.473** | **0.459** | 0.623 |
| Gradual unfreeze v2 (8 ep / ~38k) | 0.987 | 0.431 | 0.412 | 0.579 |

### What each experiment tested and returned

**Static partial freeze** (hypothesis: freezing the bottom 6 layers stops memorisation of domain-specific vocabulary) — **falsified decisively.** Worst variant on every cross-domain metric (0.332 recall). Freezing cost plasticity without targeting the actual overfitting, which lives in the always-trainable top layers.

**Gradual unfreeze v1** — best cross-domain performer, at a cost of in-domain F1 (0.941 vs 0.982).

**Gradual unfreeze v2** (longer schedule, more data) — best in-domain F1 of the four (0.987) and *worse* cross-domain (0.412). More training at full depth and ~2× data both traded generalisation for fit.

### The conclusion that shaped everything after

**Every change that improved in-domain fit made cross-domain generalisation worse.** Layer-freezing depth, schedule gradualness and raw data volume were all exhausted as levers. Attention moved to data composition.

### Side finding: data volume actively hurt

Expanding ICFD-31k from 1 chunk/conversation (~21k rows) to 6 chunks downsampled (~38k rows) *lowered* cross-domain recall 0.459 → 0.412. The pipeline was reverted to 1 chunk permanently.

---

## 4. The confidently-wrong problem

Deep-diving the cross-domain failures produced the most consequential diagnostic of the project.

Of **557 Tier-1 mistakes** on the cross-domain set:

| Measurement | Value |
|---|---|
| Traffic escalating to Tier-2 at threshold 0.85 | **14.7%** |
| Mistakes actually caught by escalation | **12.9%** |
| Missed scams that were **confidently** wrong | **506 of 540 (93.7%)** |
| Mean confidence of those confident mistakes | **0.983** (sd 0.020, min 0.856) |

**Threshold tuning cannot fix this:**

| Threshold | Confident misses recovered |
|---|---|
| 0.85 | 0 / 506 (0.0%) |
| 0.90 | 10 / 506 (2.0%) |
| 0.95 | 28 / 506 (5.5%) |
| 0.99 | 258 / 506 (51.0%) |

Reaching even 51% requires escalating essentially all traffic, defeating the cascade's purpose. The safety net was not firing precisely where it was needed.

---

## 5. The OOD detector — a well-designed experiment that failed

**Hypothesis:** escalate on *input novelty* rather than confidence. Built Mahalanobis distance (Lee et al. 2018, class-conditional Gaussians with shared covariance over `[CLS]` embeddings, 12,000 reference rows) plus kNN-cosine.

**Experimental design:** rather than asking "does OOD detect unseen domains" (trivially yes), it was framed as a detection problem — target = *"did Tier-1 get this wrong?"*, score = the candidate escalation signal. A signal only earns deployment if it beats the incumbent, `1 − softmax confidence`.

### ROC-AUC for predicting Tier-1 is wrong

| Evaluation set | 1 − confidence | Mahalanobis | kNN-cosine |
|---|---|---|---|
| In-distribution test | **0.926** | 0.898 | 0.902 |
| Generated unseen domains | 0.740 | 0.735 | 0.733 |
| **ICFD cross-domain** | **0.413** | **0.504** | **0.514** |

**Verdict: useless on the case that matters.** Both OOD scores sit at chance. The escalation table confirms it — escalating the top 30% by OOD score catches 30.0% of mistakes, exactly what escalating a *random* 30% would catch.

Also note `1 − confidence` at **0.413 — worse than random**. On unseen domains the model is, if anything, slightly *more* confident when wrong.

### Why it failed — the useful part

**OOD detection measures *stylistic* novelty; this problem is *semantic* novelty.** ICFD's cross-domain conversations come from the same generator as training — same "Agent:/Customer:" format, same length, same synthetic register — differing only in *what the scam is about*. In embedding space they are genuinely in-distribution. There is nothing for a distance metric to detect.

The generated-holdout column is the control that proves the mechanism: that set *is* stylistically different (different LLM, different prompt scaffolding), and there OOD works fine (0.735) — but plain confidence matches it (0.740), so it still adds nothing.

**Consequence:** we cannot triage which inputs need Tier-2. This rules out OOD detection, threshold tuning, and MC-dropout (a variance estimate over the same uninformative confidence). The detector is saved but deliberately **not wired into the backend**.

---

## 6. The benchmark was measuring the wrong thing

Inspecting actual cross-domain samples revealed the split conflates **two** distribution shifts.

The five unseen categories are: crypto-investment fraud, bank security-question harvesting, fake charity, pyramid/MLM, tax-refund identity theft. But several samples reference **"Bank of America", "Social Security number", "the IRS"** — while all training data is India-centric (UPI, Aadhaar, PAN, RBI). A low score could mean "new scam type" *or* "wrong country".

**The controlled experiment:** generate the *same five scam types* in Indian context (chit funds for MLM, PAN/Income-Tax for SSN/IRS, temple and flood-relief charities), reusing the identical prompt template — deliberately *not* diverging it.

| Metric | ICFD cross-domain (US-flavoured) | Same 5 types, Indian context | Δ |
|---|---|---|---|
| Accuracy | 0.443 | 0.777 | **+0.334** |
| Precision | 0.969 | 0.703 | −0.266 |
| **Recall** | **0.427** | **0.852** | **+0.426** |
| F1 | 0.593 | 0.770 | +0.178 |
| ROC-AUC | 0.689 | 0.863 | +0.174 |

**Recall doubles.** The headline "misses more than half of unseen scams" was substantially a **geography** failure, not a scam-type failure.

### Cross-checking with older models

Running **all six trained variants** on the Indian set provided a control for the obvious objection (that the generated set is easy because it shares a generator with generated training data). The four `call_classifier*` variants **never saw a single LLM-generated row in training** — yet they score 0.689–0.820 recall:

| Variant | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.669 | 0.588 | 0.820 | 0.685 | 0.759 |
| Static partial freeze | 0.604 | 0.538 | 0.689 | 0.604 | 0.703 |
| Gradual unfreeze v1 | 0.626 | 0.549 | 0.820 | 0.658 | 0.759 |
| Gradual unfreeze v2 | 0.691 | 0.615 | 0.787 | 0.691 | 0.791 |
| Message classifier | 0.655 | 0.571 | 0.852 | 0.684 | 0.745 |
| **Unified** | **0.777** | **0.703** | **0.852** | **0.770** | **0.863** |

Since models with no exposure to generated text still perform far better here than on ICFD cross-domain (0.33–0.46 recall), the conclusion holds: **Indian-context novel scams genuinely are easier than US-flavoured ones.** It is a property of the data, not an artifact.

---

## 7. Architectural consolidation

### 7.1 Two classifiers → one

The message classifier had **140 unique training rows** (117 scam / 23 legit) — the weakest component. Its 5-fold CV compared two initialisations:

| Init | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Vanilla MuRIL | 0.836 | 0.836 | 1.000 | 0.910 | 0.783 |
| **Transfer from call classifier** | **0.914** | **0.942** | 0.957 | **0.949** | **0.911** |

Transfer learning won decisively — but the real fix was merging the models entirely, turning 140 rows into tens of thousands.

### 7.2 The user-facing mode toggle was removed

An "Auto / Online / Offline" dropdown contradicted the product's own "Zero-UI for seniors" premise. The server-side cascade now decides.

### 7.3 Postgres replaced in-memory state

`SpamNumber` and `ScanLog` tables; every cascade decision persisted, building the retraining flywheel.

---

## 8. Production bugs found and fixed

| Bug | Discovery | Fix |
|---|---|---|
| **Groq retired the entire Llama-3 family** | `llama-3.3-70b-versatile` returned HTTP 404 — Tier-2 escalation was **silently dead** | Switched to `openai/gpt-oss-120b`, made overridable via `GROQ_MODEL` |
| **Model path off by one directory** | Backend was silently falling back to escalate-everything; DB showed `tier1_confidence = 0.000` on every row | Corrected `SCAMBUST_DIR` resolution |
| **`usesCleartextTraffic` was inert** | Sat as raw *text content* inside `<activity>`, not as an attribute — while the app talks to `http://` and Android 9+ blocks cleartext by default | Moved to `<application>` as a proper attribute |
| **LLM non-determinism** | At `temperature=0.1` the LLM flip-flopped between runs on borderline messages, making verdicts unstable | Set `temperature=0.0` — verified identical across consecutive runs |
| **Degenerate calibration** | Temperature fit returned **0.05** — the clamp floor — making the model *more* confident and breaking the escalation threshold | Optimise `log(T)`; reject fits outside [0.25, 5.0] with fallback to T=1.0 |
| **Mojibake in generated data** | `requests`' charset guess mangled Ollama's UTF-8 into U+FFFD, which would become a spurious training signal | Explicit UTF-8 decode + punctuation normalisation |
| **Concurrent writers** | Two generator processes rewrote the same CSV, silently clobbering rows | Enforced single-process generation |
| **`head -40` in a pipeline** | Would SIGPIPE-kill a long training run mid-flight | Removed; logs written to files |

---

## 9. Data expansion — the real-data breakthrough

### 9.1 Adding real human corpora

| Source | Type | Contribution |
|---|---|---|
| `ucirvine/sms_spam` | **Real** | 4,518 ham + 642 spam |
| `talkmap/banking-conversation-corpus` | **Real** | 5,000 legitimate bank calls (hard negatives) |
| `festvox/cmu_hinglish_dog` | **Real** | 256 genuine Hinglish conversations |

**Design rule applied:** UCI's *spam* half was included, not just its ham. If a source's label were perfectly predictable from the source itself, the model could learn writing style instead of fraud semantics.

> **Bug caught here:** the CMU Hinglish grouping initially produced **12 conversations instead of ~256**. `uid` is only the speaker *role* ("user1"/"user2") and `docIdx` is a turn-group index — neither identifies a conversation. The real key is `(uid1LogInTime, user2_id)`.

### 9.2 Real Indian SMS — the single biggest gain

Found via GitHub: a **crowdsourced Indian SMS corpus** (1,000 ham + 1,000 spam, Hindi/English). Verified before ingesting:

- **Zero overlap** with UCI (independent, not a copy)
- **Zero Devanagari** — all romanised, matching language scope
- Authentically Indian: *"I BRO. . . Tera cell recharge nhe ho paya so mom two 25-25 wale cards lae hai"*

Adding **925 real Indian ham messages** improved every metric:

| Evaluation | Before | After |
|---|---|---|
| In-distribution F1 | 0.949 | **0.962** |
| Held-out synthetic F1 | 0.975 | **0.980** |
| Generated unseen F1 | 0.870 | **0.876** |
| **ICFD cross-domain recall** | 0.427 | **0.478** |

925 real rows moved cross-domain recall further than 18,000 synthetic ones had.

### 9.3 A labelling trap avoided

The corpus's "spam" half is **commercial marketing, not fraud** — Hangover movie promos, Noida flat adverts, Center Fresh quiz contests, coaching-institute ads. Labelling those as scam would have taught the model that *"reply within 24 hrs to win"* equals fraud.

They were instead held out as a **false-positive probe** — which immediately exposed a catastrophic, previously invisible failure.

### 9.4 Sources evaluated and rejected

| Source | Reason |
|---|---|
| `shaghayegh-hp/Smishing_Dataset` (84,863 rows) | Real, but African telecom (Glo, "Smallie") — reintroduces the geography confound |
| `animeshdinda12/indian-scam-detection` | 127 rows, empty dataset card, unverifiable provenance |
| `jencyyy/fraudcall` | Bengali translations of *Chinese* fraud calls (China Construction Bank) |
| Kaggle `narayanyadav/fraud-call-india-dataset` | Requires Kaggle API credentials, not configured |

---

## 10. The 87.9% false-positive discovery

The marketing probe fed the model **1,000 real, legitimate Indian commercial SMS**.

**It flagged 879 of them as scams — an 87.9% false-positive rate.**

Every prior evaluation set was scam-heavy or used synthetic legitimate data. This was the first test with volume of *real, legitimate-but-promotional* text. An Indian senior receives dozens of these weekly; the app would have screamed "SCAM" at nine out of ten and been uninstalled within a day.

**Root cause:** the marketing messages were deliberately excluded from training (to avoid mislabelling them as fraud). Correct as a *diagnostic* call, wrong as a *training* call — the model had never seen legitimate promotional urgency labelled not-fraud.

---

## 11. Multi-seed — establishing the noise floor

Every comparison until this point was single-seed. Three seeds (42, 7, 123) on the 31,499-row build:

| Evaluation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| In-distribution test | 0.982 ±0.001 | 0.937 ±0.002 | 0.986 ±0.004 | 0.961 ±0.002 |
| Held-out synthetic | 0.964 ±0.002 | 0.965 ±0.003 | 0.994 ±0.000 | 0.979 ±0.001 |
| Generated unseen domains | 0.825 ±0.014 | 0.861 ±0.013 | 0.879 ±0.011 | 0.870 ±0.010 |
| ICFD cross-domain | 0.497 ±0.004 | 0.970 ±0.002 | 0.484 ±0.004 | 0.646 ±0.004 |

**The noise floor is tighter than feared.** On ICFD cross-domain, seed spread is only ±0.004 (range 0.011). Retroactively this validates the earlier single-seed comparisons: gradual-v1 (0.459) vs baseline (0.431) is a ~6σ gap, and partial-freeze's collapse to 0.332 is far outside noise.

Only the generated-unseen set is genuinely noisy (±0.014) — differences under ~0.03 there are meaningless.

---

## 12. Rebalancing — quality over quantity

Goal: maximise real and Indian data share, accepting fewer rows.

### 12.1 First attempt failed — and the failure was instructive

Composition looked excellent (58.1% real, 38.3% Indian) but the model became **globally trigger-happy**:

| Set | Actual scam | Predicted scam |
|---|---|---|
| In-distribution test | 21.9% | **62.6%** |
| ICFD cross-domain | 94.9% | 97.1% |
| Indian unseen | 43.9% | 77.0% |

Its apparently spectacular "ICFD cross-domain accuracy 0.922" was an **artifact**: it answered "scam" to 97% of everything, and that set is 95% scam. The same failure signature as the old TF-IDF baseline.

**Root cause — a bug in the balancing logic.** The ICFD top-up used 1,500 scam + 750 legit, *inverting* ICFD's natural 72/28 legit/scam lean. Combined with dropping ~15,000 legit conversations, this made **transcript length correlate with the label**: long call transcripts → scam, short SMS → safe. The model learned the shortcut.

### 12.2 Corrected

ICFD legit was set to out-number ICFD scam (1,500 scam + 2,250 legit, matching its natural lean). ICFD is the *only* supplier of legitimate long conversations and cannot be starved.

| Set | Actual | Flawed balance | **Corrected** |
|---|---|---|---|
| In-dist test | 21.9% | 62.6% (2.9×) | **29.1% (1.33×)** |
| Indian unseen | 43.9% | 77.0% | **66.2% (1.51×)** |
| Marketing probe | 0% | 6.0% | **10.7%** |

### 12.3 Final composition

**10,008 rows — 49.4% real, 32.6% Indian, 27.1% scam** (was 31,499 rows / 21% real / 7% Indian).

| Evaluation | Unified (31.5k) | **Balanced (10k)** |
|---|---|---|
| Held-out synthetic F1 | 0.980 | **0.986** |
| Generated unseen F1 | 0.876 | **0.936** |
| ICFD cross-domain recall | 0.478 | **0.648** |
| Indian unseen recall | 0.820 | **0.902** |
| **Marketing false-positive rate** | **89.3%** | **10.7%** |

**Cross-domain recall 0.478 → 0.648** is the largest single generalisation gain of the project, achieved on **a third of the data**. Well-composed 10k beat poorly-composed 31.5k.

**Structural ceiling identified:** every real Indian corpus available is *legitimate*. No real Indian scam text exists in any public source found, so the positive class stays synthetic regardless of rebalancing. ~38% Indian is the practical ceiling until real Indian fraud data exists.

---

## 13. Tier-2 benchmarked, and the cascade validated

**Tier-2 standalone** (139 Indian unseen-domain rows): accuracy 0.842, precision 0.741, **recall 0.984**, F1 0.845, 2.53 s/row. It misses **1 scam in 61**.

**Full cascade simulation:**

| Escalation policy | To cloud | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Never | 0% | 0.777 | 0.703 | 0.852 |
| conf < 0.85 | 18.0% | 0.806 | 0.713 | 0.934 |
| conf < 0.95 | 34.5% | 0.827 | 0.734 | **0.951** |
| Always | 100% | 0.842 | 0.741 | 0.984 |

The cascade genuinely works in Indian context — my earlier "the safety net never fires" conclusion came from the US-flavoured ICFD set, where it truly didn't.

**Tier-1 threshold sweep** showed 0.50 is already the F1 optimum; dropping to 0.20 buys recall 0.852 → 0.885 at precision 0.703 → 0.651. No free win there.

### Changes shipped from this

1. **Threshold 0.85 → 0.95** (recall 0.934 → 0.951)
2. **Asymmetric escalation** — *any* scam prediction escalates regardless of confidence, because Tier-1 rated a genuine chit-fund reminder 0.987 scam, and because a scam verdict auto-texts the user's family
3. **Three-way verdict** — scam / suspicious / safe

The originally-reported false positive (*"Beta paisa mil gaya kya, thoda confusion ho raha hai bank statement me"*) now resolves correctly: Tier-1 uncertain at 0.704 → escalates → Tier-2 overrules with *"Lagta hai parivaar ka sawal, koi direct fraud ya OTP maang nahi"* → **suspicious**, not a red alarm.

---

## 14. Calibration repaired

| | Before | After |
|---|---|---|
| Temperature | **0.05** (clamp floor — degenerate) | **0.6371** |
| Expected Calibration Error | 0.0674 | **0.0132** (5× better) |

Mean confidence 0.955 against accuracy 0.945 — confidence now approximately means what it claims.

---

## 15. Comparison with the IISc Team04 paper

A seven-author IISc submission on the same challenge (`AI Enabled Scam Call Detection`) was reviewed.

| Approach | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Their BERT (fine-tuned) | 0.91 | 1.00 | 0.83 | 0.91 |
| Their Gemma-2B LoRA | 0.63 | 0.62 | 0.94 | 0.75 |
| Their Few-Shot+CoT GPT | 0.98 | 1.00 | 0.98 | 0.98 |
| **Our unified model** | **0.982** | 0.937 | **0.986** | **0.961** |

**Honest reading:** their fine-tuned BERT is *weaker* than ours (0.91 vs 0.961 F1), and their headline 0.98 comes from **GPT** — architecturally identical to our Tier-2. Critically, **they never ran a cross-domain test**, so their numbers have no counterpart to our 0.648.

**What they did better, and we adopted or should:**
1. **Translate-to-English normalisation** (IndicConformer ASR → IndicTrans2 → English → classifier) — turns language coverage into a solved translation problem rather than a classification problem
2. **They actually built the audio path**, benchmarking 3 ASR models (IndicConformer-600M won: 15.8% Hindi WER)
3. **Separate prompts for dev vs test generation** to prevent leakage — precisely the confound in our generated holdout
4. **RAG over a scam-pattern vector store** — new scam types added without retraining
5. **Follow-up questions + 3-way "Suspicious" output** — the three-way verdict was adopted directly

---

## 16. The audio path — investigated, and it is a platform wall

Direct call-audio access is **impossible** for a normal Android app:

- `MediaRecorder.AudioSource.VOICE_CALL` requires `CAPTURE_AUDIO_OUTPUT`, a signature-level permission for platform-signed or OEM-preinstalled apps only
- Google's **May 11 2022** Play policy banned `AccessibilityService` for call recording
- Android 10's `AudioPlaybackCapture` explicitly excludes voice-call audio
- As of 2026 only native/preinstalled dialers can record; workarounds need Shizuku + wireless ADB or root — non-starters for a senior

**Viable paths:** force speakerphone + microphone capture (only real-time option; `MODIFY_AUDIO_SETTINGS` now declared); `CallScreeningService` for pre-call number screening (no audio, but wires into the existing `/check_number`); post-call analysis of OEM recording folders (device-dependent, but would yield **real Indian scam audio**).

**Still untested end-to-end.** No live-audio-to-verdict test has been run. Remains the largest untested assumption.

---

## 17. Methodological lessons

1. **Test on data your model has never seen a relative of.** Every headline number was fine until the first genuine generalisation test.
2. **Check whether a benchmark measures what you think.** ICFD cross-domain conflated scam-type novelty with geography; separating them doubled the apparent recall.
3. **When two independent models make the same "mistake", suspect the label.** Both our classifier and a frontier LLM flagged the same chit-fund messages — the ground truth was arguable, not the models.
4. **Precision equal to base rate = constant classifier.** This caught the old TF-IDF model *and* the flawed rebalance.
5. **A metric can hide a degenerate model.** F1 0.935 alongside AUC 0.585 on a 98.9%-scam set; report both.
6. **Composition beats volume.** 10k well-composed rows beat 31.5k poorly-composed. 925 real Indian rows beat 18,000 synthetic.
7. **Build the probe for the failure you can't see.** The marketing probe existed for one run before exposing an 87.9% catastrophe invisible to every other eval.
8. **Negative results are results.** OOD detection, static partial freeze, and volume scaling all failed — each ruled out a family of approaches and redirected effort.

---

## 18. Open problems

| Problem | Status |
|---|---|
| Audio path untested end-to-end | Platform limits mapped; speakerphone is the only path; **no live test run** |
| No real scam data | All positive-class data synthetic/generated; no real Indian scam corpus found |
| Cross-domain recall 0.648 | Best achieved, still imperfect |
| Precision on financial topics (0.598 Indian unseen) | Partly a ground-truth problem, not purely a model problem |
| Cannot triage uncertainty | OOD ruled out by measurement; ensemble disagreement is the one untested mechanism left |
| Multi-seed on balanced build | Only seed 42 complete |
| Production hardening | No auth, plain HTTP, `DEBUG=True`, dev server, synchronous model load |
