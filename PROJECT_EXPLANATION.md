# Scam Shield AI - Project Deep Dive

This document serves as an in-depth technical and architectural guide to the **Scam Shield AI** project. It is written to provide a comprehensive understanding for both human developers and Large Language Models (LLMs) working on or analyzing this codebase.

---

## 1. High-Level Project Overview

**Scam Shield AI** is a privacy-first, hybrid mobile application tailored for senior citizens to protect them against digital fraud, specifically phishing messages and vishing (voice phishing) calls.

It acts as a "Zero-UI" protection layer by automatically scanning messages and monitoring live calls. If a potential scam is detected (e.g., coercive language, "Digital Arrest" threats, request for OTPs), it alerts the user (e.g., via distinct device vibrations) and can automatically notify trusted family members.

> **See also:** `TECHNICAL_DEEP_DIVE.md` (full technical description of every component) and `PROGRESS_AND_FINDINGS.md` (every experiment, measurement and negative result). This file is the architectural summary plus the operational handoff.

The system relies on a **two-tier cascade** architecture, chosen specifically so the user never has to think about "online vs. offline" mode:
- **Tier 1 (always runs, local):** A single **fine-tuned MuRIL** binary classifier handling SMS, WhatsApp and call transcripts alike. Fast, works without internet access to any cloud AI, and returns a calibrated confidence score.
- **Tier 2 (escalation only):** the request is escalated to a cloud LLM (`openai/gpt-oss-120b` via Groq API) for a second opinion when Tier 1's confidence is below `TIER1_CONFIDENCE_THRESHOLD` (**0.95**) **or whenever Tier 1 predicts "scam" at all**, regardless of confidence.

**Why scam predictions always escalate.** Tier-1 is *confidently* wrong on ambiguous financial text -- it rated a genuine chit-fund instalment reminder 0.987 scam, above any workable threshold -- so confidence alone can never catch these. And a scam verdict auto-sends an SMS to the user's family contact, which is the consequential branch and worth a ~2.5 s second opinion. Scam predictions are a minority of traffic, which bounds the cost.

**Output is three-way, not binary:** `scam` / `suspicious` / `safe`. "Suspicious" fires when Tier-1 and Tier-2 disagree -- the strongest ambiguity signal available. It exists because a real chit-fund reminder and a chit-fund scam are near-identical in text, and forcing a binary answer on such a message is dishonest. Only `scam` triggers the family alert.

**Why one classifier, not two:** an earlier version had separate message and call models. The message model had only ~140 unique training rows and was measurably the weakest part of the system -- it false-positived on an ordinary family message about a bank statement at 93.7% confidence. SMS, WhatsApp and call text are similar enough in kind that pooling every source into one model turns 140 rows into tens of thousands, and lets short-message classification benefit from the entire call corpus.

**Language scope:** English and romanised Hinglish only. Bengali and Tamil were removed from the app, because no Bengali or Tamil text exists anywhere in the training data -- shipping those UI languages implied a level of protection the model could not deliver.

---

## 2. Repository Structure

The repository (`scam_bust_ktj_challenge`) is split into two primary components:

### A. `scam_shield_app/` (The Frontend Client)
This directory contains the **Flutter (Dart)** application. It is cross-platform but primarily targets mobile devices (Android).
- **Key File:** `lib/main.dart` contains the main user interface and application logic.
- **Role:** Handles the user interface, speech-to-text conversion, hardware interactions (microphone, haptic feedback/vibration), local notifications, and HTTP communication with the backend intelligence server.
- There is no client-facing mode toggle -- the app always just asks the server, and the server's cascade decides whether local inference was sufficient or an LLM opinion was needed.

### B. `scambust/` (The Backend Intelligence Engine)
This directory is split into two parts:
- **`scambust/dataset/`** -- the Django REST API server.
  - `api_server.py`: thin entry point (prints LAN IPs, starts `manage.py runserver`).
  - `manage.py`: standard Django management entry point.
  - `backend/`: the actual Django app --
    - `settings.py` -- Django + Postgres configuration (reads `.env`).
    - `models.py` -- `SpamNumber` (crowdsourced/known spam number registry) and `ScanLog` (records every cascade decision: Tier-1 verdict + confidence, whether it escalated, Tier-2 verdict, final verdict).
    - `views.py` -- the three HTTP endpoints (`/predict`, `/check_number`, `/analyze_call`).
    - `ml_inference.py` -- the Tier-1 -> Tier-2 cascade logic itself.
    - `management/commands/seed_spam_numbers.py` -- seeds the demo spam numbers into Postgres.
  - `train_scam_shield.py`: **legacy** TF-IDF + LogisticRegression/RandomForest training script, kept only for historical reference / the baseline comparison script. No longer used in production inference.
- **`scambust/ml/`** -- the MuRIL fine-tuning pipeline (separate from the Django app, own Python venv at `scambust/.venv/`).
  - `data_prep/`
    - `prepare_legit_data.py` -- real human corpora (UCI SMS, CMU Hinglish, legit banking calls, real Indian SMS)
    - `generate_novel_domains.py` -- Groq-primary / Ollama-fallback generation for scam domains outside ICFD's 10
    - `generate_indian_crossdomain_eval.py` -- Indian-context unseen-domain evaluation set
    - `prepare_call_data.py` -- ICFD-31k streaming + local CSVs
    - `prepare_unified_data.py` -- "use everything" build (31,499 rows)
    - **`prepare_balanced_data.py` -- real/Indian-weighted build (10,008 rows). This is the current production dataset.**
  - `training/`
    - `common.py` -- dataset class, metrics, gradual-unfreeze callback, temperature fitting, ECE
    - `train_unified_classifier.py` -- the production trainer (`--seed`, `--data-prefix`, `--out-name`)
    - `recalibrate_and_report.py` -- post-hoc calibration + the full evaluation table
    - `build_ood_detector.py` -- the OOD experiment (negative result; not wired in)
    - `eval_cascade_and_tier2.py`, `eval_indian_crossdomain.py`, `eval_all_variants_indian.py`, `eval_cross_domain.py`, `compare_baseline.py`
  - `models_out/` -- trained artifacts + JSON results. **`balanced_classifier/` is in production**; `unified_classifier*/`, `call_classifier*/` and `message_classifier/` are kept for the historical comparisons in Section 5. Gitignored (large / regenerable).

---

## 3. Technology Stack Breakdown

### Frontend (Mobile App)
- **Framework:** Flutter (Dart language).
- **State & UI:** Standard Flutter widgets, Material Design principles with high-contrast, senior-centric design.
- **Key Packages (`pubspec.yaml`):**
  - `http`: For making API calls to the Django backend.
  - `speech_to_text`: For real-time on-device voice transcription during live calls.
  - `vibration`: For haptic feedback, alerting the user silently.
  - `url_launcher`: To trigger automatic SMS alerts to trusted family contacts.
  - `permission_handler`: To manage critical permissions like Microphone.
  - `shared_preferences`: To store user settings (trusted contact).
- **Localization:** Custom dictionary (`_AppTranslations` in `main.dart`) supporting English and Hindi/Hinglish only (see language-scope note in Section 1).

### Backend (Intelligence Server)
- **Framework:** Django, backed by **PostgreSQL** (`backend/settings.py`). Run `python manage.py migrate` before first use.
- **AI / LLM Integration (Tier 2):** `openai` Python client pointed at Groq's API (`https://api.groq.com/openai/v1`) using `openai/gpt-oss-120b`, with structured JSON output (`response_format={"type": "json_object"}`) rather than regex-scraping free text, at **`temperature=0.0`**. **Two notes, both from real failures:** (1) Groq retired the entire Llama-3 family; the original `llama-3.3-70b-versatile` now returns 404, which silently broke Tier-2 escalation until it was caught -- the model id is overridable via `GROQ_MODEL` in `.env`. (2) Temperature was originally 0.1, at which the LLM flip-flopped between runs on borderline financial messages, making the `suspicious` verdict itself non-deterministic. Classification is not generation; sampling buys nothing.
- **Measured Tier-2 standalone performance** (139 Indian unseen-domain rows): accuracy 0.842, precision 0.741, **recall 0.984**, F1 0.845, at 2.53 s/row -- it misses 1 scam in 61.
- **Machine Learning (Tier 1):**
  - **Libraries:** `transformers`, `torch`, `datasets`, `scikit-learn` (for metrics/splits), `accelerate`.
  - **Models:** `google/muril-base-cased` fine-tuned into one unified binary classifier -- see Section 5.
  - **Environment Variables:** `python-dotenv` loads `GROQ_API_KEY` and Postgres credentials from `scambust/dataset/.env` (see `.env.example`). Note: env vars already set at the OS level take precedence over `.env` (`python-dotenv` doesn't override existing vars).

---

## 4. How the Systems Interact (The Workflow)

### Workflow 1: Message Shield (SMS / WhatsApp)
1. The user pastes a suspicious message and taps "CHECK SAFETY" in the Flutter app.
2. The app POSTs to `/predict` on the Django server.
3. **Cascade:**
   - The unified classifier (Tier 1, MuRIL) scores it locally and returns a calibrated confidence.
   - Escalate to Tier 2 if confidence < `TIER1_CONFIDENCE_THRESHOLD` (0.95) **or** if Tier 1 says "scam" at all.
   - Otherwise the local verdict is final -- no network call to any LLM.
   - If the two tiers disagree, the verdict becomes `suspicious`.
   - Every decision (Tier-1 verdict/confidence, whether it escalated, Tier-2 verdict, final verdict) is persisted to `ScanLog` in Postgres.
4. The result (`scam` / `suspicious` / `safe`) plus a plain-language reason is returned. The API also still returns the legacy `is_scam` boolean for older app builds.
5. **Only on `scam`**, and only if a trusted contact is saved, the app prepares an SMS alert via `url_launcher`. A `suspicious` verdict deliberately does *not* alert relatives -- that would cry wolf on genuinely ambiguous, often legitimate, financial messages.

### Workflow 2: Live Call Shield
1. The user activates "START LIVE MONITOR" in the app before or during a phone call.
2. The Flutter app uses `speech_to_text` to continuously listen and transcribe the live conversation.
3. Transcript chunks are periodically POSTed to `/analyze_call`.
4. **Analysis pipeline (fastest-to-slowest):**
   - A keyword pre-filter (`otp`, `cvv`, `police`, `arrest`, `cbi`, ...) runs first -- near-zero latency, catches the obvious cases immediately.
   - If no keyword hits, the same unified classifier (Tier 1, MuRIL) scores the transcript -- `channel` no longer selects a model.
   - Escalation to Groq (Tier 2) follows the same rule as the message flow.
5. Haptics differ by verdict: `scam` -> `vibrate_strong`, `suspicious` -> `vibrate_gentle` (warn without asserting fraud), `safe` -> nothing.

> **Important limitation on this workflow.** On stock Android the app **cannot hear the caller**. `MediaRecorder.AudioSource.VOICE_CALL` needs a signature-level permission held only by platform-signed / OEM-preinstalled apps; Google's May 2022 Play policy banned `AccessibilityService` for call recording; and Android 10's `AudioPlaybackCapture` excludes voice-call audio. `speech_to_text` therefore captures the **device microphone**, which on a normal earpiece call picks up mostly the *user's* voice. The only viable workaround is forcing speakerphone (`MODIFY_AUDIO_SETTINGS`, now declared in the manifest). **No end-to-end test from live audio to verdict has ever been run** -- this is the largest untested assumption in the system.

### Workflow 3: Call Number Check
1. `/check_number` looks up the incoming phone number against the `SpamNumber` Postgres table (replacing the old hardcoded in-memory list) and returns whether it's known spam plus a message.

---

## 5. Machine Learning: One Unified MuRIL Classifier (`scambust/ml/`)

The old TF-IDF + LogisticRegression/RandomForest ensemble (`train_scam_shield.py`) has been replaced by a single fine-tuned `google/muril-base-cased` binary classifier serving every channel.

### Data sources (`ml/data_prep/`)

Rows shown are the contribution to the current 10,008-row production build.

| Source | Real? | Indian? | Rows | Scam |
|---|---|---|---|---|
| `rishia2220/icfd-31k` (HF) | No | No | 3,750 | 1,500 |
| `ucirvine/sms_spam` (HF) | **Yes** | No | ~1,538 | 190 |
| `talkmap/banking-conversation-corpus` (HF) | **Yes** | No | ~1,462 | 0 |
| **Indian crowdsourced SMS ham** (GitHub, princebari) | **Yes** | **Yes** | 986 | 0 |
| `India_Cyber_Scam_Hinglish_Dataset.csv` | No (templated) | **Yes** | 743 | 633 |
| **Indian marketing SMS** (same GitHub corpus) | **Yes** | **Yes** | 698 | 0 |
| `generate_novel_domains.py` (Groq/Ollama) | Generated | **Yes** | 289 | 155 |
| `festvox/cmu_hinglish_dog` (HF) | **Yes** | **Yes** | 256 | 0 |
| scambust `public_{sms,whatsapp,calls,audio_transcripts}.csv` | No | **Yes** | 286 | 237 |

> **Structural constraint.** *Every* real Indian corpus available is **legitimate**. No real Indian scam text was found in any public source. The positive class is therefore entirely synthetic or LLM-generated regardless of how the mix is rebalanced -- roughly 38% Indian is the practical ceiling until real Indian fraud data exists. Acquiring it is the single highest-value thing that could be added.

Decisions driven by measured failures rather than intuition:

1. **Real legitimate data was added** because the model had seen far more fraud than normal life (the Hinglish CSV contains only ~110 distinct "safe" templates), and it false-positived on an ordinary family message about a bank statement. The legitimate *banking* corpus matters most: real conversations about payments and account access are exactly the shape most likely to trip a fraud detector.
2. **UCI's spam half is included, not just its ham.** If a source's label were perfectly predictable from the source itself, the model could learn writing style instead of fraud semantics -- the same shortcut that template repetition already caused once.
3. **ICFD-31k is capped at one chunk per conversation.** Expanding it from ~21k to ~38k rows made cross-domain generalisation *worse* (recall 0.459 -> 0.412). Capacity is better spent on new, diverse sources than on more of the same 10 domains.
4. **Real Indian ham was the single biggest win.** Adding 925 real Indian messages moved cross-domain recall 0.427 -> 0.478 (~13 sigma against a ±0.004 seed noise floor) -- further than 18,000 synthetic rows had.
5. **Indian marketing SMS is labelled `0`, not `1`.** It is real, Indian, legitimate commercial messaging. Labelling it fraud would teach the model that "reply within 24 hrs to win" equals scam -- see the false-positive section below.

Note: ICFD-31k's own `validation`/`test` splits are 100% fraud-scenario conversations (verified via the HF `datasets-server` statistics endpoint), so they serve only as a fraud-recall stress test. The primary train/val/test split is carved from the realistically-balanced pool.

### Training approach
- Gradual unfreezing (ULMFiT-style): epoch 0 trains only the pooler + classification head, then one or two more encoder layers thaw per epoch, top-down. Implementation note: HF's `Trainer` builds its optimizer parameter groups **once**, filtered by `requires_grad`, so the full eventual trainable range must be registered *before* training starts or later-unfrozen layers silently never update.
- Class-weighted cross-entropy (the balanced build is ~73% legit / 27% scam).
- Confidence calibrated by temperature scaling (Guo et al. 2017) on a held-out split, so the Tier-1/Tier-2 escalation threshold means what it says.

### The rebalance: quality beat quantity (`prepare_balanced_data.py`)

The 31,499-row "use everything" build was only ~21% real and ~7% genuinely Indian; ICFD-31k alone supplied 63%. Since the single largest measured gain of the project came from adding just 925 real Indian messages, the mix was rebuilt to prioritise real and Indian data even at a much smaller row count.

**Result: 10,008 rows -- 49.4% real, 32.6% Indian context, 27.1% scam** (from 31,499 / 21% / 7%).

Built in priority tiers: (1) all real Indian data, never subsampled; (2) all Indian-context scam data, since those are the only positives available; (3) real non-Indian capped at 3,000; (4) ICFD topped up to hit the class ratio, dropping from 19,752 rows to 3,750.

> **A bug worth knowing about.** The first attempt used 1,500 ICFD scam + 750 ICFD legit, inverting ICFD's natural 72/28 lean. Because every other legit source is short-form, that made **transcript length correlate with the label** -- long conversation => scam, short SMS => safe. The model learned the shortcut and flagged 62.6% of a 21.9%-scam test set. ICFD is the only supplier of legitimate long conversations and must not be starved; the ratio is now 1,500 scam + 2,250 legit.

### Results of the production model (`models_out/balanced_classifier/calibration.json`)

Seed 42, temperature 0.6371:

| Evaluation set | n | Acc | Prec | Rec | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| In-distribution test | 401 | 0.913 | 0.798 | 0.908 | 0.850 | 0.963 |
| Held-out synthetic | 206 | 0.976 | 0.977 | 0.994 | 0.986 | 0.989 |
| Generated unseen domains | 170 | 0.912 | 0.902 | 0.973 | 0.936 | 0.951 |
| ICFD cross-domain (US-flavoured) | 1,000 | 0.643 | 0.964 | 0.648 | 0.775 | 0.682 |
| Indian unseen domains | 139 | 0.691 | 0.598 | 0.902 | 0.719 | 0.837 |
| Real marketing FP probe | 300 | 0.893 | -- | -- | -- | n/a |
| ICFD fraud-only stress test | 4,499 | 0.878 | 0.989 | 0.886 | 0.935 | 0.585 |

**Balanced (10k) vs the previous unified (31.5k) build, on identical eval sets:**

| Evaluation | Unified | Balanced |
|---|---|---|
| Held-out synthetic F1 | 0.980 | **0.986** |
| Generated unseen F1 | 0.876 | **0.936** |
| ICFD cross-domain recall | 0.478 | **0.648** |
| Indian unseen recall | 0.820 | **0.902** |
| **Marketing false-positive rate** | **89.3%** | **10.7%** |

**How to read these:**
- In-distribution **AUC 0.963 against accuracy 0.913** means the model *ranks* well but its decision threshold is placed sub-optimally -- free headroom via threshold tuning, no retraining.
- The fraud-only stress test's strong F1 (0.935) beside **AUC 0.585** shows why F1 alone misleads on a 98.9%-scam set: a near-constant "scam" answer scores well. AUC catches it.
- The marketing probe has no positives, so precision/recall/AUC are undefined by construction; the meaningful figure is its **10.7% false-positive rate**.

### The false-positive discovery that forced the rebalance

A probe of **1,000 real, legitimate Indian commercial SMS** (movie promos, Noida flat ads, quiz contests, coaching adverts) found the then-current model flagged **879 of them as scams -- an 87.9% false-positive rate**. Every prior eval set was scam-heavy or used synthetic legitimate data, so this failure was invisible. An Indian senior receives dozens of such messages weekly; the app would have been uninstalled within a day.

Root cause: those marketing messages had been *deliberately excluded* from training to avoid mislabelling them as fraud. Correct as a diagnostic call, wrong as a training call -- the model had never seen legitimate promotional urgency labelled not-fraud. Feeding 698 of them back in as `label=0` (holding 300 back as the probe) cut the rate to **10.7%**.

### Multi-seed: the noise floor

Three seeds (42, 7, 123) on the 31,499-row build:

| Evaluation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| In-distribution test | 0.982 ±0.001 | 0.937 ±0.002 | 0.986 ±0.004 | 0.961 ±0.002 |
| Held-out synthetic | 0.964 ±0.002 | 0.965 ±0.003 | 0.994 ±0.000 | 0.979 ±0.001 |
| Generated unseen domains | 0.825 ±0.014 | 0.861 ±0.013 | 0.879 ±0.011 | 0.870 ±0.010 |
| ICFD cross-domain | 0.497 ±0.004 | 0.970 ±0.002 | 0.484 ±0.004 | 0.646 ±0.004 |

Seed spread on ICFD cross-domain is only **±0.004**, which retroactively validates the single-seed variant comparisons below (gradual-v1 vs baseline is a ~6σ gap). Only the generated-unseen set is genuinely noisy (±0.014) -- differences under ~0.03 there are meaningless. **Only seed 42 has been run on the balanced build.**

### Calibration

Temperature scaling initially returned **0.05 -- the clamp floor**, a degenerate fit that makes the model *more* confident and silently breaks the escalation threshold. Two fixes: optimise `log(T)` so temperature cannot cross zero and flip logit signs, and reject fits outside [0.25, 5.0] in favour of `T = 1.0`. Current model: **T = 0.6371**, Expected Calibration Error **0.0674 → 0.0132**.

### The cross-domain number was measuring the wrong thing (`eval_indian_crossdomain.py`)

Inspecting the ICFD `cross_domain` samples revealed that it conflates **two** shifts, not one. The five unseen categories are crypto-investment fraud, bank security-question harvesting, fake charity, pyramid/MLM, and tax-refund identity theft -- but several samples reference **"Bank of America", "Social Security number" and "the IRS"**, while every training row is India-centric (UPI, Aadhaar, PAN, RBI). So a low score there could mean "new scam type" *or* "wrong country".

To separate them we generated the **same five scam types in Indian context** (chit funds instead of MLM, PAN/Income-Tax instead of SSN/IRS, temple and flood-relief charities), reusing the identical prompt template as the training generation -- deliberately *not* diverging it, so the question asked is simply "how does the model do if we naturally extend our own pipeline?"

| Metric | ICFD cross-domain (US-flavoured) | Same 5 types, Indian context | Delta |
|---|---|---|---|
| Accuracy | 0.443 | 0.777 | **+0.334** |
| Precision | 0.969 | 0.703 | -0.266 |
| **Recall** | **0.427** | **0.852** | **+0.426** |
| F1 | 0.593 | 0.770 | +0.178 |
| ROC-AUC | 0.689 | 0.863 | +0.174 |

**Recall doubles.** The headline "misses more than half of unseen scams" was substantially a *geography* failure, not a scam-type failure. On novel scam types in the context the app actually serves, the model catches ~85%.

Two things that finding does **not** let us off the hook for:

1. **Precision collapses to 0.703.** Per-domain, legitimate-conversation accuracy is 0.625 for chit-funds and 0.667 for crypto -- the model flags roughly a third of *genuine* conversations about those topics as scams. It has learned "Indian + money topic => probably scam". That is a real false-alarm problem on unfamiliar financial subjects.
2. **The style confound is unresolved.** This set came from the same generator and prompt template as the 273 generated rows in training, so some of the 0.852 may be generator-style recognition rather than true generalisation. Sample is also small (139 rows / 61 scams, so recall carries roughly +/-9%). Treat 0.852 as an optimistic bound and 0.427 as a pessimistic one.

### OOD detection: tested, and it does NOT work here (`build_ood_detector.py`)

The obvious fix for "confidently wrong on unseen domains" is to escalate on *input novelty* rather than on confidence. We built that (Mahalanobis over `[CLS]` embeddings, plus kNN-cosine) and measured it properly: framed as a detection problem where the target is **"did Tier-1 get this wrong?"** and the score is the candidate escalation signal. A signal is only worth shipping if it beats the incumbent, `1 - softmax confidence`.

ROC-AUC for predicting that Tier-1 is wrong:

| Evaluation set | 1 - confidence | Mahalanobis | kNN-cosine |
|---|---|---|---|
| In-distribution test | **0.926** | 0.898 | 0.902 |
| Generated unseen domains | 0.740 | 0.735 | 0.733 |
| **ICFD-31k cross-domain (5 unseen)** | **0.413** | 0.504 | 0.514 |

**The verdict: OOD detection is useless on the case that matters.** On ICFD cross-domain both OOD scores sit at chance (0.504 / 0.514), and the escalation table confirms it — escalating the top 30% by OOD score catches 30.0% of mistakes, exactly what escalating a *random* 30% would catch. Note also that `1 - confidence` scores **0.413**, i.e. *worse than random*: on unseen domains the model is, if anything, slightly more confident when it is wrong.

**Why it fails, and this is the useful part:** OOD detection measures *stylistic* novelty, not *semantic* novelty. The ICFD cross-domain conversations come from the same generator as the training bulk -- same "Agent:/Customer:" format, same length, same synthetic register -- and differ only in what the scam is *about*. In embedding space they are genuinely in-distribution, so no distance metric can flag them. The generated-holdout column is the control that proves this: that set *is* stylistically different (different LLM, different prompt scaffolding), and there OOD works fine (0.735) -- but so does plain confidence (0.740), so it still adds nothing.

Practical consequence: **we cannot currently triage which inputs need Tier-2.** Any strategy that depends on identifying "hard" cases at Tier-1 is blocked. What remains is either escalating broadly where latency permits, or ensemble disagreement (which measures decision-boundary uncertainty rather than input novelty, and is therefore a genuinely different mechanism -- untested).

The detector is saved (`models_out/unified_classifier/ood_detector.npz`) and the full numbers are in `models_out/ood_evaluation.json`, but it is **not wired into the backend**, because the measurement says it would not help.

### What we learned from the fine-tuning experiments

Four training variants were compared on 1,000 held-out ICFD-31k conversations from **five domains absent from training**:

| Variant | In-domain F1 | Cross-domain recall |
|---|---|---|
| Near-full fine-tune (baseline) | 0.982 | 0.431 |
| Static partial freeze (bottom 6 layers) | 0.974 | 0.332 |
| Gradual unfreeze, 4 epochs / ~20k rows | 0.941 | **0.459** |
| Gradual unfreeze, 8 epochs / ~38k rows | 0.987 | 0.412 |

The conclusion is uncomfortable but clear: **every change that improved in-domain fit made cross-domain generalisation worse.** Layer-freezing depth, schedule gradualness and raw data volume are all exhausted as levers. That is what motivated the data-centric work above. (Caveat: these are single-seed runs; differences of a few points may be noise. `train_unified_classifier.py` takes a `--seed` flag so results can be repeated and averaged.)

Against the *old* TF-IDF+LR/RF model on the same test set: 29.6% accuracy/precision -- precision equalling the test set's scam prevalence almost exactly, revealing it had collapsed into a near-constant "always scam" classifier outside its narrow training vocabulary.

### Known limitations
- **Cross-domain recall is the open problem, and we do not currently have a fix.** On genuinely unseen scam categories the model misses over half of them, and worse, it is *confidently* wrong: 93.7% of missed scams scored above the escalation threshold (mean confidence 0.983), so the LLM safety net almost never engaged. Three routes have now been ruled out **by measurement**: raising the threshold (0.95 recovers only 5.5% of misses), fine-tuning changes (four variants, all in a 33-46% recall band), and OOD-based escalation (at chance -- see the OOD subsection above). The only untested mechanism left is **ensemble disagreement**, which measures decision-boundary uncertainty rather than input novelty and so is not refuted by the OOD result. The practical workaround is to escalate broadly on channels that can afford the latency (SMS) rather than trying to triage.
- **The audio path is untested.** Training text is clean, speaker-labelled transcripts; production input is raw phone speech-to-text output. No end-to-end test from live audio to verdict has been run.
- **Almost all scam data is synthetic.** No real scam call transcript has ever been seen by this model.

---

## 6. Summary for LLMs Context

If you are an AI reading this codebase:
- **To modify the UI or Mobile Logic:** Look into `scam_shield_app/lib/main.dart`. There is no mode toggle anymore -- don't reintroduce one; the cascade on the server handles online/offline transparently.
- **To modify the API or persistence:** Look into `scambust/dataset/backend/` (`views.py`, `models.py`, `ml_inference.py`). Any new endpoint logic should log to `ScanLog` the same way `/predict` and `/analyze_call` do.
- **To modify the ML Logic:** Look into `scambust/ml/`. The live pipeline is `data_prep/prepare_balanced_data.py` -> `training/train_unified_classifier.py --data-prefix balanced --out-name balanced_classifier` -> `models_out/balanced_classifier/`. There is ONE model now -- don't reintroduce a separate message classifier; that split is what starved it of data. `train_scam_shield.py`, the old `.pkl` files, and `train_call_classifier*.py` / `train_message_classifier.py` are kept only for historical comparison -- do not wire them back into `api_server.py`.
- **Which model the server loads:** `MODEL_SEARCH_PATH` in `ml_inference.py`, tried in order -- `balanced_classifier`, then `unified_classifier`, then `call_classifier`. First present wins, and it prints which one it loaded.
- **Confidence threshold:** `TIER1_CONFIDENCE_THRESHOLD` in `ml_inference.py` (default **0.95**, env-overridable). **Do not attempt to fix cross-domain misses by raising it** -- measured, and it does not work (0.95 recovers only 5.5% of confident misses; even 0.99 recovers 51% while escalating nearly everything). Note escalation is also asymmetric: any scam prediction escalates regardless of confidence.
- **Do not re-attempt OOD-based escalation** without reading Section 5 first. Mahalanobis and kNN-cosine detectors were both built and measured; both score at chance (0.504 / 0.514 AUC) on the unseen-domain set, because those inputs are stylistically identical to training data and differ only in scam topic. The artefacts exist but are deliberately not wired in.
- **Before trusting any new result:** run more than one seed. The measured noise floor is ±0.004 on ICFD cross-domain but ±0.014 on the generated-unseen set -- differences under ~0.03 on the latter are meaningless. Only seed 42 exists for the balanced build.
- **Adding data beats tuning training.** Four fine-tuning variants all landed in a 33-46% cross-domain recall band. Composition moved the needle; hyperparameters did not. 10k well-composed rows beat 31.5k poorly-composed ones, and 925 real Indian rows beat 18,000 synthetic ones.
- **Always evaluate on the marketing probe.** It is the only set containing real, legitimate, promotional text, and it exposed an 87.9% false-positive rate that every other evaluation missed. It is wired into the standard eval loop -- don't remove it.
- **Beware metrics on scam-saturated sets.** The fraud-only stress test yields F1 0.935 with AUC 0.585. If precision ≈ the set's scam prevalence, you are looking at a near-constant classifier, not a good one. This pattern caught both the legacy TF-IDF model and a flawed rebalance.

---

## 7. Operational Handoff

Everything needed to actually run this, plus the environment quirks that will otherwise cost an hour to rediscover.

### Running things

There is one virtualenv at `scambust/.venv`. ML scripts are Python **modules**, so they must be run with `-m` **from the `scambust/` directory** (not from `ml/`, and not as file paths):

```bash
cd scambust
# --- data ---
./.venv/Scripts/python.exe -u -m ml.data_prep.prepare_legit_data          # real human corpora (+ marketing probe)
./.venv/Scripts/python.exe -u -m ml.data_prep.generate_novel_domains N    # generate N novel-domain rows (resumable)
./.venv/Scripts/python.exe -u -m ml.data_prep.prepare_balanced_data       # PRODUCTION build -> balanced_{train,val,test}.csv
./.venv/Scripts/python.exe -u -m ml.data_prep.prepare_unified_data        # legacy "use everything" build

# --- training (production command) ---
./.venv/Scripts/python.exe -u -m ml.training.train_unified_classifier \
    --seed 42 --data-prefix balanced --out-name balanced_classifier

# --- evaluation ---
./.venv/Scripts/python.exe -u -m ml.training.recalibrate_and_report ml/models_out/balanced_classifier balanced
./.venv/Scripts/python.exe -u -m ml.training.eval_cascade_and_tier2       # threshold sweep + Tier-2 + cascade
./.venv/Scripts/python.exe -u -m ml.training.eval_all_variants_indian     # every variant on the Indian set
./.venv/Scripts/python.exe -u -m ml.training.build_ood_detector           # OOD experiment (negative result)
```

**Run one training process at a time.** Three sequential runs in a single chained command produced segfaults (exit 139) after training completed, during the eval phase -- fresh processes per seed are reliable. Also avoid piping long runs through `head`, which can SIGPIPE-kill training mid-flight; write to a log file instead.

The Django server runs from `scambust/dataset/`:

```bash
cd scambust/dataset
../.venv/Scripts/python.exe manage.py migrate
../.venv/Scripts/python.exe manage.py seed_spam_numbers
../.venv/Scripts/python.exe api_server.py        # serves on 0.0.0.0:8000, prints LAN IPs
```

### Environment quirks (each of these actually bit us)

- **Postgres is NOT running as a Windows service.** `Get-Service postgresql-x64-17` reports *Stopped* and cannot be started (access denied), yet the DB is up because it was launched manually. After a reboot you must restart it yourself:
  ```bash
  "/c/Program Files/PostgreSQL/17/bin/pg_ctl.exe" start -D "/c/Program Files/PostgreSQL/17/data" -l "/c/Program Files/PostgreSQL/17/data/current.log"
  ```
  Symptom if you forget: Django dies with a connection-refused error.
- **DB credentials are gitignored.** `scambust/dataset/.env` holds the Postgres password and `GROQ_API_KEY`; a fresh clone has neither. Copy `.env.example` and refill. DB `scamshield`, role `scamshield_app`.
- **`GROQ_API_KEY` also exists as an OS-level environment variable**, and `python-dotenv` does **not** override already-set env vars -- so the OS one wins over `.env`. Confusing if you edit `.env` and nothing changes.
- **`transformers` is 5.15.0**, which is a major version with breaking changes: `TrainingArguments(warmup_ratio=...)` no longer exists (use `warmup_steps`), and `Trainer(tokenizer=...)` is now `processing_class=`. Pin or check before copying recipes from older tutorials.
- **Always run Python with `-u`.** Without it, stdout buffers and long background jobs appear to produce nothing for their entire duration.
- **Decode Ollama responses explicitly as UTF-8.** `requests`' charset guess mangles smart quotes into U+FFFD, which then becomes a spurious signal in training text. `generate_novel_domains.py` handles this; anything new talking to Ollama must too.
- **Only run one generator process at a time.** The script rewrites the whole CSV on each save, so two concurrent processes silently clobber each other's rows.
- **GPU is 6GB.** Training uses ~5.2GB. Do not run Ollama (qwen2.5:7b needs ~4.7GB) and training simultaneously.

### Generation backends

`generate_novel_domains.py` tries **Groq (`openai/gpt-oss-120b`) first**, falling back to **local Ollama (`qwen2.5:7b`)** on any error. Measured: Groq ~2s/call vs ~60s/call locally, with better Hinglish. Observed Groq success rate ~88%; the rest fall back automatically. Output is appended incrementally, so the script is safely resumable after an interrupt.

### Current repository state

- Branch **`main`**, with **13 uncommitted paths**. Nothing from this work has been committed yet. **Branch before committing** rather than committing directly to `main`.
  - Modified: `.gitignore`, `README.md`, `scam_shield_app/lib/main.dart`, `scam_shield_app/android/app/src/main/AndroidManifest.xml`, `scambust/dataset/api_server.py`
  - New: `PROJECT_EXPLANATION.md`, `TECHNICAL_DEEP_DIVE.md`, `PROGRESS_AND_FINDINGS.md`, `scambust/dataset/backend/`, `scambust/dataset/manage.py`, `scambust/dataset/.env.example`, `scambust/ml/`, `scambust/requirements.txt`
- Gitignored and therefore absent from a fresh clone: `scambust/ml/data/`, `scambust/ml/models_out/`, `scambust/dataset/.env`. All of `ml/data/` and `ml/models_out/` is regenerable from the scripts above (ICFD-31k re-streams from HF and caches locally; expect ~15-20 min for the first full pass).

### Where the work stopped

The last completed steps were: rebalancing the dataset toward real/Indian data (10,008 rows), training and calibrating `balanced_classifier`, pointing the backend at it, and writing `TECHNICAL_DEEP_DIVE.md` + `PROGRESS_AND_FINDINGS.md`.

**Open items, roughly by value:**

1. **Multi-seed the balanced build.** Only seed 42 exists. Seeds 7 and 123 segfaulted after training completed; re-run them as separate processes.
2. **The audio path** -- still the single largest untested assumption. Platform limits are now mapped (Section 4): Android blocks call-audio capture entirely, so forcing speakerphone is the only real-time option, and `CallScreeningService` is the supported way to do pre-call number screening. **No live-audio-to-verdict test has ever been run.**
3. **Acquire real Indian scam text.** Every real Indian corpus we have is legitimate; the positive class is 100% synthetic. This is the binding constraint on further data work.
4. **Ensemble disagreement** -- the one untested uncertainty mechanism left. It measures decision-boundary uncertainty rather than input novelty, so it is *not* refuted by the OOD result.
5. **Threshold tuning on the balanced model.** In-distribution AUC 0.963 against accuracy 0.913 says the operating point is sub-optimal -- free gains, no retraining.
6. **Production hardening** -- no endpoint auth, plain HTTP, `DEBUG=True`, permissive CORS, Django dev server, synchronous model load at import.

**A decision that was deliberately left open:** whether to make SMS **LLM-first**. Tier-2 alone scores 0.984 recall at ~2.5 s/row, and SMS is user-initiated and latency-tolerant, so it would sidestep the unsolved triage problem for half the product. The cost is that message content leaves the device, which cuts against the "privacy-first" positioning. Live-call monitoring should stay local-first regardless -- continuous chunks make per-chunk LLM calls impractical.
