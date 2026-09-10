# Scam Shield AI — Technical Deep Dive

A complete technical description of what this system does and how every part of it works.

For the story of *how it got here* — the experiments, failures and measurements that produced these design choices — see `PROGRESS_AND_FINDINGS.md`. For operational instructions (run commands, environment quirks), see Section 7 of `PROJECT_EXPLANATION.md`.

---

## 1. What the product does

Scam Shield AI protects Indian senior citizens from phone and message fraud. Seniors are disproportionately targeted by "digital arrest" scams, fake KYC calls, courier-customs fraud and similar social-engineering attacks, and the existing defences (blocklists, keyword filters) fail against attackers who simply rephrase.

The system provides three user-facing capabilities:

| Capability | Trigger | What happens |
|---|---|---|
| **Message Shield** | User pastes a suspicious SMS/WhatsApp text and taps "CHECK SAFETY" | Text is classified; user sees a verdict plus a plain-language reason in English or Hinglish |
| **Live Call Monitor** | User starts monitoring during a call | On-device speech-to-text transcribes continuously; chunks are analysed; the phone vibrates distinctively on danger |
| **Caller Check** | Incoming number entered/simulated | Number is looked up against a known-spam registry in Postgres |

On a scam verdict, if the user has saved a trusted contact, the app automatically composes an SMS alert to that family member describing the message and why it was flagged.

**Design constraint that shapes everything:** the user is a senior citizen who cannot be asked to make technical decisions. There is no "online/offline mode" switch, no model picker, no confidence slider. The system decides internally and shows one of three plain answers.

---

## 2. System architecture

```
┌─────────────────────────────┐
│  Flutter app (Android)      │
│  • speech_to_text (ASR)     │
│  • vibration (haptics)      │
│  • url_launcher (SMS alert) │
│  • shared_preferences       │
└──────────────┬──────────────┘
               │ HTTP JSON  (same-WiFi LAN)
               ▼
┌─────────────────────────────────────────────┐
│  Django REST server (api_server.py)         │
│                                             │
│  /predict        /analyze_call   /check_number
│       │                │                │   │
│       ▼                ▼                ▼   │
│  ┌──────────────────────────┐   ┌──────────┐│
│  │  Keyword pre-filter      │   │ Postgres ││
│  │  (calls only, ~0 ms)     │   │SpamNumber││
│  └───────────┬──────────────┘   └──────────┘│
│              ▼                               │
│  ┌──────────────────────────┐                │
│  │ TIER 1  fine-tuned MuRIL │  local, fast   │
│  │ binary classifier        │                │
│  └───────────┬──────────────┘                │
│              │ low confidence OR predicts scam│
│              ▼                               │
│  ┌──────────────────────────┐                │
│  │ TIER 2  cloud LLM (Groq) │  ~2.5 s        │
│  │ openai/gpt-oss-120b      │                │
│  └───────────┬──────────────┘                │
│              ▼                               │
│      verdict: scam / suspicious / safe       │
│              │                               │
│              ▼                               │
│      ScanLog row written to Postgres         │
└─────────────────────────────────────────────┘
```

### Why a two-tier cascade

A single cloud LLM would be more accurate but adds ~2.5 s per request, requires connectivity, and sends every private message off-device. A single local model is fast and private but weaker. The cascade runs the local model always and consults the LLM only when it matters — giving most requests local speed while retaining LLM accuracy on the hard cases.

Crucially, the cascade is **invisible to the user**. An earlier version exposed an "Auto / Online / Offline" dropdown, which contradicted the product's own "Zero-UI" premise; it was removed.

---

## 3. Tier 1 — the local classifier

### Model

`google/muril-base-cased` — a BERT-architecture encoder (12 layers, 768 hidden, ~236M parameters) pre-trained by Google on 17 Indian languages **including transliterated/romanised text**. That last property is why it was chosen over vanilla BERT or XLM-R: the target users write Hinglish in Latin script ("Aapka KYC pending hai, turant OTP bhejein"), which is exactly what MuRIL's transliterated pre-training covers.

A binary classification head (2 classes: not-scam / scam) sits on the pooled `[CLS]` representation.

### One model, not two

An earlier design used **two** classifiers — one for SMS/WhatsApp, one for call transcripts — on the reasoning that short messages and long conversations are different data shapes. This was abandoned. The message model had only ~140 unique training rows, making it by far the weakest component; it false-positived on an ordinary family message about a bank statement at 93.7% confidence. Pooling every source into one model turned 140 rows into tens of thousands.

`channel` (`"sms"` / `"call"`) is still passed through the API, but it now only selects the Tier-2 prompt wording and tags the `ScanLog` row. It does **not** select a model.

### Fine-tuning procedure

**Gradual unfreezing (ULMFiT-style).** Training runs 4 epochs with a schedule `[12, 10, 8, 6]` meaning "minimum trainable encoder layer per epoch":

| Epoch | Trainable | Params trained |
|---|---|---|
| 0 | pooler + classifier head only | 592,130 (0.2%) |
| 1 | + encoder layers 10–11 | 14,767,874 (6.2%) |
| 2 | + encoder layers 8–11 | 28,943,618 (12.2%) |
| 3 | + encoder layers 6–11 | 43,119,362 (18.2%) |

The embedding layer and encoder layers 0–5 stay frozen throughout.

> **Implementation trap.** HuggingFace's `Trainer` builds its optimizer parameter groups **once**, filtered by `requires_grad` at construction time. If only the head is marked trainable at that moment, layers unfrozen later by the callback compute gradients but are never updated — training silently does nothing for them. The fix is to register the *full eventual* trainable range before `Trainer` is constructed, then have the callback restrict/release within that already-registered set.

**Other training settings:**
- Class-weighted cross-entropy (`sklearn.utils.class_weight.compute_class_weight("balanced")`), since data is ~73/27 legit/scam
- Learning rate 2e-5, weight decay 0.01, warmup ≈10% of steps
- Batch size 8 with gradient accumulation 4 (effective batch 32) — sized for a 6 GB RTX 4050
- FP16 mixed precision
- `max_length` 256 tokens, padded
- Early stopping on validation F1, patience 2

### Confidence calibration

Raw softmax outputs are overconfident, which matters because the escalation threshold is expressed in confidence units. A single scalar **temperature** `T` is fitted post-hoc on held-out validation logits by minimising NLL (Guo et al., 2017), and inference uses `softmax(logits / T)`.

Two hardening details, both added after a real failure:

1. **Optimise `log(T)`, not `T`.** The unconstrained parameter can otherwise cross zero and flip the sign of every logit.
2. **Reject degenerate fits.** If the calibration set is small and the model separates it nearly perfectly, NLL is genuinely minimised as `T → 0` (infinite confidence). That is overfitting the calibration set, not calibrating. Fits outside `[0.25, 5.0]` are rejected in favour of `T = 1.0`, with a printed warning.

Current production model: **T = 0.6371**, which reduced Expected Calibration Error from **0.0674 → 0.0132**.

---

## 4. Tier 2 — the LLM escalation

**Model:** `openai/gpt-oss-120b` via Groq's OpenAI-compatible API.

> Groq **retired the entire Llama-3 family**. The original `llama-3.3-70b-versatile` now returns HTTP 404, which silently broke Tier-2 escalation — the cascade degraded to Tier-1-only with no error surfaced to the user. The model ID is now overridable via `GROQ_MODEL` in `.env` so the next retirement is a config change.

**Request shape:** structured JSON output (`response_format={"type": "json_object"}`) rather than regex-scraping free text, and **`temperature=0.0`**. Temperature was originally 0.1; on genuinely borderline financial messages the model flip-flopped between runs, making the verdict non-deterministic. Classification is not generation — sampling buys nothing here.

**Measured standalone performance** (139 Indian unseen-domain rows): accuracy 0.842, precision 0.741, **recall 0.984**, F1 0.845, at **2.53 s/row**. It misses 1 scam in 61.

### Escalation policy

```python
needs_escalation = (
    tier1_label is None                      # model unavailable
    or tier1_confidence < CONFIDENCE_THRESHOLD   # default 0.95
    or tier1_label is True                   # ANY positive scam call
)
```

The threshold is 0.95 (raised from 0.85 on measured evidence). The third clause — **escalate every scam prediction regardless of confidence** — exists for two reasons:

- Tier-1 is *confidently* wrong on ambiguous financial text. It rated a genuine chit-fund instalment reminder **0.987 scam**, far above any workable threshold, so confidence alone can never catch these.
- A scam verdict **auto-sends an SMS to the user's family contact**. That is the consequential branch and warrants a second opinion.

Scam predictions are the minority of traffic, which bounds the extra cost.

---

## 5. The three-way verdict

Output is `scam` / `suspicious` / `safe`, not a boolean.

```python
if escalated and tier1_label is not None and tier2_label is not None and tier1_label != tier2_label:
    verdict = "suspicious"     # two independent models disagree
else:
    verdict = "scam" if final_label else "safe"
```

**Why "suspicious" exists.** A real chit-fund instalment reminder and a chit-fund scam are near-identical in text. Inspection of false positives showed messages labelled "legitimate" that contained referral-reward-for-recruiting mechanics, payment deadlines and penalty threats — and **both** our classifier and the LLM flagged them. When two independent strong models agree on a "mistake", the label is arguable, not the model. Forcing a binary answer on such text is dishonest; "verify this independently" is the correct advice.

**Behavioural consequences:**
- `scam` → red alert + strong haptic pattern + **family SMS auto-alert**
- `suspicious` → amber "BE CAREFUL" + gentle haptic + advice to verify; **no family alert** (alerting relatives about ambiguous, often-legitimate financial messages would cry wolf)
- `safe` → green, no action

API responses retain `is_scam` (boolean) for backward compatibility with older app builds, and add `verdict`.

---

## 6. Backend implementation

### Stack
- **Django 5.2.17** + **PostgreSQL 17** (`psycopg2-binary`)
- **PyTorch 2.6.0+cu124**, **transformers 5.15.0**, CUDA on an RTX 4050 (6 GB)
- `openai` client pointed at Groq
- `python-dotenv` for secrets

### Layout
```
scambust/dataset/
  api_server.py     thin entry point; prints LAN IPs, starts runserver
  manage.py         standard Django entry point
  backend/
    settings.py     Django + Postgres config, reads .env
    models.py       SpamNumber, ScanLog
    views.py        /predict, /analyze_call, /check_number, /
    ml_inference.py the Tier-1 → Tier-2 cascade
    management/commands/seed_spam_numbers.py
```

### Data model

**`SpamNumber`** — `phone` (unique, indexed), `report_count`, `label`, `first_reported_at`, `last_reported_at`. Replaces a hardcoded in-memory Python list that reset on every restart.

**`ScanLog`** — records *every* cascade decision:

| Field | Purpose |
|---|---|
| `channel` | sms / call |
| `text_snippet` | first 200 chars only |
| `tier1_label`, `tier1_confidence` | local model's answer |
| `escalated_to_llm` | did Tier-2 run |
| `tier2_label`, `tier2_reason` | LLM's answer |
| `final_label` | what the user saw |

This is the **retraining flywheel**: every Tier-1/Tier-2 *disagreement* is, by construction, a hard example. `text_snippet` is deliberately truncated because message bodies contain OTPs and account numbers.

### Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /predict` | `{message, trusted_contact?}` | `{is_scam, verdict, reason}` |
| `POST /analyze_call` | `{transcript}` | `{action, verdict, risk_score, reason}` |
| `POST /check_number` | `{phone}` | `{show_popup, is_known_spam, message}` |
| `GET /` | — | status + endpoint list |

`/analyze_call` runs a **keyword pre-filter** first — `otp`, `cvv`, `card number`, `expiry`, `police`, `arrest`, `cbi`, `drugs` — at essentially zero latency, before any model loads. `action` is `vibrate_strong` / `vibrate_gentle` / `none`.

---

## 7. Frontend implementation

**Flutter (Dart)**, targeting Android. Single-file `lib/main.dart`.

**Packages:** `http`, `speech_to_text` (on-device ASR), `vibration`, `url_launcher` (SMS intents), `permission_handler`, `shared_preferences`.

**Screens:**
1. **MessageShieldScreen** — trusted-contact field (persisted), message box, CHECK SAFETY, colour-coded result card
2. **CallShieldScreen** — number entry, incoming-call simulation, entry to live monitor
3. **LiveCallScreen** — pulsing mic animation, live transcript, risk meter, haptic alerts

**Localisation:** custom `_AppTranslations` dictionary — **English and Hindi only**. Bengali and Tamil were removed: no Bengali or Tamil text exists anywhere in the training data, so offering those languages implied protection the model could not deliver.

**Senior-centric UI:** high contrast, large type, minimal navigation, haptics so alerts land without looking at the screen.

### The audio path — a known hard limitation

The Live Call Monitor **cannot hear the caller** on stock Android. This is a platform restriction, not a bug:

- `MediaRecorder.AudioSource.VOICE_CALL` requires `CAPTURE_AUDIO_OUTPUT`, a signature-level permission for platform-signed or OEM-preinstalled apps only
- Google's May 11 2022 Play policy banned `AccessibilityService` for call recording — the last loophole
- Android 10's `AudioPlaybackCapture` explicitly excludes voice-call audio

`speech_to_text` therefore captures the **device microphone**, which during a normal earpiece call picks up mostly the *user's* voice. The only viable workaround is forcing speakerphone (`AudioManager.setSpeakerphoneOn(true)`, requires `MODIFY_AUDIO_SETTINGS` — now declared in the manifest) so the mic hears both sides.

**No end-to-end test from live audio to verdict has ever been run.** All model evaluation used clean written text. This is the single largest untested assumption in the system.

---

## 8. The ML pipeline

```
ml/
  data_prep/
    prepare_legit_data.py           real human corpora (UCI, CMU Hinglish, banking, Indian SMS)
    generate_novel_domains.py       Groq/Ollama generation for unseen scam domains
    generate_indian_crossdomain_eval.py   Indian-context evaluation set
    prepare_call_data.py            ICFD-31k streaming + local CSVs
    prepare_unified_data.py         "use everything" build (31,499 rows)
    prepare_balanced_data.py        real/Indian-weighted build (10,008 rows)  ← current
  training/
    common.py                       dataset class, metrics, gradual-unfreeze callback,
                                    temperature fitting, ECE
    train_unified_classifier.py     the production trainer (--seed, --data-prefix)
    recalibrate_and_report.py       post-hoc calibration + full eval table
    build_ood_detector.py           OOD experiment (negative result)
    eval_*.py                       cross-domain, per-variant, cascade/Tier-2 evaluations
  models_out/                       trained artifacts + JSON results (gitignored)
```

### Training data (current "balanced" build — 10,008 rows)

| Source | Real? | Indian? | Rows | Scam |
|---|---|---|---|---|
| ICFD-31k (HF `rishia2220/icfd-31k`) | No | No | 3,750 | 1,500 |
| UCI SMS Spam (`ucirvine/sms_spam`) | **Yes** | No | ~1,538 | 190 |
| Banking corpus (`talkmap/banking-conversation-corpus`) | **Yes** | No | ~1,462 | 0 |
| Indian crowdsourced ham (GitHub, princebari) | **Yes** | **Yes** | 986 | 0 |
| India_Cyber Hinglish CSV | No | **Yes** | 743 | 633 |
| Indian marketing SMS | **Yes** | **Yes** | 698 | 0 |
| Groq/Ollama generated novel domains | Generated | **Yes** | 289 | 155 |
| CMU Hinglish DoG (`festvox/cmu_hinglish_dog`) | **Yes** | **Yes** | 256 | 0 |
| scambust SMS/WhatsApp/calls/audio | No | **Yes** | 286 | 237 |

**Composition: 49.4% real, 32.6% Indian context, 27.1% scam.**

**Structural constraint:** *every* real Indian corpus available is **legitimate**. No real Indian scam text exists in any public source found. The positive class is therefore entirely synthetic or LLM-generated, which caps how "real" this dataset can become.

### Held-out evaluation sets

| Set | Rows | What it measures |
|---|---|---|
| `balanced_test` | 401 | In-distribution performance |
| `unified_eval_synthetic` | 206 | Held-out synthetic slice |
| `unified_eval_generated_holdout` | 170 | 3 generated scam domains never trained on |
| `_icfd31k_raw_cross_domain` | 1,000 | 5 unseen domains (US-flavoured) |
| `indian_crossdomain_eval` | 139 | Same 5 scam types, Indian context |
| `marketing_probe` | 300 | Real legitimate commercial SMS (false-positive probe) |
| `call_dataset_icfd_stress_test` | 4,499 | Fraud-only recall stress test |

---

## 9. Current performance

Production model — balanced build, seed 42, T = 0.6371:

| Evaluation set | n | Acc | Prec | Rec | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| In-distribution test | 401 | 0.913 | 0.798 | 0.908 | 0.850 | 0.963 |
| Held-out synthetic | 206 | 0.976 | 0.977 | 0.994 | 0.986 | 0.989 |
| Generated unseen domains | 170 | 0.912 | 0.902 | 0.973 | 0.936 | 0.951 |
| ICFD cross-domain (US) | 1,000 | 0.643 | 0.964 | 0.648 | 0.775 | 0.682 |
| Indian unseen domains | 139 | 0.691 | 0.598 | 0.902 | 0.719 | 0.837 |
| Marketing FP probe | 300 | 0.893 | — | — | — | n/a |
| ICFD fraud-only stress | 4,499 | 0.878 | 0.989 | 0.886 | 0.935 | 0.585 |

**Reading these correctly:**
- In-distribution **AUC 0.963 vs accuracy 0.913** means the model *ranks* well but its decision threshold is placed sub-optimally — free headroom via threshold tuning.
- The fraud-only stress test's strong F1 (0.935) alongside **AUC 0.585** shows why F1 alone misleads on a 98.9%-scam set: near-constant "scam" scores well. AUC catches it.
- Marketing probe has no positives, so precision/recall/AUC are undefined by construction; the meaningful number is its **10.7% false-positive rate**.

### Cascade end-to-end (Indian unseen domains, 139 rows)

| Escalation policy | Traffic to cloud | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Never (Tier-1 only) | 0% | 0.777 | 0.703 | 0.852 |
| conf < 0.85 | 18.0% | 0.806 | 0.713 | 0.934 |
| **conf < 0.95 (current)** | 34.5% | 0.827 | 0.734 | **0.951** |
| Always (Tier-2 only) | 100% | 0.842 | 0.741 | 0.984 |

---

## 10. Known limitations

1. **The audio path is untested end-to-end.** Android blocks call-audio capture; only speakerphone + microphone is viable, and no live-audio test has been run.
2. **All scam training data is synthetic or generated.** No real scam call transcript has ever been seen by this model.
3. **Cross-domain generalisation remains imperfect** — 0.648 recall on unseen scam categories, better than the 0.478 it started at but far from solved.
4. **Precision on unfamiliar financial topics is weak** (0.598 on Indian unseen domains), partly because ground truth on chit-fund/crypto messages is genuinely arguable.
5. **Uncertainty cannot be triaged.** OOD detection was built and measured; it scores at chance on the unseen-domain set (Section 5 of `PROGRESS_AND_FINDINGS.md`). We cannot currently identify *which* inputs need Tier-2.
6. **Language scope is English + romanised Hinglish only.**
7. **Not production-hardened**: no endpoint authentication, plain HTTP, `DEBUG=True`, permissive CORS, Django dev server, model loaded synchronously at import.
