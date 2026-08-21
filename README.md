# Scam Shield AI

Scam Shield AI is a privacy-first, hybrid mobile application designed to safeguard senior citizens from digital fraud. It uses a two-tier cascade architecture -- fast local MuRIL models first, cloud LLM only when uncertain -- to detect and neutralize financial scams across SMS, WhatsApp, and live voice calls in real-time.

---

## Project Overview

Senior citizens are disproportionately targeted by sophisticated phishing and vishing (voice phishing) attacks. Existing solutions often rely on static databases or complex user interfaces unsuitable for non-tech-savvy users.

Scam Shield provides a "Zero-UI" protection layer that:

- Analyzes incoming text messages for phishing patterns.
- Monitors live calls for coercive language and financial keywords.
- Automatically alerts trusted family members when high-risk activity is detected.
- Decides for itself, per request, whether local analysis is confident enough or an LLM opinion is worth the extra round-trip -- no manual online/offline toggle for the user to think about.

---

## Key Features

### 1. Message Shield (SMS & WhatsApp)

- **Cascade Analysis:** A fine-tuned MuRIL model classifies the message locally first; only escalates to the cloud LLM when its confidence is low.
- **Instant Verification:** Classifies messages as Safe or Suspicious with a plain-language explanation.
- **Auto-Alert:** Automatically triggers an SMS warning to a pre-configured family contact if a scam is detected.

### 2. Live Call Shield

- **Real-time Transcription:** Converts voice audio to text on the fly using on-device speech recognition.
- **Keyword Pre-Filter:** Instantly flags high-risk terms (e.g., OTP, CVV, Police, CBI) before any model even runs.
- **Contextual Analysis:** The same unified classifier detects threats, urgency, or coercion typical of "Digital Arrest" scams, with the cloud LLM as a fallback for uncertain cases.
- **Haptic Feedback:** Vibrates the device distinctively to warn the user without requiring them to look at the screen.

### 3. One Unified Classifier

SMS, WhatsApp and call transcripts all run through a **single** binary scam/not-scam model. An earlier design used two separate models, which left the message classifier with only ~140 training rows — the weakest component in the system, and the one that produced a false positive on an ordinary family message about a bank statement. Pooling every source turns that into tens of thousands of examples.

### 4. Accessibility & Localization

- **Language Support:** English and Hindi/Hinglish. (Bengali and Tamil were deliberately removed — the detection model is trained only on English and romanised Hinglish, and offering those languages in the UI implied protection the model could not actually provide.)
- **Senior-Centric Design:** High-contrast interface, large typography, and simplified navigation.

---

## Technical Architecture

The system follows a Client-Server architecture:

- **Client (Mobile App):** Built with Flutter. Handles UI, Speech-to-Text, Local Notifications, and Hardware interaction (Vibration, Microphone).
- **Server (Intelligence Engine):** Django + PostgreSQL. Runs the local MuRIL model, calls Groq only on escalation, and logs every decision for future retraining.

### AI Models

- **Tier 1 (local, always runs):** One fine-tuned `google/muril-base-cased` binary classifier covering SMS, WhatsApp and calls.
- **Tier 2 (escalation only):** `openai/gpt-oss-120b` via Groq API, used only when Tier 1's calibrated confidence is below threshold. (Groq retired the Llama-3 family; override with `GROQ_MODEL` in `.env` if this one is retired too.)

### Training data

Deliberately mixed so the model sees both fraud and ordinary life, from both synthetic and **real** sources:

| Source | Kind | Role |
|---|---|---|
| ICFD-31k (HF) | synthetic call conversations, 10 fraud domains | bulk scam + legit |
| scambust CSVs | synthetic SMS/WhatsApp/call/audio | scam-heavy |
| India_Cyber Hinglish CSV | template-generated Hinglish | scam-heavy |
| `ucirvine/sms_spam` | **real** human SMS (ham + spam) | ordinary messages |
| `festvox/cmu_hinglish_dog` | **real** casual Hinglish chat | ordinary conversation |
| `talkmap/banking-conversation-corpus` | **real** legitimate bank calls | hard negatives |
| Ollama/Groq generated | novel scam domains + matching legit | domain coverage |

See `PROJECT_EXPLANATION.md` for the full architecture writeup, methodology, and evaluation numbers — including the known cross-domain generalisation limitation.

---

## Tech Stack

- **Frontend:** Flutter (Dart)
- **Backend:** Python, Django, PostgreSQL
- **AI/ML:** PyTorch, Hugging Face Transformers (MuRIL fine-tuning), OpenAI client (for Groq), scikit-learn
- **External APIs:** Groq Cloud API
- **Android Permissions:** Microphone, Internet, Vibration, URL Handling

---

## Installation and Setup

### Prerequisites

- Flutter SDK (v3.0+)
- Python 3.11+
- PostgreSQL 17
- Android Device or Emulator (API Level 26+)
- Groq API Key
- (Optional but recommended) NVIDIA GPU with CUDA for fine-tuning the MuRIL models faster

### Backend setup

```
cd scambust
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd dataset
copy .env.example .env   # fill in GROQ_API_KEY and Postgres credentials
python manage.py migrate
python manage.py seed_spam_numbers
python api_server.py
```

### Training the ML models (optional -- pretrained artifacts are gitignored)

```
cd scambust
.venv\Scripts\python -m ml.data_prep.prepare_message_data
.venv\Scripts\python -m ml.data_prep.prepare_call_data
.venv\Scripts\python -m ml.training.train_call_classifier
.venv\Scripts\python -m ml.training.train_message_classifier
```

---
