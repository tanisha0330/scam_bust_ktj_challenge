# Scam Shield AI

Scam Shield AI is a privacy-first, hybrid mobile application designed to safeguard senior citizens from digital fraud. It utilizes a dual-engine AI architecture (Cloud Llama-3 + On-Device ML) to detect and neutralize financial scams across SMS, WhatsApp, and live voice calls in real-time.

---

## Project Overview

Senior citizens are disproportionately targeted by sophisticated phishing and vishing (voice phishing) attacks. Existing solutions often rely on static databases or complex user interfaces unsuitable for non-tech-savvy users.

Scam Shield provides a "Zero-UI" protection layer that:

- Analyzes incoming text messages for phishing patterns.
- Monitors live calls for coercive language and financial keywords.
- Automatically alerts trusted family members when high-risk activity is detected.
- Operates effectively with or without an active internet connection.

---

## Key Features

### 1. Message Shield (SMS & WhatsApp)

- **Hybrid Analysis:** Uses Groq (Llama-3) for deep semantic analysis when online, falling back to a local Naive Bayes model during network outages.
- **Instant Verification:** Classifies messages as Safe or Suspicious with a plain-language explanation.
- **Auto-Alert:** Automatically triggers an SMS or WhatsApp warning to a pre-configured family contact if a high-confidence scam is detected.

### 2. Live Call Shield

- **Real-time Transcription:** Converts voice audio to text on the fly using on-device speech recognition.
- **Keyword Detection:** Instantly flags high-risk terms (e.g., OTP, CVV, Police, CBI, RBI) using offline logic.
- **Contextual Analysis:** Sends transcripts to the AI engine to detect threats, urgency, or coercion typical of "Digital Arrest" scams.
- **Haptic Feedback:** Vibrates the device distinctively to warn the user without requiring them to look at the screen.

### 3. Accessibility & Localization

- **Multi-Language Support:** Full UI localization for English, Hindi, Bengali, and Tamil.
- **Senior-Centric Design:** High-contrast interface, large typography, and simplified navigation.

---

## Technical Architecture

The system follows a Client-Server architecture:

- **Client (Mobile App):** Built with Flutter. Handles UI, Speech-to-Text, Local Notifications, and Hardware interaction (Vibration, Microphone).
- **Server (Intelligence Engine):** Built with Python (Django). Acts as a secure gateway to the LLM and hosts the fallback ML model.

### AI Models

- **Online:** Llama-3.3-70b via Groq API (High accuracy, context-aware).
- **Offline:** NLP with Random forest and linear regression (Low latency, keyword-focused).

---

## Tech Stack

- **Frontend:** Flutter (Dart)
- **Backend:** Python, Django
- **AI/ML:** OpenAI Client (for Groq), Scikit-learn, Pandas
- **External APIs:** Groq Cloud API
- **Android Permissions:** Microphone, Internet, Vibration, URL Handling

---

## Installation and Setup

### Prerequisites

- Flutter SDK (v3.0+)
- Python (v3.8+)
- Android Device or Emulator (API Level 26+)
- Groq API Key

---

