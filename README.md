Scam Shield AI

Scam Shield AI is a privacy-first, hybrid mobile application designed to safeguard senior citizens from digital fraud. It utilizes a dual-engine AI architecture (Cloud Llama-3 + On-Device ML) to detect and neutralize financial scams across SMS, WhatsApp, and live voice calls in real-time.

Project Overview

Senior citizens are disproportionately targeted by sophisticated phishing and vishing (voice phishing) attacks. Existing solutions often rely on static databases or complex user interfaces unsuitable for non-tech-savvy users.

Scam Shield provides a "Zero-UI" protection layer that:

Analyzes incoming text messages for phishing patterns.

Monitors live calls for coercive language and financial keywords.

Automatically alerts trusted family members when high-risk activity is detected.

Operates effectively with or without an active internet connection.

Key Features

1. Message Shield (SMS & WhatsApp)

Hybrid Analysis: Uses Groq (Llama-3) for deep semantic analysis when online, falling back to a local Naive Bayes model during network outages.

Instant Verification: Classifies messages as Safe or Suspicious with a plain-language explanation.

Auto-Alert: Automatically triggers an SMS or WhatsApp warning to a pre-configured family contact if a high-confidence scam is detected.

2. Live Call Shield

Real-time Transcription: Converts voice audio to text on the fly using on-device speech recognition.

Keyword Detection: Instantly flags high-risk terms (e.g., OTP, CVV, Police, CBI, RBI) using offline logic.

Contextual Analysis: Sends transcripts to the AI engine to detect threats, urgency, or coercion typical of "Digital Arrest" scams.

Haptic Feedback: Vibrates the device distinctively to warn the user without requiring them to look at the screen.

3. Accessibility & Localization

Multi-Language Support: Full UI localization for English, Hindi, Bengali, and Tamil.

Senior-Centric Design: High-contrast interface, large typography, and simplified navigation.

Technical Architecture

The system follows a Client-Server architecture:

Client (Mobile App): Built with Flutter. Handles UI, Speech-to-Text, Local Notifications, and Hardware interaction (Vibration, Microphone).

Server (Intelligence Engine): Built with Python (Django). Acts as a secure gateway to the LLM and hosts the fallback ML model.

AI Models:

Online: Llama-3.3-70b via Groq API (High accuracy, context-aware).

Offline: Scikit-learn Multinomial Naive Bayes (Low latency, keyword-focused).

Tech Stack

Frontend: Flutter (Dart)

Backend: Python, Django

AI/ML: OpenAI Client (for Groq), Scikit-learn, Pandas

External APIs: Groq Cloud API

Android Permissions: Microphone, Internet, Vibration, URL Handling

Installation and Setup

Prerequisites

Flutter SDK (v3.0+)

Python (v3.8+)

Android Device or Emulator (API Level 26+)

Groq API Key

1. Backend Setup (Server)

Navigate to the server directory and install dependencies:

cd backend
pip install django openai pandas scikit-learn requests


Train the local fallback model (required for offline mode):

python train_scam_shield.py


Configure the API Key:
Open api_server.py and replace the placeholder with your Groq API key:
API_KEY = "gsk_..."

Start the server:

python api_server.py runserver 0.0.0.0:8000


2. Frontend Setup (App)

Navigate to the app directory:

cd scam_shield_app


Install Flutter dependencies:

flutter pub get


Configuration:
Open lib/main.dart and update the ipController variable with your computer's local IP address (e.g., 192.168.1.X:8000).

Run the application:

flutter run


Note: Ensure both the mobile device and the server (computer) are connected to the same Wi-Fi network.

Usage Guide

Select Language: Choose your preferred language from the top bar dropdown.

Configure Trusted Contact: In the SMS Shield tab, enter the phone number of a family member. This saves automatically.

Scan Messages: Paste a text message to check its safety status.

Monitor Calls: Navigate to the Call Shield tab. Use the "Simulate Incoming Call" feature to test the protection popup. In a real scenario, click "Start Live Monitor" to begin analyzing conversation audio.

Privacy & Security

Data Minimization: No audio recordings are stored on the server. Only transient text transcripts are processed for analysis.

Local Fallback: The offline model ensures basic protection without sending data to the cloud.

Permission Transparency: The app explicitly requests microphone permissions only when the user opts into the Live Shield feature.
