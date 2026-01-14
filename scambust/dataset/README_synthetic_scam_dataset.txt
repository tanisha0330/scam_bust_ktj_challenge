TECOF × IIT Kharagpur (Kshitij) Hackathon — Synthetic Scam Dataset
==================================================================
This dataset is fully synthetic (fabricated) for hackathon use. It contains no real incidents or personal data.

What this dataset represents
----------------------------
Teams will receive examples across FOUR modalities that commonly appear in digital-fraud situations. Each modality is intentionally different, so solutions can be designed for:
- single short messages (SMS),
- multi-message chat threads (WhatsApp),
- two-way voice conversations transcribed into text (Calls),
- audio-origin transcripts such as robocalls/voicemails/ASR text (Audio Transcripts).

Files included (and what each one means)
----------------------------------------
1) public_sms.csv  —  SMS messages
   • ONE ROW = ONE SMS message (short, single message)
   • Text column: `message_text`
   • Typical use: message-level classification, scam signal extraction, intent detection

2) public_whatsapp.csv  —  WhatsApp conversation threads
   • ONE ROW = ONE WhatsApp chat thread (multiple messages stitched together)
   • Text column: `conversation_text` stored as multi-line “Speaker: message” format
   • Typical use: conversation-level reasoning, escalation tracking, multi-turn detection

3) public_calls.csv  —  Phone call transcripts (human-style dialogues)
   • ONE ROW = ONE phone call transcript written as a TWO-WAY dialogue
   • Text column: `call_transcript` stored as multi-line “Speaker: utterance” format
   • Includes call metadata (e.g., `call_duration_seconds`)
   • Typical use: modelling persuasion patterns, back-and-forth resistance, threat escalation in calls

4) public_audio_transcripts.csv  —  Audio-origin transcripts (robocall/voicemail/ASR-style text)
   • ONE ROW = ONE audio transcript sample (TEXT ONLY)
   • Text column: `audio_transcript_text`
   • Written like speech-to-text output and may include markers such as:
     “[Automated Voice]”, “[Voicemail]”, “[inaudible]”
   • Key difference vs public_calls.csv:
     - `public_calls.csv` = human-style TWO-WAY call dialogue (structured turn-taking)
     - `public_audio_transcripts.csv` = audio-derived transcript style (often ONE-WAY / noisier ASR-like text)
   • Typical use: robust detection on “audio-to-text” style data (robocalls, voicemails), even without raw audio

5) public_unified_multimodal.csv  (optional helper file)
   • All modalities combined into one table
   • Key columns: `modality`, `sample_id`, `timestamp`, `text`
   • Typical use: one ingestion pipeline + one unified model/evaluation loop

How labels are provided (high level)
------------------------------------
Each row includes:
- `is_scam` (0 = legitimate, 1 = scam)
- `scam_type` (category label)
- `scam_stage` (approx stage: lure / action request / threat, etc.)
- `requested_action` (what the scammer tries to get the victim to do)
- `severity` (1–5; higher = higher pressure/stakes)
- Lightweight signal flags derived from text (e.g., `has_url`, `has_upi`, `has_otp`, `has_threat`, `has_urgency`)

Important notes
---------------
- This dataset is for hackathon experimentation and learning; it is not a substitute for real-world law-enforcement data.
- There is intentional overlap between scam and legitimate content (e.g., legitimate OTP messages exist; scammers may also mention OTPs).
- No personal data is used; placeholders are synthetic.

Good luck — we look forward to seeing your solutions!
