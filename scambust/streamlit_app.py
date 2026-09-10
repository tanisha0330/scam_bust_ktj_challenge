"""
Minimal Streamlit demo for Scam Shield AI.

Runs the real Tier-1 -> Tier-2 cascade (same code the Django backend uses),
so what you see here is what the app would show.

Run:
    cd scambust
    ./.venv/Scripts/python.exe -m streamlit run streamlit_app.py
"""
import os
import sys

import streamlit as st

# ml_inference lives inside the Django app folder but has no Django imports,
# so it can be loaded directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))

st.set_page_config(page_title="Scam Shield AI", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading MuRIL classifier…")
def get_cascade():
    from backend import ml_inference
    ml_inference._get_model()          # warm the model once
    return ml_inference


EXAMPLES = {
    "— pick an example —": "",
    "Clear scam (KYC/OTP)": "URGENT: Your KYC will expire in 2 hours. Share your OTP now to avoid account block. Click https://bit.ly/verify-kyc",
    "Digital arrest call": "Caller: Main CBI se bol raha hoon, aapke naam par drugs parcel mila hai. Turant paisa transfer karein warna arrest hoga.",
    "Family message (legit)": "Beta paisa mil gaya kya, thoda confusion ho raha hai bank statement me",
    "Real marketing SMS (legit)": "Chivas Studio is back! Do you know what it is? 1) Fashion Show 2) Music Fest. Reply within 24hrs to win passes.",
    "Chit fund reminder (ambiguous)": "Rs 7500 pending hai aapke March chit installment ke liye, kripya 15 March se pehle clear kar dein taki penalty na lage.",
    "Everyday Hinglish (legit)": "Bhai tu kaisa hai? Kal milte hain, khana kha liya?",
}

STYLE = {
    "scam":       ("🚨", "SCAM", "#b3261e", "Do not pay, do not share any code. Tell a family member."),
    "suspicious": ("⚠️", "BE CAREFUL", "#a06800", "This could be genuine, but we are not sure. Verify independently before paying."),
    "safe":       ("✅", "LOOKS SAFE", "#1b5e20", "No fraud signals detected."),
}

st.title("🛡️ Scam Shield AI")
st.caption("Fine-tuned MuRIL (local) → LLM escalation. English & Hinglish.")

choice = st.selectbox("Try an example", list(EXAMPLES.keys()))
text = st.text_area(
    "Message or call transcript",
    value=EXAMPLES[choice],
    height=140,
    placeholder="Paste a suspicious SMS, WhatsApp message or call transcript…",
)
channel = st.radio("Channel", ["sms", "call"], horizontal=True)

if st.button("CHECK SAFETY", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter a message first.")
    else:
        ml = get_cascade()
        with st.spinner("Analysing…"):
            r = ml.classify(text, channel=channel)

        verdict = r.get("verdict") or "safe"
        icon, label, colour, advice = STYLE.get(verdict, STYLE["safe"])

        st.markdown(
            f"<div style='padding:18px;border-radius:10px;border:2px solid {colour};'>"
            f"<div style='font-size:28px;color:{colour};font-weight:700'>{icon} {label}</div>"
            f"<div style='margin-top:8px;font-size:16px'>{r.get('reason') or ''}</div>"
            f"<div style='margin-top:8px;color:#666;font-size:14px'>{advice}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        conf = r.get("tier1_confidence")
        c1.metric("Tier-1 confidence", f"{conf:.1%}" if conf else "n/a")
        c2.metric("Tier-1 says", "scam" if r.get("tier1_label") else "safe")
        c3.metric("Asked the LLM?", "yes" if r.get("escalated_to_llm") else "no")

        if r.get("escalated_to_llm"):
            t2 = r.get("tier2_label")
            st.info(
                f"**Tier-2 (LLM):** {'scam' if t2 else 'not a scam' if t2 is not None else 'unavailable'}"
                + (f" — {r.get('tier2_reason')}" if r.get("tier2_reason") else "")
            )
            if r.get("tier1_label") is not None and t2 is not None and r["tier1_label"] != t2:
                st.warning("The two tiers disagreed → verdict downgraded to **suspicious**.")

with st.expander("How this works"):
    st.markdown(
        """
1. **Tier 1** — a fine-tuned `google/muril-base-cased` classifier runs locally and returns a calibrated confidence.
2. **Escalation** — the cloud LLM is consulted if confidence is below **0.95**, *or* whenever Tier-1 predicts "scam"
   (a scam verdict would text the user's family, so it is always double-checked).
3. **Tier 2** — `openai/gpt-oss-120b` via Groq gives a second opinion.
4. If the two tiers **disagree**, the answer is **suspicious** rather than a confident yes/no — genuine chit-fund
   reminders and chit-fund scams read almost identically, and pretending otherwise would be dishonest.

Calls also run a zero-latency keyword pre-filter (`otp`, `cvv`, `police`, `arrest`, …) before any model.
        """
    )
