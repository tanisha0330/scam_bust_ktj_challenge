import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from . import ml_inference
from .models import ScanLog, SpamNumber

DANGER_WORDS = ["otp", "cvv", "card number", "expiry", "police", "arrest", "cbi", "drugs"]


def cors_response(data, status=200):
    response = JsonResponse(data, status=status)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _log_scan(channel: str, text: str, cascade_result: dict):
    if cascade_result.get("final_label") is None:
        return  # analysis unavailable entirely -- nothing meaningful to log
    ScanLog.objects.create(
        channel=channel,
        text_snippet=text[:200],
        tier1_label=bool(cascade_result["tier1_label"]) if cascade_result["tier1_label"] is not None else False,
        tier1_confidence=cascade_result.get("tier1_confidence") or 0.0,
        escalated_to_llm=cascade_result["escalated_to_llm"],
        tier2_label=cascade_result["tier2_label"],
        tier2_reason=cascade_result.get("tier2_reason", ""),
        final_label=cascade_result["final_label"],
    )


@csrf_exempt
def check_number(request):
    if request.method == "OPTIONS":
        return cors_response({})
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            phone = data.get("phone", "")
            entry = SpamNumber.objects.filter(phone=phone).first()
            is_spam = entry is not None
            if is_spam:
                message = entry.label or "⚠️ Suspected Spam"
            else:
                message = "Unknown Caller. Activate AI Shield?"
            return cors_response({"show_popup": True, "is_known_spam": is_spam, "message": message})
        except Exception:
            return cors_response({"error": "Error"}, 400)
    return cors_response({})


@csrf_exempt
def analyze_call(request):
    if request.method == "OPTIONS":
        return cors_response({})
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            transcript = data.get("transcript", "")
            if not transcript:
                return cors_response({"action": "none"})

            # Fast keyword pre-filter -- near-zero latency, runs before any model.
            lowered = transcript.lower()
            for w in DANGER_WORDS:
                if w in lowered:
                    return cors_response({
                        "action": "vibrate_strong",
                        "risk_score": 99,
                        "reason": f"Scammer said '{w}'",
                    })

            result = ml_inference.classify(transcript, channel="call")
            _log_scan("call", transcript, result)

            if result["final_label"] is None:
                return cors_response({"error": "Analysis unavailable", "details": result["reason"]}, 503)

            confidence = result.get("tier1_confidence") or 0.9
            if result["verdict"] == "scam":
                return cors_response({
                    "action": "vibrate_strong",
                    "verdict": "scam",
                    "risk_score": round(confidence * 100),
                    "reason": result["reason"],
                })
            if result["verdict"] == "suspicious":
                # Gentler haptic: warn without asserting fraud outright.
                return cors_response({
                    "action": "vibrate_gentle",
                    "verdict": "suspicious",
                    "risk_score": 60,
                    "reason": result["reason"],
                })
            return cors_response({"action": "none", "verdict": "safe", "risk_score": 0})
        except Exception:
            return cors_response({"error": "Error"}, 400)
    return cors_response({})


@csrf_exempt
def predict_scam(request):
    if request.method == "OPTIONS":
        return cors_response({})
    if request.method == "POST":
        try:
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return cors_response({"error": "Invalid JSON"}, 400)

            message = data.get("message", "")
            if not message:
                return cors_response({"error": "Empty message"}, 400)

            result = ml_inference.classify(message, channel="sms")
            _log_scan("sms", message, result)

            if result["final_label"] is None:
                return cors_response({"error": "Analysis unavailable", "details": result["reason"]}, 503)

            # `is_scam` kept for backward compatibility with older app builds;
            # `verdict` is the three-way answer (scam / suspicious / safe).
            return cors_response({
                "is_scam": result["final_label"],
                "verdict": result["verdict"],
                "reason": result["reason"],
            })
        except Exception as e:
            return cors_response({"error": "Server Error", "details": str(e)}, 400)
    return cors_response({})


def home(request):
    return cors_response({
        "status": "Online 🟢",
        "endpoints": ["/predict", "/check_number", "/analyze_call"],
    })
