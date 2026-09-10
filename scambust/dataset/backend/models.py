from django.db import models


class SpamNumber(models.Model):
    """Crowdsourced/known spam number registry, replacing the old hardcoded
    in-memory list in check_number()."""

    phone = models.CharField(max_length=20, unique=True, db_index=True)
    report_count = models.PositiveIntegerField(default=1)
    label = models.CharField(max_length=100, blank=True, default="")  # e.g. "Suspected Spam"
    first_reported_at = models.DateTimeField(auto_now_add=True)
    last_reported_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.phone} ({self.report_count} reports)"


class ScanLog(models.Model):
    """Records every cascade decision (tier-1 verdict, whether it escalated to
    the LLM, and the final verdict). This is the data source for the future
    fine-tuning flywheel -- disagreements between tier-1 and tier-2 are the
    highest-value examples for the next training round.

    NOTE: `text_snippet` is intentionally truncated, not the full message/
    transcript -- these can contain OTPs, account numbers, etc. Keep this
    bounded, and revisit retention policy before this table grows large.
    """

    CHANNEL_CHOICES = [("sms", "SMS/WhatsApp"), ("call", "Call")]

    created_at = models.DateTimeField(auto_now_add=True)
    channel = models.CharField(max_length=10, choices=CHANNEL_CHOICES)
    text_snippet = models.CharField(max_length=200)

    tier1_label = models.BooleanField()
    tier1_confidence = models.FloatField()
    escalated_to_llm = models.BooleanField(default=False)
    tier2_label = models.BooleanField(null=True, blank=True)
    tier2_reason = models.CharField(max_length=500, blank=True, default="")

    final_label = models.BooleanField()

    class Meta:
        indexes = [models.Index(fields=["created_at"]), models.Index(fields=["escalated_to_llm"])]

    def __str__(self):
        return f"[{self.channel}] scam={self.final_label} (escalated={self.escalated_to_llm})"
