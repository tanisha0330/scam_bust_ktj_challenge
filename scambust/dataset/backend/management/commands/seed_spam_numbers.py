from django.core.management.base import BaseCommand

from backend.models import SpamNumber

SEED_NUMBERS = ["+919876543210", "1400", "+1234567890"]


class Command(BaseCommand):
    help = "Seeds the SpamNumber table with the original demo numbers."

    def handle(self, *args, **options):
        created = 0
        for phone in SEED_NUMBERS:
            _, was_created = SpamNumber.objects.get_or_create(phone=phone, defaults={"label": "⚠️ Suspected Spam"})
            created += int(was_created)
        self.stdout.write(self.style.SUCCESS(f"Seeded {created} new spam numbers (of {len(SEED_NUMBERS)} total)."))
