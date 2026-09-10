"""
Entry point for the Scam Shield backend. Thin wrapper around manage.py that
also prints the LAN IPs the app should connect to (handy since the phone and
this server just need to be on the same WiFi).

Actual logic lives in backend/ (models.py, views.py, ml_inference.py).
Run migrations first: python manage.py migrate
"""
import os
import socket
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

# Windows consoles default to cp1252, which can't encode the emoji in the
# startup banner below -- reconfigure to UTF-8 so this doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def get_all_ips():
    ip_list = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_list.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        if host_ip not in ip_list and not host_ip.startswith("127."):
            ip_list.append(host_ip)
    except OSError:
        pass
    return ip_list


if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    ips = get_all_ips()
    port = "8000"
    print("\n🚀 SCAM SHIELD SERVER READY (MuRIL Tier-1 + LLM Tier-2 Cascade)")
    print("👇 Server IPs:")
    if not ips:
        print(f"   1. 127.0.0.1:{port}")
    else:
        for i, ip in enumerate(ips):
            print(f"   {i + 1}. {ip}:{port}")
    print("\n")

    if "runserver" not in sys.argv:
        sys.argv += ["runserver", f"0.0.0.0:{port}"]
    elif not any("0.0.0.0" in arg for arg in sys.argv):
        sys.argv.append(f"0.0.0.0:{port}")

    execute_from_command_line(sys.argv)
