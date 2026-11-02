import os
import time
import threading
import requests

_started = False
_lock = threading.Lock()


def start_keep_alive():
    """Start a background daemon thread that pings the Render public URL to prevent free-tier sleep."""
    global _started
    with _lock:
        if _started:
            return
        url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("APP_URL")
        if not url:
            return
        _started = True

    health_url = f"{url.rstrip('/')}/_stcore/health"

    def _ping_loop():
        # Wait 90 seconds after boot before sending first ping
        time.sleep(90)
        while True:
            try:
                requests.get(health_url, timeout=15)
            except Exception:
                pass
            # Render free tier sleeps after 15 minutes of inactivity; ping every 9 minutes
            time.sleep(9 * 60)

    thread = threading.Thread(target=_ping_loop, daemon=True, name="render_keep_alive")
    thread.start()
