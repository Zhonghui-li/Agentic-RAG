"""Abuse / cost controls for the public demo (OpenAI + GCP are shared lab
accounts, so the app itself must cap how many LLM calls the public can trigger).

Three layers, all tunable via env vars:
  - MAX_INPUT_CHARS    : reject over-long questions (token-inflation abuse)
  - RATE_LIMIT_PER_MIN : per-IP requests/minute (one bot can't dominate)
  - DAILY_QUOTA        : hard ceiling on requests/day -> bounds daily $ cost

Counters are in-memory (per Cloud Run instance). With --max-instances=2 the
effective caps are ~2x; that's an accepted tradeoff for a demo. A truly global
quota would need a shared store (Redis/Firestore).

Worst-case cost (gpt-4o-mini ~$0.005/query): DAILY_QUOTA=200 x 2 instances
~= $2/day ~= $60/month, regardless of the lab key's high limit.
"""
import os
import time
import threading
from collections import defaultdict, deque
from datetime import date

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
DAILY_QUOTA = int(os.environ.get("DAILY_QUOTA", "200"))

_lock = threading.Lock()
_hits = defaultdict(deque)            # ip -> recent request timestamps
_day = {"date": date.today(), "count": 0}


def check_rate_limit(ip: str) -> bool:
    """False if this IP exceeded RATE_LIMIT_PER_MIN in the last 60s."""
    now = time.time()
    with _lock:
        dq = _hits[ip]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MIN:
            return False
        dq.append(now)
        return True


def check_daily_quota() -> bool:
    """False once DAILY_QUOTA requests have been served today (resets daily)."""
    with _lock:
        today = date.today()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_QUOTA:
            return False
        _day["count"] += 1
        return True
