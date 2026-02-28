import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_FILE = Path("logs") / "rag_logs.jsonl"

def log_event(event: Dict[str, Any]) -> None:
    """
    Writes one JSON log line per event (JSONL format).
    Why JSONL:
    - easy to append
    - easy to parse later
    - standard in production logging pipelines
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **event,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")