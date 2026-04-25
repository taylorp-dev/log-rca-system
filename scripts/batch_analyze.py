#!/usr/bin/env python3
"""
Batch analyze a log file against the running anomaly_detector service.

Usage:
    python scripts/batch_analyze.py logs_sample/sample.log

Output: colored terminal table showing score and anomaly flag per line.
"""

import sys
import json
import urllib.request
import urllib.error

SERVICE_URL = "http://localhost:8000"
ANOMALY_COLOR = "\033[91m"   # red
NORMAL_COLOR  = "\033[92m"   # green
RESET         = "\033[0m"
BOLD          = "\033[1m"


def analyze_line(log_line: str) -> dict:
    payload = json.dumps({"log_line": log_line}).encode()
    req = urllib.request.Request(
        f"{SERVICE_URL}/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_analyze.py <logfile>")
        sys.exit(1)

    log_file = sys.argv[1]

    try:
        urllib.request.urlopen(f"{SERVICE_URL}/health", timeout=5)
    except Exception:
        print(f"ERROR: anomaly_detector not reachable at {SERVICE_URL}")
        print("       Run: docker compose up --build")
        sys.exit(1)

    print(f"\n{BOLD}{'SCORE':>10}  {'ANOMALY':>8}  LOG LINE{RESET}")
    print("─" * 80)

    with open(log_file) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                result = analyze_line(line)
                score   = result["score"]
                flag    = result["is_anomaly"]
                color   = ANOMALY_COLOR if flag else NORMAL_COLOR
                label   = "ANOMALY" if flag else "normal"
                display = line[:70] + ("…" if len(line) > 70 else "")
                print(f"{color}{score:>10.6f}  {label:>8}  {display}{RESET}")
            except Exception as e:
                print(f"{'ERROR':>10}  {'?':>8}  {line[:60]}  ({e})")

    print("─" * 80)
    print(f"\nService docs: {SERVICE_URL}/docs\n")


if __name__ == "__main__":
    main()
