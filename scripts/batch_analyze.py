#!/usr/bin/env python3
"""
IFE Log RCA Analyzer — Day 7 CLI tool
--------------------------------------
Reads a log file line by line, calls /full-analysis on each line,
prints a color-coded terminal table, and saves full results to JSON.

Usage:
    python scripts/batch_analyze.py logs_sample/ife_system.log
    python scripts/batch_analyze.py logs_sample/ife_system.log --output results/my_run.json
    python scripts/batch_analyze.py logs_sample/ife_system.log --anomalies-only

Services required (docker compose up):
    anomaly_detector  http://localhost:8000
    rca_service       http://localhost:8001
"""

import sys
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RCA_SERVICE_URL = "http://localhost:8001"

# Terminal colors
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Service calls
# ---------------------------------------------------------------------------

def check_services():
    """Verify both services are reachable before processing."""
    for name, url in [("rca_service", RCA_SERVICE_URL)]:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
        except Exception:
            print(f"{RED}ERROR: {name} not reachable at {url}{RESET}")
            print(f"       Run: docker compose up")
            sys.exit(1)


def full_analysis(log_line: str) -> dict:
    """Call /full-analysis and return the full response dict."""
    payload = json.dumps({"log_line": log_line}).encode()
    req = urllib.request.Request(
        f"{RCA_SERVICE_URL}/full-analysis",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_header():
    print(f"\n{BOLD}{'SCORE':>10}  {'STATUS':>8}  {'CONF':>6}  LOG LINE{RESET}")
    print("─" * 100)


def print_result(result: dict, line_num: int):
    log_line = result["log_line"]
    anomaly  = result["anomaly"]
    rca      = result["rca"]

    score      = anomaly["score"]
    is_anomaly = anomaly["is_anomaly"]
    performed  = rca["performed"]
    confidence = rca.get("confidence", "").upper()[:4] if performed else ""

    if not is_anomaly:
        color  = GREEN
        status = "normal"
        conf   = DIM + "─" + RESET
    elif confidence in ("HIGH", "MED", "MEDI"):
        color  = RED
        status = "ANOMALY"
        conf   = f"{RED}{confidence}{RESET}"
    else:
        color  = YELLOW
        status = "ANOMALY"
        conf   = f"{YELLOW}{confidence}{RESET}"

    display = log_line[:72] + ("…" if len(log_line) > 72 else "")
    print(f"{color}{score:>10.6f}  {status:>8}{RESET}  {conf:>6}  {display}")

    # Print RCA hypothesis indented below anomalous lines
    if is_anomaly and performed and rca.get("root_cause_hypothesis"):
        hypothesis = rca["root_cause_hypothesis"]
        # Wrap at 90 chars
        words = hypothesis.split()
        line_buf = "             ↳ "
        for word in words:
            if len(line_buf) + len(word) > 100:
                print(f"{CYAN}{line_buf}{RESET}")
                line_buf = "               " + word + " "
            else:
                line_buf += word + " "
        if line_buf.strip():
            print(f"{CYAN}{line_buf}{RESET}")


def print_summary(results: list, output_path: Path):
    total     = len(results)
    anomalies = [r for r in results if r["anomaly"]["is_anomaly"]]
    normals   = [r for r in results if not r["anomaly"]["is_anomaly"]]
    high_conf = [r for r in anomalies if r["rca"].get("confidence") == "high"]
    latencies = [r["total_latency_ms"] for r in results if "total_latency_ms" in r]
    avg_lat   = round(sum(latencies) / len(latencies)) if latencies else 0

    top_anomalies = sorted(anomalies, key=lambda r: r["anomaly"]["score"], reverse=True)[:3]

    print("\n" + "─" * 100)
    print(f"\n{BOLD}SUMMARY{RESET}")
    print(f"  Log lines analyzed : {total}")
    print(f"  Normal             : {GREEN}{len(normals)}{RESET}")
    print(f"  Anomalies detected : {RED}{len(anomalies)}{RESET}  "
          f"({round(len(anomalies)/total*100)}% of lines)")
    print(f"  High confidence    : {RED}{len(high_conf)}{RESET}")
    print(f"  Avg latency        : {avg_lat}ms per line")

    if top_anomalies:
        print(f"\n{BOLD}TOP ANOMALIES BY SCORE{RESET}")
        for i, r in enumerate(top_anomalies, 1):
            score = r["anomaly"]["score"]
            conf  = r["rca"].get("confidence", "?").upper()
            line  = r["log_line"][:80]
            print(f"  {i}. [{RED}{score:.4f}{RESET}] [{conf}] {line}")

    print(f"\n{BOLD}OUTPUT{RESET}")
    print(f"  Full results saved : {CYAN}{output_path}{RESET}")
    print(f"  Service docs       : {RCA_SERVICE_URL}/docs\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze a log file using the IFE Log RCA pipeline."
    )
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument(
        "--output", "-o",
        help="Path to save JSON results (default: results/rca_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--anomalies-only", "-a",
        action="store_true",
        help="Only print anomalous lines to terminal (still saves all to JSON)"
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"rca_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify services
    check_services()

    # Read log file
    log_file = Path(args.logfile)
    if not log_file.exists():
        print(f"{RED}ERROR: log file not found: {log_file}{RESET}")
        sys.exit(1)

    lines = [
        line.strip()
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    print(f"\n{BOLD}IFE Log RCA Analyzer{RESET}")
    print(f"  Log file  : {log_file}  ({len(lines)} lines)")
    print(f"  Service   : {RCA_SERVICE_URL}")
    print(f"  Output    : {output_path}")

    print_header()

    results = []
    errors  = []

    for i, line in enumerate(lines, 1):
        try:
            result = full_analysis(line)
            results.append(result)
            if not args.anomalies_only or result["anomaly"]["is_anomaly"]:
                print_result(result, i)
        except Exception as e:
            error_entry = {"log_line": line, "error": str(e), "line_num": i}
            errors.append(error_entry)
            print(f"{'ERROR':>10}  {'?':>8}  {'?':>6}  {line[:60]}  ({e})")

    # Save JSON results
    output_data = {
        "meta": {
            "log_file"    : str(log_file),
            "analyzed_at" : datetime.now().isoformat(),
            "service_url" : RCA_SERVICE_URL,
            "total_lines" : len(lines),
            "errors"      : len(errors),
        },
        "results": results,
        "errors" : errors,
    }
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print_summary(results, output_path)


if __name__ == "__main__":
    main()