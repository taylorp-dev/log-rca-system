# Log RCA System

AI-powered log anomaly detection and root cause analysis pipeline for safety-critical systems.

Combines a PyTorch autoencoder for anomaly scoring with a ChromaDB RAG pipeline and LLM-powered
root cause analysis — turning hours of manual log triage into seconds.

## Architecture

```
raw log line
    │
    ▼
[anomaly_detector]  PyTorch autoencoder → reconstruction error score
    │  if anomalous (score > threshold)
    ▼
[rca_service]       ChromaDB RAG retrieval + LLM (Claude) → root cause hypothesis
    │
    ▼
structured JSON verdict
{
  "anomaly": { "score", "threshold", "is_anomaly" },
  "rca":     { "root_cause_hypothesis", "confidence", "similar_failures", "latency_ms" },
  "total_latency_ms"
}
```

**Design principle:** deterministic anomaly gate before non-deterministic LLM component.
The autoencoder score must exceed the calibrated threshold before the LLM runs.
This keeps the certifiable surface small and bounded — relevant for safety-critical deployments.

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `anomaly_detector` | 8000 | PyTorch autoencoder — scores log lines for anomalies |
| `rca_service` | 8001 | ChromaDB RAG + LLM — retrieves similar failures, generates RCA |
| `chromadb` | 8002 | Vector store — persists failure pattern embeddings |

---

## Quickstart

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ (for CLI tools and local tests)
- Anthropic API key (for LLM-powered RCA)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Build and run

```bash
git clone https://github.com/taylorp-dev/log-rca-system
cd log-rca-system
docker compose up --build
```

All three services start automatically in dependency order:
`chromadb` → `anomaly_detector` → `rca_service`

### 3. Test the full pipeline

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health

# Full end-to-end analysis (single log line)
curl -X POST http://localhost:8001/full-analysis \
  -H "Content-Type: application/json" \
  -d '{"log_line": "CRITICAL null pointer exception in VideoDecoder traceback follows"}'

# Normal log line (anomaly gate prevents LLM from running)
curl -X POST http://localhost:8001/full-analysis \
  -H "Content-Type: application/json" \
  -d '{"log_line": "INFO system startup complete all checks passed"}'
```

### 4. Batch analyze a log file

```bash
# Full output — all lines
python scripts/batch_analyze.py logs_sample/ife_system.log

# Anomalies only — cleaner for demo
python scripts/batch_analyze.py logs_sample/ife_system.log --anomalies-only

# Custom output path
python scripts/batch_analyze.py logs_sample/ife_system.log --output results/my_run.json
```

### 5. Run tests

```bash
pip install fastapi uvicorn torch numpy pydantic pytest httpx
pytest tests/ -v
```

### 6. Explore API docs

FastAPI auto-generates interactive docs:
- Anomaly detector: `http://localhost:8000/docs`
- RCA service: `http://localhost:8001/docs`

---

## Project structure

```
log-rca-system/
├── anomaly_detector/         # PyTorch autoencoder service
│   ├── Dockerfile
│   ├── main.py               # FastAPI app + autoencoder + /retrain endpoint
│   └── requirements.txt
├── rca_service/              # LLM + RAG root cause service
│   ├── Dockerfile
│   ├── main.py               # FastAPI app + ChromaDB + Claude API
│   └── requirements.txt
├── scripts/
│   └── batch_analyze.py      # CLI tool — batch log analysis with RCA
├── logs_sample/
│   ├── sample.log            # Generic sample log file
│   └── ife_system.log        # Realistic IFE/avionics system log (66 lines)
├── results/                  # JSON output from batch_analyze.py (gitignored)
├── tests/
│   └── test_anomaly_detector.py
├── .env.example              # Environment variable template
├── docker-compose.yml        # Wires all three services together
└── README.md
```

---

## Key endpoints

### anomaly_detector (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Score a single log line |
| POST | `/analyze/batch` | Score multiple log lines |
| POST | `/retrain` | Retrain model + recalibrate threshold |
| GET | `/version` | Model and app version |

### rca_service (port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/full-analysis` | End-to-end: anomaly score + RCA in one call |
| POST | `/rca` | RCA only (caller supplies anomaly score) |
| GET | `/knowledge-base/count` | Number of failure patterns in ChromaDB |

---

## How anomaly detection works

The autoencoder is trained exclusively on normal log lines. At inference time,
reconstruction error measures how different an incoming log line is from the
learned normal distribution. High error = anomalous.

**Threshold calibration:** `mean(errors) + 3 * std(errors)` over representative
normal samples, with a minimum floor of 0.02. Derived from measured data,
not set arbitrarily — the same principle as setting acceptance criteria in
safety-critical verification.

**Training data** covers both generic service logs and IFE/avionics-style structured
logs (timestamps, subsystem tags, key=value pairs) to minimize false positives on
domain-specific log formats.

---

## How RCA works

1. Anomalous log line is embedded using `sentence-transformers/all-MiniLM-L6-v2`
2. ChromaDB retrieves the 2 most similar past failures by vector distance
3. Log line + anomaly score + retrieved failures are passed to the LLM as context
4. LLM returns structured `ROOT_CAUSE` / `ACTION` / `CONFIDENCE` response
5. Response is parsed and returned as structured JSON

Knowledge base ships with 12 failure patterns covering generic service failures
and IFE-specific patterns (ARINC 429 bus faults, video decoder crashes, content
server outages, memory overflow, seat display faults, passenger call system faults).

---

## Results on IFE system log (66 lines)

| Metric | Value |
|--------|-------|
| Lines analyzed | 66 |
| Normal (correctly filtered) | 51 (77%) |
| Anomalies detected | 15 (23%) |
| High confidence RCA | 14 |
| Avg latency per line | ~1200ms |

All real failure events correctly identified: ARINC 429 bus fault + failover,
VideoDecoder crash sequence, content server outage + cache fallback, memory
overflow, seat display timeout, passenger call fault.

---

## DO-178C connections

This project applies several concepts from safety-critical software engineering:

| Project decision | DO-178C parallel |
|-----------------|-----------------|
| Deterministic gate before LLM | Partitioning — certifiable surface stays bounded |
| Threshold from 3-sigma analysis | Acceptance criteria derived from measured data |
| RAG retrieval from known failures | Traceability to known failure modes |
| LLM structured hypothesis | FMEA automation — failure mode and effects analysis |
| Version fields in every response | Configuration management — traceability per response |
| Per-component latency tracking | Execution time budgeting |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required — Claude API key for LLM-powered RCA |
| `ANOMALY_SERVICE_URL` | Internal URL for anomaly_detector (default: `http://anomaly_detector:8000`) |
| `CHROMA_URL` | Internal URL for chromadb (default: `http://chromadb:8000`) |

---

## License

MIT