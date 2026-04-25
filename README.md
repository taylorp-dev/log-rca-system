# Log RCA System

AI-powered log anomaly detection and root-cause analysis pipeline.  
Built as a 2-week sprint project targeting Tesla AI Test Systems Engineer skills.

**Architecture (end state — Week 2):**
```
raw log line
    │
    ▼
[anomaly_detector]  PyTorch autoencoder → reconstruction error score
    │  if anomalous
    ▼
[rca_service]       ChromaDB RAG retrieval + LLM (Claude) → root cause hypothesis
    │
    ▼
structured JSON verdict  { is_anomaly, score, root_cause, similar_past_failures }
```

**Connects to DO-178C concepts:** threshold calibration = pass/fail criteria,
RAG retrieval = traceability to known failure modes, LLM hypothesis = failure
mode and effects analysis (FMEA) automation.

---

## Day 1 quickstart

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ (for running tests locally)

### 1. Build and run

```bash
git clone <your-repo>
cd log-rca-system

docker compose up --build
```

Service will be available at `http://localhost:8000`.

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Analyze a normal log line
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log_line": "INFO request complete status 200"}'

# Analyze an anomalous log line
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log_line": "CRITICAL null pointer exception crash traceback"}'

# Batch analyze the sample log file
python scripts/batch_analyze.py logs_sample/sample.log
```

### 3. Run tests locally

```bash
pip install fastapi uvicorn torch numpy pydantic pytest httpx
pytest tests/ -v
```

### 4. Explore the API docs

FastAPI auto-generates interactive docs:  
`http://localhost:8000/docs`

---

## Project structure

```
log-rca-system/
├── anomaly_detector/         # Week 1: PyTorch autoencoder service
│   ├── Dockerfile
│   ├── main.py               # FastAPI app + model
│   └── requirements.txt
├── rca_service/              # Week 2: LLM + RAG root cause service
│   └── (coming Day 10)
├── logs_sample/              # Sample log files for testing
│   └── sample.log
├── tests/
│   └── test_anomaly_detector.py
├── docker-compose.yml        # Wires all services together
└── README.md
```

---

## Week-by-week build plan

| Day | Milestone |
|-----|-----------|
| 1   | Docker fundamentals, run first containers |
| 2   | docker-compose, networking, health checks |
| 3   | PyTorch tensors, autograd, basic classifier |
| 4   | Autoencoder for log anomaly detection |
| 5   | LLM API calls, RAG pipeline basics |
| 6–7 | LLM RCA CLI prototype |
| 8   | Containerize anomaly detector as FastAPI service ← **here now** |
| 9   | GitHub Actions CI/CD pipeline |
| 10  | Wire LLM RCA into full pipeline |
| 11  | Integration tests in CI |
| 12  | Polish, README, Loom demo |
| 13–14 | Distributed systems vocabulary + interview prep |
