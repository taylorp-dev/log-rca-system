"""
RCA Service - v0.2.0
--------------------
Day 6: Refactored for full-pipeline integration.

Changes from v0.1.0:
- Anthropic client moved to module level (instantiated once at startup)
- Added httpx for async calls to anomaly_detector
- Added run_rca() as single source of truth for RCA logic
- /rca endpoint updated to use run_rca()
- Added /full-analysis endpoint: calls anomaly_detector, gates RCA on score
- All endpoints are now async
- Added latency tracking (time.perf_counter)
- Added FullAnalysisRequest/Response Pydantic models
"""

import os
import time
import httpx
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Log RCA Service", version="0.2.0")

ANOMALY_SERVICE_URL = os.getenv("ANOMALY_SERVICE_URL", "http://anomaly_detector:8000")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
CHROMA_URL          = os.getenv("CHROMA_URL", "http://chromadb:8000")

# ---------------------------------------------------------------------------
# Embedding function using sentence-transformers
# ---------------------------------------------------------------------------
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------
try:
    chroma_host = CHROMA_URL.replace("http://", "").split(":")[0]
    chroma_port = int(CHROMA_URL.split(":")[-1])
    chroma_client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port
    )
    collection = chroma_client.get_or_create_collection(
        name="failure_history",
        embedding_function=ef
    )
    print("[startup] connected to ChromaDB")
except Exception as e:
    print(f"[startup] ChromaDB not available yet: {e}")
    chroma_client = None
    collection = None

# ---------------------------------------------------------------------------
# Anthropic client — instantiated once at startup, not per request
# ---------------------------------------------------------------------------
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
if anthropic_client:
    print("[startup] Anthropic client ready")
else:
    print("[startup] WARNING: ANTHROPIC_API_KEY not set — LLM calls will fail")

# ---------------------------------------------------------------------------
# Seed knowledge base
# ---------------------------------------------------------------------------
KNOWN_FAILURES = [
    {
        "log": "ERROR connection refused host=ml-inference port=9000 timeout=30s",
        "root_cause": "ML inference service down. Last occurred Jan 15 due to certificate expiry. Fix: restart ml-inference service and renew cert.",
        "id": "failure_001"
    },
    {
        "log": "CRITICAL null pointer exception in VideoDecoder traceback follows",
        "root_cause": "VideoDecoder crash due to malformed input frame. Last occurred Feb 3 with corrupted H264 stream. Fix: add input validation before decoder.",
        "id": "failure_002"
    },
    {
        "log": "ERROR unauthorized access denied user=svc-account path=/admin",
        "root_cause": "Service account token expired. Last occurred Mar 1. Fix: rotate svc-account credentials and update secret store.",
        "id": "failure_003"
    },
    {
        "log": "WARN memory overflow detected heap_used=98% gc_pressure=high",
        "root_cause": "Memory leak in request handler. Last occurred Feb 20 after deploy v2.3.1. Fix: rollback to v2.3.0 or patch memory leak in handler.",
        "id": "failure_004"
    },
    {
        "log": "ERROR crash detected process=encoder exit_code=139 signal=SIGSEGV",
        "root_cause": "Encoder segfault due to buffer overflow. Last occurred Jan 28. Fix: update encoder library to v3.2.1 which patches the overflow.",
        "id": "failure_005"
    },
]


def seed_knowledge_base():
    if collection is None:
        return
    try:
        existing = collection.count()
        if existing > 0:
            print(f"[startup] knowledge base already has {existing} entries")
            return
        collection.add(
            ids=[f["id"] for f in KNOWN_FAILURES],
            documents=[f["log"] for f in KNOWN_FAILURES],
            metadatas=[{"root_cause": f["root_cause"]} for f in KNOWN_FAILURES],
        )
        print(f"[startup] seeded knowledge base with {len(KNOWN_FAILURES)} failure patterns")
    except Exception as e:
        print(f"[startup] could not seed knowledge base: {e}")


seed_knowledge_base()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class RCARequest(BaseModel):
    log_line: str
    score: float

class RCAResponse(BaseModel):
    log_line: str
    score: float
    similar_failures: list[dict]
    root_cause_hypothesis: str
    confidence: str

class FullAnalysisRequest(BaseModel):
    log_line: str

class AnomalyResult(BaseModel):
    score: float
    threshold: float
    is_anomaly: bool

class RCAResult(BaseModel):
    performed: bool
    root_cause_hypothesis: str | None = None
    confidence: str | None = None
    similar_failures: list[dict] | None = None
    latency_ms: int | None = None

class FullAnalysisResponse(BaseModel):
    log_line: str
    anomaly: AnomalyResult
    rca: RCAResult
    total_latency_ms: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def call_anomaly_detector(log_line: str) -> dict:
    """
    Call anomaly_detector service. Returns full response dict on success.
    Raises HTTPException on timeout or service unavailability.

    DO-178C note: explicit timeout + typed error handling = partition fault
    containment. A failed anomaly_detector cannot corrupt rca_service state.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(
                f"{ANOMALY_SERVICE_URL}/analyze",
                json={"log_line": log_line}
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="anomaly_detector timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"anomaly_detector error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"anomaly_detector unavailable: {str(e)}")


def retrieve_similar_failures(log_line: str, n: int = 2) -> list[dict]:
    if collection is None:
        return []
    try:
        results = collection.query(
            query_texts=[log_line],
            n_results=min(n, collection.count())
        )
        similar = []
        for i, doc in enumerate(results["documents"][0]):
            similar.append({
                "log": doc,
                "root_cause": results["metadatas"][0][i]["root_cause"],
                "distance": round(results["distances"][0][i], 4)
            })
        return similar
    except Exception as e:
        print(f"[rca] retrieval error: {e}")
        return []


def generate_hypothesis(log_line: str, score: float, similar: list[dict]) -> tuple[str, str]:
    if anthropic_client is None:
        return "API key not configured", "low"

    context = ""
    if similar:
        context = "\n\nSimilar past failures from our knowledge base:\n"
        for i, f in enumerate(similar, 1):
            context += f"\n{i}. Log: {f['log']}\n   Root cause: {f['root_cause']}\n   Similarity distance: {f['distance']}\n"

    prompt = f"""You are a systems reliability engineer analyzing a log anomaly in a safety-critical system.

Anomalous log line (anomaly score: {score:.4f}):
{log_line}
{context}
Based on the log and any similar past failures, provide:
1. Most likely root cause (1-2 sentences)
2. Immediate action to take (1 sentence)
3. Confidence level: high/medium/low

Respond in this exact format:
ROOT_CAUSE: <your analysis>
ACTION: <immediate step>
CONFIDENCE: <high/medium/low>"""

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text

        root_cause = "Unable to parse response"
        confidence = "low"
        action = ""

        for line in response.split("\n"):
            if line.startswith("ROOT_CAUSE:"):
                root_cause = line.replace("ROOT_CAUSE:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip().lower()

        if action:
            root_cause = f"{root_cause} Action: {action}"

        return root_cause, confidence

    except Exception as e:
        return f"LLM error: {str(e)}", "low"


async def run_rca(log_line: str, score: float) -> dict:
    """
    Single source of truth for RCA logic.
    Called by both /rca (manual/debug) and /full-analysis (automated pipeline).

    DO-178C note: one implementation, multiple callers = one requirement,
    multiple test cases. Adding a new trigger (e.g. Kafka consumer) means
    adding a new caller, not duplicating logic.
    """
    t0 = time.perf_counter()
    similar = retrieve_similar_failures(log_line)
    hypothesis, confidence = generate_hypothesis(log_line, score, similar)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    return {
        "performed": True,
        "root_cause_hypothesis": hypothesis,
        "confidence": confidence,
        "similar_failures": similar,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/knowledge-base/count")
async def kb_count():
    if collection is None:
        return {"count": 0, "status": "chromadb unavailable"}
    return {"count": collection.count(), "status": "ok"}


@app.post("/rca", response_model=RCAResponse)
async def analyze(req: RCARequest):
    """Manual RCA endpoint — caller supplies the anomaly score directly.
    Useful for debugging and testing RCA in isolation."""
    result = await run_rca(req.log_line, req.score)
    return RCAResponse(
        log_line=req.log_line,
        score=req.score,
        similar_failures=result["similar_failures"],
        root_cause_hypothesis=result["root_cause_hypothesis"],
        confidence=result["confidence"],
    )


@app.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(req: FullAnalysisRequest):
    """
    End-to-end pipeline endpoint.
    Stage 1: calls anomaly_detector to score the log line.
    Stage 2: runs RCA only if anomaly score exceeds threshold.

    DO-178C note: deterministic component (autoencoder) gates the
    non-deterministic component (LLM). Keeps certifiable surface small.
    RCA cost (latency + API $) is only paid when warranted.
    """
    t_start = time.perf_counter()

    # Stage 1: anomaly detection
    anomaly_raw = await call_anomaly_detector(req.log_line)
    anomaly = AnomalyResult(
        score=anomaly_raw["score"],
        threshold=anomaly_raw["threshold"],
        is_anomaly=anomaly_raw["is_anomaly"],
    )

    # Stage 2: RCA — only if anomalous
    if anomaly.is_anomaly:
        rca_data = await run_rca(req.log_line, anomaly.score)
        rca = RCAResult(**rca_data)
    else:
        rca = RCAResult(performed=False)

    total_ms = round((time.perf_counter() - t_start) * 1000)

    return FullAnalysisResponse(
        log_line=req.log_line,
        anomaly=anomaly,
        rca=rca,
        total_latency_ms=total_ms,
    )