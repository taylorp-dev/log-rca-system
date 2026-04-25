"""
RCA Service
-----------
Day 5: LLM-powered root cause analysis.
"""

import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Log RCA Service", version="0.1.0")

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


# ---------------------------------------------------------------------------
# Core RCA logic
# ---------------------------------------------------------------------------

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
    if not ANTHROPIC_API_KEY:
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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
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


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/knowledge-base/count")
def kb_count():
    if collection is None:
        return {"count": 0, "status": "chromadb unavailable"}
    return {"count": collection.count(), "status": "ok"}

@app.post("/rca", response_model=RCAResponse)
def analyze(req: RCARequest):
    similar = retrieve_similar_failures(req.log_line)
    hypothesis, confidence = generate_hypothesis(req.log_line, req.score, similar)
    return RCAResponse(
        log_line=req.log_line,
        score=req.score,
        similar_failures=similar,
        root_cause_hypothesis=hypothesis,
        confidence=confidence,
    )