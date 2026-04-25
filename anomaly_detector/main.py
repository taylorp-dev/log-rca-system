"""
Anomaly Detector Service
------------------------
Day 4 capstone: PyTorch autoencoder that scores log lines for anomalies.
Reconstruction error > threshold => anomalous.

POST /analyze   { "log_line": "..." }  -> { "score": float, "is_anomaly": bool }
GET  /health    -> { "status": "ok" }
"""
import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION = os.getenv("ANOMALY_MODEL_VERSION", "unknown")
print(f"[startup] model version: {MODEL_VERSION}")

app = FastAPI(title="Log Anomaly Detector", version="0.1.0")


# ---------------------------------------------------------------------------
# Feature extraction
# Simple bag-of-words style feature vector from a log line.
# Replace this with embeddings (sentence-transformers) in Week 2.
# ---------------------------------------------------------------------------

VOCAB = [
    "error", "warn", "info", "debug", "critical", "fail", "timeout",
    "exception", "traceback", "null", "none", "refused", "reset",
    "success", "ok", "complete", "start", "stop", "retry", "crash",
    "overflow", "denied", "unauthorized", "disconnect", "reconnect",
]

def log_to_vector(log_line: str) -> torch.Tensor:
    """Convert a log line to a fixed-size feature vector."""
    lower = log_line.lower()
    features = [1.0 if word in lower else 0.0 for word in VOCAB]
    # Add a length feature (normalized)
    features.append(min(len(log_line) / 500.0, 1.0))
    return torch.tensor(features, dtype=torch.float32)

INPUT_DIM = len(VOCAB) + 1  # 26


# ---------------------------------------------------------------------------
# Autoencoder model
# ---------------------------------------------------------------------------

class LogAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            recon = self(x.unsqueeze(0)).squeeze(0)
            return float(torch.mean((x - recon) ** 2).item())


# ---------------------------------------------------------------------------
# Quick synthetic training on startup
# In Week 2 you'll replace this with a saved checkpoint.
# ---------------------------------------------------------------------------

def train_model(model: LogAutoencoder, epochs: int = 200):
    """
    Train on synthetic 'normal' log vectors so the model learns what
    normal reconstruction looks like. Anomalous logs will have higher error.
    """
    normal_logs = [
    # Normal INFO logs
    "INFO service started successfully",
    "INFO request complete status 200",
    "INFO health check ok latency=2ms",
    "INFO connection established host=db.internal",
    "INFO stop graceful shutdown complete",
    "INFO reconnect successful host=cache.internal",
    "INFO processing complete items=150 duration=120ms",
    "INFO user login successful id=1234",
    # Normal DEBUG logs
    "DEBUG processing item id=5678 queue_depth=3",
    "DEBUG cache hit key=user_session_1234",
    "DEBUG config loaded from environment",
    "DEBUG heartbeat sent to coordinator",
    # Normal WARN logs — these are operational, not anomalous
    "WARN retry attempt 1 of 3 target=cache.internal",
    "WARN retry attempt 2 of 3 target=cache.internal",
    "WARN high latency detected threshold=500ms actual=620ms",
    "WARN disk usage 75 percent consider cleanup",
    "WARN connection pool near capacity used=45 max=50",
    # WARN retry variations — teach the model the pattern, not just one example
    "WARN retry attempt 1 of 3",
    "WARN retry attempt 2 of 3",
    "WARN retry attempt 3 of 3",
    "WARN retry attempt 1 of 3 target=cache.internal",
    "WARN retry attempt 1 of 3 target=db.internal",
    "WARN retry attempt 1 of 5 target=ml-service",
    ] * 40  # repeat to give the model enough signal

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for log in normal_logs:
            x = log_to_vector(log)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()


model = LogAutoencoder(input_dim=INPUT_DIM)

if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt"))
    print("[startup] loaded existing model weights")
else:
    train_model(model)
    torch.save(model.state_dict(), "model.pt")
    print("[startup] trained and saved new model weights")

# Calibrate threshold on normal data
NORMAL_SAMPLES = [
    "INFO request complete status 200",
    "INFO health check ok",
    "DEBUG processing item 5678",
    "INFO connection established",
]
errors = [model.reconstruction_error(log_to_vector(s)) for s in NORMAL_SAMPLES]
THRESHOLD = float(np.mean(errors) + 3 * np.std(errors))
print(f"[startup] anomaly threshold calibrated: {THRESHOLD:.6f}")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class LogRequest(BaseModel):
    log_line: str

class LogResponse(BaseModel):
    log_line: str
    score: float
    is_anomaly: bool
    threshold: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=LogResponse)
def analyze(req: LogRequest):
    vec = log_to_vector(req.log_line)
    score = model.reconstruction_error(vec)
    return LogResponse(
        log_line=req.log_line,
        score=round(score, 6),
        is_anomaly=score > THRESHOLD,
        threshold=round(THRESHOLD, 6),
    )

@app.post("/analyze/batch")
def analyze_batch(logs: list[LogRequest]):
    return [analyze(r) for r in logs]

@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION}