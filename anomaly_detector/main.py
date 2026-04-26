"""
Anomaly Detector Service
------------------------
Day 4 capstone: PyTorch autoencoder that scores log lines for anomalies.
Reconstruction error > threshold => anomalous.

Day 8 update: added IFE-style normal log training data, expanded threshold
calibration samples, added /retrain endpoint.

POST /analyze        { "log_line": "..." }  -> { "score": float, "is_anomaly": bool }
POST /analyze/batch  [ { "log_line": "..." }, ... ]
POST /retrain        {} -> { "threshold": float, "samples": int }
GET  /health         -> { "status": "ok" }
GET  /version        -> { "model_version": str }
"""
import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION = os.getenv("ANOMALY_MODEL_VERSION", "unknown")
print(f"[startup] model version: {MODEL_VERSION}")

app = FastAPI(title="Log Anomaly Detector", version="0.2.0")


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
# Training data
# ---------------------------------------------------------------------------

NORMAL_LOGS = [
    # --- Generic service logs (original) ---
    "INFO service started successfully",
    "INFO request complete status 200",
    "INFO health check ok latency=2ms",
    "INFO connection established host=db.internal",
    "INFO stop graceful shutdown complete",
    "INFO reconnect successful host=cache.internal",
    "INFO processing complete items=150 duration=120ms",
    "INFO user login successful id=1234",
    "DEBUG processing item id=5678 queue_depth=3",
    "DEBUG cache hit key=user_session_1234",
    "DEBUG config loaded from environment",
    "DEBUG heartbeat sent to coordinator",
    "WARN retry attempt 1 of 3 target=cache.internal",
    "WARN retry attempt 2 of 3 target=cache.internal",
    "WARN high latency detected threshold=500ms actual=620ms",
    "WARN disk usage 75 percent consider cleanup",
    "WARN connection pool near capacity used=45 max=50",
    "WARN retry attempt 1 of 3",
    "WARN retry attempt 2 of 3",
    "WARN retry attempt 3 of 3",
    "WARN retry attempt 1 of 3 target=cache.internal",
    "WARN retry attempt 1 of 3 target=db.internal",
    "WARN retry attempt 1 of 5 target=ml-service",

    # --- IFE system startup logs ---
    "INFO  2026-03-31T06:00:01Z [SYSTEM   ] ife_controller startup complete version=4.2.1 seat_count=312",
    "INFO  2026-03-31T06:00:02Z [NETWORK  ] arinc429_bus_init ok bus=A bus=B channel=1 channel=2",
    "INFO  2026-03-31T06:00:03Z [CONTENT  ] content_server connected host=ife-media.aa.internal port=8443",
    "INFO  2026-03-31T06:00:04Z [DISPLAY  ] seat_display_init ok zone=A seats=1-78 status=ready",
    "INFO  2026-03-31T06:00:05Z [DISPLAY  ] seat_display_init ok zone=B seats=79-198 status=ready",
    "INFO  2026-03-31T06:00:06Z [DISPLAY  ] seat_display_init ok zone=C seats=199-312 status=ready",
    "INFO  2026-03-31T06:00:07Z [AUDIO    ] audio_subsystem init ok sample_rate=48000 channels=stereo",
    "INFO  2026-03-31T06:00:08Z [HEALTH   ] self_test pass cpu=12% mem=34% storage=61%",
    "INFO  2026-03-31T06:00:10Z [NETWORK  ] passenger_wifi_init ok ssid=AA-Inflight band=5GHz clients=0",
    "INFO  2026-03-31T06:00:15Z [CONTENT  ] content_index loaded titles=847 languages=14 size_gb=312",

    # --- IFE normal operations ---
    "INFO  2026-03-31T06:00:20Z [DISPLAY  ] seat 14A display heartbeat ok latency=3ms",
    "INFO  2026-03-31T06:00:20Z [DISPLAY  ] seat 22C display heartbeat ok latency=2ms",
    "INFO  2026-03-31T06:00:20Z [DISPLAY  ] seat 31B display heartbeat ok latency=4ms",
    "INFO  2026-03-31T06:00:25Z [AUDIO    ] seat 8A headphone jack connected impedance=32ohm",
    "INFO  2026-03-31T06:00:30Z [CONTENT  ] video stream started seat=14A title_id=MOV-2041 codec=H264",
    "INFO  2026-03-31T06:00:31Z [CONTENT  ] video stream started seat=22C title_id=MOV-0892 codec=H264",
    "INFO  2026-03-31T06:00:35Z [NETWORK  ] arinc429 frame ok bus=A label=270 data=0x0A12 latency=1ms",
    "INFO  2026-03-31T06:00:40Z [HEALTH   ] watchdog tick ok uptime=39s all_services=nominal",
    "INFO  2026-03-31T06:01:00Z [CONTENT  ] video stream started seat=31B title_id=MOV-1103 codec=H265",
    "INFO  2026-03-31T06:01:05Z [DISPLAY  ] seat 45D display heartbeat ok latency=3ms",
    "INFO  2026-03-31T06:01:10Z [NETWORK  ] wifi clients=47 throughput=24Mbps signal=good",
    "INFO  2026-03-31T06:01:15Z [AUDIO    ] seat 14A volume adjusted level=7 source=passenger",
    "INFO  2026-03-31T06:01:20Z [CONTENT  ] subtitle track loaded seat=22C language=es title_id=MOV-0892",
    "INFO  2026-03-31T06:01:30Z [HEALTH   ] self_test pass cpu=28% mem=41% storage=61%",
    "INFO  2026-03-31T06:02:00Z [NETWORK  ] arinc429 frame ok bus=B label=270 data=0x0B44 latency=1ms",
    "INFO  2026-03-31T06:03:15Z [HEALTH   ] watchdog tick ok uptime=194s all_services=nominal",
    "INFO  2026-03-31T06:03:20Z [NETWORK  ] wifi clients=134 throughput=38Mbps signal=good",
    "INFO  2026-03-31T06:05:10Z [NETWORK  ] arinc429 frame ok bus=B label=041 data=0x0C33 latency=2ms",
    "INFO  2026-03-31T06:05:20Z [DISPLAY  ] seat 200A display heartbeat ok latency=3ms",
    "INFO  2026-03-31T06:05:30Z [HEALTH   ] watchdog tick ok uptime=329s all_services=nominal",
    "INFO  2026-03-31T06:06:20Z [HEALTH   ] self_test pass cpu=31% mem=43% storage=61%",
    "INFO  2026-03-31T06:06:30Z [NETWORK  ] wifi clients=201 throughput=52Mbps signal=good",
    "INFO  2026-03-31T06:07:00Z [DISPLAY  ] zone_A heartbeat ok seats=78 failures=0",
    "INFO  2026-03-31T06:07:00Z [DISPLAY  ] zone_B heartbeat ok seats=120 failures=0",
    "INFO  2026-03-31T06:07:00Z [DISPLAY  ] zone_C heartbeat ok seats=114 failures=0",
    "INFO  2026-03-31T06:07:10Z [AUDIO    ] seat 14A headphone jack disconnected",
    "INFO  2026-03-31T06:07:15Z [CONTENT  ] video stream ended seat=14A title_id=MOV-2041 watched_pct=94",
    "INFO  2026-03-31T06:07:20Z [HEALTH   ] watchdog tick ok uptime=439s all_services=nominal",
    "INFO  2026-03-31T06:07:30Z [SYSTEM   ] ife_controller status=nominal active_streams=201 wifi_clients=201",
]

# Threshold calibration samples — representative normal IFE lines
NORMAL_CALIBRATION_SAMPLES = [
    "INFO  2026-03-31T06:00:08Z [HEALTH   ] self_test pass cpu=12% mem=34% storage=61%",
    "INFO  2026-03-31T06:00:40Z [HEALTH   ] watchdog tick ok uptime=39s all_services=nominal",
    "INFO  2026-03-31T06:01:10Z [NETWORK  ] wifi clients=47 throughput=24Mbps signal=good",
    "INFO  2026-03-31T06:00:20Z [DISPLAY  ] seat 14A display heartbeat ok latency=3ms",
    "INFO  2026-03-31T06:00:30Z [CONTENT  ] video stream started seat=14A title_id=MOV-2041 codec=H264",
    "INFO  2026-03-31T06:07:00Z [DISPLAY  ] zone_A heartbeat ok seats=78 failures=0",
    "INFO  2026-03-31T06:00:35Z [NETWORK  ] arinc429 frame ok bus=A label=270 data=0x0A12 latency=1ms",
    "INFO service started successfully",
    "INFO health check ok latency=2ms",
    "INFO request complete status 200",
    "DEBUG heartbeat sent to coordinator",
    "WARN retry attempt 1 of 3 target=cache.internal",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: LogAutoencoder, epochs: int = 200):
    """
    Train on normal log vectors so the model learns what normal looks like.
    Includes both generic service logs and IFE-style structured logs.

    DO-178C note: training data diversity = coverage of the normal operating
    envelope. Narrow training data = high false positive rate = untrustworthy
    tool. Same principle as test case coverage in DO-178C MC/DC.
    """
    training_data = NORMAL_LOGS * 40  # repeat to give the model enough signal

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for log in training_data:
            x = log_to_vector(log)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()


def calibrate_threshold(model: LogAutoencoder) -> float:
    errors = [model.reconstruction_error(log_to_vector(s)) for s in NORMAL_CALIBRATION_SAMPLES]
    threshold = float(np.mean(errors) + 3 * np.std(errors))
    # Floor prevents collapse when model trains too cleanly on homogeneous data.
    # 0.02 chosen empirically — known anomalies score 0.15-0.27, normal IFE logs score ~0.04-0.12.
    MIN_THRESHOLD = 0.02
    return max(threshold, MIN_THRESHOLD)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

model = LogAutoencoder(input_dim=INPUT_DIM)

if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    print("[startup] loaded existing model weights")
else:
    print(f"[startup] training on {len(NORMAL_LOGS)} normal log patterns...")
    train_model(model)
    torch.save(model.state_dict(), "model.pt")
    print("[startup] trained and saved new model weights")

THRESHOLD = calibrate_threshold(model)
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

class RetrainResponse(BaseModel):
    threshold: float
    training_samples: int
    calibration_samples: int
    message: str

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

@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    """
    Retrain the model on the current NORMAL_LOGS dataset and recalibrate
    the threshold. Saves new model.pt. Useful during dev without restarting.

    DO-178C note: in a certified system, retraining would require full
    reverification. Here it's a dev convenience — document the distinction.
    """
    global THRESHOLD
    print("[retrain] starting retrain...")
    train_model(model)
    torch.save(model.state_dict(), "model.pt")
    THRESHOLD = calibrate_threshold(model)
    print(f"[retrain] complete — new threshold: {THRESHOLD:.6f}")
    return RetrainResponse(
        threshold=round(THRESHOLD, 6),
        training_samples=len(NORMAL_LOGS),
        calibration_samples=len(NORMAL_CALIBRATION_SAMPLES),
        message="retrain complete — model.pt updated, threshold recalibrated",
    )

@app.post("/analyze/batch")
def analyze_batch(logs: list[LogRequest]):
    return [analyze(r) for r in logs]

@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION, "app_version": app.version}