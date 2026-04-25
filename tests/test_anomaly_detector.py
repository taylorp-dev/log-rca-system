"""
Tests for the anomaly detector service.
Run locally:  pytest tests/
Run in CI:    docker compose run anomaly_detector pytest /app/../tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../anomaly_detector"))

import pytest
from fastapi.testclient import TestClient
from main import app, log_to_vector, INPUT_DIM

client = TestClient(app)


# ── Unit tests ───────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_log_to_vector_shape():
    vec = log_to_vector("INFO service started successfully")
    assert vec.shape[0] == INPUT_DIM

def test_log_to_vector_contains_info_feature():
    vec = log_to_vector("INFO request complete")
    # "info" is in VOCAB; its position should be 1.0
    from main import VOCAB
    idx = VOCAB.index("info")
    assert vec[idx].item() == 1.0

def test_log_to_vector_length_feature_capped():
    long_log = "x" * 1000
    vec = log_to_vector(long_log)
    assert vec[-1].item() == 1.0  # capped at 1.0

# ── Integration tests ────────────────────────────────────────────────────────

def test_analyze_normal_log_not_anomaly():
    resp = client.post("/analyze", json={"log_line": "INFO request complete status 200"})
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "is_anomaly" in data
    assert data["is_anomaly"] is False  # normal log should not trigger

def test_analyze_anomalous_log_flagged():
    resp = client.post("/analyze", json={
        "log_line": "CRITICAL null pointer exception crash traceback overflow error fail"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_anomaly"] is True  # should be flagged

def test_analyze_batch():
    logs = [
        {"log_line": "INFO health check ok"},
        {"log_line": "ERROR connection refused timeout crash"},
    ]
    resp = client.post("/analyze/batch", json=logs)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    # First should be normal, second anomalous
    assert results[0]["is_anomaly"] is False
    assert results[1]["is_anomaly"] is True

def test_score_is_higher_for_anomalous():
    normal = client.post("/analyze", json={"log_line": "INFO request complete status 200"}).json()
    anomalous = client.post("/analyze", json={"log_line": "CRITICAL null crash traceback fail error"}).json()
    assert anomalous["score"] > normal["score"]
