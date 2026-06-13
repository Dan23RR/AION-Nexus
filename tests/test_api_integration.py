"""API integration tests via FastAPI TestClient.

Verifies the REST API contract end-to-end: request validation, response
schema, error handling, edge cases.

Important state-management note: ``server.main.app`` is a process-level
singleton. To avoid state leakage between tests, we use an autouse fixture
that ensures every test starts with a freshly-injected `InferenceEngine`.
The single test that exercises the "no engine" code path saves and restores
state in its own scope.
"""
import os
import time

import numpy as np
import pytest
import torch

# Ensure no AION_* env vars from the outer shell influence server behavior
os.environ.pop("AION_CHECKPOINT", None)
os.environ.pop("AION_API_KEY", None)
os.environ.pop("AION_MAX_BODY_BYTES", None)
os.environ.pop("AION_CORS_ORIGINS", None)


@pytest.fixture(scope="module")
def app():
    """Module-shared app singleton."""
    from server.main import app as _app
    return _app


@pytest.fixture(autouse=True)
def fresh_engine(app):
    """Reset engine state before every test, restore after.

    This prevents state leakage between tests that mutate ``app.state.engine``.
    """
    from aion_nexus import InferenceEngine
    from aion_nexus.model import create_aion_nexus

    torch.manual_seed(0)
    saved_engine = getattr(app.state, "engine", None)
    saved_error = getattr(app.state, "startup_error", None)
    app.state.engine = InferenceEngine(create_aion_nexus())
    app.state.startup_error = None
    yield
    # Restore (mostly cosmetic; next test will re-create anyway)
    app.state.engine = saved_engine
    app.state.startup_error = saved_error


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# ---- Health / version endpoints ---------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("healthy", "degraded", "down")
        assert body["version"]
        assert body["model_param_count"] == 1_061_724

    def test_health_includes_architecture_version(self, client):
        body = client.get("/health").json()
        assert body["architecture_version"] == "v1"

    def test_health_architecture_version_v3_engine(self, client, app):
        """A v3 engine injected into the app must surface its architecture via /health
        and serve /predict (the server-side path for v3 checkpoints)."""
        from aion_nexus import InferenceEngine, create_substrate_v3

        saved = app.state.engine
        try:
            app.state.engine = InferenceEngine(
                create_substrate_v3(), architecture_version="v3"
            )
            body = client.get("/health").json()
            assert body["architecture_version"] == "v3"
            sig = np.random.default_rng(0).standard_normal((2, 2560)).tolist()
            assert client.post("/predict", json={"signal": sig}).status_code == 200
        finally:
            app.state.engine = saved

    def test_health_no_engine_returns_down(self, client, app):
        """Save engine, set None, verify down, restore — keeps fixture invariant."""
        saved = app.state.engine
        try:
            app.state.engine = None
            app.state.startup_error = "test stub"
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "down"
        finally:
            app.state.engine = saved
            app.state.startup_error = None


class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert "model" in body
        assert "api" in body


# ---- /predict endpoint ------------------------------------------------------

class TestPredictEndpointJson:
    def test_valid_signal_returns_200(self, client):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 2560)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "predicted_class_index" in body
        assert "predicted_class_name" in body
        assert "probabilities" in body
        assert "confidence" in body
        assert "model_version" in body

    def test_short_signal_returns_400(self, client):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((2, 1000)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 400

    def test_wrong_shape_returns_400(self, client):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((3, 2560)).tolist()  # 3 channels, not 2
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 400

    def test_stuck_sensor_returns_400(self, client):
        sig = np.zeros((2, 3000)).tolist()  # both channels zero -> stuck
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 400

    def test_no_body_returns_4xx(self, client):
        # FastAPI returns 422 when neither file nor JSON body is provided.
        # Accept any 4xx since exact code may depend on FastAPI version.
        r = client.post("/predict")
        assert 400 <= r.status_code < 500

    def test_response_probabilities_sum_to_one(self, client):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal((2, 2560)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        body = r.json()
        total = sum(body["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_response_confidence_in_unit_interval(self, client):
        rng = np.random.default_rng(4)
        sig = rng.standard_normal((2, 2560)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_confidence_band_valid(self, client):
        rng = np.random.default_rng(5)
        sig = rng.standard_normal((2, 2560)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        assert r.json()["confidence_band"] in ("low", "medium", "high")

    def test_invalid_signal_via_csv(self, client):
        """NaN/Inf cannot be expressed in standard JSON, so we test the
        invalid-signal path via /predict_csv, which uses np.loadtxt
        (NaN parses as np.nan, then validate_signal rejects it → 400)."""
        rows = []
        for i in range(2700):
            h = float("nan") if i == 100 else 0.5
            v = 0.3
            rows.append(f"0,0,0,0,{h},{v}")
        csv_bytes = ("\n".join(rows)).encode()
        r = client.post(
            "/predict_csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 400, r.text


class TestPredictCsvEndpoint:
    """Separate test class for CSV upload endpoint."""

    def test_valid_csv_returns_200(self, client):
        rng = np.random.default_rng(7)
        # Build valid 6-column FEMTO CSV with non-stuck data
        rows = []
        for _ in range(2700):
            h = rng.standard_normal()
            v = rng.standard_normal()
            rows.append(f"0,0,0,0,{h:.6f},{v:.6f}")
        csv_bytes = ("\n".join(rows)).encode()
        r = client.post(
            "/predict_csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "predicted_class_name" in body

    def test_short_csv_returns_400(self, client):
        rows = ["0,0,0,0,0.5,0.3" for _ in range(100)]
        csv_bytes = ("\n".join(rows)).encode()
        r = client.post(
            "/predict_csv",
            files={"file": ("short.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 400

    def test_malformed_csv_returns_400(self, client):
        csv_bytes = b"this is not csv\nat all"
        r = client.post(
            "/predict_csv",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 400


# ---- /predict_long_signal --------------------------------------------------

class TestPredictLongSignal:
    def test_long_signal_aggregation_mean(self, client):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 12_000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_windows"] >= 4
        assert body["aggregation_method"] == "mean"
        assert "aggregated_probabilities" in body

    def test_long_signal_majority_vote(self, client):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((2, 12_000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "majority", "window": 2560},
        )
        assert r.status_code == 200, r.text

    def test_invalid_aggregation_method(self, client):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((2, 12_000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "bogus", "window": 2560},
        )
        # Either 400 (validation) or 500 (raised inside handler) is acceptable
        # — both signal "invalid input" to the client.
        assert r.status_code >= 400


# ---- Security hardening ------------------------------------------------------

class TestApiKey:
    def test_401_without_key_when_set(self, client, monkeypatch):
        monkeypatch.setenv("AION_API_KEY", "test-secret")
        assert client.get("/version").status_code == 401
        sig = np.random.default_rng(0).standard_normal((2, 2560)).tolist()
        assert client.post("/predict", json={"signal": sig}).status_code == 401

    def test_200_with_correct_key(self, client, monkeypatch):
        monkeypatch.setenv("AION_API_KEY", "test-secret")
        r = client.get("/version", headers={"X-API-Key": "test-secret"})
        assert r.status_code == 200

    def test_401_with_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("AION_API_KEY", "test-secret")
        r = client.get("/version", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_health_exempt_from_api_key(self, client, monkeypatch):
        monkeypatch.setenv("AION_API_KEY", "test-secret")
        assert client.get("/health").status_code == 200

    def test_no_key_required_when_unset(self, client, monkeypatch):
        monkeypatch.delenv("AION_API_KEY", raising=False)
        assert client.get("/version").status_code == 200


class TestBodySizeLimit:
    def test_oversized_body_returns_413(self, client, monkeypatch):
        monkeypatch.setenv("AION_MAX_BODY_BYTES", "1024")
        sig = np.zeros((2, 2560)).tolist()  # JSON body far beyond 1 KiB
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 413
        assert "exceeds" in r.json()["detail"]

    def test_body_within_limit_passes(self, client, monkeypatch):
        monkeypatch.setenv("AION_MAX_BODY_BYTES", str(50 * 1024 * 1024))
        sig = np.random.default_rng(0).standard_normal((2, 2560)).tolist()
        assert client.post("/predict", json={"signal": sig}).status_code == 200


# ---- Observability -----------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_responds_and_counts_predictions(self, client):
        sig = np.random.default_rng(0).standard_normal((2, 2560)).tolist()
        assert client.post("/predict", json={"signal": sig}).status_code == 200
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "aion_predictions_total" in r.text

    def test_request_id_header_present(self, client):
        r = client.get("/health")
        assert r.headers.get("x-request-id")

    def test_request_id_propagated(self, client):
        r = client.get("/health", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("x-request-id") == "abc123"


# ---- Generic error handling -----------------------------------------------

class TestErrorHandling:
    def test_malformed_json_returns_4xx(self, client):
        r = client.post(
            "/predict",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert 400 <= r.status_code < 500

    def test_missing_signal_field_returns_4xx(self, client):
        r = client.post("/predict", json={"wrong_field": [[1, 2, 3]]})
        assert 400 <= r.status_code < 500


# ---- Red-team kill-shots: DoS, ragged-list 500, robustness -------------------

class TestKillShotDosLongSignal:
    """Kill-shot #1: /predict_long_signal CPU-DoS via huge signal + tiny stride.

    A ~11.5k-sample signal at stride=1 expands to ~8941 windows, each a full
    model forward, pinning a worker for >115 s. The schema-level window-count
    cap (AION_MAX_WINDOWS) must reject this BEFORE any allocation/forward, fast.
    """

    def test_dos_long_signal_rejected_fast(self, client):
        # The exact red-team attack: ~11.5k samples, stride=1 -> ~8941 windows.
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 11_500)).tolist()
        t0 = time.perf_counter()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560, "stride": 1},
        )
        elapsed = time.perf_counter() - t0
        # Rejected at validation (422) — never runs thousands of forwards.
        assert r.status_code == 422, r.text
        # Must be fast: schema validation only, no model work. Generous bound to
        # avoid flakiness on a loaded CI box, but far below the >115 s attack.
        assert elapsed < 5.0, f"DoS request took {elapsed:.2f}s (should reject fast)"

    def test_dos_respects_custom_max_windows(self, client, monkeypatch):
        monkeypatch.setenv("AION_MAX_WINDOWS", "10")
        rng = np.random.default_rng(1)
        # Non-overlapping windows: 12000 // 2560 ~ 4 windows -> under a cap of 10,
        # so this should PASS; then a stride=1 over the same length blows past 10.
        sig = rng.standard_normal((2, 12_000)).tolist()
        ok = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        assert ok.status_code == 200, ok.text
        bad = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560, "stride": 1},
        )
        assert bad.status_code == 422

    def test_zero_stride_rejected(self, client):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((2, 6000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560, "stride": 0},
        )
        # stride must be > 0 (gt=0) — rejected at validation, no infinite loop.
        assert r.status_code == 422

    def test_tiny_window_rejected(self, client):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal((2, 6000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 1, "stride": 1},
        )
        # window must be >= 2560 — blocks the window=0/1 stride-explosion attack.
        assert r.status_code == 422

    def test_normal_long_signal_still_works(self, client):
        """A legitimate multi-second request is unaffected by the cap."""
        rng = np.random.default_rng(4)
        sig = rng.standard_normal((2, 12_000)).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        assert r.status_code == 200, r.text
        assert r.json()["n_windows"] >= 4


class TestKillShotRaggedList:
    """Kill-shot #2: a ragged signal list ([[1,2,3],[1]]) makes np.asarray raise
    ValueError BEFORE the engine try/except -> previously an unhandled 500 with a
    stacktrace. Must now be a clean 400."""

    def test_ragged_list_predict_returns_400(self, client):
        # Long enough to look real if it parsed, but the rows are ragged.
        sig = [[1.0, 2.0, 3.0], [1.0]]
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 400, r.text
        assert "ragged" in r.json()["detail"].lower() or "malformed" in r.json()["detail"].lower()

    def test_ragged_list_long_signal_returns_400(self, client):
        sig = [[1.0, 2.0, 3.0], [1.0, 2.0]]
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        # Either a clean 400 (ragged caught in handler) or 422 (schema window
        # check on the short signal) — both are non-500, actionable rejections.
        assert r.status_code in (400, 422), r.text
        assert r.status_code != 500

    def test_ragged_does_not_500(self, client):
        """Explicitly assert NO 500 for the canonical red-team payload."""
        r = client.post("/predict", json={"signal": [[1, 2, 3], [1]]})
        assert r.status_code != 500


class TestKillShotOodAbstain:
    """Kill-shot #4: white noise must NOT escalate to an automated action.
    The /predict response now carries ood_flag/abstain and a no-escalation
    recommended_action for implausible inputs."""

    def test_noise_response_abstains(self, client):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 2560)).tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ood_flag"] is True
        assert body["abstain"] is True
        assert body["recommended_action"]["alert_level"] == 0
        assert body["recommended_action"]["stop_machine"] is False

    def test_ood_fields_present_in_response(self, client):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((2, 2560)).tolist()
        body = client.post("/predict", json={"signal": sig}).json()
        for key in ("ood_flag", "ood_score", "ood_reason", "abstain"):
            assert key in body

    def test_noise_long_signal_does_not_escalate(self, client):
        """Kill-shot #4 on the AGGREGATED path: a multi-second white-noise
        recording must not escalate /predict_long_signal to an automated action.

        Regression guard: the handler used to derive recommended_action straight
        from CLASS_ACTIONS[aggregated_class], ignoring the per-window abstain, so
        pure noise escalated to alert_level >= 1 even though every window was
        individually flagged OOD. The aggregated OOD verdict must now abstain.
        """
        rng = np.random.default_rng(0)
        sig = (rng.standard_normal((2, 7680)) * 0.5).tolist()  # 3 windows of noise
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Majority of windows are noise -> aggregated gate must fire.
        assert body["ood_flag"] is True
        assert body["abstain"] is True
        assert body["ood_windows"] == body["n_windows"]
        assert body["ood_fraction"] == 1.0
        # The action must NOT escalate: alert_level 0, no stop / inspection.
        action = body["recommended_action"]
        assert action["alert_level"] == 0
        assert action["stop_machine"] is False
        assert action.get("schedule_inspection", False) is False
        assert action.get("abstain") is True

    def test_real_long_signal_still_escalates_normally(self, client):
        """The aggregated OOD gate must be a tight net: a plausible (spectrally
        structured) signal still gets its normal, possibly-escalating action and
        is NOT spuriously abstained.

        Uses a strongly tonal signal (low spectral flatness, like real bearing
        vibration) so the gate does NOT fire and recommended_action is the
        class-derived action (whatever class the model picks)."""
        t = np.linspace(0, 0.1, 7680, endpoint=False)
        # Multi-tone + small noise: spectrally peaky => flatness well below 0.45.
        ch = (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 320 * t)
              + 0.05 * np.random.default_rng(3).standard_normal(7680))
        sig = np.stack([ch, ch * 0.9]).tolist()
        r = client.post(
            "/predict_long_signal",
            json={"signal": sig, "aggregation": "mean", "window": 2560},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ood_flag"] is False
        assert body["abstain"] is False
        # Action present and not forced to the abstain shape.
        assert "alert_level" in body["recommended_action"]
