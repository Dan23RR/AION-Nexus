"""Example 4: minimal HTTP client for the AION-NEXUS REST API.

Assumes the server is running:
    AION_CHECKPOINT=checkpoints/aion_nexus_v1.pth uvicorn server.main:app --port 8080
"""
from __future__ import annotations

import sys
import json
import requests


def main(base_url: str = "http://localhost:8080") -> int:
    # Health probe
    print(f"GET {base_url}/health")
    r = requests.get(f"{base_url}/health", timeout=5)
    print("  ", r.status_code, r.json())

    # Predict via JSON body
    import numpy as np
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2, 2560)).tolist()

    print(f"POST {base_url}/predict")
    r = requests.post(
        f"{base_url}/predict",
        json={"signal": signal},
        timeout=10,
    )
    if r.status_code == 200:
        body = r.json()
        print(f"  predicted: {body['predicted_class_name']}  "
              f"conf={body['confidence']:.3f}  "
              f"latency={body['latency_ms']:.1f} ms")
        print(f"  recommended_action: {body['recommended_action']}")
    else:
        print(f"  error: {r.status_code}  {r.text}")
        return 1
    return 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    raise SystemExit(main(base))
