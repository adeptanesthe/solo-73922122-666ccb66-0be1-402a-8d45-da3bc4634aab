"""HTTP-level smoke test for the FastAPI app (no server process needed)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.samples import SAMPLES  # noqa: E402

client = TestClient(app)


def test_index_and_health():
    assert client.get("/healthz").json() == {"status": "ok"}
    html = client.get("/")
    assert html.status_code == 200
    assert "共享词嵌入" in html.text
    assert client.get("/static/index.html").status_code == 200


def test_samples_endpoint_and_each_sample_passes():
    body = client.get("/api/samples").json()
    assert len(body["samples"]) == len(SAMPLES) == 4
    for sample in body["samples"]:
        resp = client.post("/api/run", json=sample["payload"])
        assert resp.status_code == 200, (sample["id"], resp.text)
        report = resp.json()
        assert report["ok"], (sample["id"], report["errors"], report["split"])
        assert report["errors"]["loss_abs_diff"] <= 1e-8
        assert report["errors"]["grad_max_abs_diff"] <= 1e-8
        assert report["errors"]["weight_max_abs_diff"] <= 1e-8


def _run_payload(seed):
    return {
        "seed": seed,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"name": "mb", "tokens": [[1, 2], [3, 4], [5, -100]]}],
    }


def test_seed_beyond_2_pow_53_kept_exact_over_http():
    # JSON integer input keeps working and is echoed digit-exactly.
    resp = client.post("/api/run", json=_run_payload(9007199254740993))
    assert resp.status_code == 200
    assert "9007199254740993" in resp.text  # raw JSON body is digit-exact
    report = resp.json()
    assert report["config"]["seed"] == 9007199254740993
    assert report["config"]["seed_text"] == "9007199254740993"
    assert report["ok"]

    # Decimal-string input (what the fixed web form submits) is equivalent.
    resp_str = client.post("/api/run", json=_run_payload("9007199254740993"))
    assert resp_str.status_code == 200
    assert resp_str.json()["config"]["seed"] == 9007199254740993
    assert resp_str.json()["chunked"]["loss"] == report["chunked"]["loss"]


def test_seed_out_of_range_rejected():
    for bad in (9223372036854775808, "9223372036854775808", -1, "-1"):
        resp = client.post("/api/run", json=_run_payload(bad))
        assert resp.status_code == 422, bad
        assert resp.json()["detail"]
    # The largest legal seed still runs.
    resp = client.post("/api/run", json=_run_payload(9223372036854775807))
    assert resp.status_code == 200
    assert resp.json()["config"]["seed"] == 9223372036854775807


def test_seeds_differing_only_in_high_bits_give_different_weights():
    low = client.post("/api/run", json=_run_payload(7)).json()
    high = client.post("/api/run", json=_run_payload(4294967303)).json()  # 7 + 2**32
    assert low["ok"] and high["ok"]
    assert low["chunked"]["weight_norm"] != high["chunked"]["weight_norm"]
    assert low["chunked"]["loss"] != high["chunked"]["loss"]
    # Same config re-run is bit-identical.
    again = client.post("/api/run", json=_run_payload(4294967303)).json()
    assert again["chunked"]["loss"] == high["chunked"]["loss"]
    assert again["dense"]["weight_norm"] == high["dense"]["weight_norm"]


def test_invalid_requests_return_422():
    cases = [
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, -100]]}]},
        {"seed": 0, "scale": 200, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 128,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}] * 5},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]] * 129}]},
    ]
    for payload in cases:
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["detail"]


def test_flat_tokens_shorthand():
    resp = client.post("/api/run", json={
        "seed": 1, "scale": 1, "vocab_size": 256, "tokens": [[1, 2], [3, -100]],
    })
    assert resp.status_code == 200
    assert resp.json()["ok"]
