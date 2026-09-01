import time

import pytest
import requests


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


def test_check_weights_retries_legacy_schema_on_bad_request(monkeypatch):
    """Older SGLang builds reject selector/skip-list fields on the checker."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    requests_seen = []

    response = requests.Response()
    response.status_code = 400
    response._content = b"legacy schema"
    error = requests.exceptions.HTTPError(response=response)

    def make_request(endpoint, payload):
        requests_seen.append((endpoint, payload))
        if len(requests_seen) == 1:
            raise error
        return {"ok": True}

    monkeypatch.setattr(engine, "_make_request", make_request)

    assert engine.check_weights(action="compare", selector="target", skip_list=["derived"]) == {"ok": True}
    assert requests_seen == [
        (
            "weights_checker",
            {
                "action": "compare",
                "allow_quant_error": False,
                "selector": "target",
                "skip_tensor_list": ["derived"],
            },
        ),
        ("weights_checker", {"action": "compare", "allow_quant_error": False}),
    ]
