"""The fal queue client: submit, poll, fetch, cancel.

Each test names the bug it catches.
"""

from __future__ import annotations

import pytest

from api_types import ApiActivityStatus
from services.generation_interrupt import GenerationCancelledError
from services.http_client.http_client import HttpTransportError
from services.zit_api_client.zit_api_client import ZitImageBlockedError
from services.zit_api_client import zit_api_client_impl
from services.zit_api_client.zit_api_client_impl import ZitAPIClientImpl
from tests.fakes import FakeResponse
from tests.fakes.services import FakeHTTPClient

_STATUS_URL = "https://queue.fal.run/fal-ai/z-image/requests/r1/status"
_RESULT_URL = "https://queue.fal.run/fal-ai/z-image/requests/r1"
_CANCEL_URL = "https://queue.fal.run/fal-ai/z-image/requests/r1/cancel"
_IMAGE_URL = "https://fal.test/out.png"
_SUBMITTED = {
    "request_id": "r1",
    "status_url": _STATUS_URL,
    "response_url": _RESULT_URL,
    "cancel_url": _CANCEL_URL,
}


def _status(value: str, **extra: object) -> FakeResponse:
    return FakeResponse(status_code=200, json_payload={"status": value, **extra})


def _result(**extra: object) -> FakeResponse:
    payload = {"images": [{"url": _IMAGE_URL}], "has_nsfw_concepts": [False], **extra}
    return FakeResponse(status_code=200, json_payload=payload)


def _client(http: FakeHTTPClient, *, timeout: float = 30.0) -> ZitAPIClientImpl:
    return ZitAPIClientImpl(http, queue_poll_interval_seconds=0.0, queue_timeout_seconds=timeout)


def _submit(
    client: ZitAPIClientImpl,
    *,
    expansion: bool = False,
    on_status=None,
    on_submitted=None,
    wait_cancelled=None,
):
    return client.generate_text_to_image_queued(
        api_key="k",
        prompt="a fox",
        width=512,
        height=512,
        seed=1,
        num_inference_steps=8,
        enable_prompt_expansion=expansion,
        on_status=on_status or (lambda _status: None),
        on_submitted=on_submitted or (lambda _request_id: True),
        wait_cancelled=wait_cancelled or (lambda _seconds: False),
    )


def _queue_happy_path(http: FakeHTTPClient, *polls: FakeResponse, result: FakeResponse | None = None) -> None:
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", *polls, result or _result(), FakeResponse(content=b"png-bytes"))


def test_the_urls_fal_returns_are_used_not_built_ones() -> None:
    # Bug caught: the client rebuilds the status and result URLs and misses a sub-path endpoint.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("COMPLETED"))

    image = _submit(_client(http))

    assert image.image_bytes == b"png-bytes"
    assert [(call.method, call.url) for call in http.calls] == [
        ("post", "https://queue.fal.run/fal-ai/z-image/turbo"),
        ("get", _STATUS_URL),
        ("get", _RESULT_URL),
        ("get", _IMAGE_URL),
    ]


@pytest.mark.parametrize("expansion", [False, True])
def test_the_expansion_flag_reaches_the_request(expansion: bool) -> None:
    # Bug caught: enable_prompt_expansion is dropped, so fal bills or skips an expansion by accident.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("COMPLETED"))

    _submit(_client(http), expansion=expansion)

    payload = http.calls[0].json_payload
    assert payload is not None and payload["enable_prompt_expansion"] is expansion


def test_fal_statuses_map_to_the_api_activity_statuses() -> None:
    # Bug caught: the panel shows the wrong provider phase, or none.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("IN_QUEUE"), _status("IN_PROGRESS"), _status("COMPLETED"))
    seen: list[ApiActivityStatus] = []

    _submit(_client(http), on_status=seen.append)

    assert seen == ["pending", "processing"]


def test_the_prompt_fal_used_is_returned() -> None:
    # Bug caught: an expanded prompt is lost, so the finished row shows only the typed prompt.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("COMPLETED"), result=_result(prompt="a fox, expanded"))

    assert _submit(_client(http), expansion=True).prompt == "a fox, expanded"


def test_an_error_on_a_completed_request_fails_and_cancels() -> None:
    # Bug caught: COMPLETED with an error is read as a success and the result fetch fails oddly.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", _status("COMPLETED", error="model crashed"))
    http.queue("put", FakeResponse(status_code=400, text="ALREADY_COMPLETED"))

    with pytest.raises(RuntimeError, match="model crashed"):
        _submit(_client(http))

    assert http.calls[-1].method == "put" and http.calls[-1].url == _CANCEL_URL


def test_a_request_that_never_completes_times_out_and_is_cancelled() -> None:
    # Bug caught: a stuck provider request holds the fal lane forever and keeps billing.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", _status("IN_QUEUE"))
    http.queue("put", FakeResponse(status_code=202))

    with pytest.raises(RuntimeError, match="timed out"):
        _submit(_client(http, timeout=0.0))

    assert http.calls[-1].method == "put" and http.calls[-1].url == _CANCEL_URL


@pytest.mark.parametrize(
    "cancel_response",
    [FakeResponse(status_code=202), FakeResponse(status_code=400, text="ALREADY_COMPLETED")],
    ids=["accepted", "already-completed"],
)
def test_cancel_stops_the_poll_and_asks_fal_to_drop_the_request(cancel_response: FakeResponse) -> None:
    # Bug caught: a cancelled job keeps polling, or a 400 on cancel replaces the cancel error.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", _status("IN_PROGRESS"))
    http.queue("put", cancel_response)

    answers = iter([False, True])

    with pytest.raises(GenerationCancelledError):
        _submit(_client(http), wait_cancelled=lambda _seconds: next(answers))

    assert [call.method for call in http.calls] == ["post", "get", "put"]
    assert http.calls[-1].url == _CANCEL_URL


def test_a_failed_cancel_request_does_not_hide_the_cancel() -> None:
    # Bug caught: a transport error on the cancel call turns a cancel into a failure.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", _status("IN_QUEUE"))
    http.queue("put", HttpTransportError("down"))
    answers = iter([False, True])

    with pytest.raises(GenerationCancelledError):
        _submit(_client(http), wait_cancelled=lambda _seconds: next(answers))


def test_one_dropped_status_check_is_not_an_error() -> None:
    # Bug caught: a single network blip fails a job that fal still runs.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", HttpTransportError("blip"), _status("COMPLETED"), _result(), FakeResponse(content=b"png"))

    assert _submit(_client(http)).image_bytes == b"png"


def test_three_dropped_status_checks_in_a_row_fail_the_job() -> None:
    # Bug caught: the client polls a dead connection forever.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", *[HttpTransportError("down")] * 3)
    http.queue("put", FakeResponse(status_code=202))

    with pytest.raises(RuntimeError, match="status check failed"):
        _submit(_client(http))


@pytest.mark.parametrize(
    "result",
    [
        FakeResponse(json_payload={"images": [], "has_nsfw_concepts": []}),
        FakeResponse(json_payload={"images": [{"url": _IMAGE_URL}], "has_nsfw_concepts": [True]}),
    ],
    ids=["no-images", "nsfw-flag-on-black-image"],
)
def test_a_blocked_result_raises_instead_of_returning_the_placeholder(result: FakeResponse) -> None:
    # Bug caught: a black placeholder from the safety checker is saved as the user's image.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("get", _status("COMPLETED"), result)

    with pytest.raises(ZitImageBlockedError):
        _submit(_client(http))

    # The request finished. There is nothing to cancel at the provider.
    assert [call.method for call in http.calls] == ["post", "get", "get"]


@pytest.mark.parametrize("key", ["status_url", "response_url", "cancel_url"])
def test_a_response_url_on_another_host_is_refused_so_the_key_is_never_sent_there(key: str) -> None:
    # Bug caught: the client sends the fal API key to whatever host the submit response names.
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(status_code=200, json_payload={**_SUBMITTED, key: "https://evil.example/steal"}),
    )

    with pytest.raises(RuntimeError, match="unexpected host"):
        _submit(_client(http))

    assert [call.method for call in http.calls] == ["post"]


def test_a_status_page_that_is_not_json_is_retried_like_any_transient_failure() -> None:
    # Bug caught: a proxy page with status 200 raises ValueError and ends a job fal still runs.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue(
        "get",
        FakeResponse(status_code=200, json_raises=True),
        _status("COMPLETED"),
        _result(),
        FakeResponse(content=b"png-bytes"),
    )

    assert _submit(_client(http)).image_bytes == b"png-bytes"


@pytest.mark.parametrize(
    ("answers", "expected_methods"),
    [
        ([True], ["post", "put"]),
        ([False, True], ["post", "get", "put"]),
        ([False, False, True], ["post", "get", "get", "put"]),
    ],
    ids=["cancel-before-status", "cancel-before-result", "cancel-before-download"],
)
def test_a_cancel_that_lands_during_a_request_stops_before_the_next_one(
    answers: list[bool], expected_methods: list[str]
) -> None:
    # Bug caught: a cancel during a request still runs the next status, result or download request.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("COMPLETED"))
    http.queue("put", FakeResponse(status_code=400, text="ALREADY_COMPLETED"))
    remaining = iter(answers)

    with pytest.raises(GenerationCancelledError):
        _submit(_client(http), wait_cancelled=lambda _seconds: next(remaining))

    assert [call.method for call in http.calls] == expected_methods
    assert _IMAGE_URL not in [call.url for call in http.calls]


def test_the_request_id_is_reported_before_the_first_status_check() -> None:
    # Bug caught: the id is stored late, so a crash during the first poll leaves a request fal still runs.
    http = FakeHTTPClient()
    _queue_happy_path(http, _status("COMPLETED"))
    seen_calls: list[int] = []

    _submit(
        _client(http),
        on_submitted=lambda request_id: seen_calls.append(len(http.calls)) or request_id == "r1",
    )

    assert seen_calls == [1]  # Only the submit had run.


def test_a_request_id_that_is_not_stored_cancels_the_request_at_once() -> None:
    # Bug caught: a boot cannot cancel a request it has no id for, so fal keeps running and billing it.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue("put", FakeResponse(status_code=202))

    with pytest.raises(RuntimeError, match="not stored"):
        _submit(_client(http), on_submitted=lambda _request_id: False)

    assert [(call.method, call.url) for call in http.calls] == [
        ("post", "https://queue.fal.run/fal-ai/z-image/turbo"),
        ("put", _CANCEL_URL),
    ]


@pytest.mark.parametrize("request_id", [None, "", "../x", "a/b", "a b"])
def test_a_submit_response_without_a_usable_request_id_fails_before_any_poll(request_id: object) -> None:
    # Bug caught: a bad id reaches the store and later becomes part of a cancel URL.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=200, json_payload={**_SUBMITTED, "request_id": request_id}))

    with pytest.raises(RuntimeError, match="request_id"):
        _submit(_client(http))

    assert [call.method for call in http.calls] == ["post"]


@pytest.mark.parametrize("status", [200, 202, 400, 404])
def test_a_boot_cancel_uses_the_url_fal_returns_and_accepts_every_done_status(status: int) -> None:
    # Bug caught: the derived cancel URL drifts from the one fal returns, so a boot cancel hits nothing.
    http = FakeHTTPClient()
    http.queue("put", FakeResponse(status_code=status))

    _client(http).cancel_queued(api_key="k", request_id="r1")

    assert [(call.method, call.url) for call in http.calls] == [("put", _CANCEL_URL)]
    assert http.calls[0].headers is not None and http.calls[0].headers["Authorization"] == "Key k"


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_a_boot_cancel_that_fails_raises_so_the_id_is_kept(status: int) -> None:
    # Bug caught: a failed cancel is read as done, the id is cleared and the request runs on.
    http = FakeHTTPClient()
    http.queue("put", FakeResponse(status_code=status))

    with pytest.raises(RuntimeError, match=str(status)):
        _client(http).cancel_queued(api_key="k", request_id="r1")


def test_a_boot_cancel_refuses_an_id_that_is_not_a_request_id() -> None:
    # Bug caught: an id with a path in it sends the key to another fal path.
    http = FakeHTTPClient()

    with pytest.raises(ValueError):
        _client(http).cancel_queued(api_key="k", request_id="../../x")

    assert http.calls == []


def _queue_result_then_image(http: FakeHTTPClient, image_url: str, image: FakeResponse) -> None:
    http.queue("post", FakeResponse(status_code=200, json_payload=_SUBMITTED))
    http.queue(
        "get",
        _status("COMPLETED"),
        FakeResponse(status_code=200, json_payload={"images": [{"url": image_url}]}),
        image,
    )


def test_an_image_url_that_is_not_https_is_not_fetched() -> None:
    # Bug caught: a plain-http or file-like URL in the fal result is fetched (CWE-918).
    http = FakeHTTPClient()
    _queue_result_then_image(http, "http://fal.test/out.png", FakeResponse(content=b"png-bytes"))

    with pytest.raises(RuntimeError, match="not https"):
        _submit(_client(http))

    assert "http://fal.test/out.png" not in [call.url for call in http.calls]


def test_an_image_larger_than_the_cap_fails_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bug caught: a huge body from fal is read whole into memory (CWE-770).
    monkeypatch.setattr(zit_api_client_impl, "_MAX_IMAGE_DOWNLOAD_BYTES", 8)
    http = FakeHTTPClient()
    _queue_result_then_image(http, _IMAGE_URL, FakeResponse(content=b"123456789"))

    with pytest.raises(RuntimeError, match="too large"):
        _submit(_client(http))


def test_a_failed_download_shows_the_status_and_not_the_body() -> None:
    # Bug caught: the provider body lands in the job error that the user and the logs show (CWE-209).
    http = FakeHTTPClient()
    _queue_result_then_image(
        http, _IMAGE_URL, FakeResponse(status_code=500, text="internal detail: token abc")
    )

    with pytest.raises(RuntimeError) as raised:
        _submit(_client(http))

    assert "500" in str(raised.value)
    assert "internal detail" not in str(raised.value)
