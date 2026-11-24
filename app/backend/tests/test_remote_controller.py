"""Exposure gates: auth token + settings toggle; isolated LAN bind."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from starlette.testclient import TestClient

from app_factory import create_app
from app_handler import ServiceBundle
from remote.app import create_remote_app
from remote.controller import (
    RemoteExposureController,
    _Running,
    _Stopping,
    remote_controller_from_app,
)
from services.records import UnavailableError
from services.sqlite_store import SqliteStore
from state import build_initial_state, get_state_service
from tests.conftest import DEFAULT_APP_SETTINGS
from tests.fakes.generation import FakeGenerationExecutor
from tests.fakes.services import FakeServices
from tests.http_error_assertions import assert_http_error


def _port_from_url(url: str) -> int:
    port = urlsplit(url).port
    assert port is not None and port > 0
    return port


def _wait_until_serving(client, *, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict | None = None
    while time.monotonic() < deadline:
        payload = client.get("/api/remote/status").json()
        if payload.get("serving") is True:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"remote did not start: {payload}")


def test_desktop_openapi_includes_remote_status(test_state) -> None:
    schema = create_app(handler=test_state).openapi()
    status = schema["paths"]["/api/remote/status"]["get"]
    assert status["tags"] == ["remote"]
    assert status["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RemoteStatusResponse"
    }
    assert "RemoteStatusResponse" in schema["components"]["schemas"]
    assert "/api/remote/devices" in schema["paths"]
    assert "/api/remote/devices/{device_id}/revoke" in schema["paths"]


def test_list_devices_store_outage_is_unavailable(client) -> None:
    controller = remote_controller_from_app(client.app)
    assert controller is not None

    def boom():
        raise UnavailableError()

    controller._pairing._store.list_paired_devices = boom  # type: ignore[method-assign]
    response = client.get("/api/remote/devices")
    assert_http_error(response, status_code=503, code="STORE_UNAVAILABLE")


def test_revoke_device_store_outage_is_unavailable(client) -> None:
    controller = remote_controller_from_app(client.app)
    assert controller is not None

    def boom(_device_id: str, *, now_ms: int):
        raise UnavailableError()

    controller._pairing._store.revoke_paired_device = boom  # type: ignore[method-assign]
    response = client.post("/api/remote/devices/any/revoke")
    assert_http_error(response, status_code=503, code="STORE_UNAVAILABLE")


def test_status_permitted_when_auth_token_present(client, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    payload = client.get("/api/remote/status").json()
    assert payload["permitted"] is True
    assert payload["serving"] is False
    assert payload["mode"] == "off"


def test_empty_auth_token_refuses_enable(client, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "")
    assert client.post("/api/settings", json={"remoteExposure": "lan"}).status_code == 200
    payload = client.get("/api/remote/status").json()
    assert payload["permitted"] is False
    assert payload["serving"] is False
    assert "LTX_AUTH_TOKEN" in (payload["reason"] or "")


def test_settings_on_binds_and_advertise_url(client, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    monkeypatch.setenv("LTX_REMOTE_PORT", "0")

    assert client.post("/api/settings", json={"remoteExposure": "lan"}).status_code == 200
    payload = _wait_until_serving(client)
    assert payload["permitted"] is True
    assert payload["serving"] is True
    assert payload["mode"] == "lan"
    assert payload["localUrl"] is not None
    port = _port_from_url(payload["localUrl"])
    assert f":{port}/pairing?t=" in payload["localUrl"]
    assert payload["url"] is not None
    assert payload["lanUrl"] is not None
    assert f":{port}/pairing?t=" in payload["lanUrl"]
    assert "tunnelUrl" not in payload
    assert "tunnelPending" not in payload

    # httpx, not urllib: Windows urllib can abort the socket (WinError 10053)
    # instead of returning the HTTP status for a 404 POST.
    grant = payload["localUrl"].split("?t=", 1)[1]
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2.0) as http:
        health = http.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        exchanged = http.post("/api/pairing/exchange", json={"code": grant})
        assert exchanged.status_code == 200
        token = exchanged.json()["token"]
        authed = {"Authorization": f"Bearer {token}"}

        generations = http.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers=authed,
        )
        assert generations.status_code == 200
        assert isinstance(generations.json(), list)

        unauth = http.get(
            "/api/generations",
            params={"feature": "text-to-video"},
        )
        assert unauth.status_code == 401

        legacy = http.post(
            "/api/generate",
            headers={**authed, "Content-Type": "application/json"},
            content=b"{}",
        )
        assert legacy.status_code == 404

    assert client.post("/api/settings", json={"remoteExposure": "off"}).status_code == 200
    stopped = client.get("/api/remote/status").json()
    assert stopped["serving"] is False
    controller = remote_controller_from_app(client.app)
    assert controller is not None
    assert controller.status().localUrl is None


def _independent_handler(test_state, tmp_path: Path, *, prompt_cache_size: int):
    fake_services = FakeServices()
    data_dir = tmp_path / "other-app"
    data_dir.mkdir()
    handler = build_initial_state(
        test_state.config,
        DEFAULT_APP_SETTINGS.model_copy(deep=True),
        service_bundle=ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            lora_catalog_provider=fake_services.lora_catalog_provider,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            ltx_api_client=fake_services.ltx_api_client,
            zit_api_client=fake_services.zit_api_client,
            fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            prompt_enhancer_pipeline_class=type(fake_services.prompt_enhancer_pipeline),
            store=SqliteStore(data_dir),
            generation_executors={
                "text-to-video": FakeGenerationExecutor(),
                "image-to-video": FakeGenerationExecutor(),
            },
        ),
    )
    handler.state.app_settings.prompt_cache_size = prompt_cache_size
    return handler


def test_independent_apps_do_not_share_handler_or_controller(
    test_state, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    monkeypatch.setenv("LTX_REMOTE_PORT", "0")

    handler_a = test_state
    handler_a.state.app_settings.prompt_cache_size = 11
    handler_b = _independent_handler(test_state, tmp_path, prompt_cache_size=22)

    app_a = create_app(handler=handler_a)
    app_b = create_app(handler=handler_b)

    assert remote_controller_from_app(app_a) is not remote_controller_from_app(app_b)
    assert app_a.dependency_overrides[get_state_service]() is handler_a
    assert app_b.dependency_overrides[get_state_service]() is handler_b

    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        assert client_a.get("/api/settings").json()["promptCacheSize"] == 11
        assert client_b.get("/api/settings").json()["promptCacheSize"] == 22

        assert client_a.post("/api/settings", json={"remoteExposure": "lan"}).status_code == 200
        payload_a = _wait_until_serving(client_a)
        payload_b = client_b.get("/api/remote/status").json()

        assert payload_a["serving"] is True
        assert payload_a["mode"] == "lan"
        assert payload_b["serving"] is False
        assert payload_b["mode"] == "off"
        assert handler_a.state.app_settings.remote_exposure == "lan"
        assert handler_b.state.app_settings.remote_exposure == "off"


def test_remote_app_does_not_replace_desktop_handler_or_controller(
    test_state, tmp_path: Path
) -> None:
    handler_desktop = test_state
    handler_desktop.state.app_settings.prompt_cache_size = 11
    handler_remote = _independent_handler(test_state, tmp_path, prompt_cache_size=22)

    desktop = create_app(handler=handler_desktop)
    desktop_controller = remote_controller_from_app(desktop)
    assert get_state_service() is handler_desktop
    assert desktop.dependency_overrides[get_state_service]() is handler_desktop

    remote = create_remote_app(handler=handler_remote, remote_token="pair-token")

    assert get_state_service() is handler_desktop
    assert remote_controller_from_app(desktop) is desktop_controller
    assert remote_controller_from_app(remote) is None
    assert remote.dependency_overrides[get_state_service]() is handler_remote
    assert desktop.dependency_overrides[get_state_service]() is handler_desktop

    with TestClient(desktop) as desktop_client, TestClient(remote) as remote_client:
        assert desktop_client.get("/api/settings").json()["promptCacheSize"] == 11
        listed = remote_client.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers={"Authorization": "Bearer pair-token"},
        )
        assert listed.status_code == 200
        assert listed.json() == []


class _FakeServer:
    def __init__(self, *, started: bool = True) -> None:
        self.started = started
        self.should_exit = False
        self.run_called = False

    def run(self) -> None:
        self.run_called = True


class _BlockingJoinThread:
    def __init__(self) -> None:
        self.join_started = threading.Event()
        self.release_join = threading.Event()
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_started.set()
        self.release_join.wait(timeout)
        self._alive = False


class _LiveJoinThread:
    def __init__(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self._alive = False


class _DeadThread:
    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        del timeout


class _JoinTimeoutLiveThread:
    def __init__(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self._alive = False


class _StartRecordingController(RemoteExposureController):
    def __init__(self, handler) -> None:
        super().__init__(handler)
        self.start_calls = 0
        self.started_servers: list[_FakeServer] = []

    def _start_locked(self) -> None:
        self.start_calls += 1
        server = _FakeServer(started=True)
        self.started_servers.append(server)
        self._ownership = _Running(
            server=server,
            thread=_LiveJoinThread(),
            port=41955,
        )
        self._exposure = self._handler.settings.get_settings_snapshot().remote_exposure
        self._last_reason = "starting"


def _install_running(
    controller: RemoteExposureController,
    *,
    server: _FakeServer | None = None,
    thread: object | None = None,
) -> tuple[_FakeServer, object]:
    owned_server = server or _FakeServer(started=True)
    owned_thread = thread or _LiveJoinThread()
    controller._ownership = _Running(
        server=owned_server,
        thread=owned_thread,
        port=41955,
    )
    controller._exposure = "lan"
    controller._last_reason = "starting"
    return owned_server, owned_thread


def test_status_stays_responsive_while_stop_joins(test_state, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    controller = RemoteExposureController(test_state)
    thread = _BlockingJoinThread()
    _install_running(controller, thread=thread)

    stopped = threading.Event()

    def _stop() -> None:
        controller.stop()
        stopped.set()

    threading.Thread(target=_stop, name="ltx-remote-stop-test", daemon=True).start()
    assert thread.join_started.wait(timeout=1.0)
    assert not stopped.is_set()

    started = time.monotonic()
    payload = controller.status()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert payload.serving is False
    assert payload.permitted is True
    assert not stopped.is_set()
    assert isinstance(controller._ownership, _Stopping)

    thread.release_join.set()
    assert stopped.wait(timeout=1.0)
    assert controller._ownership is None


def test_sync_does_not_start_replacement_until_stop_completes(
    test_state, monkeypatch
) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    test_state.state.app_settings.remote_exposure = "off"
    controller = _StartRecordingController(test_state)
    old_server = _FakeServer(started=True)
    thread = _BlockingJoinThread()
    _install_running(controller, server=old_server, thread=thread)

    first_sync_done = threading.Event()

    def _disable() -> None:
        controller.sync()
        first_sync_done.set()

    threading.Thread(target=_disable, name="ltx-remote-sync-stop", daemon=True).start()
    assert thread.join_started.wait(timeout=1.0)
    assert old_server.should_exit is True
    assert controller.start_calls == 0

    test_state.state.app_settings.remote_exposure = "lan"
    controller.sync()
    assert controller.start_calls == 0
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is old_server

    thread.release_join.set()
    assert first_sync_done.wait(timeout=1.0)
    assert controller.start_calls == 1
    assert controller.started_servers[0] is not old_server
    assert isinstance(controller._ownership, _Running)
    assert controller._ownership.server is controller.started_servers[0]


def test_run_server_swallows_uvicorn_bind_systemexit(test_state) -> None:
    """uvicorn.Server.startup calls sys.exit(1) on EADDRINUSE; that must not kill the thread."""
    controller = RemoteExposureController(test_state)

    class _ExitOnBind:
        def run(self) -> None:
            raise SystemExit(1)

    controller._run_server(_ExitOnBind())  # type: ignore[arg-type]


def test_run_server_uses_passed_instance_not_later_owner(test_state) -> None:
    controller = RemoteExposureController(test_state)
    old_server = _FakeServer()
    new_server = _FakeServer()
    _install_running(controller, server=new_server, thread=_DeadThread())

    controller._run_server(old_server)

    assert old_server.run_called is True
    assert new_server.run_called is False


def test_finish_stop_does_not_clear_a_replacement_server(test_state) -> None:
    controller = RemoteExposureController(test_state)
    old_server = _FakeServer()
    old_thread = _LiveJoinThread()
    new_server = _FakeServer()
    replacement = _Running(
        server=new_server,
        thread=_LiveJoinThread(),
        port=41955,
    )
    controller._ownership = replacement

    controller._finish_stop_locked(_Stopping(server=old_server, thread=old_thread))

    assert controller._ownership is replacement
    assert controller._ownership.server is new_server


def test_status_reaps_dead_running_thread(test_state, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    controller = RemoteExposureController(test_state)
    _install_running(controller, thread=_DeadThread())

    payload = controller.status()

    assert payload.serving is False
    assert payload.reason == "remote server stopped unexpectedly"
    assert controller._ownership is None


def test_status_does_not_reap_stopping_as_unexpected(test_state, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    controller = RemoteExposureController(test_state)
    server = _FakeServer(started=True)
    controller._ownership = _Stopping(server=server, thread=_DeadThread())
    controller._exposure = "lan"
    controller._last_reason = "stopped"

    payload = controller.status()

    assert payload.serving is False
    assert payload.reason == "stopped"
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is server


def test_stop_does_not_restart_when_setting_still_on(test_state, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    test_state.state.app_settings.remote_exposure = "lan"
    controller = _StartRecordingController(test_state)
    _install_running(controller, thread=_LiveJoinThread())

    controller.stop()

    assert controller._ownership is None
    assert controller.start_calls == 0
    assert controller.status().serving is False


def test_fail_start_does_not_start_replacement_until_thread_exits(
    test_state, monkeypatch
) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    test_state.state.app_settings.remote_exposure = "lan"
    controller = _StartRecordingController(test_state)
    old_server = _FakeServer(started=False)
    thread = _BlockingJoinThread()
    _install_running(controller, server=old_server, thread=thread)

    fail_done = threading.Event()

    def _fail() -> None:
        controller._fail_start(
            old_server,
            reason="remote server failed to bind",
            log_msg="Remote server did not bind on port",
            signal_exit=True,
        )
        fail_done.set()

    threading.Thread(target=_fail, name="ltx-remote-fail-start-test", daemon=True).start()
    assert thread.join_started.wait(timeout=1.0)
    assert old_server.should_exit is True
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is old_server
    assert controller.start_calls == 0

    started = time.monotonic()
    controller.sync()
    assert time.monotonic() - started < 1.0
    assert controller.start_calls == 0
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is old_server

    thread.release_join.set()
    assert fail_done.wait(timeout=1.0)
    assert not thread.is_alive()
    assert controller._ownership is None

    controller.sync()
    assert controller.start_calls == 1
    assert controller.started_servers[0] is not old_server
    assert isinstance(controller._ownership, _Running)
    assert controller._ownership.server is controller.started_servers[0]


def test_fail_start_keeps_ownership_when_join_times_out(
    test_state, monkeypatch
) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    test_state.state.app_settings.remote_exposure = "lan"
    controller = _StartRecordingController(test_state)
    old_server = _FakeServer(started=False)
    thread = _JoinTimeoutLiveThread()
    _install_running(controller, server=old_server, thread=thread)

    controller._fail_start(
        old_server,
        reason="remote server failed to bind",
        log_msg="Remote server did not bind on port",
        signal_exit=True,
    )

    assert old_server.should_exit is True
    assert thread.is_alive()
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is old_server
    assert controller.start_calls == 0

    controller.sync()
    assert controller.start_calls == 0
    assert isinstance(controller._ownership, _Stopping)
    assert controller._ownership.server is old_server

    thread.terminate()
    finish_reasons: list[str] = []
    original_finish = controller._finish_stop_locked

    def _record_finish(target: _Stopping, *, reason: str = "stopped") -> None:
        finish_reasons.append(reason)
        original_finish(target, reason=reason)

    controller._finish_stop_locked = _record_finish  # type: ignore[method-assign]
    controller.sync()
    assert finish_reasons == ["remote server failed to bind"]
    assert controller.start_calls == 1
    assert controller.started_servers[0] is not old_server
    assert isinstance(controller._ownership, _Running)
    assert controller._ownership.server is controller.started_servers[0]


def test_desktop_lists_and_revokes_paired_devices(client, monkeypatch) -> None:
    monkeypatch.setenv("LTX_AUTH_TOKEN", "desktop-secret")
    monkeypatch.setenv("LTX_REMOTE_PORT", "0")
    assert client.post("/api/settings", json={"remoteExposure": "lan"}).status_code == 200
    payload = _wait_until_serving(client)
    grant = payload["localUrl"].split("?t=", 1)[1]
    port = _port_from_url(payload["localUrl"])
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2.0) as http:
        exchanged = http.post(
            "/api/pairing/exchange",
            json={"code": grant},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        assert exchanged.status_code == 200
        body = exchanged.json()
        device_id = body["device_id"]
        token = body["token"]

        listed = client.get("/api/remote/devices").json()
        match = next(item for item in listed if item["id"] == device_id)
        assert match["revoked_at"] is None
        assert match["ip"] == "127.0.0.1"
        assert match["name"] == "Chrome on Mac"

        revoked = client.post(f"/api/remote/devices/{device_id}/revoke")
        assert revoked.status_code == 200
        listed_after = client.get("/api/remote/devices").json()
        match = next(item for item in listed_after if item["id"] == device_id)
        assert match["revoked_at"] is not None
        assert client.post("/api/remote/devices/missing/revoke").status_code == 404

        session = http.get(
            "/api/session",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert session.status_code == 401

    assert client.post("/api/settings", json={"remoteExposure": "off"}).status_code == 200
