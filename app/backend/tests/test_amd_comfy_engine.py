"""Run the private worker supervisor against a real local fixture subprocess."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from _routes.amd_comfy_engine import router
from http_exception_handlers import install_http_exception_handlers
from services import amd_comfy_provision as provision
from services.amd_comfy_client import AmdComfyClient, ComfyCancelled, ComfyError
from services.amd_comfy_engine import ManagedComfyEngine

_STUB = """\
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
base = Path(sys.argv[sys.argv.index('--base-directory') + 1])
port = int(sys.argv[sys.argv.index('--port') + 1])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/system_stats':
            device = 'cpu' if (base / 'cpu').exists() else 'cuda'
            data = {'system': {'argv': sys.argv, 'comfyui_version': '0.38.0', 'pytorch_version': '2.9.0+rocm'},
                    'devices': [{'type':device,'name':'AMD Radeon 8060S'}]}
        elif self.path == '/internal/folder_paths':
            data = {'vae': [str(base / 'models' / 'vae')]}
        else:
            data = {'queue_running': ['busy'] if (base / 'busy').exists() else [], 'queue_pending': []}
        out=json.dumps(data).encode(); self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)
    def do_POST(self):
        if self.path == '/free':
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert data == {'unload_models':True, 'free_memory':True}
            with (base / 'free_calls').open('a') as f: f.write('x')
        self.send_response(200); self.end_headers()
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
"""


@pytest.fixture
def worker(tmp_path, monkeypatch):
    layout = provision.ManagedComfyLayout(tmp_path / "owned", tmp_path / "resources")
    layout.base_dir.mkdir(parents=True)
    receipt = layout.root / "ready"
    script = tmp_path / "worker.py"
    script.write_text(_STUB)
    installed = threading.Event()

    def prepare(layout, progress):
        if receipt.exists():
            return
        progress("Checking bundled archive")
        installed.set()
        receipt.touch()

    def command(layout, port):
        return (
            [
                sys.executable,
                str(script),
                "--port",
                str(port),
                "--base-directory",
                str(layout.base_dir),
            ],
            os.environ.copy(),
            layout.root,
        )

    monkeypatch.setattr(provision, "needs_install", lambda layout: not receipt.exists())
    monkeypatch.setattr(provision, "provision_private_comfy", prepare)
    monkeypatch.setattr(provision, "private_comfy_command", command)
    engine = ManagedComfyEngine(layout)
    monkeypatch.setattr(engine, "_external", lambda: None)
    try:
        yield engine, layout, installed
    finally:
        engine.shutdown()


def test_prepare_owns_worker_idle_restart_render_lease_and_shutdown(worker):
    engine, layout, installed = worker
    assert engine.status().status == "not_installed"
    assert engine.action("start").status == "not_installed"
    assert not installed.is_set()
    assert engine.action("prepare").status == "installing"
    engine.action("prepare")
    client = engine.connect(start_prepared=True)
    assert engine.status().status == "ready" and engine.active_at(client.base_url)
    assert client.get_folder_paths() == {
        "vae": [str(layout.base_dir / "models" / "vae")]
    }
    pid = engine._process.pid
    with engine.rendering(client):
        assert not engine.action("stop").canStop
        assert engine.restart().status == "ready" and psutil.pid_exists(pid)
    (layout.base_dir / "busy").touch()
    assert engine.restart().status == "ready" and psutil.pid_exists(pid)
    (layout.base_dir / "busy").unlink()
    assert engine.restart().status == "starting"
    restarted = engine.connect(start_prepared=True)
    assert engine.active_at(restarted.base_url) and not psutil.pid_exists(pid)
    assert engine.action("stop").status == "stopped"
    assert engine._process is None
    # A saved generation starts the already prepared private worker without an extra UI visit.
    started = engine.connect(start_prepared=True)
    assert engine.active_at(started.base_url)
    pid = engine._process.pid
    engine.shutdown()
    assert not psutil.pid_exists(pid)


def test_start_avoids_other_listener_and_external_shutdown_never_touches_it(
    worker, monkeypatch
):
    engine, layout, _ = worker
    (layout.root / "ready").touch()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied = listener.getsockname()[1]
        external = AmdComfyClient(occupied)
        external.gpu_name = "external AMD"
        monkeypatch.setattr(engine, "_external", lambda: external)
        engine._port = occupied
        assert engine.connect(start_prepared=True) is external
        assert engine.action("stop").status == "external"
        # Explicit start requests a private engine and leaves the other listener up.
        assert engine.action("start").status == "starting"
        hosted = engine.connect(start_prepared=True)
        assert hosted.base_url != external.base_url
        engine.shutdown()
        assert listener.getsockname()[1] == occupied
        assert engine.connect() is external


def test_failed_device_check_can_retry_and_cancelled_start_never_submits(worker):
    engine, layout, _ = worker
    (layout.base_dir / "cpu").touch()
    engine.action("prepare")
    with pytest.raises(ComfyCancelled):
        engine.connect(start_prepared=True, cancelled=lambda: True)
    with pytest.raises(ComfyError, match="ROCm"):
        engine.connect(start_prepared=True)
    assert engine.status().canPrepare and engine._process is None
    (layout.base_dir / "cpu").unlink()
    engine.action("start")
    assert engine.active_at(engine.connect(start_prepared=True).base_url)


def test_management_requires_desktop_origin_and_admin_token(worker):
    engine, _, installed = worker
    app = FastAPI()
    app.include_router(router)
    install_http_exception_handlers(app)
    app.state.admin_token, app.state.managed_comfy = "desktop-secret", engine
    path = "/api/comfy/engine"
    with TestClient(app, client=("127.0.0.1", 49100)) as client:
        assert client.get(path).json()["status"] == "not_installed"
        assert client.post(path + "/prepare").status_code == 403
    with TestClient(app, client=("192.0.2.23", 49100)) as remote:
        assert remote.get(path).status_code == 403
        assert (
            remote.post(
                path + "/start", headers={"X-Admin-Token": "desktop-secret"}
            ).status_code
            == 403
        )
    assert not installed.is_set()
    with TestClient(app, client=("127.0.0.1", 49100)) as client:
        assert (
            client.post(
                path + "/prepare", headers={"X-Admin-Token": "desktop-secret"}
            ).json()["status"]
            == "installing"
        )
        assert installed.wait(3)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_windows_kills_private_worker_when_owner_exits_abruptly(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    code = """import os, subprocess, sys
from services.amd_worker_job import bind_worker_to_parent
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
release = bind_worker_to_parent(p)
open(sys.argv[1], 'w').write(str(p.pid))
os._exit(0)
"""
    pid_file = tmp_path / "pid"
    subprocess.run(
        [sys.executable, "-c", code, str(pid_file)], check=True, cwd=root, timeout=10
    )
    pid = int(pid_file.read_text())
    until = time.monotonic() + 8
    while psutil.pid_exists(pid) and time.monotonic() < until:
        time.sleep(0.05)
    assert not psutil.pid_exists(pid)


def test_native_switch_releases_only_previously_used_idle_private_worker(
    worker, monkeypatch
):
    engine, layout, _ = worker
    engine.action("prepare")
    client = engine.connect(start_prepared=True)
    record = layout.base_dir / "free_calls"
    engine.release_models_for_native()
    assert not record.exists()
    with engine.rendering(client):
        engine.release_models_for_native()
        assert not record.exists()
    (layout.base_dir / "busy").touch()
    engine.release_models_for_native()
    assert not record.exists()
    (layout.base_dir / "busy").unlink()
    engine.release_models_for_native()
    engine.release_models_for_native()
    assert record.read_text() == "x"
    engine.shutdown()
    monkeypatch.setattr(
        engine,
        "_external",
        lambda: pytest.fail("Memory release attempted external discovery"),
    )
    engine.release_models_for_native()
