"""ComfyUI Manager v2 HTTP fixture for the Desktop setup flow."""

import json
from dataclasses import dataclass, field
from http import HTTPStatus
from urllib.parse import parse_qs

import httpx


@dataclass
class FakeKandinskyManager:
    version: str = "0.38.0"
    installed: dict = field(default_factory=dict)
    imported: dict = field(default_factory=dict)
    nodes_loaded: bool = False
    jobs: list = field(default_factory=list)
    queued: list = field(default_factory=list)
    add_other_task: bool = False
    deny_install: bool = False
    started: bool = False
    rebooted: bool = False

    def connect(self):
        return httpx.Client(base_url="http://127.0.0.1", transport=httpx.WSGITransport(app=self.serve))

    def serve(self, environ, start_response):
        path = environ["PATH_INFO"]
        query = parse_qs(environ["QUERY_STRING"])
        status, payload = self.route(environ, path, query)
        start_response(f"{status} {HTTPStatus(status).phrase}", [("Content-Type", "application/json")])
        return [json.dumps(payload).encode()]

    def route(self, environ, path, query):
        if path == "/system_stats":
            return 200, {"system": {"comfyui_version": self.version}}
        if path.startswith("/object_info/"):
            return 200, {path.split("/")[-1]: {}} if self.nodes_loaded else {}
        if path == "/v2/customnode/installed":
            return 200, self.imported if query.get("mode") == ["imported"] else self.installed
        if path == "/queue":
            return 200, {"queue_pending": self.jobs, "queue_running": []}
        if path == "/v2/manager/queue/status":
            owner = query.get("client_id")
            pending = [task for task in self.queued if not owner or task["client_id"] == owner[0]]
            return 200, {"pending_count": len(pending), "in_progress_count": 0, "is_processing": self.started}
        if path == "/v2/manager/queue/task" and environ["REQUEST_METHOD"] == "POST":
            if self.deny_install:
                return 403, None
            task = json.load(environ["wsgi.input"])
            if task["kind"] != "install" or task["params"] != {
                "id": "kandinsky6", "version": "1.0.2", "selected_version": "1.0.2",
                "channel": "default", "mode": "cache",
            }:
                return 400, None
            self.queued.append(task)
            if self.add_other_task:
                self.queued.append({"client_id": "another-user"})
            return 200, None
        if path == "/v2/manager/queue/start" and environ["REQUEST_METHOD"] == "POST":
            self.started = True
            return 200, None
        if path == "/v2/manager/reboot" and environ["REQUEST_METHOD"] == "POST":
            self.rebooted = True
            self.imported = self.installed.copy()
            self.nodes_loaded = True
            return 200, None
        return 404, None

    def finish_install(self):
        self.installed = {"kandinsky6": {"cnr_id": "kandinsky6", "ver": "1.0.2", "enabled": True}}
        self.queued.clear()
        self.started = False


class FakePrivateKandinskyWorker(FakeKandinskyManager):
    def __init__(self, data_dir):
        super().__init__()
        self.custom_nodes_dir = data_dir / "custom_nodes"
        self.port = 18812
        self.restart_count = 0
        self.load_nodes = True
        self.leased = False
        self.paths = []

    def connect(self):
        return httpx.Client(base_url=f"http://127.0.0.1:{self.port}", transport=httpx.WSGITransport(app=self.serve))

    def owner_at(self, origin):
        return self if origin.rstrip("/") == f"http://127.0.0.1:{self.port}" else None

    def restart(self):
        from services.amd_comfy_engine import ComfyEngineStatus

        if self.leased:
            return ComfyEngineStatus(status="ready", message="A generation is being saved.")
        self.restart_count += 1
        self.port += 1
        self.nodes_loaded = self.load_nodes and (self.custom_nodes_dir / "kandinsky6/kandinsky6/nodes.py").is_file()
        return ComfyEngineStatus(status="starting", message="Starting")

    def route(self, environ, path, query):
        self.paths.append(path)
        if path.startswith("/v2/"):
            raise AssertionError("The private worker has no Manager")
        return super().route(environ, path, query)
