from fastapi import FastAPI
from starlette.testclient import TestClient

from _routes.amd_kandinsky_node import router
from http_exception_handlers import install_http_exception_handlers
from services.amd_kandinsky_node import KandinskyNodeSetup
from tests.fakes.kandinsky_manager import FakeKandinskyManager


def test_staged_registry_pack_is_loaded_only_after_explicit_idle_restart():
    manager = FakeKandinskyManager()
    setup = KandinskyNodeSetup(manager.connect)
    assert setup.run().canInstall
    assert setup.run("install").status == "installing"
    assert setup.run("install").status == "installing"
    assert len(manager.queued) == 1
    manager.finish_install()
    assert setup.run().canRestart
    manager.jobs.append("another-user-generation")
    assert setup.run("restart").status == "busy"
    assert not manager.rebooted
    manager.jobs.clear()
    assert setup.run("restart").status == "restarting"
    assert setup.run().status == "ready"


def test_foreign_task_joining_after_submission_is_not_started():
    manager = FakeKandinskyManager(add_other_task=True)
    setup = KandinskyNodeSetup(manager.connect)
    status = setup.run("install")
    assert status.status == "manual"
    assert "queued" in status.message
    assert not manager.started
    assert len(manager.queued) == 2


def test_manager_refusal_and_failed_import_return_actionable_manual_setup():
    manager = FakeKandinskyManager(deny_install=True)
    setup = KandinskyNodeSetup(manager.connect)
    assert "403" in setup.run("install").message
    assert not manager.started
    manager.finish_install()
    manager.imported = manager.installed.copy()
    status = setup.run("restart")
    assert status.status == "manual"
    assert "startup log" in status.message
    assert not manager.rebooted


def test_remote_or_unprivileged_control_request_cannot_change_comfy():
    manager = FakeKandinskyManager()
    app = FastAPI()
    app.include_router(router)
    install_http_exception_handlers(app)
    app.state.admin_token = "local-admin"
    app.state.kandinsky_node_setup = KandinskyNodeSetup(manager.connect)
    path = "/api/amd/comfy/kandinsky-node"
    with TestClient(app, client=("127.0.0.1", 9921)) as desktop:
        assert desktop.get(path).json()["canInstall"]
        assert desktop.post(path + "/install").status_code == 403
    with TestClient(app, client=("192.0.2.17", 9921)) as remote:
        assert remote.get(path).status_code == 403
        assert remote.post(path + "/install", headers={"X-Admin-Token": "local-admin"}).status_code == 403
    assert not manager.queued


def test_private_install_uses_bundled_pack_and_follows_idle_restart_to_new_port(tmp_path):
    from pathlib import Path

    from services.amd_kandinsky_pack import ARCHIVE_NAME
    from tests.fakes.kandinsky_manager import FakePrivateKandinskyWorker

    archive = Path(__file__).resolve().parents[3] / "vendor" / "ComfyUI" / ARCHIVE_NAME
    worker = FakePrivateKandinskyWorker(tmp_path)
    setup = KandinskyNodeSetup(worker.connect, worker.owner_at, archive)
    assert setup.run().canInstall
    worker.jobs.append("generation")
    assert setup.run("install", managed_only=True).status == "busy"
    assert not worker.custom_nodes_dir.exists()
    worker.jobs.clear()
    assert setup.run("install", managed_only=True).canRestart
    assert (worker.custom_nodes_dir / "kandinsky6/kandinsky6/nodes.py").is_file()
    assert setup.run("install", managed_only=True).canRestart
    worker.leased = True
    assert setup.run("restart", managed_only=True).status == "busy"
    assert worker.restart_count == 0
    worker.leased = False
    assert setup.run("restart", managed_only=True).status == "restarting"
    assert setup.run().status == "ready"
    assert setup.run("restart", managed_only=True).status == "ready"
    assert worker.restart_count == 1


def test_private_pack_refuses_altered_archive_and_preserves_existing_folder(tmp_path):
    from tests.fakes.kandinsky_manager import FakePrivateKandinskyWorker

    worker = FakePrivateKandinskyWorker(tmp_path)
    bad = tmp_path / "wrong.zip"
    bad.write_bytes(b"unverified")
    setup = KandinskyNodeSetup(worker.connect, worker.owner_at, bad)
    assert "integrity check" in setup.run("install", managed_only=True).message
    assert not worker.custom_nodes_dir.exists()
    foreign = worker.custom_nodes_dir / "kandinsky6"
    foreign.mkdir(parents=True)
    (foreign / "personal.py").write_text("# existing node")
    assert "another installation" in setup.run("install", managed_only=True).message
    assert (foreign / "personal.py").is_file()
    assert not worker.restart_count


def test_failed_private_import_shows_startup_error_without_restarting_in_a_loop(tmp_path):
    from pathlib import Path

    from services.amd_kandinsky_pack import ARCHIVE_NAME
    from tests.fakes.kandinsky_manager import FakePrivateKandinskyWorker

    archive = Path(__file__).resolve().parents[3] / "vendor" / "ComfyUI" / ARCHIVE_NAME
    worker = FakePrivateKandinskyWorker(tmp_path)
    worker.load_nodes = False
    setup = KandinskyNodeSetup(worker.connect, worker.owner_at, archive)
    assert setup.run("install", managed_only=True).canRestart
    assert setup.run("restart", managed_only=True).status == "restarting"
    assert "private engine log" in setup.run().message
    assert setup.run("restart", managed_only=True).status == "manual"
    assert worker.restart_count == 1


def test_private_only_request_via_desktop_api_cannot_change_external_manager():
    manager = FakeKandinskyManager()
    app = FastAPI()
    app.include_router(router)
    install_http_exception_handlers(app)
    app.state.admin_token = "local-admin"
    app.state.kandinsky_node_setup = KandinskyNodeSetup(manager.connect, lambda _origin: None)
    with TestClient(app, client=("127.0.0.1", 9921)) as desktop:
        for action in ("install", "restart"):
            response = desktop.post(f"/api/amd/comfy/kandinsky-node/{action}?managedOnly=true", headers={"X-Admin-Token": "local-admin"})
            assert response.status_code == 200
            assert response.json()["status"] == "busy"
    assert not manager.queued and not manager.rebooted
