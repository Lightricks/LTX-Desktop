import asyncio
import io
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from services.assistant_models import FileAssistantModels, LOCAL_ASSISTANT_MODELS
from services.llama_runtime import ASSETS, AssistantModelError, LlamaRuntime, extract_runtime
from tests.fakes.assistant import FakeLlamaRuntime
from tests.fakes.services import FakeModelDownloader


def test_existing_weights_are_reused_but_missing_or_corrupt_projector_blocks_selection(tmp_path):
    runtime = FakeLlamaRuntime(tmp_path)
    runtime.install(lambda _: None)
    downloader = FakeModelDownloader()
    models = FileAssistantModels(lambda: tmp_path, downloader, runtime)
    spec = LOCAL_ASSISTANT_MODELS["qwen3.5:9b"]
    weights = models.path("qwen3.5:9b")
    weights.parent.mkdir(parents=True)
    with weights.open("wb") as file:
        file.truncate(spec.weights.size)
    weights.with_suffix(".verified").write_text(models._receipt(weights, spec.weights))
    assert models.installed() == {}
    with pytest.raises(AssistantModelError, match="verification failed"):
        models.pull("qwen3.5:9b", lambda *_: None)
    assert weights.exists()
    assert models.installed() == {}
    assert not models.projector_path("qwen3.5:9b").exists()


def test_projector_is_deleted_with_weights(tmp_path):
    runtime = FakeLlamaRuntime(tmp_path)
    runtime.install(lambda _: None)
    models = FileAssistantModels(lambda: tmp_path, FakeModelDownloader(), runtime)
    for model, spec in LOCAL_ASSISTANT_MODELS.items():
        for path, asset in ((models.path(model), spec.weights), (models.projector_path(model), spec.projector)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as file:
                file.truncate(asset.size)
            path.with_suffix(".verified").write_text(models._receipt(path, asset))
    assert list(models.installed()) == ["qwen3.5:9b"]
    models.delete("qwen3.5:9b")
    assert models.installed() == {}
    assert not models.projector_path("qwen3.5:9b").with_suffix(".verified").exists()


def test_unsupported_platform_has_actionable_error(tmp_path):
    runtime = LlamaRuntime(tmp_path, system="Linux", machine="unknown")
    with pytest.raises(AssistantModelError, match="Apple Silicon.*Windows x64"):
        runtime.check_platform()


@pytest.mark.parametrize("suffix", ["zip", "tar.gz"])
def test_runtime_archive_rejects_path_traversal(tmp_path, suffix):
    archive = tmp_path / f"runtime.{suffix}"
    output = tmp_path / "output"
    output.mkdir()
    if suffix == "zip":
        with zipfile.ZipFile(archive, "w") as file:
            file.writestr("../escaped", "bad")
    else:
        with tarfile.open(archive, "w:gz") as file:
            member = tarfile.TarInfo("../escaped")
            member.size = 3
            file.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises((AssistantModelError, tarfile.FilterError)):
        extract_runtime(archive, output)
    assert not (tmp_path / "escaped").exists()


def test_windows_archive_preserves_runtime_dlls(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("llama-server.exe", b"executable")
        file.writestr("ggml-vulkan.dll", b"library")
    runtime = LlamaRuntime(tmp_path, system="Windows", machine="AMD64")
    extract_runtime(archive, runtime.directory)
    assert not runtime.installed()
    (runtime.directory / ".complete").write_text(ASSETS[("Windows", "AMD64")].sha256)
    assert runtime.installed()
    assert (runtime.directory / "ggml-vulkan.dll").read_bytes() == b"library"


def test_partial_or_corrupt_model_never_becomes_installed(tmp_path):
    runtime = FakeLlamaRuntime(tmp_path)
    models = FileAssistantModels(lambda: tmp_path / "models", FakeModelDownloader(), runtime)
    with pytest.raises(AssistantModelError, match="verification failed"):
        models.pull("qwen3.5:9b", lambda *_: None)
    assert models.installed() == {}
    assert not models.path("qwen3.5:9b").exists()


def test_missing_runtime_can_retry_without_a_stuck_lock(tmp_path):
    runtime = LlamaRuntime(tmp_path)

    async def check():
        for _ in range(2):
            with pytest.raises(AssistantModelError, match="Download"):
                async with runtime.serve("qwen3.5:9b", tmp_path / "missing.gguf"):
                    pytest.fail("Must not serve an uninstalled model")

    asyncio.run(check())


@pytest.mark.skipif(os.name == "nt", reason="Test child executable uses a POSIX shebang")
@pytest.mark.parametrize("failure", [False, True])
def test_child_is_authenticated_and_reaped_on_success_or_startup_failure(tmp_path, failure):
    runtime = LlamaRuntime(tmp_path, system="Darwin", machine="arm64")
    runtime.directory.mkdir(parents=True)
    child = runtime.directory / "llama-server"
    shutil.copyfile(Path(__file__).parent / "fakes" / "llama_server.py", child)
    child.chmod(0o755)
    (runtime.directory / ".complete").write_text(ASSETS[("Darwin", "arm64")].sha256)
    model = tmp_path / ("fail.gguf" if failure else "model.gguf")
    model.touch()

    async def check():
        if failure:
            with pytest.raises(AssistantModelError, match="could not start"):
                async with runtime.serve("qwen3.5:9b", model):
                    pytest.fail("Broken child must not become ready")
        else:
            async with runtime.serve("qwen3.5:9b", model) as endpoint:
                async with httpx.AsyncClient(trust_env=False) as client:
                    health = endpoint.base_url.removesuffix("/v1") + "/health"
                    assert (await client.get(health)).status_code == 401
                    assert (await client.get(health, headers={"Authorization": f"Bearer {endpoint.api_key}"})).status_code == 200
            async with httpx.AsyncClient(trust_env=False) as client:
                with pytest.raises(httpx.ConnectError):
                    await client.get(health)
        (tmp_path / "model.gguf").touch()
        async with asyncio.timeout(10):
            async with runtime.serve("qwen3.5:9b", tmp_path / "model.gguf"):
                pass

    asyncio.run(check())
