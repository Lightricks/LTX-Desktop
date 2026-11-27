from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

from handlers.h3_models_handler import H3ModelsHandler
from services.amd_h3_models import H3_MODELS, H3_TURBO_LORA, h3_license_acknowledged
from tests.fakes.services import FakeModelDownloader, FakeTaskRunner


def setup(tmp_path: Path, *, downloader=None, tasks=None, nodes=None):
    files = {w for recipe in H3_MODELS.values() for w in recipe}
    contents = bytes(1024)  # public-repo payload in FakeModelDownloader
    sha = hashlib.sha256(contents).hexdigest()
    recipes = {
        key: tuple(
            replace(w, size=len(contents), sha256=sha, sample_sha256=None) for w in ws
        )
        for key, ws in H3_MODELS.items()
    }
    folders = {w.category: [str(tmp_path / w.category)] for w in files}
    if nodes is None:
        nodes = {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"}
    return H3ModelsHandler(
        tmp_path / "appdata",
        lambda: None,
        downloader or FakeModelDownloader(),
        tasks or FakeTaskRunner(),
        lambda: (folders, nodes),
        recipes,
    )


def test_read_only_catalog_does_not_acknowledge_or_schedule_a_download(
    client, test_state, tmp_path, windows_local_gpu
):
    tasks = FakeTaskRunner()
    test_state.h3_models = setup(tmp_path, tasks=tasks)
    path = "/api/comfy/models/h3"
    catalog = client.get(path).json()
    assert not catalog["licenseAcknowledged"]
    assert {k: v["totalFiles"] for k, v in catalog["variants"].items()} == {
        "h3-fl2va": 4,
        "h3-fl2va-distill": 5,
        "h3-ref2va": 4,
    }
    for url in [
        path + "/license",
        path + "/license?acceptedLicense=false",
        path + "/download?variant=h3-ref2va&acceptedLicense=true",
    ]:
        denied = client.post(url)
        assert (
            denied.status_code == 403
            and denied.json()["code"] == "H3_LICENSE_ACK_REQUIRED"
        )
    assert not (tmp_path / "appdata").exists() and tasks.jobs_run == 0
    assert client.post(path + "/license?acceptedLicense=true").json()[
        "licenseAcknowledged"
    ]
    assert setup(tmp_path).license_acknowledged()  # restart
    assert tasks.jobs_run == 0
    assert client.post(path + "/download?variant=h3-ref2va").status_code == 403
    assert (
        client.post(path + "/download?variant=h3-max&acceptedLicense=true").status_code
        == 422
    )
    result = client.post(path + "/download?variant=h3-ref2va&acceptedLicense=true")
    result.raise_for_status()
    assert result.json()["variants"]["h3-ref2va"]["status"] == "ready"
    receipt = tmp_path / "appdata" / "h3-license-v1.json"
    saved = json.loads(receipt.read_text())
    for bad in [
        {**saved, "licenseBlob": "outdated"},
        {**saved, "accepted": 1},
        "partial receipt",
    ]:
        receipt.write_text(json.dumps(bad))
        assert not h3_license_acknowledged(receipt.parent)


def test_nested_manual_weights_reused_and_only_new_siblings_downloaded(tmp_path):
    transfer = FakeModelDownloader()
    h3 = setup(tmp_path, downloader=transfer)
    enc = H3_MODELS["h3-fl2va"][1]
    existing = tmp_path / enc.category / "my-library" / enc.filename
    existing.parent.mkdir(parents=True)
    existing.write_bytes(bytes(1024))
    assert h3.status().variants["h3-fl2va"].completedFiles == 1
    h3.acknowledge(True)
    first = h3.start("h3-fl2va", accepted=True).variants
    assert (
        first["h3-fl2va"].status == "ready" and first["h3-ref2va"].completedFiles == 3
    )
    assert first["h3-fl2va-distill"].completedBytes == 4 * 1024
    assert len(transfer.calls) == 3
    h3.start("h3-fl2va-distill", accepted=True)
    assert transfer.calls[-1]["filename"] == f"loras/{H3_TURBO_LORA}"
    final = h3.start("h3-ref2va", accepted=True)
    assert all(v.status == "ready" for v in final.variants.values())
    assert all(
        c["revision"] == H3_MODELS["h3-fl2va"][0].revision and c["token"] is None
        for c in transfer.calls
    )
    assert all(
        c["filename"] != f"{enc.category}/{enc.filename}" for c in transfer.calls
    )
    assert len(transfer.calls) == 5
    h3.start("h3-fl2va", accepted=True)
    assert len(transfer.calls) == 5


def test_missing_comfy_node_blocks_only_its_recipe(tmp_path):
    tasks = FakeTaskRunner()
    h3 = setup(tmp_path, tasks=tasks, nodes={"MiniMaxH3ImageToVideo"})
    h3.acknowledge(True)
    state = h3.start("h3-ref2va", accepted=True)
    assert (
        state.variants["h3-ref2va"].status == "error"
        and "update" in state.variants["h3-ref2va"].message
    )
    assert state.variants["h3-fl2va"].status == "missing" and tasks.jobs_run == 0


def test_concurrent_jobs_share_downloads_report_progress_and_retry_failure(tmp_path):
    started, resume = Event(), Event()

    class PausedTransfer(FakeModelDownloader):
        def download_file(self, **kwargs):
            if not started.is_set():
                started.set()
                assert resume.wait(timeout=8)
            return super().download_file(**kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []

        class Runner:
            def run_background(self, target, *, on_error, **kwargs):
                def capture():
                    try:
                        target()
                    except Exception as exc:
                        on_error(exc)

                futures.append(pool.submit(capture))

        transfer = PausedTransfer()
        h3 = setup(tmp_path, tasks=Runner(), downloader=transfer)
        h3.acknowledge(True)
        h3.start("h3-fl2va", accepted=True)
        assert started.wait(timeout=8)
        try:
            transfer.fail_next = RuntimeError("temporary connection failure")
            assert (
                h3.start("h3-fl2va-distill", accepted=True)
                .variants["h3-fl2va-distill"]
                .status
                == "downloading"
            )
            progress = h3.start("h3-fl2va", accepted=True).variants["h3-fl2va"]
            assert progress.status == "downloading" and progress.currentFile is not None
            assert len(futures) == 2
        finally:
            resume.set()
        for f in futures:
            f.result(timeout=10)
        # The other job publishes the first job's missing files. Retrying never writes them twice.
        for variant in ("h3-fl2va-distill", "h3-fl2va"):
            h3.start(variant, accepted=True)
        for f in futures:
            f.result(timeout=10)
        final = h3.status(fresh=True).variants
        assert final["h3-fl2va"].status == final["h3-fl2va-distill"].status == "ready"
        assert list(Counter(c["filename"] for c in transfer.calls).values()) == [1] * 5


def test_wrong_model_of_identical_size_with_same_header_does_not_enable_a_recipe(
    tmp_path,
):
    import struct
    from handlers.comfy_models_handler import locate_comfy_weight

    # FL2VA and Ref2VA even have byte-identical safetensors headers.
    block = 256 * 1024
    weight = H3_MODELS["h3-fl2va"][0]
    parts = [
        struct.pack("<Q", 2) + b"{}" + bytes(block - 10),
        bytes(block),
        bytes(block),
        bytes(block),
    ]
    sample = hashlib.sha256(parts[0] + parts[2] + parts[3]).hexdigest()
    fixture = replace(weight, size=4 * block, sample_sha256=sample)
    model_file = tmp_path / weight.filename
    model_file.write_bytes(b"".join(parts))
    dirs = {weight.category: [str(tmp_path)]}
    assert locate_comfy_weight(fixture, dirs).installed
    with model_file.open("r+b") as stream:
        stream.seek(2 * block)
        stream.write(b"other checkpoint")
    assert not locate_comfy_weight(fixture, dirs).installed
    assert model_file.read_bytes().endswith(parts[3])


def test_corrupt_transfer_never_published_and_retry_continues(tmp_path):
    class CorruptOnce(FakeModelDownloader):
        def download_file(self, **kwargs):
            output = super().download_file(**kwargs)
            if len(self.calls) == 2:
                output.write_bytes(b"x" * 1024)
            return output

    transfer = CorruptOnce()
    h3 = setup(tmp_path, downloader=transfer)
    h3.acknowledge(True)
    failed = h3.start("h3-fl2va", accepted=True).variants["h3-fl2va"]
    assert failed.status == "error" and "SHA-256" in failed.message
    completed, rejected, *_ = failed.models
    assert Path(completed.path).is_file() and not Path(rejected.path).exists()
    assert h3.start("h3-fl2va", accepted=True).variants["h3-fl2va"].status == "ready"
    assert len(transfer.calls) == 5
