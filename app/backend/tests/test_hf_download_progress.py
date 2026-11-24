"""Regressions for concurrent model setup and resumed Hub transfers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from huggingface_hub import file_download
from huggingface_hub.utils import tqdm as hub_tqdm

from services.model_downloader import hugging_face_downloader as downloader


def test_simultaneous_downloads_keep_their_own_byte_progress(monkeypatch, tmp_path):
    # Both callers enter the Hub together. Snapshots report files, reconstructed
    # bytes and network bytes; HTTP range resumes start from an existing offset.
    both_started = Barrier(2)
    original_http = file_download.http_get

    def fetch_file(*, tqdm_class, **kwargs):
        assert issubclass(tqdm_class, hub_tqdm)
        with tqdm_class(name="huggingface_hub.http_get", unit="B", initial=3) as bar:
            both_started.wait(timeout=10)
            bar.update(5)
            assert file_download.http_get is original_http
        return str(tmp_path / "model.bin")

    def fetch_snapshot(*, tqdm_class, **kwargs):
        with tqdm_class(name="huggingface_hub.snapshot_download", unit="B") as data:
            with tqdm_class(name="huggingface_hub.snapshot_download.transfer", unit="B") as network:
                with tqdm_class(total=2) as files:
                    both_started.wait(timeout=10)
                    network.update(20)
                    files.update(1)
                    data.update(10)
                    data.update(10)
        return str(tmp_path)

    monkeypatch.setattr(downloader, "hf_hub_download", fetch_file)
    monkeypatch.setattr(downloader, "snapshot_download", fetch_snapshot)
    single, snapshot = [], []
    hub = downloader.HuggingFaceDownloader()
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(hub.download_file, "owner/a", "model.bin", str(tmp_path), None, single.append)
        two = pool.submit(hub.download_snapshot, "owner/b", str(tmp_path), None, snapshot.append)
        assert one.result() == tmp_path / "model.bin"
        assert two.result() == tmp_path
    assert single == [3, 8]
    assert snapshot == [10, 20]
