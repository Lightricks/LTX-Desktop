"""Rules for a pasted custom LoRA link: which hosts are refused and what is saved."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from runtime_config.custom_lora_urls import (
    CustomLoraUrlError,
    check_fetch_url,
    hugging_face_repo_of,
    resolve_custom_lora_url,
    saved_source_url,
)
from runtime_config.custom_loras import CustomLoraFileError, validate_safetensors_weights


@pytest.mark.parametrize(
    "url",
    [
        "https://127.1/x.safetensors",  # short form of 127.0.0.1
        "https://2130706433/x.safetensors",  # the same address as one number
        "https://0x7f.1/x.safetensors",  # hex form
        "https://localhost./x.safetensors",  # trailing dot is the same host
        "https://intranet/x.safetensors",  # one label: a private name
        "https://[::1]/x.safetensors",
    ],
)
def test_private_hosts_in_their_other_spellings_are_refused(url: str) -> None:
    with pytest.raises(CustomLoraUrlError, match="private network"):
        check_fetch_url(url)


def test_a_normal_host_with_a_trailing_dot_is_accepted() -> None:
    assert check_fetch_url("https://cdn.example.com./x.safetensors").host == "cdn.example.com"


def test_a_civitai_download_link_is_saved_without_its_token_but_fetched_with_it() -> None:
    pasted = "https://civitai.com/api/download/models/456?type=Model&token=SECRET"

    fetched = resolve_custom_lora_url(pasted).url

    assert "token=SECRET" in fetched
    assert saved_source_url(fetched) == "https://civitai.com/api/download/models/456"


@pytest.mark.parametrize(
    "link",
    [
        "https://cdn.example.com/f/lora.safetensors?X-Amz-Signature=SECRET&X-Amz-Expires=60",
        "https://huggingface.co/acme/dolly/resolve/main/dolly.safetensors?token=SECRET#frag",
        "https://civitai.com/api/download/models/456?token=SECRET",
    ],
)
def test_a_saved_link_keeps_no_query_or_fragment_on_any_host(link: str) -> None:
    saved = saved_source_url(link)

    assert "SECRET" not in saved
    assert "?" not in saved and "#" not in saved
    assert saved.split("?")[0].split("#")[0] == saved


def _header_file(tmp_path: Path, offsets: list[object]) -> Path:
    header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": offsets}}).encode()
    path = tmp_path / "w.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return path


@pytest.mark.parametrize("offsets", [[5, 2], [-1, 4], [True, 4]])
def test_a_header_with_reversed_negative_or_boolean_offsets_is_refused(
    tmp_path: Path, offsets: list[object]
) -> None:
    with pytest.raises(CustomLoraFileError, match="not a .safetensors file"):
        validate_safetensors_weights(_header_file(tmp_path, offsets))


@pytest.mark.parametrize(
    ("link", "repo_id", "revision"),
    [
        ("https://huggingface.co/acme/dolly", "acme/dolly", "main"),
        ("https://huggingface.co/acme/dolly/tree/v2", "acme/dolly", "v2"),
        ("https://huggingface.co/acme/dolly/blob/main/dolly.safetensors", "acme/dolly", "main"),
        ("https://huggingface.co/acme/dolly/resolve/v2/dolly.safetensors", "acme/dolly", "v2"),
        ("https://hf.co/acme/dolly", "acme/dolly", "main"),
    ],
)
def test_every_kind_of_hugging_face_repo_link_names_its_repo_and_revision(
    link: str, repo_id: str, revision: str
) -> None:
    repo = hugging_face_repo_of(link)

    assert repo is not None
    assert (repo.repo_id, repo.revision) == (repo_id, revision)


@pytest.mark.parametrize(
    "link",
    [
        "https://huggingface.co/datasets/acme/data",  # a dataset is not a model card
        "https://huggingface.co/spaces/acme/demo",
        "https://example.com/acme/dolly",
        "http://huggingface.co/acme/dolly",
    ],
)
def test_a_link_that_is_not_a_hugging_face_model_repo_has_no_card(link: str) -> None:
    assert hugging_face_repo_of(link) is None
