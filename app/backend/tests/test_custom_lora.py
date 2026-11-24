"""Custom LoRA: import from a file or a link, run, and delete."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

from api_types import (
    CreateCustomLoraRequest,
    CustomLoraParams,
    ImageToVideoRecipeParams,
    InputAssetRef,
    LoraEntry,
    LoraRecipeInputs,
    TextToVideoParams,
)
from pydantic import ValidationError
from remote.app import create_remote_app
from runtime_config.custom_loras import (
    CustomLoraRecord,
    custom_lora_dir,
    read_custom_lora,
    write_meta,
)
from services.features.custom_loras import (
    CustomLoraCreateError,
    make_custom_lora_resolver,
    resolve_custom_lora_create,
)
from services.features.video import TextToVideoExecutor
from services.features.video.custom_lora import CustomLoraExecutor
from services.features.video.image_to_video import ImageToVideoExecutor
from services.generation_queue.types import OutputAllocation
from services.records import CapabilityFailedError
from services.sqlite_store import SqliteStore
from services.url_downloader import UrlDownloadError
from state.app_state_types import CatalogDownloadSession, DownloadSessionId
from tests.fakes.generation import FakeJobSession
from tests.http_error_assertions import assert_http_error
from tests.test_generation_queue import _generation

AUTH = {"Authorization": "Bearer pair-token"}
_IC_LORA_MARKER = {"reference_downscale_factor": "1"}
_DESCRIPTION = "A soft ink wash style."


def _safetensors(*, metadata: dict[str, str] | None = None, cut_bytes: int = 0) -> bytes:
    header: dict[str, object] = {
        "transformer_blocks.0.attn1.to_q.lora_A.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16],
        },
        "transformer_blocks.0.attn1.to_q.lora_B.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [16, 32],
        },
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(raw)) + raw + b"\x00" * 32
    return blob[: len(blob) - cut_bytes]


def _weights_file(tmp_path: Path, name: str = "source.safetensors", **kwargs) -> Path:
    path = tmp_path / name
    path.write_bytes(_safetensors(**kwargs))
    return path


def _import_file(client: TestClient, path: Path, *, name: str = "Ink Wash", **about):
    return client.post(
        "/api/custom-loras",
        json={
            "name": name,
            "source_path": str(path),
            "description": _DESCRIPTION,
            **about,
        },
    )


def _import_status(client: TestClient, started) -> dict[str, object]:
    assert started.status_code == 200, started.text
    progress = client.get(
        "/api/custom-loras/import/progress", params={"sessionId": started.json()["sessionId"]}
    )
    assert progress.status_code == 200, progress.text
    return progress.json()


def _install_lora(
    test_state,
    *,
    name: str = "Ink Wash",
    lora_id: str = "ink-wash",
    trigger: str | None = None,
    examples: tuple[str, ...] = (),
) -> None:
    directory = custom_lora_dir(test_state.config.default_models_dir, lora_id)
    directory.mkdir(parents=True)
    filename = f"{lora_id}.safetensors"
    (directory / filename).write_bytes(_safetensors())
    write_meta(
        directory,
        CustomLoraRecord(
            id=lora_id,
            name=name,
            filename=filename,
            size_bytes=0,
            source="file",
            trigger=trigger,
            description=_DESCRIPTION,
            examples=examples,
        ),
    )


def _create_body(lora_id: str = "ink-wash", **params):
    return {
        "params": {
            "prompt": "a fox on a skateboard",
            "model": "ltx-2.5-fast",
            "customLoraId": lora_id,
            "scale": 0.8,
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
            **params,
        }
    }


# --- import ---


def test_import_with_only_a_name_and_weights_is_accepted_and_lists_empty_info(
    client: TestClient, tmp_path: Path
) -> None:
    started = client.post(
        "/api/custom-loras",
        json={"name": "Ink Wash", "source_path": str(_weights_file(tmp_path))},
    )
    assert _import_status(client, started)["status"] == "complete"

    (item,) = client.get("/api/custom-loras").json()["loras"]
    assert item["trigger"] is None
    assert item["description"] == ""
    assert item["examples"] == []


def test_the_trigger_description_and_examples_of_an_import_come_back_in_the_list(
    client: TestClient, tmp_path: Path
) -> None:
    started = _import_file(
        client,
        _weights_file(tmp_path),
        trigger="InkW4sh",
        examples=["a fox in InkW4sh style"],
    )
    assert _import_status(client, started)["status"] == "complete"

    (item,) = client.get("/api/custom-loras").json()["loras"]
    assert item["trigger"] == "InkW4sh"
    assert item["description"] == _DESCRIPTION
    assert item["examples"] == ["a fox in InkW4sh style"]


def test_import_rejects_truncated_weights_instead_of_failing_at_run_time(
    client: TestClient, tmp_path: Path
) -> None:
    started = _import_file(client, _weights_file(tmp_path, cut_bytes=8))

    status = _import_status(client, started)
    assert status["status"] == "error"
    assert "incomplete" in str(status["error"])
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_import_rejects_ic_lora_weights(client: TestClient, tmp_path: Path) -> None:
    started = _import_file(client, _weights_file(tmp_path, metadata=_IC_LORA_MARKER))

    status = _import_status(client, started)
    assert status["status"] == "error"
    assert "IC-LoRA" in str(status["error"])
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_import_rejects_safetensors_without_lora_tensors(
    client: TestClient, tmp_path: Path
) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
    non_lora = tmp_path / "plain.safetensors"
    non_lora.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)

    started = _import_file(client, non_lora)

    status = _import_status(client, started)
    assert status["status"] == "error"
    assert "LoRA weights" in str(status["error"])
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_import_name_that_slugs_to_an_existing_lora_is_refused_and_keeps_the_original(
    client: TestClient, tmp_path: Path, test_state
) -> None:
    first = _import_file(client, _weights_file(tmp_path, "a.safetensors"), name="Ink Wash")
    assert _import_status(client, first)["status"] == "complete"
    original = custom_lora_dir(test_state.config.default_models_dir, "ink-wash") / "ink-wash.safetensors"
    original_bytes = original.read_bytes()

    second = _import_file(
        client, _weights_file(tmp_path, "b.safetensors", metadata={"x": "y"}), name="ink-wash"
    )

    assert_http_error(
        second,
        status_code=409,
        code="CUSTOM_LORA_NAME_TAKEN",
        message="A custom LoRA with this name already exists.",
    )
    assert original.read_bytes() == original_bytes
    assert [item["id"] for item in client.get("/api/custom-loras").json()["loras"]] == ["ink-wash"]


def test_url_import_refuses_a_private_network_host(client: TestClient) -> None:
    for url in ("https://192.168.1.5/lora.safetensors", "https://localhost/lora.safetensors"):
        response = client.post(
            "/api/custom-loras", json={"name": "Local", "url": url, "description": _DESCRIPTION}
        )
        assert response.status_code == 422
        assert response.json()["code"] == "CUSTOM_LORA_URL_INVALID"
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_url_import_refuses_a_civitai_page_without_a_version(client: TestClient) -> None:
    response = client.post(
        "/api/custom-loras",
        json={
            "name": "Page",
            "url": "https://civitai.com/models/123/some-lora",
            "description": _DESCRIPTION,
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "CUSTOM_LORA_URL_INVALID"
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_url_import_turns_a_civitai_page_with_a_version_into_the_download_link(
    client: TestClient, fake_services, test_state
) -> None:
    fake_services.url_downloader.payload = _safetensors()

    started = client.post(
        "/api/custom-loras",
        json={
            "name": "Versioned",
            "description": _DESCRIPTION,
            "url": "https://civitai.com/models/123/some-lora?modelVersionId=456",
        },
    )

    assert _import_status(client, started)["status"] == "complete"
    saved = read_custom_lora(test_state.config.default_models_dir, "versioned")
    assert saved is not None
    assert saved.source_url == "https://civitai.com/api/download/models/456"


def test_url_import_shows_the_downloader_message_for_a_sign_in_wall(
    client: TestClient, fake_services
) -> None:
    fake_services.url_downloader.error = UrlDownloadError("This file needs a sign-in.")

    started = client.post(
        "/api/custom-loras",
        json={
            "name": "Gated",
            "description": _DESCRIPTION,
            "url": "https://huggingface.co/acme/lora/resolve/main/lora.safetensors",
        },
    )

    status = _import_status(client, started)
    assert status["status"] == "error"
    assert status["error"] == "This file needs a sign-in."
    assert client.get("/api/custom-loras").json() == {"loras": []}


# --- Hugging Face repo link ---

_REPO_URL = "https://huggingface.co/acme/dolly-in"


def _import_url(client: TestClient, url: str, *, name: str = "Dolly In"):
    return client.post(
        "/api/custom-loras",
        json={"name": name, "url": url, "description": _DESCRIPTION},
    )


def test_repo_link_downloads_the_one_weights_file_and_not_the_readme(
    client: TestClient, fake_services, test_state
) -> None:
    fake_services.url_downloader.repo_files = [".gitattributes", "README.md", "dolly-in.safetensors"]
    fake_services.url_downloader.payload = _safetensors()

    started = _import_url(client, _REPO_URL)

    assert _import_status(client, started)["status"] == "complete"
    saved = read_custom_lora(test_state.config.default_models_dir, "dolly-in")
    assert saved is not None
    assert saved.source_url == "https://huggingface.co/acme/dolly-in/resolve/main/dolly-in.safetensors"


def test_repo_link_without_a_weights_file_tells_the_user_and_downloads_nothing(
    client: TestClient, fake_services
) -> None:
    fake_services.url_downloader.repo_files = [".gitattributes", "README.md"]

    response = _import_url(client, _REPO_URL)

    assert response.status_code == 422
    assert response.json()["code"] == "CUSTOM_LORA_URL_INVALID"
    assert "no .safetensors file" in response.json()["message"]
    assert client.get("/api/custom-loras").json() == {"loras": []}


def test_repo_link_with_two_weights_files_names_both_instead_of_guessing(
    client: TestClient, fake_services
) -> None:
    fake_services.url_downloader.repo_files = ["dolly-in-v1.safetensors", "dolly-in-v2.safetensors"]

    response = _import_url(client, _REPO_URL)

    assert response.status_code == 422
    assert response.json()["code"] == "CUSTOM_LORA_URL_INVALID"
    assert "dolly-in-v1.safetensors" in response.json()["message"]
    assert "dolly-in-v2.safetensors" in response.json()["message"]
    assert client.get("/api/custom-loras").json() == {"loras": []}


# --- Hugging Face page link ---


def test_only_a_lora_from_a_hugging_face_link_has_a_model_page_link(
    client: TestClient, fake_services, tmp_path: Path
) -> None:
    fake_services.url_downloader.payload = _safetensors()
    fake_services.url_downloader.repo_files = ["dolly-in.safetensors"]
    assert _import_status(client, _import_url(client, _REPO_URL, name="From HF"))["status"] == "complete"
    civitai = _import_url(
        client, "https://civitai.com/models/123/some-lora?modelVersionId=456", name="From Civitai"
    )
    assert _import_status(client, civitai)["status"] == "complete"
    from_file = _import_file(client, _weights_file(tmp_path), name="From File")
    assert _import_status(client, from_file)["status"] == "complete"

    pages = {
        item["name"]: item["page_url"] for item in client.get("/api/custom-loras").json()["loras"]
    }

    assert pages == {
        "From HF": "https://huggingface.co/acme/dolly-in",
        "From Civitai": None,
        "From File": None,
    }


# --- delete ---


def test_deleting_the_weights_keeps_the_lora_listed_and_refuses_a_run(
    client: TestClient, test_state
) -> None:
    _install_lora(test_state)

    deleted = client.delete("/api/custom-loras/ink-wash/weights")

    assert deleted.status_code == 200
    listed = client.get("/api/custom-loras").json()["loras"]
    assert [(item["id"], item["downloaded"]) for item in listed] == [("ink-wash", False)]
    run = client.post("/api/generations/custom-lora", json=_create_body())
    assert run.status_code == 422
    assert run.json()["code"] == "CUSTOM_LORA_NOT_DOWNLOADED"


def test_delete_is_refused_while_the_same_lora_is_importing(client: TestClient, test_state) -> None:
    _install_lora(test_state)
    test_state.state.custom_lora_import_session = CatalogDownloadSession(
        id=DownloadSessionId("session-1"),
        item_id="ink-wash",
        downloaded_bytes=0,
        expected_bytes=0,
    )

    response = client.delete("/api/custom-loras/ink-wash")

    assert response.status_code == 409
    assert response.json()["message"] == "CUSTOM_LORA_IMPORT_RUNNING"
    assert custom_lora_dir(test_state.config.default_models_dir, "ink-wash").is_dir()


def test_delete_is_refused_while_a_queued_run_holds_the_lora_and_only_for_that_lora(
    client: TestClient, test_state
) -> None:
    _install_lora(test_state)
    _install_lora(test_state, name="Moss", lora_id="moss")
    queued = client.post("/api/generations/custom-lora", json=_create_body("ink-wash"))
    assert queued.status_code == 200, queued.text

    for path in ("/api/custom-loras/ink-wash/weights", "/api/custom-loras/ink-wash"):
        refused = client.delete(path)
        assert refused.status_code == 409
        assert refused.json()["code"] == "CUSTOM_LORA_IN_USE"
    assert client.delete("/api/custom-loras/moss").status_code == 200
    ink = custom_lora_dir(test_state.config.default_models_dir, "ink-wash")
    assert (ink / "ink-wash.safetensors").is_file()


# --- run ---


def test_retrying_a_run_after_its_weights_were_deleted_fails_with_a_typed_error_not_a_path(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    _install_lora(test_state)
    models_dir = test_state.config.default_models_dir
    resolver = make_custom_lora_resolver(lambda: models_dir)
    executor = CustomLoraExecutor(
        TextToVideoExecutor(test_state.video_generation, lora_resolver=resolver),
        ImageToVideoExecutor(
            test_state.video_generation,
            SqliteStore(test_state.config.app_data_dir),
            lora_resolver=resolver,
        ),
    )
    generation = _generation(
        feature="custom-lora",
        params={
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
            "loras": [
                {
                    "ref": "",
                    "scale": 0.8,
                    "customLoraId": "ink-wash",
                    "displayName": "Ink Wash",
                }
            ],
        },
    )
    shutil.rmtree(custom_lora_dir(models_dir, "ink-wash"))
    plan = executor.plan_outputs(generation)[0]
    outputs = (OutputAllocation(plan=plan, asset_id="asset-1", dest_path=str(tmp_path / "out.mp4")),)

    with pytest.raises(CapabilityFailedError) as raised:
        executor.execute(generation, outputs, FakeJobSession())

    assert raised.value.code == "CUSTOM_LORA_NOT_DOWNLOADED"
    assert "Ink Wash" in str(raised.value)
    assert "loras/custom" not in str(raised.value)


def test_custom_lora_with_auto_aspect_ratio_and_no_start_frame_is_rejected(
    client: TestClient, test_state
) -> None:
    _install_lora(test_state)

    response = client.post("/api/generations/custom-lora", json=_create_body(aspectRatio="auto"))

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"


def test_a_start_frame_makes_the_run_image_to_video_and_no_start_frame_makes_it_text_to_video(
    test_state,
) -> None:
    _install_lora(test_state)
    models_dir = test_state.config.default_models_dir
    params = CustomLoraParams.model_validate(_create_body()["params"])
    start = LoraRecipeInputs(startFrame=InputAssetRef(assetId="asset-1"))

    _, text_params, text_inputs = resolve_custom_lora_create(
        CreateCustomLoraRequest(params=params), models_dir=models_dir
    )
    _, image_params, image_inputs = resolve_custom_lora_create(
        CreateCustomLoraRequest(params=params, inputs=start), models_dir=models_dir
    )

    assert isinstance(text_params, TextToVideoParams)
    assert text_inputs is None
    assert isinstance(image_params, ImageToVideoRecipeParams)
    assert image_inputs is not None
    assert image_inputs.startFrame.assetId == "asset-1"


def test_custom_lora_with_an_end_frame_is_refused(test_state) -> None:
    _install_lora(test_state)
    inputs = LoraRecipeInputs(
        startFrame=InputAssetRef(assetId="asset-1"), endFrame=InputAssetRef(assetId="asset-2")
    )

    with pytest.raises(CustomLoraCreateError):
        resolve_custom_lora_create(
            CreateCustomLoraRequest(
                params=CustomLoraParams.model_validate(_create_body()["params"]), inputs=inputs
            ),
            models_dir=test_state.config.default_models_dir,
        )


def test_a_lora_saved_with_the_old_mode_key_still_loads(test_state) -> None:
    _install_lora(test_state)
    meta_path = custom_lora_dir(test_state.config.default_models_dir, "ink-wash") / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["mode"] = "i2v"
    meta_path.write_text(json.dumps(meta))

    assert read_custom_lora(test_state.config.default_models_dir, "ink-wash") is not None


def test_run_after_the_lora_is_deleted_returns_not_found(client: TestClient, test_state) -> None:
    _install_lora(test_state)
    assert client.delete("/api/custom-loras/ink-wash").status_code == 200

    response = client.post("/api/generations/custom-lora", json=_create_body())

    assert response.status_code == 404
    assert response.json()["code"] == "CUSTOM_LORA_NOT_FOUND"


def test_a_cover_is_stored_with_the_lora_and_the_phone_gets_a_signed_url_not_the_path(
    client: TestClient, test_state, tmp_path: Path
) -> None:
    cover = tmp_path / "chosen.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(cover)
    started = _import_file(client, _weights_file(tmp_path), cover_path=str(cover))
    assert _import_status(client, started)["status"] == "complete"

    stored = custom_lora_dir(test_state.config.default_models_dir, "ink-wash") / "cover.png"
    assert stored.is_file()
    (item,) = client.get("/api/custom-loras").json()["loras"]
    assert item["cover_url"] == str(stored.resolve())

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        listed = remote.get("/api/custom-loras", headers=AUTH)
        cover_url = listed.json()["loras"][0]["cover_url"]
        assert cover_url.startswith("/api/custom-loras/ink-wash/cover?")
        assert str(test_state.config.default_models_dir) not in listed.text
        fetched = remote.get(cover_url)
    assert fetched.status_code == 200
    assert fetched.headers["content-type"].startswith("image/png")


def test_the_phone_router_cannot_add_restore_or_delete_a_lora(test_state, tmp_path: Path) -> None:
    _install_lora(test_state)
    weights = _weights_file(tmp_path)
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    with TestClient(app) as remote:
        attempts = [
            remote.post(
                "/api/custom-loras",
                headers=AUTH,
                json={"name": "Phone", "source_path": str(weights)},
            ),
            remote.post(
                "/api/custom-loras/ink-wash/weights", headers=AUTH, json={"source_path": str(weights)}
            ),
            remote.delete("/api/custom-loras/ink-wash/weights", headers=AUTH),
            remote.delete("/api/custom-loras/ink-wash", headers=AUTH),
        ]
        listed = remote.get("/api/custom-loras", headers=AUTH)

    assert [attempt.status_code in (404, 405) for attempt in attempts] == [True] * 4
    assert [item["id"] for item in listed.json()["loras"]] == ["ink-wash"]
    assert listed.json()["loras"][0]["downloaded"] is True


def test_run_stores_an_empty_ref_until_execute_and_the_phone_never_sees_loras(test_state) -> None:
    _install_lora(test_state)
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    with TestClient(app) as remote:
        created = remote.post("/api/generations/custom-lora", headers=AUTH, json=_create_body())

    assert created.status_code == 200, created.text
    body = created.json()
    params = body["spec"]["params"]
    assert params["customLoraId"] == "ink-wash"
    assert params["customLoraName"] == "Ink Wash"
    assert params["scale"] == 0.8
    assert "loras" not in params
    assert "safetensors" not in created.text
    stored = SqliteStore(test_state.config.app_data_dir).get_generation(body["id"])
    assert stored is not None
    stored_params = stored.spec["params"]
    assert isinstance(stored_params, dict)
    lora = stored_params["loras"][0]
    assert lora["ref"] == ""
    assert lora["customLoraId"] == "ink-wash"


# --- link info (model card) ---


def test_a_failed_card_request_gives_empty_suggestions_and_not_an_error(
    client: TestClient, fake_services
) -> None:
    fake_services.url_downloader.card_error = UrlDownloadError("Could not reach Hugging Face.")

    response = client.post(
        "/api/custom-loras/link-info", json={"url": "https://huggingface.co/acme/dolly-in"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "trigger": None,
        "description": "",
        "examples": [],
    }


def test_link_info_marks_accepted_links_and_rejects_local_or_unversioned_links(
    client: TestClient,
) -> None:
    accepted = (
        "https://huggingface.co/owner/repo",
        "https://huggingface.co/owner/repo/resolve/main/model.safetensors",
        "https://civitai.com/api/download/models/123",
        "https://civitai.com/models/123?modelVersionId=9",
        "https://example.com/weights/model.safetensors",
    )
    rejected = (
        "https://localhost/model.safetensors",
        "https://127.0.0.1/model.safetensors",
        "https://10.0.0.1/model.safetensors",
        "https://192.168.1.5/model.safetensors",
        "https://huggingface.co/owner/repo/resolve/model.safetensors",
        "https://civitai.com/models/123",
        "https://example.com/weights/model.bin",
    )

    for url in accepted:
        response = client.post("/api/custom-loras/link-info", json={"url": url})
        assert response.status_code == 200
        assert response.json()["accepted"] is True, url
    for url in rejected:
        response = client.post("/api/custom-loras/link-info", json={"url": url})
        assert response.status_code == 200
        assert response.json()["accepted"] is False, url


# --- prompt enhancement with a saved LoRA ---

_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"


def test_manual_enhance_gives_the_enhancer_the_trigger_and_description_of_a_custom_lora(
    client: TestClient, fake_services, test_state, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_lora(test_state, trigger="InkW4sh")
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a fox in a garden"

    response = client.post(
        "/api/enhance-prompt", json={"prompt": "a fox", "customLoraIds": ["ink-wash"]}
    )

    assert response.status_code == 200
    assert response.json()["enhancedPrompt"] == "a fox in a garden InkW4sh"
    call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
    assert _DESCRIPTION in call["system_prompt"]
    assert "Ink Wash" not in call["system_prompt"]


def test_manual_enhance_with_an_unknown_custom_lora_is_refused_and_not_rewritten_generically(
    client: TestClient, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)

    response = client.post(
        "/api/enhance-prompt", json={"prompt": "a fox", "customLoraIds": ["missing"]}
    )

    assert_http_error(response, status_code=404, code="CUSTOM_LORA_NOT_FOUND")


def test_lora_entry_rejects_path_traversal_custom_lora_id_and_dir_helper_refuses_it(
    tmp_path: Path,
) -> None:
    for bad_id in ("../other", "/tmp/evil", r"\\host\share", "Upper", "a" * 65, ""):
        with pytest.raises(ValidationError):
            LoraEntry(ref="", customLoraId=bad_id)
        with pytest.raises(ValueError):
            custom_lora_dir(tmp_path, bad_id)
        assert read_custom_lora(tmp_path, bad_id) is None
