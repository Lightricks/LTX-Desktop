"""LoRA recipe registry + create-path behavior (LTXP-499 foundation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

from _routes._errors import HTTPError
from api_types import DownloadVariant, LoraEntry
from runtime_config.model_download_specs import resolve_lora_path
from services.features import lora_recipes
from services.features.lora_recipes import (
    apply_recipe_prompt_template,
    get_lora_recipe,
    i2v_lora_recipes,
    make_recipe_lora_resolver,
    t2v_lora_recipes,
)


def _store(test_state):
    from services.sqlite_store import SqliteStore

    return SqliteStore(test_state.config.app_data_dir)


def _install_cozy_felt_weights(test_state) -> Path:
    """Write the recipe's catalog weights where resolve_lora_path expects them."""
    path = resolve_lora_path(
        test_state.config.default_models_dir, "cozy-felt-style", "CozyFelt.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _recipe_body(prompt: str = "a fox on a skateboard", scale: float = 1.0):
    return {
        "params": {
            "prompt": prompt,
            "model": "ltx-2.5-fast",
            "catalogId": "cozy-felt-style",
            "scale": scale,
            "variantId": "default",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
        }
    }


@pytest.fixture
def supported_recipe_device(monkeypatch, create_fake_model_files):
    # The gate uses the real accelerator, which is "cpu" on CI (linux/windows) — pin a device
    # cozy-felt supports so these tests reach their target check, not the device gate.
    # Also install a runnable LTX bundle so the supported_models filter doesn't hide the recipe.
    from handlers import queued_generation_handler

    create_fake_model_files()
    monkeypatch.setattr(queued_generation_handler, "accelerator_backend", lambda: "mps")


# --- registry / wrap (pure) ---


def test_registry_maps_cozy_felt_to_style_t2v() -> None:
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    assert recipe.catalog_id == "cozy-felt-style"
    assert recipe.mode == "t2v"
    assert recipe.devices == ("cuda", "mps")


def test_registry_unmapped_id_is_none() -> None:
    assert get_lora_recipe("not-a-real-recipe") is None


def test_registry_maps_dolly_in_to_i2v() -> None:
    recipe = get_lora_recipe("dolly-in")
    assert recipe is not None
    assert recipe.catalog_id == "dolly-in"
    assert recipe.mode == "i2v"
    assert recipe.devices == ("cuda", "mps")


def test_t2v_lora_recipes_match_shared_contract() -> None:
    import json

    contract_path = Path(__file__).resolve().parents[2] / "shared" / "lora-recipes.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = [entry["id"] for entry in contract if entry["mode"] == "t2v"]
    assert [r.recipe_id for r in t2v_lora_recipes()] == expected


def test_i2v_lora_recipes_match_shared_contract() -> None:
    import json

    contract_path = Path(__file__).resolve().parents[2] / "shared" / "lora-recipes.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = [entry["id"] for entry in contract if entry["mode"] == "i2v"]
    assert [r.recipe_id for r in i2v_lora_recipes()] == expected


def test_registry_matches_shared_contract() -> None:
    # shared/lora-recipes.json is the single source of truth for the id / catalogId /
    # mode / devices contract; the FE registry is asserted against the same file
    # (frontend/lib/lora-recipes.test.ts), so the two can't drift. `listed` is a
    # frontend-only Home concern except `requiresEndFrame`, which both sides
    # must agree on so Transition cannot drift.
    import json

    contract_path = (
        Path(__file__).resolve().parents[2] / "shared" / "lora-recipes.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        entry["id"]: {
            "catalogId": entry["catalogId"],
            "mode": entry["mode"],
            "devices": tuple(entry["devices"]),
            "requiresEndFrame": bool(entry.get("requiresEndFrame", False)),
        }
        for entry in contract
    }
    actual = {
        recipe.recipe_id: {
            "catalogId": recipe.catalog_id,
            "mode": recipe.mode,
            "devices": recipe.devices,
            "requiresEndFrame": recipe.requires_end_frame,
        }
        for recipe in lora_recipes.LORA_RECIPES.values()
    }
    assert actual == expected


def test_prompt_wrap_applies_felt_scaffold_when_trigger_absent() -> None:
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    wrapped = apply_recipe_prompt_template(recipe, "a fox on a skateboard")
    assert wrapped.startswith("F3ltCut0u7 handcrafted felt")
    assert "a fox on a skateboard" in wrapped


def test_prompt_wrap_applies_scaffold_even_when_trigger_already_present() -> None:
    # Regression (LTXP-499): local enhancement enforces the trigger on every
    # prompt but never adds the felt scaffold. The wrap must still apply the
    # scaffold — a trigger-based skip dropped it on every enhanced generation.
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    enhanced = "A long enhanced felt scene with F3ltCut0u7 stitched in by enhancement"
    wrapped = apply_recipe_prompt_template(recipe, enhanced)
    assert wrapped.startswith("F3ltCut0u7 handcrafted felt")
    assert enhanced in wrapped


def test_prompt_wrap_is_idempotent_when_scaffold_already_present() -> None:
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    already_wrapped = apply_recipe_prompt_template(recipe, "a fox on a skateboard")
    assert apply_recipe_prompt_template(recipe, already_wrapped) == already_wrapped


def test_prompt_wrap_applies_transition_suffix() -> None:
    recipe = get_lora_recipe("transition")
    assert recipe is not None
    wrapped = apply_recipe_prompt_template(recipe, "a wave becomes a mountain")
    assert wrapped.endswith("zhuanchang")
    assert wrapped.startswith("a wave becomes a mountain")
    assert apply_recipe_prompt_template(recipe, wrapped) == wrapped


def test_prompt_wrap_applies_scaffold_when_it_appears_mid_prompt() -> None:
    # Catalog-aware enhancement can echo the prompting example, embedding the
    # scaffold sentence mid-prompt. The wrap must still prepend the style lock —
    # a substring guard would wrongly skip it.
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    assert recipe.prompt_template is not None
    prefix = recipe.prompt_template.split("{prompt}", 1)[0].strip()
    mid = f"A fox on a skateboard. {prefix} And more scene details."
    wrapped = apply_recipe_prompt_template(recipe, mid)
    assert wrapped.startswith("F3ltCut0u7 handcrafted felt")
    assert wrapped != mid
    assert mid in wrapped


def test_prompt_wrap_fails_closed_when_template_has_no_placeholder() -> None:
    # A template without {prompt} must not drop the user's scene.
    recipe = lora_recipes.LoraRecipe(
        recipe_id="broken",
        catalog_id="broken",
        mode="t2v",
        devices=("cuda",),
        prompt_template="F3ltCut0u7 style with no placeholder",
    )
    assert apply_recipe_prompt_template(recipe, "a fox") == "a fox"


# --- create path (HTTP) ---


def test_create_recipe_resolves_catalog_id_and_stores_feature(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    _install_cozy_felt_weights(test_state)

    response = client.post(
        "/api/generations/recipes/cozy-felt", json=_recipe_body(scale=1.5)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feature"] == "cozy-felt"

    lora = body["spec"]["params"]["loras"][0]
    assert lora["catalogId"] == "cozy-felt-style"
    assert lora["displayName"] == "Cozy Felt Style"
    assert lora["scale"] == 1.5
    # The stored spec carries catalogId + scale only; the absolute weights path is never
    # persisted (resolved fresh at execution) so no server filesystem path leaks to the API.
    assert lora["ref"] == ""
    # The stored prompt is the raw user scene, not the wrapped scaffold.
    assert body["spec"]["params"]["prompt"] == "a fox on a skateboard"
    assert body["spec"]["params"]["model"] == "ltx-2.5-fast"
    assert "F3ltCut0u7" not in body["spec"]["params"]["prompt"]

    assert len(_store(test_state).list_generations("cozy-felt")) == 1


def test_create_recipe_without_a_variant_id_uses_the_only_checkpoint(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    _install_cozy_felt_weights(test_state)
    body = _recipe_body()
    del body["params"]["variantId"]

    response = client.post("/api/generations/recipes/cozy-felt", json=body)

    assert response.status_code == 200, response.text
    stored = response.json()["spec"]["params"]["loras"][0]["variantId"]
    assert stored == "default"


def test_create_recipe_with_an_explicit_variant_requires_that_file_not_index_zero(
    client: TestClient, test_state, fake_services, supported_recipe_device
) -> None:
    catalog = fake_services.lora_catalog_provider._catalog
    cozy = next(lora for lora in catalog.loras if lora.id == "cozy-felt-style")
    default_filename = cozy.download.variants[0].filename
    alt = DownloadVariant(
        id="ltx-2.3__cozy-felt-style__alt",
        label="Alt",
        filename="alt.safetensors",
        size_bytes=10,
        base_model="LTX-2.3",
    )
    loras = []
    for lora in catalog.loras:
        if lora.id != "cozy-felt-style":
            loras.append(lora)
            continue
        loras.append(
            lora.model_copy(
                update={
                    "download": lora.download.model_copy(
                        update={"variants": [*lora.download.variants, alt]}
                    )
                }
            )
        )
    fake_services.lora_catalog_provider._catalog = catalog.model_copy(update={"loras": loras})
    default_path = resolve_lora_path(
        test_state.config.default_models_dir, "cozy-felt-style", default_filename
    )
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_bytes(b"\x00" * 64)
    body = _recipe_body()
    body["params"]["variantId"] = alt.id
    missing = client.post("/api/generations/recipes/cozy-felt", json=body)
    assert missing.status_code == 422
    assert missing.json()["code"] == "LORA_NOT_DOWNLOADED"

    alt_path = resolve_lora_path(
        test_state.config.default_models_dir, "cozy-felt-style", alt.filename
    )
    alt_path.write_bytes(b"\x00" * 64)
    # Index 0 is gone; the requested checkpoint is the one that must exist.
    default_path.unlink()
    created = client.post("/api/generations/recipes/cozy-felt", json=body)
    assert created.status_code == 200, created.text
    assert created.json()["spec"]["params"]["loras"][0]["variantId"] == alt.id


def test_create_recipe_rejects_when_weights_not_downloaded(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    response = client.post("/api/generations/recipes/cozy-felt", json=_recipe_body())

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_NOT_DOWNLOADED"
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_rejects_when_required_ltx_family_missing(
    client: TestClient, test_state, monkeypatch, create_fake_model_files, fake_services
) -> None:
    from handlers import queued_generation_handler

    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    monkeypatch.setattr(queued_generation_handler, "accelerator_backend", lambda: "mps")
    _install_cozy_felt_weights(test_state)
    catalog = fake_services.lora_catalog_provider._catalog
    fake_services.lora_catalog_provider._catalog = catalog.model_copy(
        update={
            "loras": [
                lora.model_copy(update={"supported_models": ["LTX-2.3"]})
                if lora.id == "cozy-felt-style"
                else lora
                for lora in catalog.loras
            ]
        }
    )

    response = client.post("/api/generations/recipes/cozy-felt", json=_recipe_body())

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_UNSUPPORTED_MODEL"
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_rejects_mismatched_catalog_id(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    _install_cozy_felt_weights(test_state)
    body = _recipe_body()
    body["params"]["catalogId"] = "some-other-lora"

    response = client.post("/api/generations/recipes/cozy-felt", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_UNKNOWN"
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_body_forbids_client_supplied_ref(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    # Install weights so the same body *without* a ref is a valid create — otherwise
    # this test would green even if extra="forbid" were dropped, because the request
    # would 422 LORA_NOT_DOWNLOADED regardless.
    _install_cozy_felt_weights(test_state)
    body = _recipe_body()
    body["params"]["ref"] = "loras/cozy-felt-style/CozyFelt.safetensors"

    response = client.post("/api/generations/recipes/cozy-felt", json=body)

    # extra="forbid": a filesystem ref is not part of the recipe wire contract.
    assert response.status_code == 422
    # Rejected for the forbidden field, not because weights are missing.
    assert response.json().get("code") != "LORA_NOT_DOWNLOADED"
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_rejects_malformed_recipe_id(
    client: TestClient, test_state
) -> None:
    # The {recipe_id} path param is pattern-constrained (^[a-z0-9-]+$), so a
    # malformed id 422s at the route before create_lora_recipe runs.
    response = client.post("/api/generations/recipes/Bad_Id", json=_recipe_body())

    assert response.status_code == 422
    assert _store(test_state).list_generations("Bad_Id") == []


def test_create_recipe_rejects_unsupported_device(
    client: TestClient, test_state, monkeypatch
) -> None:
    from handlers import queued_generation_handler

    _install_cozy_felt_weights(test_state)
    # This machine reports mps, but the recipe only supports cuda.
    monkeypatch.setattr(
        queued_generation_handler, "accelerator_backend", lambda: "mps"
    )
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    monkeypatch.setitem(
        lora_recipes.LORA_RECIPES,
        "cozy-felt",
        lora_recipes.LoraRecipe(
            recipe_id="cozy-felt",
            catalog_id="cozy-felt-style",
            mode="t2v",
            devices=("cuda",),
            prompt_template=recipe.prompt_template,
        ),
    )

    response = client.post("/api/generations/recipes/cozy-felt", json=_recipe_body())

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_UNSUPPORTED_DEVICE"
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_rejects_when_local_generation_unsupported(
    client: TestClient, test_state
) -> None:
    _install_cozy_felt_weights(test_state)
    # A machine that can't generate locally must not enqueue a local-only recipe job.
    test_state.config.local_generations_mode = "unsupported"

    response = client.post("/api/generations/recipes/cozy-felt", json=_recipe_body())

    assert response.status_code == 422
    assert _store(test_state).list_generations("cozy-felt") == []


def test_create_recipe_rejects_when_hf_login_required(
    client: TestClient, test_state, monkeypatch, supported_recipe_device
) -> None:
    from state.app_state_types import HfNotAuthenticated

    _install_cozy_felt_weights(test_state)
    # Default test state is HF-authenticated; sign out so a gated item is refused.
    test_state.state.hf_auth_state = HfNotAuthenticated()
    item = test_state._lora_catalog_provider.get_lora("cozy-felt-style")
    assert item is not None
    monkeypatch.setattr(item, "requires_hf_login", True)

    response = client.post("/api/generations/recipes/cozy-felt", json=_recipe_body())

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_HF_AUTH_REQUIRED"
    assert _store(test_state).list_generations("cozy-felt") == []


# --- execution-time ref hydration (make_recipe_lora_resolver) ---


def _cozy_felt_resolver(test_state):
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    return make_recipe_lora_resolver(
        recipe,
        catalog=test_state._lora_catalog_provider,
        models_dir=lambda: test_state.config.default_models_dir,
    )


def test_recipe_resolver_hydrates_ref_from_catalog_id(test_state) -> None:
    resolved = _install_cozy_felt_weights(test_state)
    resolver = _cozy_felt_resolver(test_state)

    out = resolver([LoraEntry(ref="", scale=1.5, catalogId="cozy-felt-style")])

    # Stored spec carries no path; the resolver fills the current on-disk ref, scale intact.
    assert out[0].ref == str(resolved)
    assert out[0].scale == 1.5
    assert out[0].catalogId == "cozy-felt-style"


def test_recipe_resolver_fails_when_weights_removed(test_state) -> None:
    # Weights never installed: a retry after the adapter is gone fails on this job rather
    # than replaying a stale path.
    resolver = _cozy_felt_resolver(test_state)

    with pytest.raises(HTTPError) as exc:
        resolver([LoraEntry(ref="", scale=1.0, catalogId="cozy-felt-style")])

    assert exc.value.code == "LORA_NOT_DOWNLOADED"


def test_recipe_resolver_passes_through_foreign_entries(test_state) -> None:
    resolver = _cozy_felt_resolver(test_state)

    entry = LoraEntry(ref="loras/user/custom.safetensors", scale=0.8, catalogId=None)
    out = resolver([entry])

    assert out == [entry]


def _png(path: Path) -> Path:
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path)
    return path


def _install_dolly_in_weights(test_state) -> Path:
    path = resolve_lora_path(
        test_state.config.default_models_dir,
        "dolly-in",
        "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _dolly_in_body(start_asset_id: str, *, prompt: str = "push in on her face"):
    return {
        "params": {
            "prompt": prompt,
            "model": "ltx-2.5-fast",
            "catalogId": "dolly-in",
            "scale": 1.0,
            "variantId": "default",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
            "aspectRatio": "auto",
        },
        "inputs": {"startFrame": {"assetId": start_asset_id}},
    }


def test_create_dolly_in_requires_start_frame(
    client: TestClient, test_state, supported_recipe_device
) -> None:
    _install_dolly_in_weights(test_state)
    body = _dolly_in_body("unused")
    del body["inputs"]

    response = client.post("/api/generations/recipes/dolly-in", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("dolly-in") == []


def test_create_dolly_in_stores_start_frame_and_empty_lora_ref(
    client: TestClient, test_state, supported_recipe_device, tmp_path: Path
) -> None:
    _install_dolly_in_weights(test_state)
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "dolly-start.png"))}
    ).json()

    response = client.post(
        "/api/generations/recipes/dolly-in", json=_dolly_in_body(start["id"])
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feature"] == "dolly-in"
    assert body["spec"]["inputs"]["startFrame"]["assetId"] == start["id"]
    lora = body["spec"]["params"]["loras"][0]
    assert lora["catalogId"] == "dolly-in"
    assert lora["ref"] == ""
    assert body["spec"]["params"]["prompt"] == "push in on her face"
    assert "dolly-in camera move" not in body["spec"]["params"]["prompt"]
    assert len(_store(test_state).list_generations("dolly-in")) == 1


def test_create_cozy_felt_rejects_start_frame(
    client: TestClient, test_state, supported_recipe_device, tmp_path: Path
) -> None:
    _install_cozy_felt_weights(test_state)
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "felt-start.png"))}
    ).json()
    body = _recipe_body()
    body["inputs"] = {"startFrame": {"assetId": start["id"]}}

    response = client.post("/api/generations/recipes/cozy-felt", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("cozy-felt") == []


def _install_transition_weights(test_state) -> Path:
    path = resolve_lora_path(
        test_state.config.default_models_dir,
        "transition",
        "ltx2.3-transition.safetensors",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _transition_body(start_asset_id: str, end_asset_id: str, *, prompt: str = "morph"):
    return {
        "params": {
            "prompt": prompt,
            "model": "ltx-2.5-fast",
            "catalogId": "transition",
            "scale": 1.0,
            "variantId": "default",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
            "aspectRatio": "auto",
        },
        "inputs": {
            "startFrame": {"assetId": start_asset_id},
            "endFrame": {"assetId": end_asset_id},
        },
    }


def test_create_transition_requires_end_frame(
    client: TestClient, test_state, supported_recipe_device, tmp_path: Path
) -> None:
    _install_transition_weights(test_state)
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "transition-start.png"))}
    ).json()
    body = _transition_body(start["id"], "unused")
    del body["inputs"]["endFrame"]

    response = client.post("/api/generations/recipes/transition", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("transition") == []


def test_create_transition_stores_start_and_end_frames(
    client: TestClient, test_state, supported_recipe_device, tmp_path: Path
) -> None:
    _install_transition_weights(test_state)
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "transition-start.png"))}
    ).json()
    end = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "transition-end.png"))}
    ).json()

    response = client.post(
        "/api/generations/recipes/transition",
        json=_transition_body(start["id"], end["id"]),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feature"] == "transition"
    assert body["spec"]["inputs"]["startFrame"]["assetId"] == start["id"]
    assert body["spec"]["inputs"]["endFrame"]["assetId"] == end["id"]
    assert body["spec"]["params"]["loras"][0]["catalogId"] == "transition"
    assert body["spec"]["params"]["loras"][0]["ref"] == ""
    assert "zhuanchang" not in body["spec"]["params"]["prompt"]


def test_create_dolly_in_rejects_end_frame(
    client: TestClient, test_state, supported_recipe_device, tmp_path: Path
) -> None:
    _install_dolly_in_weights(test_state)
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "dolly-start.png"))}
    ).json()
    end = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "dolly-end.png"))}
    ).json()
    body = _dolly_in_body(start["id"])
    body["inputs"]["endFrame"] = {"assetId": end["id"]}

    response = client.post("/api/generations/recipes/dolly-in", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("dolly-in") == []
