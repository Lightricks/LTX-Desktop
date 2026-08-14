from pathlib import Path

import pytest
from pydantic import ValidationError
from api_types import parse_lora_catalog
from services.lora_catalog.file_lora_catalog import FileLoraCatalogProvider

_VALID = """
{"schema_version": 1, "ic_loras": [{
  "id": "ingredients-v1", "name": "Ingredients", "description": "d",
  "download": {"repo_id": "Lightricks/x", "variants": [{"id": "default", "label": "Default", "filename": "i.safetensors", "size_bytes": 10}]},
  "requires_hf_login": true,
  "input": {"kind": "image"},
  "preprocessing": [{"utility": "image_to_frames", "params": {}}],
  "instructions": [{"kind": "summary", "title": "What it does", "body": "use it"}],
  "controls": [{"id": "duration", "label": "Duration", "default": 5, "options": [5, 8]}],
  "default_settings": {"skip_stage_2": true, "resolution_factor": 1.0,
                       "audio_mode": "off", "lora_strength": 1.0, "conditioning_strength": 1.0}
}]}
"""

def test_parse_valid_catalog():
    cat = parse_lora_catalog(_VALID)
    assert len(cat.ic_loras) == 1
    r = cat.ic_loras[0]
    assert r.id == "ingredients-v1"
    assert r.input.kind == "image"
    assert r.preprocessing[0].utility == "image_to_frames"
    assert r.default_settings.skip_stage_2 is True
    assert r.controls[0].id == "duration"
    assert r.controls[0].options == [5, 8]
    assert r.supported_models == ["LTX-2.3", "LTX-2.5"]
    assert r.supports_family("LTX-2.3") is True
    assert r.supports_family("LTX-2.5") is True


def test_supported_models_rejects_empty_and_duplicates():
    empty = _VALID.replace(
        '"requires_hf_login": true,',
        '"requires_hf_login": true, "supported_models": [],',
    )
    with pytest.raises(ValidationError, match="supported_models"):
        parse_lora_catalog(empty)
    dup = _VALID.replace(
        '"requires_hf_login": true,',
        '"requires_hf_login": true, "supported_models": ["LTX-2.3", "LTX-2.3"],',
    )
    with pytest.raises(ValidationError, match="supported_models"):
        parse_lora_catalog(dup)


def test_shipped_catalog_allows_2_3_and_2_5():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    for item in [*cat.loras, *cat.ic_loras]:
        assert item.supported_models == ["LTX-2.3", "LTX-2.5"], item.id

def test_controls_default_to_empty():
    no_controls = _VALID.replace(
        '"controls": [{"id": "duration", "label": "Duration", "default": 5, "options": [5, 8]}],',
        "",
    )
    assert parse_lora_catalog(no_controls).ic_loras[0].controls == []

def test_shipped_ingredients_ic_lora_is_wired_for_image_input():
    # Guards the Ingredients wiring: image input expanded to a control video via
    # image_to_frames, with a duration control so the UI shows a seconds selector.
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    ing = next(r for r in cat.ic_loras if r.id == "ingredients")
    assert ing.input.kind == "image"
    assert [s.utility for s in ing.preprocessing] == ["image_to_frames"]
    duration = next(c for c in ing.controls if c.id == "duration")
    assert duration.default in duration.options

def test_parse_rejects_unknown_input_kind():
    bad = _VALID.replace('"kind": "image"', '"kind": "audio"')
    with pytest.raises(ValidationError):
        parse_lora_catalog(bad)


_DURATION_CONTROL = '"controls": [{"id": "duration", "label": "Duration", "default": 5, "options": [5, 8]}],'


def test_parse_select_control():
    j = _VALID.replace(
        _DURATION_CONTROL,
        '"controls": [{"id": "outpaint_region", "kind": "select", "label": "Extend", '
        '"default": "16:9", "options": ["16:9", "all"]}],',
    )
    c = parse_lora_catalog(j).ic_loras[0].controls[0]
    assert c.kind == "select"
    assert c.default == "16:9"
    assert c.options == ["16:9", "all"]


def test_select_control_rejects_int_options():
    bad = _VALID.replace(
        _DURATION_CONTROL,
        '"controls": [{"id": "x", "kind": "select", "label": "X", "default": "a", "options": [1, 2]}],',
    )
    with pytest.raises(ValidationError):
        parse_lora_catalog(bad)


def test_shipped_outpainting_ic_lora_is_wired():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    op = next(r for r in cat.ic_loras if r.id == "outpainting")
    assert op.input is not None and op.input.kind == "video"
    assert [s.utility for s in op.preprocessing] == ["outpaint_canvas"]
    canvas = next(c for c in op.controls if c.kind == "position_canvas")
    assert canvas.id == "outpaint_pads"
    assert canvas.default is None and canvas.options is None
    # Outpainting fills from the scene, so it opts into promptless generation.
    assert op.allows_empty_prompt is True


def test_allows_empty_prompt_defaults_false():
    assert parse_lora_catalog(_VALID).ic_loras[0].allows_empty_prompt is False


def test_allows_reference_image_default_false_and_shipped_3d_render_true():
    assert parse_lora_catalog(_VALID).ic_loras[0].allows_reference_image is False
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    render = next(r for r in cat.ic_loras if r.id == "3d-render-to-real")
    assert render.allows_reference_image is True

def test_file_provider_reads_local_file(tmp_path: Path):
    f = tmp_path / "catalog.json"
    f.write_text(_VALID, encoding="utf-8")
    provider = FileLoraCatalogProvider(str(f))
    cat = provider.get_catalog()
    assert cat.ic_loras[0].id == "ingredients-v1"


_VALID_LORA = """
{"schema_version": 1, "loras": [{
  "id": "my-style", "name": "My Style", "description": "d",
  "download": {"repo_id": "Lightricks/s", "variants": [{"id": "default", "label": "Default", "filename": "s.safetensors", "size_bytes": 5}]},
  "requires_hf_login": false,
  "instructions": [{"kind": "prompting", "title": "Prompt", "body": "use TRIGGER"}],
  "license": {"name": "OpenRAIL-M", "url": "https://example/license"},
  "author": {"name": "Someone"},
  "media": {"thumbnail": "https://example/t.png"},
  "trigger": "TRIGGER",
  "trigger_placement": "anywhere",
  "recommended_strength": 0.8
}]}
"""


def test_parse_plain_lora_entry():
    cat = parse_lora_catalog(_VALID_LORA)
    assert len(cat.loras) == 1 and len(cat.ic_loras) == 0
    lora = cat.loras[0]
    assert lora.input is None  # plain LoRAs need no driving input
    assert lora.license is not None and lora.license.name == "OpenRAIL-M"
    assert lora.author is not None and lora.author.name == "Someone"
    assert lora.trigger == "TRIGGER"
    assert lora.trigger_placement == "anywhere"
    assert lora.recommended_strength == 0.8


def test_trigger_without_placement_rejected():
    bad = _VALID_LORA.replace('"trigger_placement": "anywhere",\n  ', "")
    with pytest.raises(ValidationError):
        parse_lora_catalog(bad)


def test_placement_without_trigger_rejected():
    bad = _VALID_LORA.replace('"trigger": "TRIGGER",\n  ', "")
    with pytest.raises(ValidationError):
        parse_lora_catalog(bad)


def test_prompt_template_placeholders_must_match_template():
    from api_types import PromptTemplateSpec
    from pydantic import ValidationError as VErr

    PromptTemplateSpec(template="upscale", placeholders={})
    PromptTemplateSpec(template="COLORIZE {result}", placeholders={"result": {}})
    with pytest.raises(VErr):
        PromptTemplateSpec(template="COLORIZE {result}", placeholders={})
    with pytest.raises(VErr):
        PromptTemplateSpec(template="upscale", placeholders={"unused": {}})


def test_prompt_template_supports_enum_choices():
    from api_types import PromptTemplateSpec

    spec = PromptTemplateSpec(
        template="crossview. new camera angle: {azimuth}, {elevation}, {distance}.",
        placeholders={
            "azimuth": {"choices": ["same angle", "to the left", "to the right"]},
            "elevation": {"choices": ["lower", "same height", "higher"]},
            "distance": {"choices": ["closer", "same distance", "further"]},
        },
    )
    assert spec.placeholders["azimuth"].choices == ["same angle", "to the left", "to the right"]


def test_prompt_template_rejects_trigger_placement():
    bad = _VALID_LORA.replace(
        '"trigger_placement": "anywhere",',
        '"trigger_placement": "anywhere", "prompt_template": {"template": "TRIGGER", "placeholders": {}},',
    )
    with pytest.raises(ValidationError):
        parse_lora_catalog(bad)


def test_file_provider_falls_back_to_bundled_on_remote_failure(tmp_path: Path):
    bundled = tmp_path / "bundled.json"
    bundled.write_text(_VALID, encoding="utf-8")
    # Unreachable URL (connection refused) -> falls back to the bundled file.
    provider = FileLoraCatalogProvider("http://127.0.0.1:9/catalog.json", fallback_path=str(bundled))
    cat = provider.get_catalog()
    assert cat.ic_loras[0].id == "ingredients-v1"


def test_resolve_media_url_joins_relative_paths():
    from services.lora_catalog.media_urls import LORA_CATALOG_MEDIA_BASE_URL, resolve_media_url

    assert resolve_media_url("fpv-motion_web.mp4") == f"{LORA_CATALOG_MEDIA_BASE_URL}fpv-motion_web.mp4"
    assert resolve_media_url("ltx-team/DayToNight-light.mp4") == (
        f"{LORA_CATALOG_MEDIA_BASE_URL}ltx-team/DayToNight-light.mp4"
    )
    absolute = "https://videos.ltx.io/other/clip.mp4"
    assert resolve_media_url(absolute) == absolute
    assert resolve_media_url(None) is None


def test_with_resolved_media_copies_item():
    from services.lora_catalog.media_urls import LORA_CATALOG_MEDIA_BASE_URL, with_resolved_media

    cat = parse_lora_catalog(_VALID_LORA.replace(
        '"media": {"thumbnail": "https://example/t.png"}',
        '"media": {"demo_video": "my-style_web.mp4"}',
    ))
    original = cat.loras[0]
    resolved = with_resolved_media(original)
    assert original.media is not None and original.media.demo_video == "my-style_web.mp4"
    assert resolved.media is not None
    assert resolved.media.demo_video == f"{LORA_CATALOG_MEDIA_BASE_URL}my-style_web.mp4"


def test_shipped_catalog_uses_relative_demo_videos_and_author_affiliation():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    for item in [*cat.loras, *cat.ic_loras]:
        if item.author is not None:
            assert item.author.name != "Lightricks"
            assert item.author.affiliation in ("ltx", "community")
            if item.author.name == "LTX":
                assert item.author.affiliation == "ltx"
            else:
                assert item.author.affiliation == "community"
        if item.media and item.media.demo_video:
            assert not item.media.demo_video.startswith("http://")
            assert not item.media.demo_video.startswith("https://")
    official = [r for r in cat.ic_loras if r.author and r.author.affiliation == "ltx"]
    assert len(official) == 10
    missing = next(r for r in cat.loras if r.id == "fantasy-anime-style")
    assert missing.media is not None and missing.media.demo_video == "fantasy_anime_style_demo.mp4"


def test_author_affiliation_defaults_to_community():
    cat = parse_lora_catalog(_VALID_LORA)
    assert cat.loras[0].author is not None
    assert cat.loras[0].author.affiliation == "community"


def test_download_variants_must_be_non_empty():
    from api_types import DownloadSpec
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DownloadSpec(repo_id="org/repo", variants=[])


def test_download_variants_require_unique_filenames_and_ids():
    from api_types import DownloadSpec, DownloadVariant
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DownloadSpec(
            repo_id="org/repo",
            variants=[
                DownloadVariant(id="a", label="A", filename="a.safetensors", size_bytes=1),
                DownloadVariant(id="b", label="B", filename="a.safetensors", size_bytes=2),
            ],
        )
    with pytest.raises(ValidationError):
        DownloadSpec(
            repo_id="org/repo",
            variants=[
                DownloadVariant(id="a", label="A", filename="a.safetensors", size_bytes=1),
                DownloadVariant(id="a", label="B", filename="b.safetensors", size_bytes=2),
            ],
        )


def test_download_resolve_variant_defaults_to_first():
    from api_types import DownloadSpec, DownloadVariant

    spec = DownloadSpec(
        repo_id="org/repo",
        variants=[
            DownloadVariant(id="strong", label="Strong", filename="strong.safetensors", size_bytes=2),
            DownloadVariant(id="light", label="Light", filename="light.safetensors", size_bytes=1),
        ],
    )
    assert spec.resolve_variant(None).id == "strong"
    assert spec.default_variant().filename == "strong.safetensors"
    assert spec.resolve_variant("light").filename == "light.safetensors"
    assert spec.candidate_filenames() == ["strong.safetensors", "light.safetensors"]
    assert spec.resolve_variant("x") is None


def test_shipped_3dreal_has_strong_v2_default_and_light_variant():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    item = next(r for r in cat.ic_loras if r.id == "3d-render-to-real")
    assert item.download.default_variant().filename == "3DREAL-strong-v2.safetensors"
    ids = {v.id for v in item.download.variants}
    assert ids == {"strong-v2", "light"}


def test_shipped_catalog_every_download_has_variants():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    for item in [*cat.loras, *cat.ic_loras]:
        assert len(item.download.variants) >= 1, item.id
        assert "filename" not in type(item.download).model_fields
        assert "size_bytes" not in type(item.download).model_fields

def test_shipped_catalog_every_instruction_has_kind():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    for item in [*cat.loras, *cat.ic_loras]:
        for instr in item.instructions:
            assert instr.kind is not None, (item.id, instr.title)


def test_shipped_catalog_every_trigger_has_placement_or_template():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    for item in [*cat.loras, *cat.ic_loras]:
        if item.trigger is not None:
            assert item.trigger_placement is not None or item.prompt_template is not None, item.id


def test_shipped_crossview_prompt_template_has_enum_placeholders():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    crossview = next(r for r in cat.ic_loras if r.id == "crossview-prompt")
    assert crossview.prompt_template is not None
    assert set(crossview.prompt_template.placeholders) == {"azimuth", "elevation", "distance"}
    assert crossview.prompt_template.placeholders["elevation"].choices == ["lower", "same height", "higher"]


def test_shipped_upscale_prompt_template_has_no_placeholders():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    upscale = next(r for r in cat.ic_loras if r.id == "upscale")
    assert upscale.prompt_template is not None
    assert upscale.prompt_template.template == "upscale"
    assert upscale.prompt_template.placeholders == {}


def test_enhancement_examples_default_to_empty():
    assert parse_lora_catalog(_VALID_LORA).loras[0].enhancement_examples == []


def test_enhancement_examples_parse_when_present():
    j = _VALID_LORA.replace(
        '"trigger_placement": "anywhere",',
        '"trigger_placement": "anywhere", "enhancement_examples": ["ex one", "ex two"],',
    )
    assert parse_lora_catalog(j).loras[0].enhancement_examples == ["ex one", "ex two"]


def test_input_optional_defaults_false():
    assert parse_lora_catalog(_VALID).ic_loras[0].input.optional is False


def test_shipped_product_ad_style_accepts_optional_image_input():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    item = next(r for r in cat.loras if r.id == "product-ad-style")
    assert item.input is not None
    assert item.input.kind == "image"
    assert item.input.optional is True


def test_shipped_cinemagraph_requires_non_optional_image_input():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    item = next(r for r in cat.loras if r.id == "cinemagraph-motion")
    assert item.input is not None
    assert item.input.optional is False


def test_shipped_vrgamedevgirl84_loras_have_ltx2_community_license():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    vrg = [r for r in cat.loras if r.author and r.author.name == "vrgamedevgirl84"]
    assert len(vrg) == 9
    for item in vrg:
        assert item.license is not None, item.id
        assert item.license.name == "LTX-2 Community License"
        assert item.license.url == "https://github.com/Lightricks/LTX-2/blob/main/LICENSE"


def test_shipped_cinemagraph_lora_requires_image_input():
    catalog = Path(__file__).parent.parent / "runtime_config" / "lora_catalog.json"
    cat = parse_lora_catalog(catalog.read_text(encoding="utf-8"))
    lora = next(r for r in cat.loras if r.id == "cinemagraph-motion")
    assert lora.input is not None and lora.input.kind == "image"
    assert lora.trigger == "CINEMAGRAPH_MOTION"
    assert lora.trigger_placement == "first_token"
    assert lora.requires_hf_login is True


def test_downloaded_variant_ids_lists_only_installed_checkpoints(tmp_path: Path):
    from runtime_config.model_download_specs import (
        downloaded_ic_lora_variant_ids,
        resolve_ic_lora_path,
    )

    variants = [("strong-v2", "3DREAL-strong-v2.safetensors"), ("light", "3DREAL-light.safetensors")]
    assert downloaded_ic_lora_variant_ids(tmp_path, "3d-render-to-real", variants) == []
    light = resolve_ic_lora_path(tmp_path, "3d-render-to-real", "3DREAL-light.safetensors")
    light.parent.mkdir(parents=True)
    light.write_bytes(b"x")
    assert downloaded_ic_lora_variant_ids(tmp_path, "3d-render-to-real", variants) == ["light"]
