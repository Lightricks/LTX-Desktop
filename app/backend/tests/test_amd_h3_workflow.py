import pytest

from services.amd_h3_models import (
    H3_AUDIO_VAE, H3_FL2VA_MODEL, H3_REF2VA_MODEL,
    H3_TEXT_ENCODER, H3_TURBO_LORA, H3_VIDEO_VAE,
)
from services.amd_h3_workflow import build_h3_workflow


def info():
    result = {kind: {} for kind in (
        "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "BasicGuider", "KSamplerSelect",
        "BasicScheduler", "RandomNoise", "SamplerCustomAdvanced", "LoadImage", "SaveVideo",
        "VAEDecode", "VAEDecodeAudio", "CreateVideo",
    )}
    for kind, field, names in (
        ("UNETLoader", "unet_name", [H3_FL2VA_MODEL, H3_REF2VA_MODEL]),
        ("LoraLoaderModelOnly", "lora_name", [H3_TURBO_LORA]),
        ("CLIPLoader", "clip_name", [H3_TEXT_ENCODER]),
        ("VAELoader", "vae_name", [H3_VIDEO_VAE, H3_AUDIO_VAE]),
    ):
        result[kind] = {"input": {"required": {field: ["COMBO", {"options": ["minimax\\" + name for name in names]}]}}}
    return result


def build(engine, **kwargs):
    options = dict(
        prompt="Rain on slate roofs.", width=864, height=480, frames=124, fps=24, seed=12, object_info=info(),
    )
    return build_h3_workflow(engine=engine, **(options | kwargs))


@pytest.mark.parametrize("first,last", [(None, None), ("first.png", None), (None, "last.png"), ("a.png", "b.png")])
def test_fl2va_anchors_keep_their_positions_and_audio_stream(first, last):
    graph, output = build("h3-fl2va", first_image=first, last_image=last)
    inputs = graph["condition"]["inputs"]
    for filename, field in ((first, "first_frame"), (last, "last_frame")):
        if filename is None:
            assert field not in inputs
        else:
            assert graph[inputs[field][0]] == {"class_type": "LoadImage", "inputs": {"image": filename}}
    assert graph["transformer"]["inputs"]["unet_name"] == "minimax\\" + H3_FL2VA_MODEL
    assert graph["encoder"]["inputs"]["clip_name"] == "minimax\\" + H3_TEXT_ENCODER
    assert graph["encoder"]["inputs"]["type"] == "minimax"
    assert graph["frames"]["inputs"]["samples"] == graph["audio"]["inputs"]["samples"] == ["sample", 0]
    assert graph["movie"]["inputs"]["images"] == ["frames", 0]
    assert graph["movie"]["inputs"]["audio"] == ["audio", 0]
    assert graph[output]["inputs"]["video"] == ["movie", 0]
    assert "turbo" not in graph


def test_distillation_patches_model_for_both_guider_and_scheduler():
    graph, _ = build("h3-fl2va-distill", last_image="last.png")
    assert graph["turbo"]["inputs"]["lora_name"] == "minimax\\" + H3_TURBO_LORA
    assert graph["guider"]["inputs"]["model"] == graph["schedule"]["inputs"]["model"] == ["turbo", 0]
    assert graph["schedule"]["inputs"]["steps"] == 8


def test_ref2va_preserves_order_and_uses_its_own_checkpoint():
    reference_names = [f"folder/{i}.png" for i in reversed(range(9))]
    graph, _ = build("h3-ref2va", width=768, height=1344, frames=243, reference_images=reference_names)
    assert graph["transformer"]["inputs"]["unet_name"] == "minimax\\" + H3_REF2VA_MODEL
    assert graph["condition"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["schedule"]["inputs"]["steps"] == 20
    for index, name in enumerate(reference_names):
        image_link = graph["condition"]["inputs"][f"ref_images.ref_image_{index}"]
        assert graph[image_link[0]]["inputs"]["image"] == name


@pytest.mark.parametrize("engine,kwargs", [
    ("h3-ref2va", {}), ("h3-ref2va", {"reference_images": ["r.png"] * 10}),
    ("h3-ref2va", {"reference_images": "r.png"}),
    ("h3-ref2va", {"reference_images": ["r.png"], "first_image": "f.png"}),
    ("h3-fl2va", {"reference_images": ["r.png"]}), ("h3-fl2va", {"first_image": "../x.png"}),
    ("h3-fl2va", {"fps": 30}), ("h3-fl2va", {"frames": 121}),
    ("h3-fl2va", {"negative_prompt": "do not zoom"}),
])
def test_unsupported_inputs_are_refused_instead_of_dropped(engine, kwargs):
    with pytest.raises(ValueError):
        build(engine, **kwargs)


def test_missing_distillation_lora_cannot_silently_run_base_at_eight_steps():
    installed = info()
    del installed["LoraLoaderModelOnly"]
    assert build("h3-fl2va", object_info=installed)
    with pytest.raises(ValueError, match="LoraLoaderModelOnly"):
        build("h3-fl2va-distill", object_info=installed)
