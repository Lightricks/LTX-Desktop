"""Regression checks for the flattened Comfy API graph (no downloaded weights)."""

import pytest

from services.amd_comfy_models import AUDIO_VAE, TEXT_ENCODER, TRANSFORMER, UPSCALER, VIDEO_VAE
from services.amd_comfy_workflow import (
    BASE_SAMPLER_ID,
    REFINE_SAMPLER_ID,
    SAVE_VIDEO_ID,
    build_workflow,
    mp4_save_inputs,
    resolve_model_name,
)


def make_graph(**changes):
    args = dict(
        prompt="The drummer strikes a cymbal.", negative_prompt="extra drumsticks",
        width=576, height=960, frames=481, fps=24, seed=123,
    )
    return build_workflow(**{**args, **changes})


@pytest.mark.parametrize("source_image", [None, "incoming/portrait.png"])
@pytest.mark.parametrize("single", [False, True])
def test_ltx_graph_routes_real_audio_and_video_to_one_saved_clip(source_image, single):
    graph = make_graph(image_name=source_image, single_stage=single)

    def graph_inputs(key):
        return graph[key]["inputs"]

    base = graph_inputs(BASE_SAMPLER_ID)
    base_video = graph_inputs("20")
    base_audio = graph_inputs("21")
    assert base_video == {
        "width": 576 if single else 288, "height": 960 if single else 480,
        "length": 481, "batch_size": 1,
    }
    assert base_audio["frames_number"] == 481
    assert base_audio["frame_rate"] == graph_inputs("12")["frame_rate"] == graph_inputs("75")["fps"] == 24
    assert graph_inputs(base["noise"][0])["noise_seed"] == 123
    assert len(graph_inputs(base["sigmas"][0])["sigmas"].split(",")) == 9
    assert graph_inputs("10")["text"] == "The drummer strikes a cymbal."
    assert graph_inputs("11")["text"] == "extra drumsticks"
    assert graph_inputs("1")["unet_name"] == TRANSFORMER
    assert graph_inputs("2")["clip_name"] == TEXT_ENCODER
    assert graph_inputs("13")["video_cfg"] == graph_inputs("13")["audio_cfg"] == 1
    assert not any(n["class_type"] == "TextGenerateLTX2Prompt" for n in graph.values())

    if source_image:
        assert graph_inputs("25") == {"image": source_image}
        assert graph_inputs("26")["resize_type.longer_size"] == 1536
        assert graph_inputs("27")["img_compression"] == 18
        assert graph_inputs("28")["strength"] == 0.7
        assert graph_inputs("30")["video_latent"] == ["28", 0]
    else:
        assert graph_inputs("30")["video_latent"] == ["20", 0]
        assert not any(n["class_type"] in {"LoadImage", "LTXVImgToVideoInplace"} for n in graph.values())
    if single:
        assert REFINE_SAMPLER_ID not in graph
        assert not any(n["class_type"] == "LatentUpscaleModelLoader" for n in graph.values())
        last_split = "41"
    else:
        refine = graph_inputs(REFINE_SAMPLER_ID)
        assert graph_inputs("50")["model_name"] == UPSCALER
        assert graph_inputs("51")["samples"] == ["41", 0]
        assert graph_inputs("53")["audio_latent"] == ["41", 1]
        assert graph_inputs(refine["noise"][0])["noise_seed"] == 42
        assert len(graph_inputs(refine["sigmas"][0])["sigmas"].split(",")) == 4
        if source_image:
            assert graph_inputs("52")["strength"] == 1.0
            assert graph_inputs("52")["image"] == graph_inputs("28")["image"]
        last_split = "61"
    assert graph_inputs("70")["samples"] == [last_split, 0]
    assert graph_inputs("71")["samples"] == [last_split, 1]
    assert graph_inputs("75")["audio"] == ["71", 0]
    assert graph_inputs(SAVE_VIDEO_ID)["video"] == ["75", 0]

    # A Comfy prompt executes only dependencies of an output. Catch dangling
    # references and disconnected nodes after changes to either sampling branch.
    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for value in graph_inputs(node_id).values():
            if isinstance(value, list):
                assert len(value) == 2
                visit(value[0])

    visit(SAVE_VIDEO_ID)
    assert reachable == set(graph)


def test_server_model_index_uses_exact_choices_and_reports_missing_weights():
    graph = make_graph()
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    for loader, field, names in (
        ("UNETLoader", "unet_name", [TRANSFORMER]),
        ("VAELoader", "vae_name", [AUDIO_VAE, VIDEO_VAE]),
        ("CLIPLoader", "clip_name", [TEXT_ENCODER]),
        ("LatentUpscaleModelLoader", "model_name", [UPSCALER]),
    ):
        names = [f"library\\LTX\\{name}" for name in names]
        info[loader]["input"]["required"][field] = (
            ["COMBO", {"options": names}] if loader == "LatentUpscaleModelLoader" else [names]
        )
    resolved = make_graph(object_info=info)
    assert resolved["1"]["inputs"]["unet_name"] == f"library\\LTX\\{TRANSFORMER}"
    assert resolved["50"]["inputs"]["model_name"] == f"library\\LTX\\{UPSCALER}"
    del info["LatentUpscaleModelLoader"]
    assert make_graph(object_info=info, single_stage=True)
    with pytest.raises(ValueError, match="LatentUpscaleModelLoader"):
        make_graph(object_info=info)
    info["UNETLoader"]["input"]["required"]["unet_name"] = [[]]
    with pytest.raises(ValueError, match=TRANSFORMER):
        make_graph(object_info=info, single_stage=True)


def test_model_resolver_matches_complete_requested_variant():
    names = ["root/pro/model.safetensors", "X\\lite\\model.safetensors", "lite/MODEL.safetensors"]
    info = {"Loader": {"input": {"required": {"weight": [names]}}}}
    assert resolve_model_name(info, "Loader", "weight", "lite/model.safetensors") == names[2]
    names.append("lite/model.safetensors")
    assert resolve_model_name(info, "Loader", "weight", "lite/model.safetensors") == names[3]
    with pytest.raises(ValueError, match="missing/model.safetensors"):
        resolve_model_name(info, "Loader", "weight", "missing/model.safetensors")


def test_save_video_chooses_server_codec_schema():
    info = {"SaveVideo": {"input": {"required": {"format": ["COMFY_DYNAMICCOMBO_V3", {}]}}}}
    assert mp4_save_inputs(info) == {"format": "mp4", "format.codec": "auto"}
    info["SaveVideo"]["input"]["required"]["format"] = [["auto", "mp4"]]
    assert mp4_save_inputs(info) == {"format": "mp4", "codec": "auto"}


@pytest.mark.parametrize("changes", [
    {"width": 540}, {"height": 32}, {"frames": 480}, {"frames": 1001},
    {"fps": 0}, {"fps": float("nan")}, {"seed": -1}, {"image_name": "../example.png"},
])
def test_wrong_shapes_and_inputs_fail_before_submission(changes):
    with pytest.raises(ValueError):
        make_graph(**changes)
