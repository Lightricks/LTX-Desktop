from services.amd_kandinsky_workflow import LITE_MODEL, build_kandinsky_workflow


def test_image_graph_uses_lite_with_similarly_named_pro_installed_and_decodes_generated_audio():
    node_types = (
        "Kandinsky6Sampler", "Kandinsky6TextEncode", "Kandinsky6EmptyLatent", "LoadImage",
        "ImageScale", "VAEEncode", "Kandinsky6ImageToVideoAudio", "Kandinsky6RemoveReferenceLatent",
        "VAEDecode", "Kandinsky6AudioVAEDecode", "CreateVideo", "SaveVideo",
    )
    info = {name: {} for name in node_types}

    def choices(**inputs):
        return {"input": {"required": {field: [names] for field, names in inputs.items()}}}

    installed_lite = "Local\\" + LITE_MODEL.replace("/", "\\")
    info.update(
        UNETLoader=choices(unet_name=[LITE_MODEL.replace("Lite", "Pro"), installed_lite]),
        DualCLIPLoader=choices(clip_name1=["qwen_2.5_vl_7b.safetensors"], clip_name2=["clip_l.safetensors"]),
        VAELoader=choices(vae_name=["hunyuan_video_vae_bf16.safetensors"]),
        Kandinsky6AudioVAELoader=choices(tod_vae=["v1-44.pth"], bigvgan_dir=["bigvgan_vocoder"]),
    )
    graph, out = build_kandinsky_workflow(prompt="A cymbal rings.", width=480, height=864, fps=24,
                                         frames=121, seed=4, image_name="inputs/reference.png", object_info=info)

    assert graph["denoiser"]["inputs"]["unet_name"] == installed_lite
    assert graph["k_sampler"]["inputs"]["latent_image"] == ["condition", 2]
    assert graph["decode"]["inputs"]["samples"] == graph["sound"]["inputs"]["joint_latent"] == ["trim_reference", 0]
    assert graph["video"]["inputs"]["audio"] == ["sound", 0]
    assert graph[out]["inputs"]["video"] == ["video", 0]
