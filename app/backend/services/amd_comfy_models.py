"""Pinned files for the LTX INT8 and Kandinsky Lite ComfyUI graphs."""

from __future__ import annotations

from dataclasses import dataclass

TRANSFORMER = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
HF_REPO = "Lightricks/LTX-2.5"
HF_REVISION = "49b093a39c19169a8a81721dcc541e07e945232f"


@dataclass(frozen=True)
class ComfyWeight:
    category: str
    filename: str
    size: int
    sha256: str
    repo: str = HF_REPO
    revision: str = HF_REVISION
    remote_path: str | None = None
    sample_sha256: str | None = None


# Byte counts and hashes are the LFS metadata at the pinned revision.
REQUIRED_MODELS = (
    ComfyWeight(
        "diffusion_models",
        TRANSFORMER,
        21504034224,
        "c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334",
    ),
    ComfyWeight(
        "text_encoders",
        TEXT_ENCODER,
        15372969374,
        "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f",
    ),
    ComfyWeight(
        "vae",
        VIDEO_VAE,
        1472223346,
        "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
    ),
    ComfyWeight(
        "vae",
        AUDIO_VAE,
        364866540,
        "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    ),
    ComfyWeight(
        "latent_upscale_models",
        UPSCALER,
        995778752,
        "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
    ),
)

# Official Kandinsky 6 ComfyUI Lite 5s SD recipe. Use the full transformer path
# because the Pro checkpoint has the same filename.
LITE_MODEL = "Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors"
KANDINSKY_MODELS = (
    ComfyWeight(
        "diffusion_models",
        LITE_MODEL,
        6381599096,
        "b1b52668561d9138d93bb4da13c9999e615bf569be9f8cf65732d54652447f53",
        repo="kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers",
        revision="4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81",
        remote_path="transformer/diffusion_pytorch_model.safetensors",
    ),
    ComfyWeight(
        "text_encoders",
        "qwen_2.5_vl_7b.safetensors",
        16584415576,
        "cfafd739459bc86257397259f612a9aee88e5b98e85b5c0d0d1717e898b3463a",
        repo="Comfy-Org/Qwen-Image",
        revision="1f12b17be14c89b026c51a91d67c32f84bb047bc",
        remote_path="split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    ),
    ComfyWeight(
        "text_encoders",
        "clip_l.safetensors",
        246144152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        repo="Comfy-Org/Hunyuan-Video",
        revision="533a43813f01517d9e624f9e824a85bb94094120",
        remote_path="split_files/text_encoders/clip_l.safetensors",
    ),
    ComfyWeight(
        "vae",
        "hunyuan_video_vae_bf16.safetensors",
        492984198,
        "e8f8553275406d84ccf22e7a47601650d8f98bdb8aa9ccfdd6506b57a9701aed",
        repo="Comfy-Org/Hunyuan-Video",
        revision="533a43813f01517d9e624f9e824a85bb94094120",
        remote_path="split_files/vae/hunyuan_video_vae_bf16.safetensors",
    ),
    ComfyWeight(
        "audio_vae",
        "v1-44.pth",
        1221942998,
        "ab6cc15dc31947675f75c950c41f4dcfd0d6d1817555ac871f809ec388e4651a",
        repo="hkchengrex/MMAudio",
        revision="eb13a1a98fdbec91753775c57b074ccdfc60587c",
        remote_path="ext_weights/v1-44.pth",
    ),
    ComfyWeight(
        "audio_vae",
        "bigvgan_vocoder/bigvgan_generator.pt",
        489041291,
        "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357",
        repo="nvidia/bigvgan_v2_44khz_128band_512x",
        revision="95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
        remote_path="bigvgan_generator.pt",
    ),
    ComfyWeight(
        "audio_vae",
        "bigvgan_vocoder/config.json",
        1403,
        "65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb",
        repo="nvidia/bigvgan_v2_44khz_128band_512x",
        revision="95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
        remote_path="config.json",
    ),
)
