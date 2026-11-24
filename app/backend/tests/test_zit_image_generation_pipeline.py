"""Device-placement behavior for the local Z-Image pipeline."""

from __future__ import annotations

from services.image_generation_pipeline.zit_image_generation_pipeline import ZitImageGenerationPipeline


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def enable_model_cpu_offload(self) -> None:
        self.calls.append("offload")

    def to(self, device: str) -> None:
        self.calls.append(("to", device))


def _pipeline_stub() -> ZitImageGenerationPipeline:
    pipe = object.__new__(ZitImageGenerationPipeline)
    pipe.pipeline = _FakePipeline()
    pipe._img2img = object()
    pipe._device = None
    pipe._cpu_offload_active = False
    return pipe


def test_cuda_uses_model_cpu_offload() -> None:
    pipe = _pipeline_stub()
    pipe.to("cuda")
    assert pipe.pipeline.calls == ["offload"]
    assert pipe._cpu_offload_active is True
    assert pipe._device == "cuda"
    assert pipe._img2img is None


def test_mps_is_resident_and_does_not_offload() -> None:
    pipe = _pipeline_stub()
    pipe.to("mps")
    assert pipe.pipeline.calls == [("to", "mps")]
    assert pipe._cpu_offload_active is False
    assert pipe._device == "mps"
    assert pipe._img2img is None


def test_cpu_moves_without_offload() -> None:
    pipe = _pipeline_stub()
    pipe.to("cpu")
    assert pipe.pipeline.calls == [("to", "cpu")]
    assert pipe._cpu_offload_active is False
    assert pipe._device == "cpu"
