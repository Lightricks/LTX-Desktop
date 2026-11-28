"""MiDaS DPT-Hybrid depth pipeline for IC-LoRA preprocessing."""

from __future__ import annotations

from typing import Any, cast

import torch

from services.services_utils import FrameArray


class MidasDPTPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: torch.device,
    ) -> "MidasDPTPipeline":
        return MidasDPTPipeline(
            model_path=model_path,
            device=device,
        )

    def __init__(
        self,
        model_path: str,
        device: torch.device,
    ) -> None:
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        self._device = device
        processor_class = cast(Any, DPTImageProcessor)
        self._image_processor = processor_class.from_pretrained(model_path)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._dtype = dtype
        model_class = cast(Any, DPTForDepthEstimation)
        self._model = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    @torch.inference_mode()
    def apply(self, frame: FrameArray) -> FrameArray:
        import cv2
        import numpy as np
        from PIL import Image

        h, w = frame.shape[:2]

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs_obj = cast(dict[str, torch.Tensor], self._image_processor(images=image, return_tensors="pt"))
        inputs = {k: v.to(device=self._device, dtype=self._dtype) for k, v in inputs_obj.items()}

        predicted_depth = self._model(**inputs).predicted_depth

        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        del predicted_depth

        depth_np = depth.detach().float().cpu().numpy()
        del depth
        min_depth = float(depth_np.min())
        max_depth = float(depth_np.max())
        if max_depth - min_depth <= 1e-6:
            depth_uint8 = np.zeros((h, w), dtype=np.uint8)
        else:
            normalized = (depth_np - min_depth) / (max_depth - min_depth)
            depth_uint8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return cast(FrameArray, colored)
