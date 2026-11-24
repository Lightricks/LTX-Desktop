"""Videoseal video marker wrapping Meta's ``Videoseal`` module."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from videoseal.utils.cfg import SubModelConfig, VideosealConfig, setup_model

from ltx_pipelines.marking.score import score_bits
from ltx_pipelines.marking.types import MarkDetectResult, VideoMarker

logger = logging.getLogger(__name__)

# Baked payload seed shared by embed and detect. Not a signature and not a model version.
VIDEOSEAL_PAYLOAD_SEED = 0x4C5458  # ASCII "LTX"

# Max sampled frames held on GPU for one ``detect`` call. Logits are accumulated.
_DETECT_FRAME_BATCH = 8


class VideoSealMarker(VideoMarker):
    """Adapter around a loaded Meta ``Videoseal`` module (PixelSeal, ChunkySeal, …)."""

    def __init__(self, model: torch.nn.Module, *, name: str, nbits: int) -> None:
        self._model = model
        self._name = name
        self._payload = baked_videoseal_payload(nbits)

    @classmethod
    def create(
        cls,
        checkpoint: Path | str,
        *,
        card: Path | str,
        attenuation_config: Path | str,
        device: torch.device,
    ) -> VideoSealMarker:
        """Load local checkpoint, card, and attenuation; ``eval``, drop grad, move to ``device``."""
        card_path = Path(card)
        logger.info("Loading Videoseal card %s from %s", card_path, checkpoint)
        model, nbits = load_videoseal(
            checkpoint,
            card=card_path,
            attenuation_config=attenuation_config,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        return cls(model, name=card_path.stem, nbits=nbits)

    @property
    def name(self) -> str:
        return self._name

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        """Stamp one ``[F, H, W, C]`` RGB tensor; Videoseal resizes internally to 256 and back."""
        device = next(self._model.parameters()).device
        nchw = rearrange(frames.to(device=device, dtype=torch.float32), "f h w c -> f c h w")
        payload = self._payload.to(device=device, dtype=torch.float32)
        stamped = self._model.embed(nchw, msgs=payload, is_video=True, lowres_attenuation=True)["imgs_w"]
        return rearrange(stamped, "f c h w -> f h w c").clamp(0.0, 1.0).to(device=frames.device)

    def detect(self, frames: Iterator[torch.Tensor]) -> MarkDetectResult:
        """Sample at ``step_size``, average logits; ``detected`` is ``log10(p_value) < -6``."""
        device = next(self._model.parameters()).device
        step_size = int(getattr(self._model, "step_size", 8))
        logits: list[torch.Tensor] = []
        sampled = _nchw_sampled_frames(frames, step_size=step_size, device=device)
        for video in _nchw_batches(sampled, _DETECT_FRAME_BATCH):
            bit_logits = self._model.detect(video, is_video=True)["preds"][:, 1:]
            while bit_logits.ndim > 2:
                bit_logits = bit_logits.mean(dim=-1)
            logits.append(bit_logits)
        if not logits:
            raise ValueError("no frames to detect")
        mean_logits = torch.cat(logits, dim=0).mean(dim=0)
        decoded = (mean_logits > 0).to(dtype=torch.float32)
        bit_accuracy, p_value = score_bits(decoded, self._payload)
        detected = p_value == 0.0 or math.log10(p_value) < -6
        return MarkDetectResult(
            detected=detected,
            bit_accuracy=bit_accuracy,
            p_value=p_value,
            payload=self._payload.detach().cpu(),
        )


def _nchw_sampled_frames(
    frames: Iterator[torch.Tensor], *, step_size: int, device: torch.device
) -> Iterator[torch.Tensor]:
    for index, frame in enumerate(frames):
        if index % step_size != 0:
            continue
        rgb = frame.to(device=device, dtype=torch.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"expected frame [H, W, C]; got {tuple(rgb.shape)}")
        yield rearrange(rgb, "h w c -> c h w")


def _nchw_batches(frames: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    while batch := list(islice(frames, batch_size)):
        yield torch.stack(batch, dim=0)


def baked_videoseal_payload(nbits: int) -> torch.Tensor:
    """Return the baked payload as a ``[1, nbits]`` float ``{0, 1}`` tensor."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VIDEOSEAL_PAYLOAD_SEED)
    return torch.randint(0, 2, (1, nbits), generator=generator).float()


def load_videoseal(
    checkpoint: Path | str,
    *,
    card: Path | str,
    attenuation_config: Path | str,
) -> tuple[torch.nn.Module, int]:
    """Build a Videoseal module from local checkpoint, card YAML, and attenuation YAML paths.
    Videoseal's ``setup_model_from_checkpoint`` looks up repo-root ``configs/*.yaml``
    that the installed package does not ship. Pass the JND attenuation YAML from the pack
    (``videoseal_attenuation.yaml``).
    """
    card_path = Path(card)
    if not card_path.is_file():
        raise FileNotFoundError(f"Videoseal model card not found: {card_path}")
    raw = OmegaConf.load(card_path)
    raw.args.attenuation_config = str(Path(attenuation_config))
    nbits = int(raw.args.nbits)
    config = VideosealConfig(
        args=raw.args,
        embedder=SubModelConfig(model=str(raw.embedder.model), params=raw.embedder.params),
        extractor=SubModelConfig(model=str(raw.extractor.model), params=raw.extractor.params),
    )
    return setup_model(config, Path(checkpoint)), nbits
