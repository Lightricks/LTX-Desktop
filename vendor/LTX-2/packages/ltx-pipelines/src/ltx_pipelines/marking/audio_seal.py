"""AudioSeal marker wrapping Meta's generator and detector."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from audioseal import AudioSeal
from einops import rearrange
from torch import nn

from ltx_core.types import Audio
from ltx_pipelines.marking.score import score_bits
from ltx_pipelines.marking.types import AudioMarker, MarkDetectResult

logger = logging.getLogger(__name__)

# Baked payload seed shared by embed and detect. Not a signature and not a model version.
AUDIOSEAL_PAYLOAD_SEED = 0x4C5458  # ASCII "LTX"
AUDIOSEAL_NBITS = 16


class AudioSealMarker(AudioMarker):
    """Adapter around loaded AudioSeal generator and detector modules."""

    def __init__(self, generator: nn.Module, detector: nn.Module, *, nbits: int) -> None:
        self._generator = generator
        self._detector = detector
        self._nbits = nbits
        self._payload = baked_audioseal_payload(nbits)

    @classmethod
    def create(
        cls,
        generator: Path | str,
        detector: Path | str,
        *,
        device: torch.device,
        nbits: int = AUDIOSEAL_NBITS,
    ) -> AudioSealMarker:
        """Load local generator and detector; ``eval``, drop grad, move to ``device``."""
        logger.info("Loading AudioSeal from %s and %s", generator, detector)
        gen, det = load_audioseal(generator, detector, nbits=nbits)
        for module in (gen, det):
            module.eval()
            module.requires_grad_(False)
            module.to(device)
        return cls(gen, det, nbits=nbits)

    @property
    def name(self) -> str:
        return "audioseal"

    def embed(self, audio: Audio) -> Audio:
        """Stamp one stereo ``Audio`` as a batch of two mono channels (no downmix)."""
        device = next(self._generator.parameters()).device
        wav, pattern = waveform_to_channel_batch(audio.waveform)
        wav = wav.to(device=device, dtype=torch.float32)
        message = self._payload.to(device=device, dtype=torch.long).unsqueeze(0).expand(wav.shape[0], -1)
        stamped = self._generator(wav, message=message)
        return Audio(
            waveform=rearrange(stamped, f"b 1 t -> {pattern}").to(
                device=audio.waveform.device, dtype=audio.waveform.dtype
            ),
            sampling_rate=audio.sampling_rate,
        )

    def detect(self, audio: Audio) -> MarkDetectResult:
        """Run the detector; ``detected`` is the clip-level presence score (mean >= 0.5)."""
        device = next(self._detector.parameters()).device
        wav, _ = waveform_to_channel_batch(audio.waveform)
        wav = wav.to(device=device, dtype=torch.float32)
        detection, message = self._detector(wav)
        detect_score = float((detection[:, 1, :] > 0.5).to(dtype=torch.float32).mean().item())
        decoded_bits = (message.mean(dim=0) > 0.5).to(dtype=torch.float32)
        bit_accuracy, p_value = score_bits(decoded_bits, self._payload)
        return MarkDetectResult(
            detected=detect_score >= 0.5,
            bit_accuracy=bit_accuracy,
            p_value=p_value,
            payload=self._payload.detach().cpu(),
        )


def baked_audioseal_payload(nbits: int = AUDIOSEAL_NBITS) -> torch.Tensor:
    """Return the baked payload as a ``[nbits]`` integer ``{0, 1}`` tensor."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(AUDIOSEAL_PAYLOAD_SEED)
    return torch.randint(0, 2, (nbits,), generator=generator)


def waveform_to_channel_batch(waveform: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Convert PCM to AudioSeal ``(b, 1, t)`` (stereo as batch-2, no downmix).
    The returned string is the einops source pattern (``1 b t``, ``t b``, or ``b t``).
    """
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError(f"expected batched waveform (1, C, T); got {tuple(waveform.shape)}")
        pattern = "1 b t"
    elif waveform.ndim != 2:
        raise ValueError(f"audio.waveform must be (C, T), (T, C), or (1, C, T); got {tuple(waveform.shape)}")
    elif waveform.shape[1] == 2:
        pattern = "t b"
    elif waveform.shape[0] == 2:
        pattern = "b t"
    else:
        raise ValueError(f"audio.waveform must be stereo (2, N) or (N, 2); got {tuple(waveform.shape)}")
    return rearrange(waveform, f"{pattern} -> b 1 t"), pattern


def load_audioseal(
    generator: Path | str,
    detector: Path | str,
    *,
    nbits: int = AUDIOSEAL_NBITS,
) -> tuple[nn.Module, nn.Module]:
    """Load generator and detector from local file paths (no Hub download)."""
    generator_path = Path(generator)
    detector_path = Path(detector)
    missing = [str(path) for path in (generator_path, detector_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AudioSeal checkpoints missing: {', '.join(missing)}")
    generator = AudioSeal.load_generator(str(generator_path), nbits=nbits)
    detector = AudioSeal.load_detector(str(detector_path), nbits=nbits)
    return generator, detector
