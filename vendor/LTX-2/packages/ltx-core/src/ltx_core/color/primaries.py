"""Linear primaries: gamut basis, conversion, and EXR chromaticity tags.
Shared by the VAE HDR working-space path (AP1 ↔ Rec.709) and the HLG encode
path (Rec.709 / ACEScg → Rec.2020).
Matrices and chromaticities come from ``colour-science`` (Bradford CAT), the
same source OpenColorIO-Config-ACES uses for its utility CLFs.
"""

from __future__ import annotations

import enum

import colour
import torch
from torch import Tensor

_CAT = "Bradford"
_CS_ACESCG = colour.RGB_COLOURSPACES["ACEScg"]
_CS_REC709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
_CS_REC2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]


def _rgb_to_rgb(src: colour.RGB_Colourspace, dst: colour.RGB_Colourspace) -> Tensor:
    """Linear RGB→RGB matrix from colour-science, as float32."""
    return torch.as_tensor(
        colour.matrix_RGB_to_RGB(src, dst, chromatic_adaptation_transform=_CAT),
        dtype=torch.float32,
    )


def _exr_chroma(cs: colour.RGB_Colourspace) -> tuple[float, ...]:
    """EXR ``chromaticities`` order: Rxy, Gxy, Bxy, Wxy."""
    r, g, b = cs.primaries
    w = cs.whitepoint
    return (float(r[0]), float(r[1]), float(g[0]), float(g[1]), float(b[0]), float(b[1]), float(w[0]), float(w[1]))


_ACESCG_TO_SRGB = _rgb_to_rgb(_CS_ACESCG, _CS_REC709)
_SRGB_TO_ACESCG = torch.linalg.inv(_ACESCG_TO_SRGB.double()).to(torch.float32)
_REC709_TO_2020 = _rgb_to_rgb(_CS_REC709, _CS_REC2020)
_ACESCG_TO_2020 = _rgb_to_rgb(_CS_ACESCG, _CS_REC2020)

# Public matrix aliases (tests / advanced callers). Prefer ``Primaries.to``.
ACESCG_TO_SRGB = _ACESCG_TO_SRGB
SRGB_TO_ACESCG = _SRGB_TO_ACESCG
REC709_TO_2020 = _REC709_TO_2020
ACESCG_TO_2020 = _ACESCG_TO_2020


def _apply_primaries_matrix(video: Tensor, mat: Tensor) -> Tensor:
    """Apply a 3x3 primaries matrix. Channel axis: dim 1 for 5D, else ``..., C, H, W``."""
    mat = mat.to(device=video.device, dtype=video.dtype)
    if video.ndim == 5:
        return torch.einsum("bcfhw,dc->bdfhw", video, mat)
    return torch.einsum("...chw,dc->...dhw", video, mat)


class Primaries(enum.Enum):
    """Authoring / working-space colour primaries (linear light).
    Rec.2020 is a conversion *target* (HLG master), not an EXR authoring case —
    use :meth:`to_rec2020`.
    """

    REC709 = "rec709"
    AP1 = "ap1"

    @property
    def _colourspace(self) -> colour.RGB_Colourspace:
        match self:
            case Primaries.REC709:
                return _CS_REC709
            case Primaries.AP1:
                return _CS_ACESCG

    @property
    def exr_chromaticities(self) -> tuple[float, ...]:
        """EXR ``chromaticities`` header (R/G/B/W x,y)."""
        return _exr_chroma(self._colourspace)

    @property
    def matrix_to_rec2020(self) -> Tensor:
        """3x3 linear map from this basis to Rec.2020 (for HLG)."""
        match self:
            case Primaries.REC709:
                return _REC709_TO_2020
            case Primaries.AP1:
                return _ACESCG_TO_2020

    def to(self, target: Primaries, video: Tensor) -> Tensor:
        """Convert linear light from this basis into ``target``. Identity when same."""
        if self is target:
            return video
        match (self, target):
            case (Primaries.REC709, Primaries.AP1):
                return _apply_primaries_matrix(video, _SRGB_TO_ACESCG)
            case (Primaries.AP1, Primaries.REC709):
                return _apply_primaries_matrix(video, _ACESCG_TO_SRGB)
        raise ValueError(f"No primaries conversion from {self!r} to {target!r}.")

    def to_rec2020(self, video: Tensor) -> Tensor:
        """Linear light in this basis → Rec.2020 (HLG master space)."""
        return _apply_primaries_matrix(video, self.matrix_to_rec2020)


# Back-compat helpers used by older call sites / re-exports.
def apply_acescg_to_srgb(video: Tensor) -> Tensor:
    """Linear ACEScg (AP1) → linear sRGB/Rec.709 primaries."""
    return Primaries.AP1.to(Primaries.REC709, video)


def apply_srgb_to_acescg(video: Tensor) -> Tensor:
    """Linear sRGB/Rec.709 → linear ACEScg (AP1) primaries."""
    return Primaries.REC709.to(Primaries.AP1, video)


# String-keyed maps for EXR tag lookup / transitional call sites.
# Prefer ``Primaries.matrix_to_rec2020`` / ``Primaries.exr_chromaticities``.
PRIMARY_MATRIX_TO_2020 = {
    Primaries.REC709.value: REC709_TO_2020,
    Primaries.AP1.value: ACESCG_TO_2020,
}

EXR_CHROMATICITIES = {
    Primaries.REC709.value: Primaries.REC709.exr_chromaticities,
    Primaries.AP1.value: Primaries.AP1.exr_chromaticities,
}
