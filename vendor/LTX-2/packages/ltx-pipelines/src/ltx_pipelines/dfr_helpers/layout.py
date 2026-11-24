"""DFR canvas layout: keyframe segment grid and coordinate helpers."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import NamedTuple, overload

from ltx_core.tiling import DimensionInterval, split_at_seams
from ltx_core.types import VIDEO_SCALE_FACTORS

SEGMENT_CANDIDATES = (24, 32)


def padding_to_segment(content_frames: int, segment: int) -> int:
    """Frames needed to round ``content_frames`` up to a whole number of ``segment``."""
    return (-content_frames) % segment


def choose_segment_length(content_frames: int) -> int:
    """Pick the keyframe segment length from ``SEGMENT_CANDIDATES``, preferring whichever pads least.
    ``content_frames`` is ``num_frames - 1``. Ties keep the larger segment (``-segment`` breaks them).
    """
    if content_frames < 1:
        raise ValueError(f"content_frames must be >= 1, got {content_frames}")
    return min(SEGMENT_CANDIDATES, key=lambda segment: (padding_to_segment(content_frames, segment), -segment))


def resolve_canvas(
    num_frames: int,
    *,
    temporal_scale: int = VIDEO_SCALE_FACTORS.time,
) -> tuple[int, int, list[int]]:
    """Pad ``(num_frames - 1)`` up to a multiple of the segment length, returning ``(N', S, positions)``.
    Positions are ``[S, 2S, ..., N' - 1]``: frame 0 is excluded (already a single-pixel-frame token
    under causal encoding) and the terminal frame is included.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(f"num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 (got {num_frames})")

    content = num_frames - 1
    if content == 0:
        raise ValueError("The canvas needs at least 2 pixel frames")

    segment = choose_segment_length(content)
    content_padded = content + padding_to_segment(content, segment)
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def pixel_to_latent_index(pixel_frame: int, temporal_scale: int = VIDEO_SCALE_FACTORS.time) -> int:
    """Map an x8-border pixel frame to its latent index."""
    if pixel_frame < 0:
        raise ValueError(f"pixel_frame must be >= 0, got {pixel_frame}")
    if pixel_frame != 0 and pixel_frame % temporal_scale != 0:
        raise ValueError(f"pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border")
    return pixel_frame // temporal_scale


def split_canvas_at_seams(
    seams: Sequence[int],
    num_tiles: int,
    overlap: int,
    dim_size: int,
) -> list[DimensionInterval]:
    """Split a DFR canvas on keyframe boundary cells, allowing a remainder segment count.
    Same handover as :func:`ltx_core.tiling.split_at_seams` (lead-in plus the shared seam cell
    land in ``left_ramp``); leftover segments go to the leading tiles.
    """
    return list(split_at_seams(seams, num_tiles, overlap)(dim_size).intervals)


class TemporalTile(NamedTuple):
    """One temporal-upsample window: latent interval plus pixel-frame anchors and slots."""

    interval: DimensionInterval
    pixel_start: int
    pixel_end: int
    anchors: tuple[int, ...]
    slots: tuple[int, ...]


class TemporalTilePlan:
    """Keyframe-seam tiles for one temporal-upsample round. Each window is a :class:`TemporalTile`."""

    tiles: tuple[TemporalTile, ...]

    def __init__(
        self,
        seam_positions: Sequence[int],
        num_frames: int,
        num_tiles: int,
        temporal_scale: int = VIDEO_SCALE_FACTORS.time,
    ) -> None:
        """Partition the canvas with :func:`split_canvas_at_seams` and attach per-window anchors/slots.
        Remainder segments go to the leading tiles.
        Slots are one per canvas segment midpoint, derived globally and then handed to whichever
        window contains them.
        The split takes **no overlap**: a tile's kept cells are exactly the segments it owns. The
        pinned prefix a non-first tile denoises before them is not a cell count at all -- it starts
        at the last keyframe before the seam, so the caller derives it from the plane positions
        (:func:`~ltx_pipelines.dfr_helpers.ops.tile_prefix`) rather than from a configured overlap.
        """
        seams = [0, *(pixel_to_latent_index(position, temporal_scale) for position in seam_positions)]
        latent_len = (num_frames - 1) // temporal_scale + 1
        intervals = split_canvas_at_seams(seams, num_tiles, 0, latent_len)
        # Midpoints of the *canvas* segments, not of each window's own marks. A window's first mark
        # is its own start, so per-window midpoints land wherever that falls -- including off the x8
        # latent border, which a plane may not do: a tile starts on one (see ``tile_prefix``).
        global_slots = [(left + right) // 2 for left, right in itertools.pairwise([0, *seam_positions])]
        tiles: list[TemporalTile] = []
        for interval in intervals:
            pixel_start = interval.start * temporal_scale
            pixel_end = (interval.end - 1) * temporal_scale
            anchors = tuple(position for position in seam_positions if pixel_start <= position <= pixel_end)
            slots = tuple(position for position in global_slots if pixel_start <= position <= pixel_end)
            tiles.append(TemporalTile(interval, pixel_start, pixel_end, anchors, slots))
        self.tiles = tuple(tiles)

    def __iter__(self) -> Iterator[TemporalTile]:
        return iter(self.tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    @overload
    def __getitem__(self, index: int) -> TemporalTile: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TemporalTile, ...]: ...

    def __getitem__(self, index: int | slice) -> TemporalTile | tuple[TemporalTile, ...]:
        return self.tiles[index]
