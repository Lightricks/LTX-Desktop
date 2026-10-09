"""Aspect-preserving fit of a source size into a box, on a pixel grid."""

from __future__ import annotations

# The grid can force a shape error: 1280x720 has no 64-grid size with the same aspect
# that is not smaller than 1280x704 (2.3% off). Accept a size up to 3% off.
_ASPECT_TOLERANCE = 0.03


def fit_in_box(width: int, height: int, box_width: int, box_height: int, grid: int) -> tuple[int, int]:
    """Largest grid size inside the box and inside the source, close to the source aspect.

    The result never has an edge above the box or above the source, so it never
    upscales. Each grid size inside that limit has an aspect error against the source.
    The largest area wins among sizes within 3% of the source aspect. If the grid leaves
    none within 3%, the smallest error wins. A limit below one grid cell gives one cell:
    the minimum canvas, the same floor that the other two-stage recipes have.
    """
    limit_width = min(width, box_width)
    limit_height = min(height, box_height)
    aspect = width / height
    sizes = [
        (candidate_w, candidate_h)
        for candidate_w in range(grid, max(grid, limit_width // grid * grid) + 1, grid)
        for candidate_h in range(grid, max(grid, limit_height // grid * grid) + 1, grid)
    ]

    def aspect_error(size: tuple[int, int]) -> float:
        return abs(size[0] / size[1] - aspect) / aspect

    close = [size for size in sizes if aspect_error(size) <= _ASPECT_TOLERANCE]
    if close:
        return max(close, key=lambda size: (size[0] * size[1], -aspect_error(size)))
    return min(sizes, key=lambda size: (aspect_error(size), -size[0] * size[1]))
