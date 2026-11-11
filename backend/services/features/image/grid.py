"""Snap an image side down to the model grid.

The side limits stay on the request DTO. This is the rounding the create path applies.
"""

from api_types import IMAGE_SIDE_GRID


def floor_to_image_grid(side: int) -> int:
    return (side // IMAGE_SIDE_GRID) * IMAGE_SIDE_GRID
