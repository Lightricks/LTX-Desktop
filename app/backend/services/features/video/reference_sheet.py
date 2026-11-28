"""Ingredients reference sheet: the panels on one black canvas, none cropped.

Port of the ltx.io layout (``layoutCanvas`` and ``createNoTrimGrid``), so a sheet
looks the same on both apps. The canvas is the size the model denoises, so the
sheet is resized once.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from server_utils.oriented_image import open_oriented_image, oriented_image_size

# ltx.io gutter at a 1920 px long edge. The gutter keeps panels apart on the black canvas.
_GAP_PX_AT_1920 = 32
# Rows within this spread of aspect ratios get one height.
_EQUAL_ROW_SPREAD = 0.1

Panel = tuple[int, float, float, float, float]

_SHEET_PREFIX = "sheet-"
_SHEET_SUFFIX = ".png"
_MAX_CACHED_SHEETS = 8


def _sheet_key(paths: Sequence[str], width: int, height: int) -> str:
    digest = hashlib.sha256(f"{width}x{height}".encode())
    for raw in paths:
        path = Path(raw).resolve()
        stat = path.stat()
        digest.update(f"|{path}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()[:16]


def _evict_old_sheets(in_flight_dir: Path, *, keep: Path) -> None:
    if not in_flight_dir.is_dir():
        return
    sheets: list[tuple[int, Path]] = []
    for child in in_flight_dir.iterdir():
        name = child.name
        if child == keep or not (name.startswith(_SHEET_PREFIX) and name.endswith(_SHEET_SUFFIX)):
            continue
        try:
            if child.is_file():
                sheets.append((child.stat().st_mtime_ns, child))
        except OSError:
            continue
    sheets.sort(key=lambda item: item[0], reverse=True)
    for _, stale in sheets[_MAX_CACHED_SHEETS - 1 :]:
        stale.unlink(missing_ok=True)


def ensure_reference_sheet(
    paths: Sequence[str], width: int, height: int, *, in_flight_dir: Path
) -> Path:
    """Return the cached reference sheet in ``.in-flight``, building it if absent.

    Only ``sheet-*.png`` files are evicted; other files in ``.in-flight`` are untouched.
    """
    in_flight = in_flight_dir
    # No parents: a gone custom assets folder must not be created without its marker.
    in_flight.mkdir(exist_ok=True)
    dest = in_flight / f"{_SHEET_PREFIX}{_sheet_key(paths, width, height)}{_SHEET_SUFFIX}"
    if not dest.is_file():
        # A unique name, so two builders of the same sheet do not replace each other's file.
        tmp = in_flight / f".{dest.name}.{uuid.uuid4().hex}.tmp"
        try:
            build_reference_sheet(paths, width, height, tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    _evict_old_sheets(in_flight, keep=dest)
    return dest


def _rows(aspects: Sequence[float], width: int, row_count: int, gap: int) -> list[tuple[int, int, float]]:
    """Split the panels into rows of about equal total aspect: ``(start, end, height)``."""
    count = len(aspects)
    row_count = max(1, min(row_count, count))
    target = sum(aspects) / row_count
    rows: list[tuple[int, int, float]] = []
    start = 0
    for row in range(row_count):
        end = count
        if row < row_count - 1:
            running, best_diff = 0.0, math.inf
            for stop in range(start + 1, count - (row_count - row - 1) + 1):
                running += aspects[stop - 1]
                if abs(target - running) < best_diff:
                    best_diff, end = abs(target - running), stop
        usable = max(1, width - (end - start - 1) * gap)
        rows.append((start, end, usable / max(1e-6, sum(aspects[start:end]))))
        start = end
    return rows


def layout_panels(aspects: Sequence[float], width: int, height: int, gap: int) -> list[Panel]:
    """``(index, x, y, w, h)`` per panel, widest first, scaled to fill the canvas height."""
    order = sorted(range(len(aspects)), key=lambda index: -aspects[index])
    ordered = [aspects[index] for index in order]
    mean = sum(ordered) / len(ordered)
    spread = math.sqrt(sum((a - mean) ** 2 for a in ordered) / len(ordered)) / mean
    rows = _rows(ordered, width, math.floor(math.sqrt(height * sum(ordered) / width) + 0.5), gap)
    if len(rows) > 1 and spread <= _EQUAL_ROW_SPREAD:
        rows = [(start, end, min(row[2] for row in rows)) for start, end, _ in rows]
    panels: list[Panel] = []
    row_widths: list[float] = []
    y = 0.0
    for start, end, row_height in rows:
        x = 0.0
        for position in range(start, end):
            panel_width = ordered[position] * row_height
            panels.append((order[position], x, y, panel_width, row_height))
            x += panel_width + gap
        row_widths.append(x - gap)
        y += row_height + gap
    content_height = y - gap
    scale = height / content_height
    # Shrink to fit the canvas. Grow only while the widest row still fits.
    if scale > 1 and max(row_widths) * scale > width + 1e-6:
        scale = 1.0
    placed: list[Panel] = []
    cursor = 0
    for (start, end, _), row_width in zip(rows, row_widths):
        # Each row sits in the middle, so the black margin is the same on the left and right.
        offset = (width - row_width * scale) / 2
        for index, x, y, w, h in panels[cursor : cursor + end - start]:
            placed.append((index, offset + x * scale, y * scale, w * scale, h * scale))
        cursor += end - start
    return placed


def build_reference_sheet(paths: Sequence[str], width: int, height: int, dest: Path) -> None:
    """Write the sheet PNG. Each panel keeps its aspect ratio and sits over black."""
    # The layout needs only the aspect ratios. Decode one image at a time: ten large
    # photos decoded together would need gigabytes.
    sizes = [oriented_image_size(path) for path in paths]
    gap = max(2, round(_GAP_PX_AT_1920 * max(width, height) / 1920))
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    for index, x, y, w, h in layout_panels([w / h for w, h in sizes], width, height, gap):
        with open_oriented_image(paths[index]) as source:
            panel = source.convert("RGBA").resize(
                (max(1, math.floor(w)), max(1, math.floor(h))), Image.Resampling.LANCZOS
            )
        sheet.paste(panel, (round(x), round(y)), panel)
    sheet.save(dest, format="PNG")
