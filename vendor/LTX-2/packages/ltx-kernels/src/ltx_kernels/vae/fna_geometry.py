"""Coordinate arithmetic shared by the two DiffVAE CuTe DSL kernels.
Pure geometry -- flat index to 3D position and back, window clamping, and the maps
from a query tile to the rows its operands come from. No MMA, no TMEM, no SMEM; those
live in :mod:`ltx_kernels.vae.fna_attn_core`. Both kernels derive the *same* per-tile
geometry from different sources, so it lives here once and neither can drift from the
other.
Both kernels are built out of nested 3D boxes, and telling them apart is most of what
reading either one takes:
``volume``
    The whole ``(T, H, W)`` token grid.
``query tile``
    The ``(TT, TH, TW)`` box of exactly ``M`` output tokens one CTA owns. Lanes tile
    it row-major, one query per lane.
``halo panel``
    A query tile grown by the NA radius to ``(PT, PH, PW)``: every key any query in
    the tile can see. Clamped inward at the volume edge, never truncated.
``NA window``
    One individual query's ``(kt, kh, kw)`` box, clamped the same way. It sits inside
    the tile's halo panel, and the offset between the two is what the per-key mask
    tests.
The fused block kernel adds two more boxes between the tile and the volume, because
it fills a K/V cache once for a run of tiles rather than once per tile:
``slab core``
    The run of query tiles sharing one cache fill -- the output region a slab owns.
``slab box``
    That core grown by the NA radius: what the cache is filled over. Every halo panel
    of every tile in the core lies inside it, which is what makes the cache reusable.
A note on the ``@cute.jit`` decorators below. The DSL lowers ``if`` on a *runtime*
value by rewriting the function's AST, which only happens for a function it owns, so
anything branching on a device value has to be decorated. Helpers that touch only
compile-time values are left undecorated and trace straight inline.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import cutlass
import cutlass.cute as cute

# --------------------------------------------------------------------------
# Flat index <-> 3D position
# --------------------------------------------------------------------------


def unflatten_3d(index, plane_stride, row_stride):
    """Row-major flat ``index`` -> ``(i0, i1, i2)`` within a ``(_, d1, d2)`` box.
    ``plane_stride`` is ``d1 * d2`` and ``row_stride`` is ``d2``. Both are passed in
    rather than derived from extents because every call site already has the product
    hoisted out of the loop it sits in.
    Each remainder is spelled ``x - q * d`` rather than ``x % d`` so that it reuses
    the quotient the division just above it produced.
    """
    i0 = index // plane_stride
    rest = index - i0 * plane_stride
    i1 = rest // row_stride
    i2 = rest - i1 * row_stride
    return i0, i1, i2


def flatten_3d(i0, i1, i2, extent_1, extent_2):
    """``(i0, i1, i2)`` -> its row-major flat index in a ``(_, extent_1, extent_2)`` box."""
    return (i0 * extent_1 + i1) * extent_2 + i2


# --------------------------------------------------------------------------
# Clamping
# Spelled as predicated assignment rather than ``cutlass.min`` / ``cutlass.max``:
# the DSL lowers this form to a single select, and it is the form that composes with
# the mixed Int32 / compile-time operands these call sites hand it.
# --------------------------------------------------------------------------


@cute.jit
def at_most(value, limit):
    """``min(value, limit)``."""
    if value > limit:
        value = limit
    return value


@cute.jit
def at_least(value, limit):
    """``max(value, limit)``."""
    if value < limit:
        value = limit
    return value


@cute.jit
def clamp_window_start(start, window_extent, volume_extent):
    """Slide a window of ``window_extent`` so it lies inside ``[0, volume_extent)``.
    This is NATTEN's inward shift: a window hanging off the low edge is pushed to 0
    and one hanging off the high edge is pulled back to
    ``volume_extent - window_extent``, rather than truncated or masked. It is the
    single rule behind both the halo panel and the per-query NA window.
    The low clamp runs first. The two disagree only if the volume is shorter than the
    window, which ``fna_supported`` / ``na_supported`` rule out up front.
    """
    if start < 0:
        start = cutlass.Int32(0)
    if start > volume_extent - window_extent:
        start = volume_extent - window_extent
    return start


# --------------------------------------------------------------------------
# Query tile geometry
# --------------------------------------------------------------------------


def query_slot_in_tile(query_idx, tile_h, tile_w):
    """Position ``(dt, dh, dw)`` of query ``query_idx`` within its query tile.
    A tile holds exactly ``M`` queries and its threads walk them row-major, one each,
    so a thread's accumulator row *is* its query index -- which is what keeps the
    softmax, the norms and RoPE register-local.
    """
    return unflatten_3d(query_idx, tile_h * tile_w, tile_w)


@cute.jit
def halo_panel_origin(tile_origin, radius, panel_shape, volume):
    """Origin of the halo panel a whole query tile reads, clamped into the volume.
    Args:
        tile_origin: ``(t0, h0, w0)`` of the query tile within the volume.
        radius: ``(rt, rh, rw)``, the NA kernel's half-extents.
        panel_shape: ``(PT, PH, PW)``, compile-time.
        volume: ``(T, H, W)``.
    """
    t0, h0, w0 = tile_origin
    radius_t, radius_h, radius_w = radius
    panel_t, panel_h, panel_w = panel_shape
    vol_t, vol_h, vol_w = volume
    return (
        clamp_window_start(t0 - radius_t, panel_t, vol_t),
        clamp_window_start(h0 - radius_h, panel_h, vol_h),
        clamp_window_start(w0 - radius_w, panel_w, vol_w),
    )


@cute.jit
def query_row_indices(query_pos, volume, lane, scratch_row_base):
    """The global rows a lane reads its query from and writes its result to.
    Returns ``(clamped_pos, read_row, write_row)``.
    A partial tile at the volume edge has lanes whose query falls outside the volume.
    They still run every MMA -- the tile is always a full ``M`` rows, and the extra
    rows are simply garbage -- so each needs a row that is safe to *read* (any
    in-bounds one; the value is discarded) and a row that is safe to *write*. The
    write goes to a scratch row past the end of the output, which is why callers
    allocate ``M`` extra rows there.
    ``clamped_pos`` is the in-volume position the read row came from; downstream
    geometry (the NA window, the RoPE index, the cached-norm lookup) uses it so that
    an out-of-range lane still computes something well-defined.
    """
    q_t, q_h, q_w = query_pos
    vol_t, vol_h, vol_w = volume
    in_volume = q_t < vol_t and q_h < vol_h and q_w < vol_w
    r_t = q_t if q_t < vol_t else vol_t - 1
    r_h = q_h if q_h < vol_h else vol_h - 1
    r_w = q_w if q_w < vol_w else vol_w - 1
    read_row = flatten_3d(r_t, r_h, r_w, vol_h, vol_w)
    write_row = read_row
    if not in_volume:
        write_row = scratch_row_base + lane
    return (r_t, r_h, r_w), read_row, write_row


@cute.jit
def window_origin_in_panel(query_pos, panel_origin, radius, kernel_shape, volume):
    """One query's clamped NA window, as an offset inside its tile's halo panel.
    The panel bounds every query in the tile; this is the ``(kt, kh, kw)`` box *this*
    query attends to. Returned panel-relative because that is the form the per-key
    mask consumes.
    NATTEN's inward shift. See :func:`window_bounds_in_panel` for the other rule and why
    both exist.
    """
    q_t, q_h, q_w = query_pos
    panel_t0, panel_h0, panel_w0 = panel_origin
    radius_t, radius_h, radius_w = radius
    kernel_t, kernel_h, kernel_w = kernel_shape
    vol_t, vol_h, vol_w = volume
    return (
        clamp_window_start(q_t - radius_t, kernel_t, vol_t) - panel_t0,
        clamp_window_start(q_h - radius_h, kernel_h, vol_h) - panel_h0,
        clamp_window_start(q_w - radius_w, kernel_w, vol_w) - panel_w0,
    )


@cute.jit
def window_bounds_in_panel(query_pos, panel_origin, radius, kernel_shape, volume):
    """One query's **clamp-and-mask** NA window, as half-open panel-relative bounds.
    The other window rule, and the one the joint keyframe operator is defined on: iterate
    the full configured kernel from ``q - radius`` and drop whatever falls outside the
    volume, so a boundary query attends to *fewer* keys and its softmax renormalises over
    them. NATTEN -- and therefore :func:`window_origin_in_panel` and every non-keyframe path
    here -- instead slides the window inward so every query sees exactly ``k`` keys.
    The two agree in the interior and genuinely differ at boundaries; at latent resolution
    with a ``(3, 7, 7)`` kernel that is about half the volume, so it is not a rounding
    detail. Which rule a call uses is a property of the operator, not of the hardware, hence
    two helpers rather than a flag buried here.
    Returns ``((lo_t, hi_t), (lo_h, hi_h), (lo_w, hi_w))``. Every bound is a subset of the
    inward-shifted window's span, so a panel sized for that rule already covers this one.
    """
    bounds = []
    for q, panel_0, radius_a, kernel_a, extent in zip(
        query_pos, panel_origin, radius, kernel_shape, volume, strict=True
    ):
        start = q - radius_a
        lo = start if start > 0 else cutlass.Int32(0)
        end = start + kernel_a
        hi = end if end < extent else extent
        bounds.append((lo - panel_0, hi - panel_0))
    return tuple(bounds)
