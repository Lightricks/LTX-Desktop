"""Which keyframe planes a video query -- and a whole query tile -- attends to.
Host-side geometry for the kernels' keyframe tail. Two maps, and the split between
them is the whole reason this module exists:
``nearest_keyframe_slots``
    Per video timestep ``t``, the ``S`` nearest keyframe planes. This is the model's
    contract: a query at ``t`` folds those planes' ``K_h x K_w`` spatial windows into
    the same softmax as its local video neighbors. Membership is by temporal distance
    only -- a plane may sit far outside the video kernel's ``K_t``.
``tile_plane_union``
    Per *query tile*, the union of those sets over the ``TT`` timesteps the tile spans.
    The kernels gather K/V one plane at a time for a whole tile, but the nearest-``S``
    set is per ``t``, and a tile holds more than one. So the tail iterates the union and
    each lane masks itself out of the planes its own ``t`` did not pick
    (``attention_tile``'s ``sel``).
The union is small because anchors are spaced: the set changes only where the ranking
of the ``S``-th nearest swaps, and those crossings are about as far apart as the
anchors themselves. At stage 5 (one latent step per pixel frame, gap 17) a 4-wide tile
crosses at most one, so three planes cover it. Coarse stages pack anchors into fewer
stage steps and churn faster, which is what ``max_planes`` is sized against -- and why
picking a tile with ``TT == 1`` there can be cheaper than a wider one (see
``default_tile_thw``).
"""

from __future__ import annotations

import torch

#: Keyframe slots per video query. Fixed at 2 by the model contract: capacity is a
#: property of the attention operator, not of the kernel window or the anchor spacing.
KEYFRAME_CONTEXT_SLOTS = 2


def nearest_keyframe_slots(
    keyframe_times: torch.Tensor,
    video_length: int,
    *,
    slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    """The ``slots`` nearest keyframe planes for each of ``video_length`` timesteps.
    Args:
        keyframe_times: ``(n_kf,)`` float stage-local plane positions -- the chunk-center
            coordinates, not pixel frame indices.
        video_length: ``T`` on the stage grid.
        slots: capacity per query.
    Returns:
        ``(video_length, slots)`` int32 plane ids, ``-1`` where a slot is unused (which
        happens only when ``n_kf < slots``). Ties break toward the lower plane id.
    """
    if video_length < 1:
        raise ValueError(f"video_length must be positive, got {video_length}")
    n_kf = int(keyframe_times.numel())
    if n_kf == 0:
        return torch.full((video_length, slots), -1, dtype=torch.int32)
    times = keyframe_times.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    query = torch.arange(video_length, dtype=torch.float32)
    return _rank_nearest(query, times, slots)


def nearest_video_frames(
    keyframe_times: torch.Tensor,
    video_length: int,
    *,
    slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    """The ``slots`` video timesteps nearest each keyframe plane -- the other direction.
    A keyframe *query* attends to its own plane's spatial window plus these frames' spatial
    windows, in one softmax. Deliberately not the video kernel's full ``K_t`` reach: the
    point is local motion context around the anchor, so the count is fixed at ``slots``
    however wide ``K_t`` happens to be.
    Returns ``(n_kf, slots)`` int32 timesteps, ``-1`` where the video is shorter than
    ``slots``. Ties break toward the earlier frame.
    """
    if video_length < 1:
        raise ValueError(f"video_length must be positive, got {video_length}")
    n_kf = int(keyframe_times.numel())
    if n_kf == 0:
        return torch.zeros((0, slots), dtype=torch.int32)
    times = keyframe_times.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    video = torch.arange(video_length, dtype=torch.float32)
    return _rank_nearest(times, video, slots)


def tile_plane_union(keyframe_slots: torch.Tensor, tile_t: int, *, max_planes: int) -> tuple[torch.Tensor, int]:
    """Per query-tile row, the distinct planes any of its timesteps selected.
    Args:
        keyframe_slots: ``(T, S)`` from :func:`nearest_keyframe_slots`.
        tile_t: the query tile's temporal extent, ``TT``.
        max_planes: compile-time cap on the union; raises if a tile needs more, since
            silently dropping a plane would change the softmax rather than slow it down.
    Returns:
        ``(planes, used)`` -- ``planes`` is ``(ceil(T / tile_t), max_planes)`` int32,
        padded with ``-1``, and ``used`` is the largest union any tile actually needs, so
        a caller can loop that far instead of to ``max_planes``.
    """
    if tile_t < 1:
        raise ValueError(f"tile_t must be positive, got {tile_t}")
    video_length, _ = keyframe_slots.shape
    n_tiles = -(-video_length // tile_t)
    planes = torch.full((n_tiles, max_planes), -1, dtype=torch.int32)
    used = 0
    for tile in range(n_tiles):
        rows = keyframe_slots[tile * tile_t : (tile + 1) * tile_t]
        distinct = sorted({int(p) for p in rows.reshape(-1).tolist() if p >= 0})
        if len(distinct) > max_planes:
            raise ValueError(
                f"query tile {tile} spans timesteps selecting {len(distinct)} distinct keyframe "
                f"planes, over the max_planes={max_planes} cap; raise the cap or use a tile with "
                f"a smaller temporal extent"
            )
        used = max(used, len(distinct))
        if distinct:
            planes[tile, : len(distinct)] = torch.tensor(distinct, dtype=torch.int32)
    return planes, used


def _rank_nearest(query_times: torch.Tensor, candidate_times: torch.Tensor, slots: int) -> torch.Tensor:
    """``(Q, slots)`` candidate indices ranked by ``(|dt|, index)``, ``-1`` where unfilled.
    A **stable argsort** on ``|dt|`` gives the index tie-break exactly. The obvious
    alternative -- ``|dt| + arange * 1e-6`` before a topk, which is how upstream writes it --
    is only approximately that: at ``|dt|`` of a few hundred the perturbation is below
    float32 eps and the tie-break silently stops ordering anything. Measured on random
    layouts that disagreed with a stable sort on 91 of 400 trials, so the two are not
    interchangeable even though they agree on the evenly spaced production layouts.
    """
    distances = (query_times[:, None] - candidate_times[None, :]).abs()
    order = torch.argsort(distances, dim=-1, stable=True)
    take = min(slots, candidate_times.numel())
    chosen = order[:, :take].to(torch.int32)
    if take == slots:
        return chosen
    pad = torch.full((chosen.shape[0], slots - take), -1, dtype=torch.int32)
    return torch.cat((chosen, pad), dim=-1)
