"""
Computes the farthest-tile-from-spawn distance for a confirmed real island,
via a real GPU-accelerated flood fill (same terrain evaluator as everything
else in this pipeline) -- used to track how close confirmed islands are
running to the RING_TILES definitional boundary over the life of a census
run (see find_islands.py's CSV `farthest_tile_dist` column).

Only ever called on CPU-verified real islands (rare -- see cpu_verify.py),
so this is not a hot-path cost; matches leader_visualizer.py's own
per-new-record cost profile.
"""
import numpy as np
import torch

import noise_gpu as ng

HALF_EXTENT = 10_000  # generous vs. the ~7,100-tile cascade domain and the
                       # 32,000,000-tile^2 CPU-verify area cap (see cpu_verify.py)
STEP = 4  # RAISED from 8 (2026-07-09): at step=8, a real narrow water strait
          # that genuinely encloses an island (confirmed by cpu_verify.py's
          # own exact, step=4 ground-truth flood fill) can fall entirely
          # between sample points, so this function's coarser flood fill
          # "leaks" through it into the surrounding mainland -- 6 confirmed
          # real islands hit this (all reported hit_border=True at exactly
          # ~half_extent*sqrt(2), i.e. this function's own search-window
          # corner, not a real measurement). step=4 matches the CPU
          # reference's resolution exactly, closing the gap; still
          # negligible cost per this module's own docstring (rare call, only
          # ever on CPU-verified real islands).


def farthest_tile_distance(seed, device, half_extent=HALF_EXTENT, step=STEP):
    """
    Returns (farthest_dist, hit_border): farthest_dist is the max Euclidean
    tile-distance from spawn among the seed's connected land component
    (flood-filled at `step`-tile resolution, matching the real game's
    4-connectivity). hit_border=True means the component reached the edge
    of this function's own render/search window -- farthest_dist is then a
    LOWER BOUND, not exact (widen half_extent and re-run if this matters).
    """
    seeds_t = torch.tensor([seed], dtype=torch.int64, device=device)
    tables = ng.build_all_tables(seeds_t)

    xs = torch.arange(-half_extent, half_extent, step, dtype=torch.float32, device=device)
    ys = torch.arange(-half_extent, half_extent, step, dtype=torch.float32, device=device)
    dim = xs.shape[0]
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    px = gx.reshape(1, -1)
    py = gy.reshape(1, -1)

    CHUNK = 800_000
    elev_chunks = []
    with torch.no_grad():
        for start in range(0, px.shape[1], CHUNK):
            end = min(start + CHUNK, px.shape[1])
            elev_chunks.append(ng.elevation_nauvis(tables, px[:, start:end], py[:, start:end]))
    elevation = torch.cat(elev_chunks, dim=1).reshape(dim, dim)
    is_water = (elevation <= 0.0).cpu().numpy()

    sr = sc = dim // 2
    visited = np.zeros((dim, dim), dtype=bool)
    stack = [(sr, sc)]
    visited[sr, sc] = True
    hit_border = False
    max_dist = 0.0
    while stack:
        r, c = stack.pop()
        if is_water[r, c]:
            continue
        wx = (c - sc) * step
        wy = (r - sr) * step
        d = (wx * wx + wy * wy) ** 0.5
        if d > max_dist:
            max_dist = d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= dim or nc < 0 or nc >= dim:
                hit_border = True
                continue
            if not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))

    return max_dist, hit_border
