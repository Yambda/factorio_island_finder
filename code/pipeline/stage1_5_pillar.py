"""
Cheap REJECT-only pre-tier ahead of the stage-1.5 lattice cascade: a single
wide, dense "candy-bar" strip (a small grid of 64-tile segments, several
columns wide) extending due East from spawn out to RING_TILES. Every tile on
every segment is individually sampled (same as the lattice cascade's own
segments) -- no stride gap, so REJECT here is exactly as sound as the
lattice cascade's REJECT.

WHY THIS EXISTS
--------------------------------------------------------------------------
Stage 1 (the old ray-cast screen) is gone as a REJECT gate: it rejected on a
20-tile-radial-step, 48-angle discrete sample, which has a real, demonstrated
geometric false-negative channel (see find_islands.py's --stage1 help and the
project history). Every seed must now reach a certified stage. But feeding
100% of seeds straight into the full lattice cascade measured at only ~1,404
seeds/sec (2M-seed benchmark) -- a ~35-day full census. This module is a
CHEAP, ALSO-CERTIFIED first look: most ordinary mainland seeds have *some*
wide-enough clear corridor within a modest distance of spawn, and checking
one direction's dense strip is far cheaper per-seed than the lattice
cascade's multi-round graph walk.

DESIGN, AND WHY THE PARAMETERS ARE WHAT THEY ARE (measured, not assumed)
--------------------------------------------------------------------------
- ONE direction (East, +x) only. Measured on 100k random seeds: adding
  South/West/North on top of East resolves only a few more percent each,
  for a full extra pass of cost -- net LOSS on combined throughput (1,824
  seeds/s stage-alone but only ~1,203 seeds/s once the unresolved residual's
  fallback-cascade cost is included, vs ~1,772 seeds/s for East alone).
  Directionally-diverse escapes exist, but they're cheaper to find via the
  lattice cascade's own full-frontier search than via more fixed pillars.
- Width = 3 nodes each side of center (7 columns, 7*64=448 tiles wide).
  Measured sweep at RING_TILES=1400: width 2 nodes (5 cols) -> 45.3% yield,
  3 nodes (7 cols) -> 55.2%, 5 nodes (11 cols) -> 68.8%, 8 nodes (17 cols)
  -> 80.3%. Yield keeps climbing with width, but so does per-seed cost;
  width=3 measured as the actual combined-throughput optimum once the
  unresolved residual's fallback cost is priced in (wider each looked
  faster "alone" but wasted more work on seeds that don't resolve here).
- Fully eager (no lazy/staged sub-batching). At RING_TILES=1400 the strip is
  only ~22 segments long -- staging's bookkeeping overhead measured a wash
  or a net loss here (unlike a much longer bar, there isn't much "wasted
  tail" to trim).
- Starts 2 nodes BEHIND spawn (not AT spawn) purely so spawn isn't sitting
  at the strip's own edge; those two columns cost little and don't change
  any REJECT logic (a REJECT witness only ever needs FORWARD progress).

MEASURED RESULT (100k random seeds, RING_TILES=1400, this exact config)
--------------------------------------------------------------------------
  55.20% of seeds REJECT here alone, at 4,075 seeds/sec standalone.
  Combined with the existing lattice cascade handling the 44.8% residual:
  ~1,772 seeds/sec end-to-end, vs ~1,404 seeds/sec with no pre-tier --
  a measured ~26% throughput improvement. See
  scratchpad/test_staged_candybar.py for the sweep this was chosen from.

SOUNDNESS
--------------------------------------------------------------------------
Identical soundness argument to stage1_5_oracle.py's segment graph: a
REJECT only fires when every one of the actually-sampled tiles along a
connected chain of segments from spawn out to >= RING_TILES clears
EPSILON_FP. This module reuses oracle.classify_edges verbatim -- the exact
same GPU-vs-CPU margin argument applies unchanged. A seed this module does
NOT reject is simply passed through to the lattice cascade unchanged
(DEFER-like, not a verdict) -- this module only ever emits REJECT or "not
resolved here", never CONFIRM.
"""
import torch

import stage1_5_oracle as oracle

SEGMENT_TILES = 64
BACK = 2
RING_TILES = oracle.RING_TILES  # tracks the lattice cascade's own definition (2000.0 as of the 1400->2000 bump)
WIDTH_NODES = 3  # -> 7 columns, 448 tiles wide
L = int((RING_TILES + SEGMENT_TILES - 1) // SEGMENT_TILES)  # ceil(RING_TILES/64); 32 at RING_TILES=2000

# East only: primary axis dir_code 0 (+x), perpendicular dir_code 2 (+y).
PRIMARY_DIR_CODE = 0
PERP_DIR_CODE = 2

_STRUCT_CACHE = {}


def _get_strip_structure(device):
    cached = _STRUCT_CACHE.get(device)
    if cached is not None:
        return cached
    h_i, h_j = [], []
    for i in range(-BACK, L):
        for j in range(-WIDTH_NODES, WIDTH_NODES + 1):
            h_i.append(i); h_j.append(j)
    v_i, v_j = [], []
    for i in range(-BACK, L + 1):
        for j in range(-WIDTH_NODES, WIDTH_NODES):
            v_i.append(i); v_j.append(j)
    struct = (
        torch.tensor(h_i, dtype=torch.int64, device=device),
        torch.tensor(h_j, dtype=torch.int64, device=device),
        torch.tensor(v_i, dtype=torch.int64, device=device),
        torch.tensor(v_j, dtype=torch.int64, device=device),
    )
    _STRUCT_CACHE[device] = struct
    return struct


def pillar_reject(seeds, tables, device):
    """
    seeds:  torch.int64 [N] -- the full batch (must match `tables`' own seed order).
    tables: dict from noise_gpu.build_all_tables(seeds) (or oracle.build_tables) --
            shared with the caller, NOT rebuilt here (same fixed-cost-amortization
            rule as everywhere else in this pipeline -- see noise_gpu.build_all_tables's
            own docstring).

    Returns bool[N]: True where this pillar found a certified REJECT witness
    (a connected chain of EDGE_LAND segments from spawn reaching >= RING_TILES
    due East). False means "not resolved here" -- caller must still send that
    seed to the lattice cascade; it is NOT a verdict of any kind.
    """
    N = seeds.shape[0]
    h_i, h_j, v_i, v_j = _get_strip_structure(device)
    n_h, n_v = h_i.shape[0], v_i.shape[0]
    n_rows = 2 * WIDTH_NODES + 1

    seed_idx = torch.arange(N, device=device)

    seed_idx_h = seed_idx.repeat_interleave(n_h)
    base_i_h = h_i.unsqueeze(0).expand(N, -1).reshape(-1)
    base_j_h = h_j.unsqueeze(0).expand(N, -1).reshape(-1)
    dir_code_h = torch.full((N * n_h,), PRIMARY_DIR_CODE, dtype=torch.int64, device=device)

    seed_idx_v = seed_idx.repeat_interleave(n_v)
    base_i_v = v_i.unsqueeze(0).expand(N, -1).reshape(-1)
    base_j_v = v_j.unsqueeze(0).expand(N, -1).reshape(-1)
    dir_code_v = torch.full((N * n_v,), PERP_DIR_CODE, dtype=torch.int64, device=device)

    res_h = oracle.classify_edges(tables, seed_idx_h, base_i_h, base_j_h, dir_code_h,
                                   node_spacing=SEGMENT_TILES, segment_tiles=SEGMENT_TILES)
    res_v = oracle.classify_edges(tables, seed_idx_v, base_i_v, base_j_v, dir_code_v,
                                   node_spacing=SEGMENT_TILES, segment_tiles=SEGMENT_TILES)

    land_h = (res_h["classification"] == oracle.EDGE_LAND).view(N, BACK + L, n_rows)
    land_v = (res_v["classification"] == oracle.EDGE_LAND).view(N, BACK + L + 1, n_rows - 1)

    reached = torch.ones(N, n_rows, dtype=torch.bool, device=device)  # column -BACK assumed reached (spawn's own strip)
    farthest = torch.zeros(N, dtype=torch.int64, device=device)
    for i in range(BACK + L):
        advanced = reached & land_h[:, i, :]
        for _ in range(n_rows):
            up = torch.zeros_like(advanced); down = torch.zeros_like(advanced)
            up[:, 1:] = advanced[:, :-1] & land_v[:, i + 1, :]
            down[:, :-1] = advanced[:, 1:] & land_v[:, i + 1, :]
            new_advanced = advanced | up | down
            if torch.equal(new_advanced, advanced):
                advanced = new_advanced
                break
            advanced = new_advanced
        col = i - BACK + 1
        any_reached = advanced.any(dim=1)
        farthest = torch.where(any_reached & (col > farthest), torch.full_like(farthest, col), farthest)
        reached = advanced

    dist = farthest.to(torch.float32) * SEGMENT_TILES
    return dist >= RING_TILES
