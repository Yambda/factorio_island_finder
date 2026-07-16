"""
Stage 2: GPU-parallel flood-fill refinement.

STATUS (2026): this stage predates the stage-1.5 RING_TILES=5000 net-walk
cascade and the 1,400 -> 2,000-tile radius bump. In the current pipeline its
only input is stage-1.5's DEFER verdicts (stage 1's own ray screen is
disabled by default now -- see find_islands.py's --stage1 help), and the
net-walk cascade's full-frontier retention means DEFER essentially never
fires: it has processed exactly zero seeds across this census's entire
production run so far (confirmed via the "defer 0" / "stage2 0.00s" lines in
every logged round). Treat it as a dormant backstop, not a proven-live path:
if it ever DOES receive a seed, its own grid domain (+-1400 tiles, see
build_land_mask's grid_radius_cells default) is narrower than islands this
census has already confirmed for real (farthest tile 1999.6 on the current
record, seed 7598754) -- a deferred seed near that size would hit this
grid's outer border and be dropped as unconfirmed rather than correctly
resolved. Enlarging grid_radius_cells to match the current ring definition
is a real memory/compute tradeoff (cost is O(radius^2)), not done here --
flag for a design decision if DEFER ever actually fires in production.

Originally: run ONLY on the (small) shortlist of seeds stage 1 flagged as
"maybe an island".

DESIGN NOTE -- why this is iterated single-step dilation, NOT a Jump Flood
Algorithm (JFA), and why that choice is load-bearing, not stylistic:
--------------------------------------------------------------------------
Classic JFA propagates a nearest-seed label via exponentially decreasing
step offsets, and its O(log N) step count relies on Euclidean distance being
a metric that supports safely skipping ahead. Real 4-connected flood-fill
connectivity on an irregular coastline does NOT support that skip-ahead
trick in general: a jump of size 2^k across a `land` cell can leap clean
over a narrow water strait and wrongly mark a disconnected region as
"reached". This was independently confirmed as a DISQUALIFYING bug during
this project's own JFA prototype: it produced a false negative on the one
confirmed real island (seed 20271579), incorrectly bridging it to the
outside through a water gap, and -- because dilation is monotonic (it can
only ever add cells, never remove a wrongly-added one) -- no amount of
step=1 "refinement" after a JFA phase can undo that error once it happens.
So: JFA/large-step doubling is DISQUALIFIED for this stage, permanently, not
merely "an option to consider" -- do not reintroduce it as an optimization
without redoing that correctness analysis from scratch.

Instead, this module implements the flood fill as iterated single-step BFS
"dilation" (grow the visited/reachable set by one 4-connected step per
iteration, intersected with the land mask, i.e. water blocks propagation) --
exactly what the CPU reference does (seed_finders/largest_island/stages.cpp,
queue-based BFS), just executed as dense, batched, parallel tensor ops
across the whole shortlist at once instead of a per-seed queue. This is
EXACT (given the underlying land/water mask), not approximate -- the only
approximation left at this stage is the land/water mask itself, which
inherits noise_gpu's approximated elevation (see noise_gpu.py's docstring).

Each iteration has an O(1) sequential dependency on the previous one (BFS
reachability grows by exactly one cell-step per iteration), so this stage
trades away stage 1's embarrassing parallelism across iterations -- the
parallelism instead comes from batching every shortlisted seed's grid
together into one 3D tensor and executing each iteration as a single,
fully vectorized elementwise op over the whole batch.

STAGE2-CHUNK-SIZE CHOICE -- 32, not the prior default of 64 (tentative, lower
confidence than the other tuning changes in this file): build_land_mask()'s
`chunk_size` only bounds a TRANSIENT per-chunk elevation buffer -- the
land_mask output tensor itself is allocated once, up front, for the WHOLE
shortlist, so this knob purely trades transient memory for nothing; output
is bit-identical at chunk_size in {1, 64, 256} (verified directly). A
parameter-tuning sweep (capped at 100,000 seeds/data-point, ~1,071-seed
shortlist actually tested) ran under confirmed heavy GPU contention from a
sibling process, so its absolute throughput numbers are unreliable, but it
did confirm a hard ceiling at chunk_size=512: 43.3GB allocated on this 32GB
card -- silent host-RAM oversubscription (a more dangerous failure mode than
a clean OOM, since it "succeeds" while collapsing throughput). The CLI's
former help text claim ("raise this if you have GPU memory to spare") is
unsupported by any measurement here -- memory scales with shortlist size
(O(shortlist), not O(chunk_size)), so raising chunk_size buys nothing even
when memory is available. 32 was picked as a conservative lowering pending a
clean re-run of this one sweep on an idle GPU. NOTE: at real 2,000,000-seed-
round shortlist sizes (~22,000, vs. ~1,071 tested here) the fixed land_mask
tensor alone would need roughly 10.8GB -- re-check headroom if --batch-size
or --water-threshold are ever pushed up enough to grow the shortlist
materially.

GRID / CAP DESIGN
------------------
- Grid step = 4 tiles, matching the CPU reference's FLOOD_FILL_STEP exactly,
  so land_cell_count * 16 tiles^2 is directly comparable to the CPU score.
- Grid radius GRID_RADIUS_CELLS (default 350 => 701x701 grid => covers
  +-1400 tiles from spawn) -- was "comfortably larger than the confirmed
  real island's footprint" when this was written, but is STALE by that
  measure now: the current record (seed 7598754) reaches 1999.6 tiles,
  outside this domain -- see the module docstring's STATUS note. Do not
  shrink this for a speed win without
  re-validating against BOTH ground-truth seeds: geodesic (land-only-path)
  distance from spawn to a point can run roughly 2x the naive straight-line
  distance on a winding coastline, a lesson learned the hard way in an
  earlier prototype (see below).
- STOPPING RULE (read this before touching it): an earlier version of this
  function capped the number of BFS iterations at roughly the straight-line
  distance from spawn to the grid border, on the theory that "if it hasn't
  touched the border by then, it's enclosed". That is WRONG for convoluted
  mainland coastlines: measured empirically as 15/15 false positives in one
  test run (all 15 stage-2-"confirmed" candidates were real, CPU-verified
  mainlands whose flood front just hadn't reached the border yet along its
  actual winding path, even though it never stops growing at any grid size).
  The FIX -- and the only correct rule -- is to mirror the CPU reference's
  own stopping criterion (stages.hpp/.cpp) literally: track total touched
  cells (land cells that propagate the flood, AND the water cells
  immediately adjacent to them, exactly like the CPU's BFS `visited` set,
  which inserts neighbor coordinates regardless of land/water and only
  skips *expanding through* water) and reject/freeze a seed once that count
  exceeds CPU_FLOOD_FILL_MAX_CELLS -- the same cell-cap constant the CPU
  reference uses (see that constant's own comment for its current value and
  history), not a spatial/geodesic proxy for it. Touching the grid's outer
  border is kept only as a secondary safety net (grid too small for this
  seed), and should essentially never fire if grid_radius_cells is generous
  -- if it starts firing often, the grid is too small, not the cap logic.

FUTURE OPTIMIZATION PRIORITY NOTE (2026, STALE -- see module docstring's
STATUS note): the "~75-78% of wall-clock" claim below describes the OLD
pre-cascade pipeline, where stage 2 ran on every stage-1 shortlist seed.
In the current RING_TILES=5000 cascade pipeline this stage processes ~0
seeds and ~0% of wall-clock (see STATUS note above) -- it is not where any
current optimization effort should go. Left below for archaeological
context only; DO NOT use the 75-78% figure to justify new work here.

Original note: a round of five per-point/per-ray
noise-evaluation cost-reduction ideas was investigated against stage 1 and
noise_gpu.py (lattice-cell dedup, per-ray early termination, macro-predicts-
sign screening, octave truncation, mip-pyramid/interval bounds -- see
stage1_screen.py's and noise_gpu.py's module docstrings for the measured
numbers and rejection reasons). None were merged: one was correctness-clean
but capped at a low single-digit-percent end-to-end ceiling even in the best
case (because stage 1 is the minority contributor to wall-clock -- THIS
stage's flood fill dominates at ~75-78%), and the rest were either net
wall-clock losses or structurally incapable of exploiting any real signal.
The takeaway: further investment in *this* stage's own per-seed cost (not
stage 1's) is where the next real end-to-end win has to come from.
"""

import torch
import torch.nn.functional as F

import noise_gpu as ng

FLOOD_FILL_STEP = 4  # must match seed_finders/largest_island/stages.hpp FLOOD_FILL_STEP exactly
CPU_FLOOD_FILL_MAX_CELLS = 2_000_000  # must match seed_finders/largest_island/stages.hpp exactly
                                       # (raised from 200,000 alongside stages.hpp / cpu_verify.py
                                       # after the confirmed record island (seed 7598754,
                                       # 2,819,120 tiles^2) left only 3.2% headroom under the
                                       # old cap -- see CLAUDE.md's cpu_verify.py bug history.
                                       # This module's own grid domain (see build_land_mask's
                                       # default grid_radius_cells=350 => +-1400 tiles) is
                                       # UNRELATED and still narrower than that same record's
                                       # farthest tile (1999.6) -- raising this constant does not
                                       # by itself fix that; see build_land_mask's docstring.


def build_land_mask(seeds, grid_radius_cells=350, chunk_size=32, verbose=False):
    """
    seeds: torch.int64 [S] (the shortlist).
    Returns: land_mask torch.bool [S, H, W], H=W=2*grid_radius_cells+1,
             grid[:, grid_radius_cells, grid_radius_cells] is the spawn tile (0,0).
    """
    device = seeds.device
    s = seeds.shape[0]
    side = 2 * grid_radius_cells + 1

    idx = torch.arange(-grid_radius_cells, grid_radius_cells + 1, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(idx, idx, indexing="ij")  # [H, W]
    pos_x_flat = (gx * FLOOD_FILL_STEP).reshape(-1)  # [H*W]
    pos_y_flat = (gy * FLOOD_FILL_STEP).reshape(-1)

    land_mask = torch.zeros(s, side, side, dtype=torch.bool, device=device)

    # See stage1_screen.py's identical comment: build tables ONCE for the
    # whole shortlist, then only chunk the memory-heavy per-point elevation
    # evaluation by slicing the already-built tables.
    all_tables = ng.build_all_tables(seeds)

    for start in range(0, s, chunk_size):
        end = min(start + chunk_size, s)
        cn = end - start

        tables = ng.slice_tables(all_tables, start, end)
        px = pos_x_flat.unsqueeze(0).expand(cn, -1)
        py = pos_y_flat.unsqueeze(0).expand(cn, -1)

        with torch.no_grad():
            elevation = ng.elevation_nauvis(tables, px, py)

        land_mask[start:end] = (elevation > 0.0).view(cn, side, side)
        if verbose:
            print(f"  stage2 land-mask chunk {start}:{end} done", flush=True)

    return land_mask


def _dilate_step(visited):
    """One 4-connected dilation step with zero-padded borders. visited: bool [S,H,W]."""
    padded = F.pad(visited, (1, 1, 1, 1), mode="constant", value=False)
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    return visited | up | down | left | right


def flood_fill(land_mask, max_steps=1500, cell_cap=CPU_FLOOD_FILL_MAX_CELLS, sync_every=8, verbose=False):
    """
    land_mask: torch.bool [S, H, W] (H=W odd, center is spawn).

    See module docstring for why the stopping rule mirrors the CPU's literal
    cell cap rather than any spatial/geodesic heuristic.

    max_steps=1500 is SWEPT and kept unchanged as the default (see
    find_islands.py's --max-flood-steps help text for the numbers): the
    early-exit above fires at each batch's actual convergence point
    regardless of this ceiling -- steps_run was measured pinned at exactly
    544 for every ceiling tried >=600, and observed convergence across
    ~1,613 sampled seeds (<=100,000-seed sweep batches, plus the 4 ground-
    truth seeds) ranged 238-589 steps. Raising or lowering max_steps above
    the true convergence point costs nothing at any scale; 1500 is a safety
    margin against a pathological non-converging seed, not a speed knob.

    Returns dict:
      area_tiles2:      torch.float32 [S] -- reachable LAND cell count * 16
                                              (matches CPU's land_cells*16)
      hit_boundary:     torch.bool [S]    -- rejected: either the CPU-cap
                                              analog (visited cells >
                                              cell_cap) or (rare/safety net)
                                              flood reached the grid's outer
                                              border before converging
      spawn_is_land:    torch.bool [S]
      confirmed_island: torch.bool [S]    -- spawn_is_land & not hit_boundary
      steps_run:        int               -- BFS iterations actually executed
    """
    s, h, w = land_mask.shape
    device = land_mask.device
    cy, cx = h // 2, w // 2

    border = torch.zeros(h, w, dtype=torch.bool, device=device)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    spawn_is_land = land_mask[:, cy, cx].clone()

    visited_land = torch.zeros_like(land_mask)  # propagatable (land) reached cells
    visited_land[:, cy, cx] = spawn_is_land
    all_touched = visited_land.clone()  # land cells reached, PLUS water cells discovered as neighbors
    touched_count = all_touched.view(s, -1).sum(dim=1)  # int64 [S]

    # Per-seed early freeze once a seed is DECIDED (cap exceeded, or -- as a
    # safety net -- grid border touched): stop updating its mask and exclude
    # it from the batch-wide "still changing" convergence check, so a few
    # slow-to-resolve mainlands in a large shortlist don't force the whole
    # batched loop to keep re-simulating already-decided seeds out to
    # max_steps.
    frozen = touched_count > cell_cap

    # Exit-check cadence: torch.equal()/.item()-style host<->device syncs
    # force the whole CUDA queue to drain, which is comparatively expensive
    # if repeated every single step over hundreds of steps (especially under
    # shared-GPU contention). The per-step mask update itself needs no such
    # sync, so we only pay the sync cost every `sync_every` steps -- worst
    # case this delays noticing "done" by that many extra (cheap,
    # already-converged/frozen) steps, it cannot change the final result.
    #
    # HARDCODED AT 8, NOT A CLI FLAG -- SWEPT and kept unchanged: a
    # parameter-tuning sweep (<=100,000 seeds/data-point) found sync_every
    # in {2, 4, 8, 16, 32} statistically indistinguishable (6.06-6.67s,
    # 81.6-89.7 seeds/s on that sweep's shortlist) -- already on the flat
    # part of the curve, so 8 is not a guess. Only sync_every=1 was a clear
    # regression (~4.7x slower: ~28.6s vs ~6.1s), which is why this stays
    # above 1. No correctness or memory effect was observed at any value
    # tested.
    steps_run = 0
    for step in range(max_steps):
        # candidate neighbor cells (land OR water) of the current land frontier
        candidates = _dilate_step(visited_land) & (~visited_land)
        newly_touched = candidates & (~all_touched)
        new_all_touched = all_touched | newly_touched
        new_visited_land = new_all_touched & land_mask

        new_touched_count = new_all_touched.view(s, -1).sum(dim=1)
        touched_border_now = (new_all_touched & border.unsqueeze(0)).any(dim=(1, 2))
        newly_frozen = (~frozen) & ((new_touched_count > cell_cap) | touched_border_now)
        frozen = frozen | newly_frozen

        freeze_mask = frozen.view(s, 1, 1)
        new_visited_land = torch.where(freeze_mask, visited_land, new_visited_land)
        new_all_touched = torch.where(freeze_mask, all_touched, new_all_touched)

        steps_run = step + 1
        check_now = (step % sync_every == (sync_every - 1)) or (step == max_steps - 1)
        no_change = torch.equal(new_visited_land, visited_land) if check_now else False
        visited_land = new_visited_land
        all_touched = new_all_touched
        if check_now:
            if verbose:
                print(f"  stage2 BFS step {step}, live land cells = {visited_land.sum().item()}, "
                      f"frozen(rejected) = {int(frozen.sum().item())}/{s}", flush=True)
            if bool(frozen.all()) or no_change:
                break

    visited = visited_land
    hit_boundary = frozen.clone()

    area_tiles2 = visited.sum(dim=(1, 2)).float() * (FLOOD_FILL_STEP * FLOOD_FILL_STEP)
    confirmed_island = spawn_is_land & (~hit_boundary)

    return {
        "area_tiles2": area_tiles2,
        "hit_boundary": hit_boundary,
        "spawn_is_land": spawn_is_land,
        "confirmed_island": confirmed_island,
        "steps_run": steps_run,
    }
