"""
Stage 1: cheap ray-cast closure screen, run on the FULL seed batch, to
produce a high-recall shortlist of "maybe an island" seeds for stage 2's
more expensive GPU flood fill.

METHOD
------
For each seed, cast N_ANGLES rays outward from spawn (0,0). Along each ray,
sample elevation at increasing radius (coarse steps) up to MAX_RADIUS. Record
the first radius at which the tile becomes water (elevation <= 0). A seed
looks like an island if:
  (a) spawn itself is land (elevation(0,0) > 0), AND
  (b) most/all rays hit water before MAX_RADIUS ("the land is surrounded").

This can never be a proof of enclosure -- a landmass can be non-star-shaped
(e.g. a ray could pass over a narrow water strait and re-enter land further
out, or a gap in the coastline could squeeze between two sampled angles
without ever crossing one of our rays), so this is only a PRE-FILTER, never
a verdict. The "fraction of rays that must hit water" threshold is
deliberately tuned toward high recall (see water_fraction_threshold below),
accepting some false positives (mainland seeds that happen to look
mostly-water-ringed from spawn, e.g. spawn near a large bay) in exchange for
very low risk of rejecting a true island.

Validated (against the two ground-truth seeds used throughout this project):
seed 20271579 (real island) scores close_score=1.0; seed 1489155600
(mainland) scores close_score=0.77 -- a wide margin either side of the
default 0.98 threshold below.

THRESHOLD CHOICE -- 0.98, kept at 0.98 after TWO rounds of investigation (not
raised, not lowered):

Round 1 (0.98 vs 0.90, lower/looser): a design review of this pipeline
argued the wide 1.0-vs-0.77 margin above meant the threshold could safely be
"loosened" to ~0.90 for extra recall margin against coastline shapes not yet
seen. Measured directly during productionization on random 5,000-seed
batches, threshold=0.90 shortlists ~29% of ALL seeds (median close_score
across random seeds is already ~0.875 -- most mainland spawns have *some*
nearby water/bay/lake pattern that scores fairly high, just not 1.0), which
defeats the entire point of a two-stage funnel (stage 2's per-seed flood
fill is far too expensive to run on 29% of a multi-million-seed batch).
threshold=0.98 reproduces the ~1-1.2% shortlist rate this design was
validated against while still keeping the real island (1.0) comfortably
clear of it.

Round 2 (0.98 vs 0.99/0.995/0.999/1.0, higher/tighter -- investigated because
stage 2 dominates wall-clock and only runs on the shortlist, so a smaller
shortlist looked like a free win): close_score is quantized in units of
1/n_angles = 1/48, so the two values bracketing the top of the range are
47/48 = 0.97917 and 48/48 = 1.0. Since the default threshold 0.98 already
satisfies 0.98 > 47/48, the gate `close_score >= 0.98` ALREADY admits only a
perfect close_score of exactly 1.0 -- there is nothing in (0.9792, 1.0) left
for a higher threshold to exclude. Verified directly, not just inferred:
shortlist masks for thresholds 0.98/0.99/0.995/0.999/1.0 were bit-identical
(torch.equal(...) == True) on a fixed 100,000-seed batch, all producing the
identical 1,105 shortlisted seeds (1.105%). On the real 2,000,000-seed
production run (1.099% measured shortlist rate) this means raising the
threshold to any of these values changes stage-2 workload and overall
throughput by exactly 0% -- CONCLUSION: --water-threshold above 0.98 is NOT
a lever for pipeline speedup; don't re-litigate this without new evidence
that changes the quantization argument above. If stage 2 needs to get
cheaper, look at a different filter dimension (e.g. approx_area) or stage
2's own per-seed cost, not this threshold.

Since 0.98 is *functionally* "require exact closure", a natural follow-up
question is whether that's even safe -- could ray-angle luck (a narrow water
strait failing to align with any of the 48 sampled angles, ~7.5 degrees
apart) ever cause a genuine island to score just under 1.0, in which case
0.98 (or 1.0) would start rejecting real islands? This was stress-tested,
not assumed: the ray-cast logic was reimplemented with an angular-phase-
offset knob and run against all 3 known real islands (20271579, 1064911508,
3627767453) plus the known mainland (1489155600):
  - angular resolution: all 3 islands scored exactly 1.0 at every n_angles
    from 16 to 128 -- a 30x range around production's 48.
  - angular phase: rotating the 48-ray fan across the full 7.5-degree
    inter-ray gap (76 offsets, 0.1-degree fine pass) -- worst-case
    close_score across ALL offsets was still exactly 1.0 for all 3 islands.
  - ultra-fine full-circle scan (1440 rays, 0.25-degree resolution): 0/1440
    rays hit land for any of the 3 islands, vs. 306/1440 for the mainland --
    no coastline gap anywhere that a much denser fan would catch and the
    production 48 rays missed.
  - radial margin: worst-case water-hit radius across all rays/islands was
    1220 of max_radius=1400 tiles -- >=180 tiles (9 radius-steps) of slack
    before max_radius would even become the limiting factor.
No configuration in this sweep produced a real island scoring below 1.0, so
treating 0.98 as "require exact closure" is well-supported as safe. CAVEAT:
this rests on n=3 known real islands -- the entire confirmed population this
project has found so far (real islands are ~1-in-1.75M rarity) -- so treat
"exact closure is safe" as well-supported but NOT ironclad, unlike the Round
2 finding above (that one is a direct fact about close_score's quantization,
with no sample-size caveat attached: "raising the threshold buys no speed"
is disproven, not just unlikely).

DECISION: left at 0.98 rather than bumping the literal default to 1.0 for
cosmetic clarity -- they are behaviorally and performance-wise identical per
the above, so there's nothing to gain from the diff. Use --water-threshold
to experiment, but per Round 1, do not LOWER it below 0.98 without
re-measuring the resulting shortlist RATE (not just recall on known islands)
on a large random batch first; and per Round 2, do not expect RAISING it to
buy any speedup -- it mathematically cannot, at n_angles=48.

CHUNK-SIZE CHOICE -- 2,000, not the prior default of 8,000: a parameter-
tuning sweep (capped at 100,000 seeds/data-point per this project's
investigation constraints, batch_size=100,000 held fixed) measured stage-1-
only throughput of 12,380 seeds/s at chunk_size=8000 vs. 17,393 seeds/s
(+40%) at chunk_size=2000 and 21,929 seeds/s (+77%, single-run best) at
chunk_size=1000; every value tried in the 500-4000 range beat 8000 by
35-77%, and all of them sit at the SAME ~11.5GB memory floor (vs ~13.8GB at
the old default of 8000) -- i.e. this range is a free win memory-wise too.
2,000 was chosen over the single-run-best 1,000 as a safer pick in the
middle of a consistently-faster plateau rather than overfitting to one
unrepeated run -- see find_islands.py's --stage1-chunk-size help text for
the full numbers. HARD RISK, confirmed not hypothetical: raising this toward
--batch-size is dangerous, not just diminishing-returns -- chunk_size=50,000
caused physical-VRAM oversubscription (37.96GB alloc on a 32GB card, a 7x
slowdown) and chunk_size=100,000 (i.e. unchunked) is a confirmed hard CUDA
OOM crash.

APPROXIMATION NOTE: this stage only cares about the *sign* of elevation
(land vs water) at sampled points, so it inherits noise_gpu's approximation
of Math::exp2f/log2f inside modified_amplitude (see noise_gpu.py's module
docstring) -- empirically this has never flipped a land/water classification
in testing. Being a screen, occasional misclassifications near elevation==0
here are exactly the kind of error this funnel is designed to tolerate
(stage 2's exact-given-the-mask flood fill, plus the mandatory final CPU
verification, catch anything that slips through).

PER-POINT/PER-RAY COST -- THREE IDEAS INVESTIGATED (2026), NONE MERGED. All
three targeted this stage's ray-cast structure specifically (48 angles x 71
radii = 3,408 points/seed); none changed correctness, but none earned their
complexity either. Do not re-attempt any of these without new evidence:

  - Lattice-cell memoization on nauvis_macro (dedup the ray grid's ~3,408
    points down to the handful of unique noise-lattice cells they actually
    land on, since this stage's ray-grid positions are IDENTICAL across
    every seed/chunk in a run): the premise is strongly true -- only ~12
    unique primary lattice cells out of 3,408 ray points for nauvis_macro_2
    (99.6% duplication), similar for macro_1's two octaves (97-99.9% dup).
    Bit-exact against all 4 ground-truth seeds. Hoisting the one-time
    torch.unique() out of the per-seed/per-chunk loop (mirroring
    build_all_tables()'s own hoisting pattern above) gave a real, tightly
    reproducible 3.37x-5.04x speedup on the isolated nauvis_macro
    computation at this stage's production chunk_size range (2000-4000).
    But nauvis_macro is only ~3 of elevation_nauvis's ~25 total octave-calls
    across all six components (the other five -- nauvis_detail,
    nauvis_persistance especially -- have lattice spacing at or finer than
    the 20-tile ray step and show little to no redundancy), and this stage
    is itself the MINORITY contributor to end-to-end wall-clock (stage 2's
    flood fill dominates at ~75-78%, see stage2_floodfill.py). Measured
    full-pipeline effect: 1.05x-1.21x, with IQRs at chunk_size>=2000
    spanning below 1.0 to 8x -- not distinguishable from GPU contention
    noise. SHELVED, not rejected: the technique is correctness-clean and
    the component-level win is real, it's just unproven to matter enough
    end-to-end (ceiling roughly 1.03-1.2x) to justify the added per-octave
    unique/gather bookkeeping. Would need (a) a clean uncontended-GPU
    remeasurement and (b) generalizing to nauvis_bridge_billows/
    nauvis_hills_cliff_level (also 97-99.5% dup) before it's worth merging.

  - Per-ray early termination (compact away rays that already found water
    before the 71-radius sweep finishes, instead of evaluating every radius
    for every ray): theoretical FLOP savings are real (47% fewer
    evaluations on random seeds, 61-68% on the 3 known real islands) and
    the compaction implementation is bit-exact against baseline shortlist/
    close_score at every batch size tested. But measured wall-clock is a
    NET LOSS at every scale tried -- 0.169x (5.9x slower) at n=1,000,
    0.277x (3.6x slower) at n=5,000, 0.393x (2.5x slower) at n=8,000 --
    because compaction turns this stage's single vectorized
    [chunk, n_angles*n_r] elevation call into a strictly sequential
    71-iteration Python loop, and the per-step CUDA-launch + boolean-mask-
    compaction overhead costs far more than the FLOPs it saves. The gap
    narrows with n (5.9x -> 3.6x -> 2.5x slower from n=1,000 to n=8,000) but
    does not close. REJECTED as built; a fundamentally different structure
    (CUDA-graph-capturing the loop, or compacting only every K steps instead
    of every step) would be needed for another attempt, not a tweak of this
    prototype.

  - Macro-predicts-sign two-tier screening (use the cheap ~3-octave
    nauvis_macro sub-term alone to decide "definitely land"/"definitely
    water" for most points, falling back to the full ~25-octave-call
    elevation_nauvis only on an ambiguous remainder): REJECTED, there is no
    exploitable signal, not merely an implementation shortfall. Raw
    macro-sign agreement with the true elevation sign was measured at 39.9%
    across 300 random seeds + all 4 ground-truth seeds (1,036,032 points) --
    worse than a coin flip -- and even the most generous safe threshold
    (calibrated with a 20% margin against the ground-truth seeds' own
    worst-case disagreement) still leaves ~99% of points in the ambiguous
    band needing full evaluation anyway. Mechanistically: once distance from
    spawn exceeds ~500 tiles (true for most of a stage-1 ray), land requires
    BOTH nauvis_main > 0 (which macro contributes to) AND an independent
    starting_lake > 0 field that carries zero information from macro -- so
    no single ~3-eval sub-component can bound that AND. Measured two-tier
    wall-clock was flat-to-slightly-worse (0.967x-1.019x, both within
    contention noise) because the compaction step couldn't actually shrink
    the batch (worst-case rows still needed ~all 3,408 points). Any future
    "cheap proxy first" idea for this noise stack must be checked against
    this same AND-of-independent-fields structure before prototyping.
"""

import math

import torch

import noise_gpu as ng


def build_ray_grid(n_angles, radii, device):
    """
    radii: 1D float list/tensor of radii to sample along each ray, MUST
        include 0.0 as the first entry (used to read spawn elevation).
    Returns pos_x, pos_y: torch.float32 [n_angles * len(radii)] flattened
        ray-grid sample points (angle varies slower than radius: point
        (a, r) is at flat index a * n_r + r).
    """
    radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device)
    n_r = radii_t.shape[0]
    angles = torch.arange(n_angles, dtype=torch.float32, device=device) * (2.0 * math.pi / n_angles)

    cos_a = torch.cos(angles).unsqueeze(1)  # [n_angles, 1]
    sin_a = torch.sin(angles).unsqueeze(1)
    r = radii_t.unsqueeze(0)  # [1, n_r]

    pos_x = (cos_a * r).reshape(-1)  # [n_angles * n_r]
    pos_y = (sin_a * r).reshape(-1)
    return pos_x, pos_y, n_angles, n_r


def screen_seeds(seeds, n_angles=48, max_radius=1400.0, radius_step=20.0,
                  water_fraction_threshold=0.98, chunk_size=2000, verbose=False):
    """
    seeds: torch.int64 [N] on target device (values in [0, 2**32)).

    Returns dict of per-seed torch tensors (all length N, same order as input `seeds`):
      spawn_land:   bool     -- elevation(0,0) > 0
      close_score:  float32  -- fraction of the n_angles rays that hit water
                                 before max_radius (1.0 = fully surrounded
                                 within search radius, at ray resolution)
      approx_area:  float32  -- shoelace-polygon area estimate (tiles^2) from
                                 the per-angle first-water-hit radius (rays
                                 that never hit water use max_radius, so this
                                 OVER-estimates area for non-closed shapes --
                                 only meaningful when close_score is high;
                                 stage 2 produces the real area estimate)
      hit_radius:   float32  -- [N, n_angles] first-water-hit radius per ray
                                 (== max_radius for a ray that never found
                                 water -- a "near-miss" ray). Exposed (rather
                                 than kept as a stage-1-internal local, as it
                                 was before this field existed) so stage 1.5
                                 (gpu/stage1_5_witness.py) can find each
                                 shortlisted seed's likely escape corridor
                                 without re-casting the same rays a second
                                 time.
      near_miss_bearing: float32 -- [N] circular mean bearing (radians, atan2
                                 convention matching build_ray_grid's angle
                                 parametrization) of whichever ray(s) never
                                 hit water within max_radius -- i.e. the
                                 direction stage 1.5's directional beam
                                 seeding should be biased toward. 0.0 for a
                                 seed with no near-miss ray at all (close_score
                                 == 1.0, e.g. a real island candidate) since
                                 there is no escape-corridor signal to give in
                                 that case -- callers must not mistake this
                                 fallback 0.0 for an actual eastward bearing;
                                 gate on close_score < 1.0 (equivalently
                                 hit_radius.min(dim=1) < max_radius) first if
                                 that distinction matters.
      shortlist:    bool     -- spawn_land & close_score >= water_fraction_threshold
                                 (the actual filter passed on to stage 2)
    """
    device = seeds.device
    n = seeds.shape[0]

    radii = [0.0] + list(torch.arange(radius_step, max_radius + radius_step, radius_step).tolist())
    pos_x_shared, pos_y_shared, n_a, n_r = build_ray_grid(n_angles, radii, device)
    radii_t = torch.tensor(radii, dtype=torch.float32, device=device)
    theta_shared = torch.arange(n_angles, dtype=torch.float32, device=device) * (2.0 * math.pi / n_angles)
    cos_theta_shared = torch.cos(theta_shared)  # [n_angles], reused for both the shoelace
    sin_theta_shared = torch.sin(theta_shared)  # boundary points and the near-miss circular mean below

    spawn_land = torch.zeros(n, dtype=torch.bool, device=device)
    close_score = torch.zeros(n, dtype=torch.float32, device=device)
    approx_area = torch.zeros(n, dtype=torch.float32, device=device)
    hit_radius_out = torch.zeros(n, n_angles, dtype=torch.float32, device=device)
    near_miss_bearing = torch.zeros(n, dtype=torch.float32, device=device)

    # Build ALL permutation tables for the whole batch ONCE up front (see
    # noise_gpu.build_all_tables's docstring for why this must never be done
    # per-chunk: the underlying Fisher-Yates shuffle is a sequential,
    # launch-overhead-dominated loop whose cost barely depends on batch
    # size, so rebuilding it once per chunk multiplies that fixed cost by
    # the chunk count for no benefit). Only the per-point elevation
    # evaluation below (genuinely memory-heavy: N x P float32 tensors) needs
    # to be chunked.
    all_tables = ng.build_all_tables(seeds)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        cn = end - start

        tables = ng.slice_tables(all_tables, start, end)
        px = pos_x_shared.unsqueeze(0).expand(cn, -1)
        py = pos_y_shared.unsqueeze(0).expand(cn, -1)

        with torch.no_grad():
            elevation = ng.elevation_nauvis(tables, px, py)  # [cn, n_a*n_r]

        elevation = elevation.view(cn, n_a, n_r)
        is_water = elevation <= 0.0

        chunk_spawn_land = ~is_water[:, 0, 0]  # radius index 0 is (0,0) for every angle

        any_water = is_water.any(dim=2)  # [cn, n_a]
        first_idx = is_water.float().argmax(dim=2)  # first True index along radius (0 if none found)
        hit_radius = torch.where(any_water, radii_t[first_idx], torch.full_like(radii_t[first_idx], max_radius))

        chunk_score = any_water.float().mean(dim=1)  # [cn]

        # shoelace polygon area from (angle, hit_radius) -> (x,y) boundary points
        bx = hit_radius * cos_theta_shared.unsqueeze(0)  # [cn, n_a]
        by = hit_radius * sin_theta_shared.unsqueeze(0)
        bx_next = torch.roll(bx, shifts=-1, dims=1)
        by_next = torch.roll(by, shifts=-1, dims=1)
        cross = bx * by_next - bx_next * by
        chunk_area = 0.5 * cross.sum(dim=1).abs()

        # near-miss bearing: circular mean angle of rays that never hit water
        # (any_water == False) -- i.e. the direction(s) that look like an
        # escape corridor from spawn. Circular (vector) mean, not arithmetic
        # mean of angles, since angles wrap at 2*pi and a plain average of
        # e.g. two near-miss rays at 179 and -179 degrees must land near
        # +-180, not near 0. Falls back to 0.0 when no ray ever missed water
        # (any_water all True, i.e. close_score == 1.0) -- see the "0.0" caveat
        # in this function's docstring.
        never_hit = ~any_water  # [cn, n_a]
        never_hit_f = never_hit.float()
        n_never_hit = never_hit_f.sum(dim=1)  # [cn]
        sum_cos = (never_hit_f * cos_theta_shared.unsqueeze(0)).sum(dim=1)
        sum_sin = (never_hit_f * sin_theta_shared.unsqueeze(0)).sum(dim=1)
        chunk_near_miss_bearing = torch.where(
            n_never_hit > 0, torch.atan2(sum_sin, sum_cos), torch.zeros_like(sum_sin),
        )

        spawn_land[start:end] = chunk_spawn_land
        close_score[start:end] = chunk_score
        approx_area[start:end] = chunk_area
        hit_radius_out[start:end] = hit_radius
        near_miss_bearing[start:end] = chunk_near_miss_bearing

        if verbose:
            print(f"  stage1 chunk {start}:{end} done", flush=True)

    shortlist = spawn_land & (close_score >= water_fraction_threshold)

    return {
        "spawn_land": spawn_land,
        "close_score": close_score,
        "approx_area": approx_area,
        "hit_radius": hit_radius_out,
        "near_miss_bearing": near_miss_bearing,
        "shortlist": shortlist,
    }
