"""
Stage 1.5 cascade -- the integration glue between stage1_5_solver.run_stage15
(the validated standalone prototype) and find_islands.py (the production
pipeline). Implements the THREE-TIER CASCADE the project owner specified and
already measured as a standalone script (see
SCRATCH/cascade_second_pass.py, not checked into this repo): run the coarse
64-tile lattice first, then re-run ONLY the residual (non-REJECT) seeds
through the finer 32-tile lattice, then combine.

WHY TWO PASSES INSTEAD OF ONE
--------------------------------------------------------------------------
stage1_5_oracle.py's EPSILON_FP CAVEAT section explains that this design's
REJECT is airtight (sound) at ANY lattice spacing, but its SENSITIVITY (how
often a real mainland actually produces a REJECT) depends on how finely the
lattice is drawn -- a coarser lattice is cheaper per seed but catches fewer
real escape corridors on-lattice. Measured on a several-hundred-thousand-seed
batch: 64-tile-alone REJECT rate is already close to the 32-tile-alone
baseline, but not identical; running 32-tile on top, but ONLY on the tiny
residual that 64-tile could not resolve, recovers most of that gap at a
fraction of running 32-tile over the whole shortlist (measured ~2.6x less
extrapolated GPU time for nearly identical efficacy: 99.89% vs 99.9% REJECT).

COMBINING RULE (per the project owner's exact spec -- do not reorder)
--------------------------------------------------------------------------
  REJECT  if EITHER pass rejected the seed. A pass-1 REJECT is exactly as
          airtight as a pass-2 REJECT (same certificate construction, just a
          coarser lattice -- see stage1_5_oracle.py's SOUNDNESS ARGUMENT,
          which is spacing-independent), so short-circuiting pass 2 for
          anything pass 1 already rejected costs nothing in soundness and is
          the entire point of running the cheap pass first.
  CONFIRM if the LAST pass that actually ran on the seed says CONFIRM (i.e.
          pass 2's verdict for anything routed to pass 2, else pass 1's).
          Still CPU-verify-backstopped downstream, same as any other CONFIRM
          in this pipeline -- combining passes changes nothing about that.
  DEFER   if the last pass that ran says DEFER -- falls through to today's
          existing dense stage 2, completely unchanged.

Neither pass ever prunes/evicts (see stage1_5_solver.py's module docstring),
so nothing here introduces a new soundness concern; this module only adds
the pass-1/residual/pass-2 bookkeeping and does not touch the underlying
REJECT/CONFIRM/DEFER machinery.
"""

import numpy as np
import torch

import noise_gpu as ng
import stage1_5_oracle as oracle  # noqa: F401  (re-exported for convenience/tests)
import stage1_5_solver as solver

# Pass 1 (coarse advisor pass): 64-tile lattice spacing. Domain radius scales
# with RING_TILES (see stage1_5_oracle.py's DOMAIN_RADIUS_NODES comment for
# the 1.417 headroom ratio this reuses): ceil(5000*1.417/64) = 111 nodes ->
# nominal domain radius 111*64 = 7,104 tiles.
PASS1_NODE_SPACING = 64
PASS1_DOMAIN_RADIUS_NODES = 111

# Pass 2 (fine recovery pass, residual-only): 32-tile lattice spacing --
# stage1_5_oracle.py's original default design. ceil(5000*1.417/32) = 222
# nodes -> nominal domain radius 222*32 = 7,104 tiles (same ratio, matches
# pass 1's own headroom).
PASS2_NODE_SPACING = 32
PASS2_DOMAIN_RADIUS_NODES = 222

DEFAULT_TIERS = [(PASS1_NODE_SPACING, PASS1_DOMAIN_RADIUS_NODES),
                 (PASS2_NODE_SPACING, PASS2_DOMAIN_RADIUS_NODES)]

# Pass 0 (EXPERIMENTAL third, coarser tier): 128-tile lattice spacing, same
# 1.417 headroom ratio: ceil(5000*1.417/128) = 56 nodes -> nominal domain
# radius 56*128 = 7,168 tiles. NOT part of DEFAULT_TIERS (find_islands.py's
# unmodified run_cascade(seeds, device) calls are untouched by this) -- opt
# in explicitly via run_cascade(..., tiers=THREE_TIER_CASCADE) pending its
# own measurement/ground-truth validation.
PASS0_NODE_SPACING = 128
PASS0_DOMAIN_RADIUS_NODES = 56
THREE_TIER_CASCADE = [(PASS0_NODE_SPACING, PASS0_DOMAIN_RADIUS_NODES)] + DEFAULT_TIERS

# Pass 3 (EXPERIMENTAL fourth, finest tier): 16-tile lattice spacing. ceil(
# 5000*1.417/16) = 443 nodes -> nominal domain radius 443*16 = 7,088 tiles.
# Measured on the 3 known false-CONFIRM (sensitivity-gap) seeds from a
# 50,000-seed RING_TILES=5000 batch: catches 1 of 3 (flips a false CONFIRM
# to the correct REJECT) that neither 64-tile nor 32-tile found -- an 8-tile
# follow-up caught ZERO further cases on the remaining 2 at ~2x the round
# count (767 vs 385), so this is a real but bounded win, not the start of a
# "just go finer" trend -- do not add a 5th, even-finer tier without new
# evidence. Cheap to add regardless of the modest catch rate: like pass 2,
# it only ever runs on whatever tiny residual survives the tier before it.
PASS3_NODE_SPACING = 16
PASS3_DOMAIN_RADIUS_NODES = 443
FOUR_TIER_CASCADE = THREE_TIER_CASCADE + [(PASS3_NODE_SPACING, PASS3_DOMAIN_RADIUS_NODES)]

REJECT = solver.REJECT
CONFIRM = solver.CONFIRM
DEFER = solver.DEFER
UNRESOLVED = solver.UNRESOLVED
VERDICT_NAMES = solver.VERDICT_NAMES


def _empty_timing():
    return {"gpu_seconds": 0.0, "cpu_seconds": 0.0, "n_gpu_calls": 0, "rounds_used": 0, "round_sizes": []}


def run_cascade(seeds, device, verbose=False, backend=oracle.DEFAULT_BACKEND, parallel_cpu=False,
                tiers=None):
    """
    seeds: torch.int64 [N] on `device` -- typically one round's stage-1
    shortlist.

    Runs an N-TIER cascade: tier 0 (coarsest) over ALL of `seeds`; each
    subsequent tier over only the residual (non-REJECT) seeds the previous
    tier left unresolved; combines per the module docstring's rule
    (REJECT if ANY tier rejected; CONFIRM/DEFER from the LAST tier that ran).

    tiers: list of (node_spacing, domain_radius_nodes) tuples, coarsest
        first. None (default) -> DEFAULT_TIERS (the original 2-tier
        64-then-32 design, UNCHANGED behavior from before this parameter
        existed -- find_islands.py's existing run_cascade(seeds, device)
        calls are not affected). Pass tiers=THREE_TIER_CASCADE for the
        128-then-64-then-32 experimental variant.

    build_tables is called ONCE for the full `seeds` batch; every
    subsequent tier gathers its residual's rows out of that same table set
    (noise_gpu.gather_tables) instead of paying build_tables' ~1,020-launch
    fixed cost again on a tiny subset -- measured ~230ms (13-seed residual)
    vs <1ms to gather, a >500x difference on that one call.

    backend / parallel_cpu: passed straight through to every tier's own
    run_stage15 call (see stage1_5_oracle.classify_edges's backend= and
    stage1_5_solver.run_stage15's parallel_cpu= for what each does).

    Returns dict, all arrays length N (numpy), aligned with `seeds`' order:
      verdict:        int8[N]  -- final combined REJECT/CONFIRM/DEFER
      pass_verdicts:  list[int8[N]] -- per-tier verdict, UNRESOLVED where
                      that tier didn't run (tier 0 always fully resolved)
      ran_pass:       list[bool[N]] -- which seeds ran each tier
      n_pass:         list[int] -- seeds that ran each tier
      timings:        list[dict] -- see stage1_5_solver.run_stage15's
                      timing_out, one per tier (all-zeros if that tier never ran)
      pass1_verdict, pass2_verdict, ran_pass2, n_pass1, n_pass2, timing1, timing2:
                      back-compat aliases to pass_verdicts[0]/[-1] etc. --
                      exact values ONLY meaningful for the 2-tier default;
                      with tiers=THREE_TIER_CASCADE, pass2_verdict/timing2
                      alias the LAST (32-tile) tier, not tier index 1.
    """
    n = int(seeds.shape[0])
    if tiers is None:
        tiers = DEFAULT_TIERS
    n_tiers = len(tiers)

    if n == 0:
        empty_i8 = np.zeros(0, dtype=np.int8)
        empty_bool = np.zeros(0, dtype=np.bool_)
        return {
            "verdict": empty_i8,
            "pass_verdicts": [empty_i8] * n_tiers, "ran_pass": [empty_bool] * n_tiers, "n_pass": [0] * n_tiers,
            "timings": [_empty_timing() for _ in range(n_tiers)],
            "pass1_verdict": empty_i8, "pass2_verdict": empty_i8, "ran_pass2": empty_bool,
            "n_pass1": 0, "n_pass2": 0, "timing1": _empty_timing(), "timing2": _empty_timing(),
        }

    # Built ONCE for the whole batch -- see docstring above.
    full_tables = oracle.build_tables(seeds)

    verdict_final = np.full(n, UNRESOLVED, dtype=np.int8)
    pass_verdicts = []
    ran_pass = []
    n_pass = []
    timings = []

    current_idx = torch.arange(n, dtype=torch.int64, device=device)  # indices into the ORIGINAL seeds/full_tables
    for node_spacing, domain_radius_nodes in tiers:
        tier_seeds = seeds[current_idx]
        tier_tables = ng.gather_tables(full_tables, current_idx)

        timing = {}
        res = solver.run_stage15(
            tier_seeds, device, node_spacing=node_spacing, domain_radius_nodes=domain_radius_nodes,
            verbose=verbose, timing_out=timing, backend=backend, parallel_cpu=parallel_cpu,
            tables=tier_tables,
        )
        tier_verdict = res["verdict"]

        current_idx_np = current_idx.cpu().numpy()
        verdict_full = np.full(n, UNRESOLVED, dtype=np.int8)
        verdict_full[current_idx_np] = tier_verdict
        ran_full = np.zeros(n, dtype=np.bool_)
        ran_full[current_idx_np] = True

        verdict_final[current_idx_np] = tier_verdict
        pass_verdicts.append(verdict_full)
        ran_pass.append(ran_full)
        n_pass.append(int(current_idx.shape[0]))
        timings.append(timing)

        residual_local = np.nonzero(tier_verdict != REJECT)[0]
        if residual_local.size == 0:
            break
        current_idx = current_idx[torch.as_tensor(residual_local, dtype=torch.int64, device=device)]

    # Any tier not reached (cascade fully resolved early) gets empty placeholders.
    while len(timings) < n_tiers:
        pass_verdicts.append(np.full(n, UNRESOLVED, dtype=np.int8))
        ran_pass.append(np.zeros(n, dtype=np.bool_))
        n_pass.append(0)
        timings.append(_empty_timing())

    return {
        "verdict": verdict_final,
        "pass_verdicts": pass_verdicts,
        "ran_pass": ran_pass,
        "n_pass": n_pass,
        "timings": timings,
        # back-compat aliases (exact only for the 2-tier default -- see docstring)
        "pass1_verdict": pass_verdicts[0],
        "pass2_verdict": pass_verdicts[-1],
        "ran_pass2": ran_pass[-1],
        "n_pass1": n_pass[0],
        "n_pass2": n_pass[-1] if n_tiers > 1 else 0,
        "timing1": timings[0],
        "timing2": timings[-1],
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"pass 1: node_spacing={PASS1_NODE_SPACING}, domain_radius_nodes={PASS1_DOMAIN_RADIUS_NODES} "
          f"(nominal radius {PASS1_NODE_SPACING * PASS1_DOMAIN_RADIUS_NODES} tiles)")
    print(f"pass 2: node_spacing={PASS2_NODE_SPACING}, domain_radius_nodes={PASS2_DOMAIN_RADIUS_NODES} "
          f"(nominal radius {PASS2_NODE_SPACING * PASS2_DOMAIN_RADIUS_NODES} tiles)")

    ground_truth = [
        (20271579, "confirmed real island -- must NEVER REJECT"),
        (3627767453, "confirmed real island -- must NEVER REJECT"),
        (1064911508, "RETRACTED (razor-thin land bridge) -- must NEVER REJECT"),
        (1489155600, "confirmed unbounded mainland -- must REJECT"),
    ]
    seeds = torch.tensor([s for s, _ in ground_truth], dtype=torch.int64, device=device)

    result = run_cascade(seeds, device, verbose=True)

    print("\n=== cascade verdicts ===")
    all_ok = True
    for i, (seed, note) in enumerate(ground_truth):
        v = int(result["verdict"][i])
        p1 = int(result["pass1_verdict"][i])
        p2 = int(result["pass2_verdict"][i])
        ran2 = bool(result["ran_pass2"][i])
        name = VERDICT_NAMES[v]
        p2_name = VERDICT_NAMES.get(p2, "UNRESOLVED(did not run)")
        print(f"  seed {seed}: final={name}  (pass1={VERDICT_NAMES[p1]}, "
              f"pass2={'did not run' if not ran2 else p2_name})  ({note})")
        if seed == 1489155600 and v != REJECT:
            print(f"    *** FAIL: {seed} must REJECT, got {name} ***")
            all_ok = False
        if seed != 1489155600 and v == REJECT:
            print(f"    *** FAIL (SAFETY-CRITICAL): {seed} must NEVER REJECT, got REJECT ***")
            all_ok = False

    print(f"\npass1: n={result['n_pass1']}, gpu={result['timing1']['gpu_seconds']:.3f}s, "
          f"cpu={result['timing1']['cpu_seconds']:.3f}s, rounds={result['timing1']['rounds_used']}")
    print(f"pass2: n={result['n_pass2']}, gpu={result['timing2']['gpu_seconds']:.3f}s, "
          f"cpu={result['timing2']['cpu_seconds']:.3f}s, rounds={result['timing2']['rounds_used']}")
    print("\nALL SAFETY CHECKS PASSED" if all_ok else "\n*** SAFETY CHECK FAILURE -- DO NOT WIRE THIS IN ***")
