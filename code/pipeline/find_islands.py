#!/usr/bin/env python3
"""
find_islands.py -- GPU-accelerated APPROXIMATE screening pipeline for
Factorio seeds that naturally produce spawn on a landmass fully enclosed by
water, under the DEFAULT elevation type (2.0) and default water settings
(water_scale=1.0, water_coverage=1.0). This is the "natural island" search
described in this project's README/CLAUDE context -- NOT the dedicated
"island" elevation type, which guarantees enclosure by construction.

*** READ README.md IN THIS DIRECTORY BEFORE TRUSTING ANY OUTPUT. ***
This tool trades bit-exactness for throughput (see noise_gpu.py's module
docstring for exactly what's approximated). By default every GPU-flagged
candidate is re-verified against the real CPU flood fill
(seed_finders/largest_island) before being written to the output CSV, so the
default output is trustworthy. If you pass --no-cpu-verify to skip that
(much slower) step for quick large-scale exploration, the output contains
UNVERIFIED candidates that must be checked by hand before being reported as
real islands.

PIPELINE (see stage1_screen.py / stage1_5_cascade.py / stage2_floodfill.py
for full design notes):
  Stage 1:   cheap ray-cast closure screen over the WHOLE batch -> shortlist
             (~1% of seeds, high recall, run entirely on GPU).
  Stage 1.5: (--stage15, on by default) certified segment-lattice cascade
             over the stage-1 shortlist ONLY -- a coarse 64-tile pass, then a
             finer 32-tile pass on just the residual seeds the coarse pass
             couldn't resolve. REJECT (either pass) is airtight and excludes
             the seed entirely, no stage 2, no CPU verify. CONFIRM routes
             straight to Stage 3 (CPU verify), skipping stage 2's dense flood
             fill. DEFER falls through to Stage 2 unchanged. See
             stage1_5_oracle.py / stage1_5_solver.py / stage1_5_cascade.py
             module docstrings for the full soundness argument -- disable
             with --no-stage15 for an A/B comparison against the pre-cascade
             pipeline.
  Stage 2:   exact-given-the-mask GPU BFS flood fill, only on whatever stage
             1.5 (if enabled) deferred, or the whole stage-1 shortlist
             otherwise -> per-seed enclosed/unbounded verdict + area estimate.
  Stage 3: mandatory CPU re-verification of every stage-2-confirmed
           candidate against the real, exact flood fill (non-negotiable
           correctness gate -- see cpu_verify.py). Runs candidates through a
           bounded ThreadPoolExecutor (--cpu-verify-concurrency) instead of
           one-subprocess-at-a-time, so a round with several candidates
           verifies them in parallel (measured 2.8x-8.6x on 3-32 concurrent
           candidates). NOTE: profiling on a real 2,000,000-seed production
           run found 0 rounds with more than one Stage-3 candidate (natural
           islands are rare -- see ground-truth notes), so this concurrency
           is a low-cost insurance policy for now, not something that moves
           typical wall-clock time; the actual bottleneck is stage1+stage2
           GPU time. A cross-round pipelining redesign (overlapping Stage 3
           of round N with Stage 1/2 of round N+1) was also prototyped and
           measured -- its hideable benefit (bounded by cpu-verify time per
           round) was smaller than normal GPU scheduling jitter at
           production round durations (30s+), so it was not merged; see git
           history if candidate density per round increases materially
           (e.g. after loosening --water-threshold well below its default).

USAGE
-----
  <venv>/bin/python3 find_islands.py --output candidates.csv --num-seeds 1000000

See --help for the full list of options (batch sizing, thresholds, grid
size, random-sampling vs. sequential scanning, etc).
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_screen as stage1  # noqa: E402
import stage1_5_cascade as stage15  # noqa: E402
import stage1_5_oracle as stage15_oracle  # noqa: E402
import stage1_5_pillar as pillar  # noqa: E402
import stage2_floodfill as stage2  # noqa: E402
import cpu_verify  # noqa: E402
import leader_visualizer  # noqa: E402
import island_extent  # noqa: E402

# Measured baseline of the exact CPU flood-fill tool (seed_finders/largest_island),
# 32 threads, random-sampled seeds under default settings -- see project context.
CPU_BASELINE_SEEDS_PER_SEC = 273.0

# farthest_tile_dist: max Euclidean tile-distance from spawn among the
# confirmed island's own connected component (island_extent.py) -- tracks
# margin against RING_TILES (see stage1_5_oracle.RING_TILES) directly in the
# output, not just as a one-off visualization. hit_border=1 means the
# component reached island_extent.HALF_EXTENT before closing off -- the
# recorded distance is then a LOWER BOUND, not exact (should not happen for
# a real CPU-verified island given HALF_EXTENT's own margin, but flagged
# rather than silently assumed).
CSV_HEADER = "rank;seed;score;water scale;water coverage;elevation type;farthest_tile_dist;hit_border"


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-o", "--output", required=True, metavar="PATH",
                     help="CSV output path. Rewritten after every batch (checkpointed), "
                          "so partial progress is never lost if interrupted.")

    ap.add_argument("--num-seeds", type=int, default=1_000_000, metavar="INT",
                     help="Total number of seeds to screen. Default: 1,000,000.")
    ap.add_argument("--first-seed", type=int, default=0, metavar="UINT32",
                     help="Lower bound of the seed range (inclusive).")
    ap.add_argument("--last-seed", type=int, default=2**32 - 1, metavar="UINT32",
                     help="Upper bound of the seed range (inclusive).")
    ap.add_argument("--sequential", action="store_true",
                     help="Scan [--first-seed, --last-seed] in order instead of sampling "
                          "--num-seeds seeds uniformly at random from it.")
    ap.add_argument("--reverse", action="store_true",
                     help="Only meaningful with --sequential: scan [--first-seed, --last-seed] "
                          "DESCENDING (starting at --last-seed, ending at --first-seed) instead "
                          "of ascending. Useful for covering a range from the opposite end of "
                          "an already-in-progress ascending scan.")
    ap.add_argument("--rng-seed", type=int, default=0, metavar="INT",
                     help="RNG seed for --num-seeds random sampling (reproducibility).")

    ap.add_argument("--batch-size", type=int, default=100_000, metavar="INT",
                     help="Seeds processed per top-level round (stage1 -> stage2 -> "
                          "CPU-verify -> checkpoint write). Default: 100,000 -- SWEPT "
                          "(<=100k seeds/point, per this project's investigation cap) and "
                          "kept unchanged: stage-1-only throughput scaled 9,713 -> 10,764 "
                          "-> 12,307 -> 12,454 seeds/s for batch_size 10k -> 25k -> 50k -> "
                          "100k, i.e. the 50k->100k step already only buys +1.2%% "
                          "throughput for +50%% peak memory (9.19GB -> 13.77GB) -- past "
                          "the knee of the curve. No data above 100k exists yet (that's "
                          "the sweep's cap, not a proven ceiling); revisit if ever pushed "
                          "past that on this card.")
    ap.add_argument("--stage1-chunk-size", type=int, default=2000, metavar="INT",
                     help="Memory-chunk size for stage 1's per-point elevation evaluation "
                          "(tables for the whole round stay resident; only this is "
                          "chunked). Default: 2,000 -- CHANGED from a prior default of "
                          "8,000 based on a sweep (<=100k seeds/point, batch_size=100,000 "
                          "fixed) that found chunk=8000 ran at 12,380 seeds/s while "
                          "chunk=2000 hit 17,393 (+40%%) and chunk=1000 hit 21,929 "
                          "(+77%%, single-run best); every value tried in 500-4000 beat "
                          "8000 by 35-77%%, all at the SAME 11.483GB memory floor (vs "
                          "13.768GB at the old default), so this range is free "
                          "memory-wise too. 2,000 was picked over the single-run-best "
                          "1,000 as a safer point in the middle of a consistently-faster "
                          "plateau rather than overfitting to one run. HARD RISK, "
                          "confirmed not hypothetical: chunk_size=50,000 caused physical-"
                          "VRAM oversubscription (37.96GB alloc on a 32GB card, a 7x "
                          "slowdown) and chunk_size=100,000 (unchunked) is a confirmed "
                          "hard CUDA OOM crash -- never raise this toward --batch-size.")
    ap.add_argument("--stage2-chunk-size", type=int, default=32, metavar="INT",
                     help="Memory-chunk size for stage 2's land-mask rasterization over "
                          "the shortlist -- only the transient per-chunk elevation buffer "
                          "is chunked; build_land_mask()'s output tensor is allocated once "
                          "for the WHOLE shortlist up front, so this knob trades transient "
                          "memory only and produces bit-identical output at any value "
                          "(verified at chunk in {1,64,256}). Default: 32 -- lowered from "
                          "a prior default of 64 based on a sweep (<=100k seeds/point, "
                          "shortlist ~1,071 seeds) that ran under confirmed heavy sibling-"
                          "process GPU contention (absolute throughput numbers from that "
                          "sweep are unreliable), but did confirm a hard ceiling at "
                          "chunk=512 (43.3GB alloc > this 32GB card -- silent host-RAM "
                          "oversubscription, a more dangerous failure mode than a clean "
                          "OOM since it 'succeeds' while collapsing throughput). This is "
                          "a lower-confidence, tentative change pending a re-run on an "
                          "idle GPU -- the old help text's claim that raising this helps "
                          "when GPU memory is available is UNSUPPORTED by any measurement "
                          "here (memory scales with shortlist size, not this chunk size). "
                          "NOTE: at real 2M-seed-round shortlist sizes (~22,000, vs. "
                          "~1,071 tested) the fixed land_mask tensor alone needs ~10.8GB "
                          "-- re-check headroom if --batch-size or --water-threshold are "
                          "pushed up.")

    ap.add_argument("--water-threshold", type=float, default=0.98, metavar="FLOAT",
                     help="Stage-1 shortlist gate: fraction of cast rays that must hit "
                          "water before max-radius. Lower = more recall, bigger (and much "
                          "more expensive, since stage 2 runs on every shortlisted seed) "
                          "shortlist. Default 0.98 (real island scores 1.0, mainland 0.77 -- "
                          "measured shortlist rate ~1-1.2%% of the batch; 0.90 was tried and "
                          "measured to shortlist ~29%% of a random batch, so it is NOT the "
                          "default). Raising this above 0.98 was also investigated as a "
                          "possible stage-2 speedup (stage 2 dominates wall-clock and only "
                          "runs on the shortlist) and found to buy NOTHING: with n_angles=48, "
                          "close_score is quantized to multiples of 1/48, and 0.98 already "
                          "exceeds 47/48=0.97917, so the gate already only admits an exact "
                          "1.0 score -- shortlist masks at 0.98/0.99/0.995/0.999/1.0 were "
                          "verified bit-identical on a 100,000-seed batch. See "
                          "stage1_screen.py's module docstring for the full "
                          "threshold-choice writeup (both the 0.90 and >0.98 investigations, "
                          "plus the ray-angle-luck robustness check on the 3 known real "
                          "islands) before changing this default.")
    ap.add_argument("--n-angles", type=int, default=48, metavar="INT",
                     help="Number of rays cast from spawn in stage 1.")
    ap.add_argument("--max-radius", type=float, default=2000.0, metavar="TILES",
                     help="Max ray length in stage 1 (tiles). Bumped from 1400 to 2000: "
                          "confirmed islands are getting large enough (e.g. seed 558632404, "
                          "an elongated 1.19M-tile^2 island) that some rays run out of budget "
                          "before finding water, dragging close_score below the 0.98 threshold "
                          "on real islands even when the other 47 rays correctly find the "
                          "enclosure. This only widens stage 1's own (uncertified) search -- "
                          "it does not change RING_TILES, the actual definitional enclosure "
                          "radius used by stage 1.5's certified REJECT/CONFIRM.")
    ap.add_argument("--radius-step", type=float, default=20.0, metavar="TILES",
                     help="Ray sampling step in stage 1 (tiles).")

    ap.add_argument("--stage1", dest="stage1", action="store_true", default=False,
                     help="Run stage 1 (ray screen) as a REJECT gate before stage 1.5. "
                          "OFF by default: stage 1's REJECT is an uncertified heuristic over a "
                          "discrete sample (48 angles x --radius-step radial steps) -- it has an "
                          "irreducible geometric false-negative channel (a moat narrower than "
                          "--radius-step, or one lying in the up-to-183-tile angular gap between "
                          "rays at r=1400, is invisible to it even when the island is genuinely "
                          "enclosed). Measured directly: of the seeds this gate would reject in a "
                          "500k random sample, some still reached a stage-1.5 CONFIRM verdict -- "
                          "i.e. the gate silently discards seeds before they reach any certified "
                          "stage. A complete, provably-no-island-missed catalog requires every "
                          "seed to reach stage 1.5's certified REJECT/CONFIRM/DEFER. Enable this "
                          "only for throughput experiments where an uncertified prefilter is "
                          "acceptable, never for a production census.")
    ap.add_argument("--no-stage1", dest="stage1", action="store_false",
                     help="(default) Skip stage 1 entirely -- every seed goes straight to "
                          "stage 1.5's certified cascade. Stage 1.5 does not consume stage 1's "
                          "outputs (no bearing-prior wiring exists in stage1_5_cascade.py), so "
                          "there is no benefit to running stage 1 once it is not gating.")

    ap.add_argument("--pillar", dest="pillar", action="store_true", default=True,
                     help="(default) Run the certified pillar pre-tier (stage1_5_pillar.py) "
                          "ahead of the lattice cascade. Measured ~26%% faster end-to-end than "
                          "no pre-tier at all, and every REJECT it issues is fully certified "
                          "(same soundness as the lattice cascade) -- mutually exclusive in "
                          "practice with --stage1 (using both is redundant; using neither means "
                          "every seed pays the full, slower lattice cascade).")
    ap.add_argument("--no-pillar", dest="pillar", action="store_false",
                     help="Disable the pillar pre-tier -- e.g. when using --stage1 as the fast "
                          "pre-filter instead (uncertified, faster, see --stage1's own warning).")

    ap.add_argument("--stage15", dest="stage15", action="store_true", default=True,
                     help="Run the stage-1.5 certified segment-lattice cascade (64-tile pass, "
                          "then a 32-tile pass on the residual) between stage 1 and stage 2. "
                          "ON by default. REJECTs from either pass are airtight (no CPU backstop "
                          "needed) and are excluded entirely; CONFIRMs skip stage 2 and go "
                          "straight to CPU verify; DEFERs fall through to stage 2 unchanged. "
                          "See stage1_5_oracle.py / stage1_5_solver.py / stage1_5_cascade.py.")
    ap.add_argument("--no-stage15", dest="stage15", action="store_false",
                     help="Disable the stage-1.5 cascade (for A/B comparison against the "
                          "pre-cascade pipeline) -- the full stage-1 shortlist goes straight "
                          "to stage 2, exactly as before stage 1.5 existed.")
    ap.add_argument("--stage15-backend", default="triton_stripe",
                     choices=["pytorch", "triton_fused", "triton_stripe"], metavar="BACKEND",
                     help="Terrain evaluator backend for the stage-1.5 cascade (see "
                          "stage1_5_oracle.py's BACKEND_* constants). Default triton_stripe: "
                          "the fully-fused, in-kernel-position/in-kernel-reduction Triton "
                          "kernel, measured ~15x faster end-to-end than pytorch (the original, "
                          "pre-Triton implementation) on this pipeline's own ground-truth "
                          "regression. pytorch is kept for A/B comparison only.")
    ap.add_argument("--stage15-no-parallel-cpu", dest="stage15_parallel_cpu",
                     action="store_false", default=True,
                     help="Disable numba-prange parallelization of apply_round (the CPU-side "
                          "graph solver) -- ON by default, measured bit-identical to the "
                          "sequential version on the standing ground-truth seeds plus large "
                          "random batches. Only useful for A/B timing comparisons.")
    ap.add_argument("--stage15-three-tier", action="store_true",
                     help="Use the experimental 3-tier cascade (128-tile -> 64-tile -> 32-tile) "
                          "instead of the default 2-tier (64-tile -> 32-tile) -- measured 2.22x "
                          "faster on a 50,000-seed RING_TILES=5000 batch with identical verdicts "
                          "(REJECT is a hard certificate at any lattice spacing, so an extra "
                          "coarser tier can only resolve more seeds earlier, never change the "
                          "final answer). Off by default pending more production-scale mileage. "
                          "Mutually exclusive with --stage15-four-tier (four-tier wins if both given).")
    ap.add_argument("--stage15-four-tier", action="store_true",
                     help="Use the experimental 4-tier cascade (128 -> 64 -> 32 -> 16-tile) -- "
                          "adds a 16-tile residual-only pass on top of --stage15-three-tier. "
                          "Measured to catch 1 of 3 known sensitivity-gap false CONFIRMs an "
                          "8-tile follow-up didn't improve on further (see stage1_5_cascade.py's "
                          "FOUR_TIER_CASCADE docstring) -- a real but bounded reduction in the "
                          "false-CONFIRM rate reaching CPU verify, cheap to run (only ever "
                          "touches the tiny residual surviving the 32-tile pass).")
    ap.add_argument("--stage15-batch-size", type=int, default=50_000, metavar="INT",
                     help="Minimum number of stage-1-shortlisted seeds to ACCUMULATE (across "
                          "as many stage-1 rounds as it takes) before invoking the stage-1.5 "
                          "cascade -- and, downstream of it, stage 2 / CPU verify / checkpoint "
                          "write -- once on the whole accumulated batch. Only relevant when "
                          "--stage15 is enabled; ignored (each stage-1 round's shortlist is "
                          "processed immediately, matching the pre-cascade pipeline) under "
                          "--no-stage15, since that A/B baseline never had this cost. "
                          "WHY THIS EXISTS: stage1_5_oracle.build_tables -> "
                          "noise_gpu.build_all_tables's per-seed permutation-table shuffle is "
                          "~1,020 sequential tiny GPU kernel launches whose wall-clock is "
                          "nearly independent of batch size (fixed launch-overhead-dominated, "
                          "see noise_gpu.py's module docstring) -- calling run_cascade "
                          "(which calls it twice, once per lattice pass) once per stage-1 "
                          "round's ~1,000-1,200-seed shortlist pays that fixed cost ~2x per "
                          "round for almost no batch-size benefit, which measured out to a "
                          "flat ~11,500 seeds/sec end-to-end (round-by-round stage1.5 time "
                          "nearly constant regardless of round shortlist size), well below "
                          "both stage-1-alone's ~17,000-17,500 seeds/sec ceiling and the "
                          "cascade's own validated throughput when given a properly large "
                          "batch. Default: 50,000 -- matches the scale (43,857-65,667 "
                          "shortlisted seeds, from 4M-6M raw seeds screened at the measured "
                          "~1.1%% shortlist rate) at which stage1_5_cascade.run_cascade / "
                          "stage1_5_solver.run_stage15 were actually validated end-to-end "
                          "(see SCRATCH/dryrun_segment_graph_generic.py runs); does not need "
                          "to be tuned finer than that order of magnitude, since the fixed "
                          "per-call cost this amortizes is ~3s regardless of batch size.")

    ap.add_argument("--grid-radius-cells", type=int, default=350, metavar="INT",
                     help="Stage-2 flood-fill grid half-width, in 4-tile cells "
                          "(350 => 701x701 grid => +-1400 tiles from spawn). Do not shrink "
                          "without re-validating against both ground-truth seeds -- see "
                          "stage2_floodfill.py's module docstring.")
    ap.add_argument("--max-flood-steps", type=int, default=1500, metavar="INT",
                     help="Max BFS dilation iterations per stage-2 round. Default: 1,500 "
                          "-- SWEPT and kept unchanged: flood_fill()'s early-exit fires at "
                          "the batch's actual convergence point regardless of this "
                          "ceiling, and a sweep found steps_run pinned at exactly 544 for "
                          "every ceiling tried >=600 (600/700/800/1000/1500 all ran "
                          "identical real iterations). Observed convergence range across "
                          "~1,613 sampled seeds (<=100k-seed sweep batches plus the 4 "
                          "ground-truth seeds) was 238-589 steps, well under this default "
                          "-- raising or lowering it above the true convergence point "
                          "costs nothing at any scale, so there's no throughput reason to "
                          "change it. 1,500 was left as a safety margin against a "
                          "pathological non-converging seed, not tuned for speed.")
    ap.add_argument("--cell-cap", type=int, default=stage2.CPU_FLOOD_FILL_MAX_CELLS, metavar="INT",
                     help="Stage-2 rejection cap on touched cells -- MUST match the CPU "
                          "reference's FLOOD_FILL_MAX_CELLS to stay CPU-faithful. "
                          f"Default: {stage2.CPU_FLOOD_FILL_MAX_CELLS} (do not change lightly).")

    ap.add_argument("--double-buffer", action="store_true",
                     help="EXPERIMENTAL: overlap round k's CPU-bound tail (stage-3 CPU "
                          "verify + CSV write + proof images) with round k+1's GPU-bound "
                          "cascade, on a background thread. Retry of an earlier attempt "
                          "that was reverted after a CUDA crash -- that crash's real root "
                          "cause (an id()-keyed cache in noise_triton_fused.py colliding "
                          "after GC) is now fixed, so this is safe to re-test. Unlike the "
                          "earlier attempt, this only threads the CPU-only tail (not the "
                          "whole batch, including the GPU-bound stage 1.5/stage 2 work), "
                          "and strictly waits for round k's tail to finish before "
                          "submitting round k+1's, so at most one tail is ever in flight -- "
                          "no two calls can race on the shared candidates dict / counters. "
                          "One known cosmetic side effect: the printed 'total screened' / "
                          "'running rate' on a flush line can be very slightly ahead of "
                          "that flush's true state (main thread has moved on to the next "
                          "round's accounting by the time the tail prints) -- this affects "
                          "only log text, never the CSV or candidate correctness. Default: "
                          "off (synchronous, matches all prior measurements in this repo).")
    ap.add_argument("--no-cpu-verify", action="store_true",
                     help="Skip the mandatory CPU re-verification stage. FAST BUT UNSAFE: "
                          "the output will contain raw, UNVERIFIED GPU candidates and must "
                          "not be trusted as real islands until independently checked. "
                          "See README.md.")
    ap.add_argument("--cpu-verify-binary", default=cpu_verify.BINARY, metavar="PATH",
                     help="Path to the seed_finders/largest_island CLI binary.")
    ap.add_argument("--cpu-verify-concurrency", type=int, default=16, metavar="INT",
                     help="Max number of Stage-3 CPU-verify subprocess calls in flight at "
                          "once (ThreadPoolExecutor size). The CPU binary itself uses "
                          "--threads 1, so this is roughly the number of cores Stage 3 "
                          "will occupy at once. Default 16. This mainly matters when a "
                          "round has multiple GPU-confirmed candidates at once (natural "
                          "islands are rare, so most rounds have zero and this is a no-op); "
                          "measured 2.8x-8.6x speedup on rounds with several candidates, "
                          "see git history for the profiling behind this default.")
    ap.add_argument("--cpu-verify-timeout", type=int, default=180, metavar="SECONDS",
                     help="Timeout for a single Stage-3 CPU-verify subprocess call. RAISED "
                          "from 60 to 180 (2026-07-10): a live production run crashed on "
                          "subprocess.TimeoutExpired for a candidate near the 2,000,000-cell "
                          "flood-fill cap (see stages.hpp -- that cap was itself raised from "
                          "200,000 earlier, specifically because real islands were found near "
                          "it, so slow-but-legitimate large candidates are expected now, not "
                          "just hangs). A timeout here no longer crashes the process -- it's "
                          "logged to needs_recheck.txt next to --output and the run continues "
                          "-- but a candidate this happens to is GPU-CERTIFIED enclosed and "
                          "genuinely needs re-verification, not silent disposal.")

    ap.add_argument("--proof-images", dest="proof_images", action="store_true", default=True,
                     help="Generate a flood-fill proof PNG (image_generator/floodfill_proof) "
                          "for every CPU-confirmed real island, as it's found. ON by default. "
                          "Filename: FLOODFILL_PROOF_2x_bbox_area_<area:07d>_seed_<seed>.png -- "
                          "same convention used throughout .output/images/.")
    ap.add_argument("--no-proof-images", dest="proof_images", action="store_false",
                     help="Disable proof-image generation (e.g. for fast exploration runs).")
    ap.add_argument("--proof-images-dir", default=None, metavar="PATH",
                     help="Output directory for proof images. Default: <repo>/.output/images/.")
    ap.add_argument("--floodfill-proof-binary", default=None, metavar="PATH",
                     help="Path to the floodfill_proof CLI binary. Default: "
                          "<repo>/build/image_generator/floodfill_proof.")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="torch device string, e.g. 'cuda', 'cuda:0', 'cpu'.")

    return ap.parse_args()


def write_csv(path, candidates, verified):
    """
    candidates: dict seed -> (score, farthest_tile_dist, hit_border).
                score: tiles^2 area (CPU-exact if verified, GPU stage-2
                estimate otherwise). farthest_tile_dist: see
                island_extent.farthest_tile_distance; None if not computed
                (written as empty fields, e.g. for old-format preloaded rows).
    Rewrites the whole file, ranked by score descending -- cheap since the
    candidate set is always small (islands are rare) even at millions of
    seeds screened.
    """
    rows = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for rank, (seed, (score, farthest_dist, hit_border)) in enumerate(rows, start=1):
            dist_str = f"{farthest_dist:.1f}" if farthest_dist is not None else ""
            border_str = "" if hit_border is None else ("1" if hit_border else "0")
            f.write(f"{rank};{seed};{score:.4f};1;1;2.0;{dist_str};{border_str}\n")
    os.replace(tmp_path, path)


_AREA_RE = re.compile(r"area = (\d+) tiles\^2")
_HITCAP_RE = re.compile(r"hit_cap=(\S+)")


def generate_proof_image(seed, binary, out_dir, timeout=120):
    """
    Runs floodfill_proof for one CPU-confirmed seed and files the PNG under
    out_dir using the project's established naming convention:
    FLOODFILL_PROOF_2x_bbox_area_<area, zero-padded to 7 digits>_seed_<seed>.png

    Best-effort: prints a warning and returns without raising on any failure
    (binary missing/crashed, unexpected output, timeout) -- a missing proof
    image is never a reason to lose an otherwise-confirmed real island result.
    """
    tmp_path = os.path.join(out_dir, f".tmp_floodfill_proof_{seed}.png")
    try:
        result = subprocess.run(
            [binary, str(seed), tmp_path],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        m_area = _AREA_RE.search(result.stdout)
        m_hitcap = _HITCAP_RE.search(result.stdout)
        if not m_area or not m_hitcap or m_hitcap.group(1) != "no":
            print(f"WARNING: floodfill_proof produced unexpected output for seed {seed}, "
                  f"skipping its proof image:\n{result.stdout}", file=sys.stderr)
            return
        area = int(m_area.group(1))
        final_path = os.path.join(out_dir, f"FLOODFILL_PROOF_2x_bbox_area_{area:07d}_seed_{seed}.png")
        os.replace(tmp_path, final_path)
    except Exception as e:
        print(f"WARNING: proof-image generation failed for seed {seed} ({e!r}), "
              f"continuing without it.", file=sys.stderr)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.first_seed > args.last_seed:
        print("ERROR: --first-seed must be <= --last-seed", file=sys.stderr)
        sys.exit(1)
    if args.num_seeds <= 0:
        print("ERROR: --num-seeds must be positive", file=sys.stderr)
        sys.exit(1)

    range_size = args.last_seed - args.first_seed + 1
    if args.sequential and args.num_seeds > range_size:
        print(f"NOTE: --num-seeds ({args.num_seeds:,}) > seed range size ({range_size:,}); "
              f"clamping to the range size.")
        args.num_seeds = range_size

    print(f"device = {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    scan_desc = "uniform random sampling"
    if args.sequential:
        scan_desc = "sequential scan, DESCENDING" if args.reverse else "sequential scan"
    print(f"seed range: [{args.first_seed}, {args.last_seed}] ({scan_desc})")
    print(f"total seeds to screen: {args.num_seeds:,}")
    if args.stage1:
        print(f"stage 1 (ray screen): ENABLED as a REJECT gate, water-fraction threshold "
              f"{args.water_threshold} -- WARNING: this is an uncertified heuristic filter, "
              f"see --stage1 help. Not suitable for a complete-catalog production census.")
    else:
        print("stage 1 (ray screen): DISABLED (default) -- every seed goes straight to "
              "stage 1.5's certified cascade, no uncertified REJECT gate in front of it.")
    print(f"pillar pre-tier: {'ENABLED' if args.pillar else 'DISABLED'}")
    if args.stage15:
        print(f"stage-1.5 cascade: ENABLED -- pass1 node_spacing={stage15.PASS1_NODE_SPACING} "
              f"domain_radius_nodes={stage15.PASS1_DOMAIN_RADIUS_NODES} "
              f"(nominal {stage15.PASS1_NODE_SPACING * stage15.PASS1_DOMAIN_RADIUS_NODES} tiles), "
              f"pass2 (residual only) node_spacing={stage15.PASS2_NODE_SPACING} "
              f"domain_radius_nodes={stage15.PASS2_DOMAIN_RADIUS_NODES} "
              f"(nominal {stage15.PASS2_NODE_SPACING * stage15.PASS2_DOMAIN_RADIUS_NODES} tiles)")
        print(f"stage-1.5 invocation batching: accumulate shortlisted seeds across stage-1 "
              f"rounds and fire the cascade (+downstream stage2/verify/checkpoint) once the "
              f"accumulator reaches {args.stage15_batch_size:,} seeds (or input is exhausted) "
              f"-- see --stage15-batch-size help for why this is decoupled from --batch-size.")
    else:
        print("stage-1.5 cascade: DISABLED (--no-stage15) -- full stage-1 shortlist goes to stage 2")
    print(f"stage-2 grid: {2 * args.grid_radius_cells + 1}x{2 * args.grid_radius_cells + 1} cells "
          f"(+-{args.grid_radius_cells * stage2.FLOOD_FILL_STEP} tiles), cell cap {args.cell_cap:,}")
    if args.no_cpu_verify:
        print("\n*** --no-cpu-verify set: output will be UNVERIFIED GPU candidates. ***")
        print("*** Re-run seed_finders/largest_island on them before trusting any result. ***\n")
    else:
        print(f"CPU re-verification binary: {args.cpu_verify_binary}")
        print(f"CPU re-verification concurrency: {args.cpu_verify_concurrency}")
        if not os.path.exists(args.cpu_verify_binary):
            print(f"\nERROR: CPU verifier binary not found at {args.cpu_verify_binary}", file=sys.stderr)
            print("Build it first, e.g.:", file=sys.stderr)
            print("  cmake --build build --config Release --target largest_island", file=sys.stderr)
            sys.exit(1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.proof_images_dir is None:
        args.proof_images_dir = os.path.join(repo_root, ".output", "images")
    if args.floodfill_proof_binary is None:
        args.floodfill_proof_binary = os.path.join(repo_root, "build", "image_generator", "floodfill_proof")

    if args.proof_images:
        print(f"proof images: ENABLED -- {args.proof_images_dir} (binary: {args.floodfill_proof_binary})")
        if not os.path.exists(args.floodfill_proof_binary):
            print(f"\nERROR: floodfill_proof binary not found at {args.floodfill_proof_binary}", file=sys.stderr)
            print("Build it first, e.g.:", file=sys.stderr)
            print("  cmake --build build --config Release --target floodfill_proof", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.proof_images_dir, exist_ok=True)
    else:
        print("proof images: DISABLED (--no-proof-images)")

    candidates = {}  # seed -> (score tiles^2, farthest_tile_dist or None, hit_border or None)
    if os.path.exists(args.output):
        with open(args.output) as f:
            next(f, None)  # header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                # farthest_tile_dist/hit_border are new columns -- old-format rows
                # (pre-dating this change) simply won't have them; None means
                # "not computed", not "computed as zero".
                farthest_dist = float(parts[6]) if len(parts) > 6 and parts[6] != "" else None
                hit_border = (parts[7] == "1") if len(parts) > 7 and parts[7] != "" else None
                candidates[int(parts[1])] = (float(parts[2]), farthest_dist, hit_border)
        print(f"preloaded {len(candidates):,} existing candidate(s) from {args.output} "
              f"(continuing onto a prior run's results, not overwriting them)")

    current_max_area = max(v[0] for v in candidates.values()) if candidates else -1.0
    leader_viz_dir_base = os.path.join(repo_root, ".output", "images")

    total_screened = 0
    total_shortlisted = 0
    total_gpu_confirmed = 0
    cpu_checked = 0
    cpu_confirmed = 0
    flush_counter = 0

    # pillar pre-tier (stage1_5_pillar.py) -- cheap certified REJECT-only
    # pre-check ahead of the lattice cascade, see that module's docstring
    total_pillar_reject = 0
    total_pillar_time = 0.0

    # stage-1.5 cumulative counters/timing (all zero/no-op if --no-stage15)
    total_stage15_reject = 0
    total_stage15_confirm = 0
    total_stage15_defer = 0
    total_stage15_pass1_seeds = 0
    total_stage15_pass2_seeds = 0
    total_stage1_time = 0.0
    total_stage15_time = 0.0
    total_stage15_pass1_time = 0.0
    total_stage15_pass2_time = 0.0
    total_stage2_time = 0.0
    total_verify_time = 0.0

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.rng_seed)
    # Descending scan starts its cursor at last_seed and each round's chunk is
    # [seq_cursor - round_size + 1, seq_cursor] -- see the round loop below.
    seq_cursor = args.last_seed if args.reverse else args.first_seed

    n_rounds = (args.num_seeds + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    # One executor for the whole run (cheap: threads, not processes; avoids
    # pool-startup cost every round). shutdown() happens in `finally` below.
    # Stage 3 candidates rarely arrive more than one at a time (natural
    # islands are rare -- see project notes), so most rounds this pool has
    # 0-1 tasks and is effectively a no-op; it only pays off on the rare
    # round with several GPU-confirmed candidates at once.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.cpu_verify_concurrency),
        thread_name_prefix="cpu_verify",
    )

    # Only used under --double-buffer; single worker because at most one
    # cpu_tail() call is ever submitted before the previous one is awaited
    # (see flush_batch() below) -- this is a strict 1-deep pipeline, not a
    # queue, so a bigger pool would buy nothing and would just make it easier
    # to accidentally violate the one-in-flight invariant.
    cpu_tail_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cpu_tail",
    )
    pending_tail_future = None

    # ---- stage-1.5 accumulator ----------------------------------------------
    # Stage 1 keeps running in --batch-size chunks (unrelated to the bug below,
    # left alone). Shortlisted seeds from each stage-1 round are appended here
    # instead of being fed to stage 1.5 immediately; stage 1.5 (and, downstream
    # of it, stage 2 / CPU-verify / the CSV checkpoint write) only fires once
    # this accumulator reaches --stage15-batch-size seeds, or the input is
    # exhausted (final partial flush below, so no accumulated seed is ever
    # silently dropped). This is the fix for the diagnosed bug: calling
    # stage1_5_cascade.run_cascade -> ... -> noise_gpu.build_all_tables once
    # per ~1,000-1,200-seed stage-1 round pays that function's ~1,020-launch,
    # batch-size-insensitive fixed cost on a batch far too small to amortize
    # it; accumulating first means that fixed cost is paid once per
    # --stage15-batch-size-sized batch instead of once per stage-1 round.
    # When --stage15 is disabled there is nothing to amortize (that A/B
    # baseline never had this cost), so each round is flushed immediately,
    # exactly reproducing the pre-cascade pipeline's cadence.
    accum_chunks = []
    accum_count = 0
    accum_screened = 0
    accum_rounds = 0

    def gpu_phase(batch_seeds):
        """
        Runs stage 1.5 (if enabled) -> stage 2 on `batch_seeds` (a stage-1
        shortlist). Touches ONLY the GPU/stage1.5/stage2-side nonlocal
        counters (disjoint from the CPU-tail's counters below), so this is
        safe to run on the main thread concurrently with a background
        cpu_tail() call for the PREVIOUS batch -- see --double-buffer help.
        Returns a plain dict of everything cpu_tail() and the summary print
        need; no shared mutable state is handed across the thread boundary
        except via this dict's own (immutable-once-returned) values.
        """
        nonlocal total_stage15_reject, total_stage15_confirm, total_stage15_defer
        nonlocal total_stage15_pass1_seeds, total_stage15_pass2_seeds
        nonlocal total_stage15_pass1_time, total_stage15_pass2_time, total_stage15_time
        nonlocal total_stage2_time
        nonlocal total_pillar_reject, total_pillar_time

        n_batch = int(batch_seeds.shape[0])

        # ---------------- pillar pre-tier (cheap, certified REJECT-only) ----------------
        # See stage1_5_pillar.py's docstring: a single wide dense East-facing
        # strip, measured ~26% faster end-to-end than sending every seed
        # straight into the lattice cascade. REJECT here is exactly as sound
        # as the lattice cascade's own REJECT (same oracle.classify_edges,
        # same per-tile EPSILON_FP margin) -- seeds it doesn't resolve are
        # forwarded to the lattice cascade below completely unchanged, not
        # given any special treatment there.
        t_pillar_start = time.time()
        if args.pillar and n_batch > 0:
            pillar_tables = stage15_oracle.build_tables(batch_seeds)
            pillar_rejected = pillar.pillar_reject(batch_seeds, pillar_tables, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            n_pillar_reject = int(pillar_rejected.sum())
            batch_seeds = batch_seeds[~pillar_rejected]
        else:
            n_pillar_reject = 0
        t_pillar_end = time.time()
        total_pillar_reject += n_pillar_reject
        total_pillar_time += (t_pillar_end - t_pillar_start)

        # ---------------- stage 1.5: certified segment-lattice cascade ----------------
        # (64-tile pass over the whole batch, then a 32-tile pass over only the
        # residual seeds that pass didn't REJECT -- see stage1_5_cascade.py). REJECT
        # from either pass is airtight (no CPU backstop) and drops the seed entirely;
        # CONFIRM skips stage 2 and goes straight to stage 3 (CPU verify); DEFER falls
        # through to stage 2 below, unchanged.
        stage15_reject_n = stage15_confirm_n = stage15_defer_n = 0
        timing15_1 = timing15_2 = None
        stage15_confirmed_seeds = batch_seeds[0:0]  # empty tensor, correct dtype/device
        seeds_for_stage2 = batch_seeds
        t1_5_start = t1_5_end = time.time()
        if args.stage15 and n_batch > 0:
            t1_5_start = time.time()
            if args.stage15_four_tier:
                stage15_tiers = stage15.FOUR_TIER_CASCADE
            elif args.stage15_three_tier:
                stage15_tiers = stage15.THREE_TIER_CASCADE
            else:
                stage15_tiers = None
            casc = stage15.run_cascade(
                batch_seeds, device, verbose=False,
                backend=args.stage15_backend, parallel_cpu=args.stage15_parallel_cpu,
                tiers=stage15_tiers,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1_5_end = time.time()

            verdict15 = casc["verdict"]
            reject_mask15_np = verdict15 == stage15.REJECT
            confirm_mask15_np = verdict15 == stage15.CONFIRM
            defer_mask15_np = verdict15 == stage15.DEFER

            stage15_reject_n = int(reject_mask15_np.sum())
            stage15_confirm_n = int(confirm_mask15_np.sum())
            stage15_defer_n = int(defer_mask15_np.sum())
            timing15_1, timing15_2 = casc["timing1"], casc["timing2"]

            confirm_mask15 = torch.as_tensor(confirm_mask15_np, device=device)
            defer_mask15 = torch.as_tensor(defer_mask15_np, device=device)
            stage15_confirmed_seeds = batch_seeds[confirm_mask15]

            if args.no_cpu_verify:
                # --no-cpu-verify still needs a GPU area estimate for anything
                # reported (stage 1.5 doesn't produce one), so route its CONFIRMs
                # through stage 2 too in that mode -- cheap, this candidate count
                # is tiny (islands are rare).
                stage2_input_mask = confirm_mask15 | defer_mask15
            else:
                stage2_input_mask = defer_mask15
            seeds_for_stage2 = batch_seeds[stage2_input_mask]

        total_stage15_reject += stage15_reject_n
        total_stage15_confirm += stage15_confirm_n
        total_stage15_defer += stage15_defer_n
        if timing15_1 is not None:
            total_stage15_pass1_seeds += casc["n_pass1"]
            total_stage15_pass2_seeds += casc["n_pass2"]
            total_stage15_pass1_time += timing15_1["gpu_seconds"] + timing15_1["cpu_seconds"]
            total_stage15_pass2_time += timing15_2["gpu_seconds"] + timing15_2["cpu_seconds"]
        total_stage15_time += (t1_5_end - t1_5_start)

        # ---------------- stage 2 ----------------
        n_for_stage2 = int(seeds_for_stage2.shape[0])
        gpu_confirmed_seeds, gpu_confirmed_areas = [], []
        if n_for_stage2 > 0:
            land_mask = stage2.build_land_mask(
                seeds_for_stage2, grid_radius_cells=args.grid_radius_cells,
                chunk_size=args.stage2_chunk_size,
            )
            s2_res = stage2.flood_fill(land_mask, max_steps=args.max_flood_steps,
                                        cell_cap=args.cell_cap)
            if device.type == "cuda":
                torch.cuda.synchronize()

            confirmed_mask = s2_res["confirmed_island"]
            gpu_confirmed_seeds = seeds_for_stage2[confirmed_mask].cpu().tolist()
            gpu_confirmed_areas = s2_res["area_tiles2"][confirmed_mask].cpu().tolist()

        # Fold in stage-1.5 CONFIRMs that skipped stage 2 entirely (only when CPU
        # verify is the actual downstream gate -- see the no-cpu-verify branch above,
        # where they were routed through stage 2 instead). Area is a placeholder
        # (unused: cpu_verify.py computes the real area itself); it is only ever
        # read from gpu_confirmed_areas in the --no-cpu-verify branch below, which
        # never applies to these seeds.
        if args.stage15 and not args.no_cpu_verify and stage15_confirmed_seeds.shape[0] > 0:
            extra_seeds = [int(s) for s in stage15_confirmed_seeds.cpu().tolist()]
            gpu_confirmed_seeds = list(gpu_confirmed_seeds) + extra_seeds
            gpu_confirmed_areas = list(gpu_confirmed_areas) + [0.0] * len(extra_seeds)
        t2 = time.time()
        total_stage2_time += (t2 - t1_5_end)

        return {
            "n_batch": n_batch, "n_pillar_reject": n_pillar_reject,
            "t_pillar_start": t_pillar_start, "t_pillar_end": t_pillar_end,
            "t1_5_start": t1_5_start, "t1_5_end": t1_5_end, "t2": t2,
            "stage15_reject_n": stage15_reject_n, "stage15_confirm_n": stage15_confirm_n,
            "stage15_defer_n": stage15_defer_n,
            "gpu_confirmed_seeds": gpu_confirmed_seeds, "gpu_confirmed_areas": gpu_confirmed_areas,
        }

    def cpu_tail(gpu_result, batch_screened, batch_rounds, round_label):
        """
        Runs stage 3 (CPU verify) + prints the flush summary + checkpoints
        the CSV, using a completed gpu_phase() result. Touches ONLY the
        CPU-tail-side nonlocal counters (disjoint from gpu_phase's), and
        under --double-buffer is only ever invoked from ONE place at a time
        (the round loop always awaits the previous tail before submitting a
        new one -- see --double-buffer help) -- so even though it CAN run
        concurrently with the NEXT round's gpu_phase on the main thread,
        it never runs concurrently with ANOTHER cpu_tail call, and the two
        phases share no mutable state, so there is no race.
        """
        nonlocal total_verify_time, total_gpu_confirmed
        nonlocal cpu_checked, cpu_confirmed, flush_counter
        nonlocal current_max_area

        flush_counter += 1
        n_batch = gpu_result["n_batch"]
        n_pillar_reject = gpu_result["n_pillar_reject"]
        t_pillar_start, t_pillar_end = gpu_result["t_pillar_start"], gpu_result["t_pillar_end"]
        t1_5_start, t1_5_end, t2 = gpu_result["t1_5_start"], gpu_result["t1_5_end"], gpu_result["t2"]
        stage15_reject_n = gpu_result["stage15_reject_n"]
        stage15_confirm_n = gpu_result["stage15_confirm_n"]
        stage15_defer_n = gpu_result["stage15_defer_n"]
        gpu_confirmed_seeds = gpu_result["gpu_confirmed_seeds"]
        gpu_confirmed_areas = gpu_result["gpu_confirmed_areas"]

        # ---------------- stage 3: mandatory CPU re-verification ----------------
        # Concurrent (bounded thread pool) rather than one-subprocess-at-a-time:
        # each cpu_verify_seed() call blocks on subprocess.run with the GIL
        # released, and each call uses its own tempfile / argv list with no
        # shared mutable state, so submitting a batch's whole candidate set
        # at once is safe. The candidates dict / counters are only ever
        # mutated here in the main thread, inside the as_completed loop below.
        batch_new_candidates = 0
        if not args.no_cpu_verify:
            futures = {}
            for sd, _gpu_area in zip(gpu_confirmed_seeds, gpu_confirmed_areas):
                sd_int = int(sd)
                fut = executor.submit(cpu_verify.cpu_verify_seed, sd_int,
                                       binary=args.cpu_verify_binary, timeout=args.cpu_verify_timeout)
                futures[fut] = sd_int
            for fut in concurrent.futures.as_completed(futures):
                sd_int = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    # By the time a seed reaches CPU-verify, the GPU cascade has
                    # ALREADY certified (full-frontier, zero-pruning exhaustion)
                    # that it doesn't escape the ring -- it IS an island by that
                    # proof. A timeout/crash here means we failed to get its
                    # exact area, NOT that it isn't real -- silently dropping it
                    # would be exactly the "silent, unrecoverable false reject"
                    # this project's correctness culture is built to avoid (see
                    # CLAUDE.md). Log loudly and persist it for re-verification
                    # (e.g. with a longer --cpu-verify-timeout) instead of
                    # crashing the whole run or pretending it was resolved.
                    print(f"*** CPU-VERIFY FAILED for seed {sd_int} ({e!r}) -- this seed is "
                          f"GPU-CERTIFIED enclosed (full-frontier exhaustion) but its exact "
                          f"area could not be measured. NOT counted as confirmed, NOT safe to "
                          f"assume mainland -- needs re-verification, e.g. with a longer "
                          f"--cpu-verify-timeout. ***", file=sys.stderr)
                    needs_recheck_path = os.path.join(os.path.dirname(args.output), "needs_recheck.txt")
                    with open(needs_recheck_path, "a") as f:
                        f.write(f"{sd_int}\t{e!r}\n")
                    cpu_checked += 1
                    continue
                cpu_checked += 1
                if r["is_island"]:
                    # farthest_tile_dist: real GPU flood-fill, only ever run on
                    # CPU-verified real islands (rare) -- see island_extent.py.
                    # Best-effort: never let this crash a confirmed real find.
                    try:
                        farthest_dist, hit_border = island_extent.farthest_tile_distance(sd_int, device)
                    except Exception as e:
                        print(f"WARNING: island_extent.farthest_tile_distance failed for seed {sd_int} "
                              f"({e!r}); recording without it.")
                        farthest_dist, hit_border = None, None
                    candidates[sd_int] = (r["area_tiles2"], farthest_dist, hit_border)
                    cpu_confirmed += 1
                    batch_new_candidates += 1
                    if args.proof_images:
                        generate_proof_image(sd_int, args.floodfill_proof_binary, args.proof_images_dir)
                    if r["area_tiles2"] > current_max_area:
                        current_max_area = r["area_tiles2"]
                        print(f"*** NEW RECORD: seed {sd_int} area {r['area_tiles2']:,.0f} tiles^2, "
                              f"farthest tile {farthest_dist and f'{farthest_dist:,.0f}'} tiles -- "
                              f"generating pipeline-stage visualization ***")
                        leader_viz_dir = os.path.join(leader_viz_dir_base, f"pipeline_stages_seed_{sd_int}")
                        leader_visualizer.generate_leader_visualization(
                            sd_int, r["area_tiles2"], leader_viz_dir, device)
        else:
            for sd, gpu_area in zip(gpu_confirmed_seeds, gpu_confirmed_areas):
                candidates[int(sd)] = (float(gpu_area), None, None)
                batch_new_candidates += 1
        t3 = time.time()
        total_verify_time += (t3 - t2)
        total_gpu_confirmed += len(gpu_confirmed_seeds)

        elapsed = time.time() - t_start
        rate = total_screened / elapsed if elapsed > 0 else 0.0
        if args.stage15:
            stage15_str = (
                f" | s1.5: rej {stage15_reject_n} conf {stage15_confirm_n} defer {stage15_defer_n}"
            )
        else:
            stage15_str = ""
        print(
            f"[{round_label}] STAGE1.5 FLUSH #{flush_counter}: {n_batch:,} shortlisted seed(s) "
            f"accumulated over {batch_rounds} stage-1 round(s) ({batch_screened:,} raw seeds) | "
            f"pillar {t_pillar_end - t_pillar_start:.2f}s (rej {n_pillar_reject}), "
            f"stage1.5 {t1_5_end - t1_5_start:.2f}s, stage2 {t2 - t1_5_end:.2f}s, verify {t3 - t2:.2f}s"
            f"{stage15_str} | gpu-confirmed {len(gpu_confirmed_seeds)} | "
            f"new candidates {batch_new_candidates} | total screened {total_screened:,}/{args.num_seeds:,} | "
            f"running rate {rate:,.0f} seeds/sec ({rate / CPU_BASELINE_SEEDS_PER_SEC:.1f}x CPU baseline)",
            flush=True,
        )
        write_csv(args.output, candidates, verified=not args.no_cpu_verify)

    def flush_batch(batch_seeds, batch_screened, batch_rounds, round_label):
        """
        Runs gpu_phase() synchronously (main thread), then either runs
        cpu_tail() synchronously too (default) or, under --double-buffer,
        waits for any PREVIOUSLY submitted tail to finish before submitting
        this one to the background cpu_tail_executor without waiting for it.
        That wait-before-submit is what keeps at most one tail in flight --
        by the time we get back here for round k+1, round k's gpu_phase
        (several seconds) has already given round k-1's tail (a fraction of
        a second, when there's anything to verify at all) ample time to
        finish, so the wait is normally a no-op and the overlap is free.
        """
        nonlocal pending_tail_future
        gpu_result = gpu_phase(batch_seeds)
        if args.double_buffer:
            if pending_tail_future is not None:
                pending_tail_future.result()
            pending_tail_future = cpu_tail_executor.submit(
                cpu_tail, gpu_result, batch_screened, batch_rounds, round_label)
        else:
            cpu_tail(gpu_result, batch_screened, batch_rounds, round_label)

    try:
        for round_idx in range(n_rounds):
            round_size = min(args.batch_size, args.num_seeds - total_screened)
            if round_size <= 0:
                break

            if args.sequential and args.reverse:
                # descending: this round covers [seq_cursor - round_size + 1, seq_cursor]
                seeds_cpu = torch.arange(seq_cursor, seq_cursor - round_size, -1, dtype=torch.int64)
                seq_cursor -= round_size
            elif args.sequential:
                seeds_cpu = torch.arange(seq_cursor, seq_cursor + round_size, dtype=torch.int64)
                seq_cursor += round_size
            else:
                seeds_cpu = torch.randint(args.first_seed, args.last_seed + 1, (round_size,),
                                           dtype=torch.int64, generator=gen)
            seeds = seeds_cpu.to(device)

            # ---------------- stage 1 ----------------
            t0 = time.time()
            if args.stage1:
                s1_res = stage1.screen_seeds(
                    seeds, n_angles=args.n_angles, max_radius=args.max_radius,
                    radius_step=args.radius_step, water_fraction_threshold=args.water_threshold,
                    chunk_size=args.stage1_chunk_size,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                shortlist_mask = s1_res["shortlist"]
                shortlist_seeds = seeds[shortlist_mask]
            else:
                # No REJECT gate before the certified stage -- every seed must reach
                # stage 1.5 for a complete, no-island-missed catalog (see --stage1 help).
                shortlist_seeds = seeds
            t1 = time.time()

            n_shortlist = int(shortlist_seeds.shape[0])

            total_screened += round_size
            total_shortlisted += n_shortlist
            total_stage1_time += (t1 - t0)

            if n_shortlist > 0:
                accum_chunks.append(shortlist_seeds)
            accum_count += n_shortlist
            accum_screened += round_size
            accum_rounds += 1

            is_last_round = (round_idx == n_rounds - 1) or (total_screened >= args.num_seeds)
            if not args.stage15:
                # No accumulation cost to amortize -- flush every round, exactly
                # reproducing the pre-cascade pipeline's per-round cadence.
                flush_now = True
            else:
                flush_now = accum_count >= args.stage15_batch_size or (is_last_round and accum_count > 0)

            if flush_now:
                batch_seeds = (
                    torch.cat(accum_chunks) if accum_chunks
                    else torch.zeros(0, dtype=torch.int64, device=device)
                )
                # See --double-buffer help for the history here: an earlier
                # whole-function-threaded attempt crashed (traced to an
                # unrelated id()-cache bug, since fixed) and was reverted.
                # flush_batch() re-attempts a narrower, race-free version --
                # only the CPU-only tail is ever threaded, strictly one in
                # flight at a time -- gated behind this opt-in flag.
                flush_batch(batch_seeds, accum_screened, accum_rounds,
                            round_label=f"round {round_idx + 1}/{n_rounds}")
                accum_chunks = []
                accum_count = 0
                accum_screened = 0
                accum_rounds = 0
            else:
                elapsed = time.time() - t_start
                rate = total_screened / elapsed if elapsed > 0 else 0.0
                print(
                    f"[stage1 round {round_idx + 1}/{n_rounds}] screened {total_screened:,}/{args.num_seeds:,} | "
                    f"stage1 {t1 - t0:.2f}s | shortlist {n_shortlist} "
                    f"({100.0 * n_shortlist / max(round_size, 1):.2f}%) | "
                    f"stage1.5 accumulator {accum_count:,}/{args.stage15_batch_size:,} seed(s) "
                    f"({accum_rounds} round(s) since last flush) | "
                    f"running rate {rate:,.0f} seeds/sec ({rate / CPU_BASELINE_SEEDS_PER_SEC:.1f}x CPU baseline)",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(f"\nInterrupted -- {accum_count:,} accumulated stage-1-shortlisted seed(s) "
              f"({accum_rounds} round(s) since the last stage-1.5 flush) have not yet been "
              "run through stage 1.5/stage 2/CPU-verify. Attempting one best-effort flush "
              "of them before writing out final candidates.")
        # Wait for (and stop tracking) any background tail BEFORE running the
        # interrupt-time flush below -- that flush calls cpu_tail() directly
        # and synchronously (simplest, most deterministic under an interrupt),
        # and it must not run concurrently with a still-in-flight background
        # tail, since both touch the same candidates dict / counters.
        if pending_tail_future is not None:
            try:
                pending_tail_future.result()
            except Exception as e:
                print(f"Previously-submitted background tail raised ({e!r}); continuing best-effort.")
            pending_tail_future = None
        if accum_count > 0:
            try:
                batch_seeds = torch.cat(accum_chunks)
                gpu_result = gpu_phase(batch_seeds)
                cpu_tail(gpu_result, accum_screened, accum_rounds, round_label="interrupt flush")
            except KeyboardInterrupt:
                print("Second interrupt during the flush -- abandoning it; "
                      "writing out only previously-completed flushes' candidates.")
            except Exception as e:
                print(f"Flush during interrupt handling failed ({e!r}); "
                      "writing out only previously-completed flushes' candidates.")
        print("Writing out whatever candidates were found so far.")
        write_csv(args.output, candidates, verified=not args.no_cpu_verify)
    finally:
        # Normal-completion path: pending_tail_future (if any) hasn't been
        # awaited yet at this point, so this is where the LAST batch's
        # background tail is guaranteed to finish before final stats print.
        # Interrupt path: already None (handled above); this is a no-op.
        if pending_tail_future is not None:
            try:
                pending_tail_future.result()
            except Exception as e:
                print(f"Background tail raised ({e!r}) during shutdown.")
        # cancel_futures=True: on interrupt, don't block waiting for
        # in-flight subprocess.run calls that we no longer care about.
        # (Python 3.9+; if older, plain shutdown() blocks until they finish,
        # which is still correct, just not instant.)
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
        cpu_tail_executor.shutdown(wait=True)

    total_time = time.time() - t_start
    print("\n=== DONE ===")
    print(f"screened {total_screened:,} seeds in {total_time:.1f}s "
          f"({total_screened / max(total_time, 1e-9):,.0f} seeds/sec, "
          f"{total_screened / max(total_time, 1e-9) / CPU_BASELINE_SEEDS_PER_SEC:.1f}x CPU baseline)")
    print(f"stage-1 shortlist total: {total_shortlisted:,} "
          f"({100.0 * total_shortlisted / max(total_screened, 1):.3f}% of screened seeds)")
    print(f"per-stage wall-clock: stage1 {total_stage1_time:.1f}s, pillar {total_pillar_time:.1f}s "
          f"(rej {total_pillar_reject:,}), stage1.5 {total_stage15_time:.1f}s "
          f"(pass1 {total_stage15_pass1_time:.1f}s over {total_stage15_pass1_seeds:,} seeds, "
          f"pass2 {total_stage15_pass2_time:.1f}s over {total_stage15_pass2_seeds:,} residual seeds), "
          f"stage2 {total_stage2_time:.1f}s, verify {total_verify_time:.1f}s")
    print(f"stage-1.5/stage-2/verify ran in {flush_counter} batch(es) "
          f"(target size --stage15-batch-size={args.stage15_batch_size:,}"
          f"{', one flush per stage-1 round since --stage15 is disabled' if not args.stage15 else ''})")
    if args.stage15:
        stage15_total = total_stage15_reject + total_stage15_confirm + total_stage15_defer
        print(f"stage-1.5 verdicts (of {stage15_total:,} shortlisted seeds fed to the cascade): "
              f"REJECT {total_stage15_reject:,} ({100.0 * total_stage15_reject / max(stage15_total, 1):.2f}%), "
              f"CONFIRM {total_stage15_confirm:,} ({100.0 * total_stage15_confirm / max(stage15_total, 1):.2f}%), "
              f"DEFER {total_stage15_defer:,} ({100.0 * total_stage15_defer / max(stage15_total, 1):.2f}%)")
    print(f"GPU-confirmed total (stage 2 + stage-1.5 CONFIRMs, forwarded to CPU verify): "
          f"{total_gpu_confirmed:,}")
    if not args.no_cpu_verify:
        precision = 100.0 * cpu_confirmed / cpu_checked if cpu_checked else 0.0
        print(f"CPU-verified: {cpu_confirmed} real islands out of {cpu_checked} GPU candidates "
              f"checked ({precision:.1f}% precision)")
        print(f"All {len(candidates)} row(s) in {args.output} are CPU-CONFIRMED real natural islands.")
    else:
        print(f"--no-cpu-verify was set: all {len(candidates)} row(s) in {args.output} are "
              f"UNVERIFIED GPU candidates -- re-run seed_finders/largest_island on them before "
              f"trusting any result!")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
