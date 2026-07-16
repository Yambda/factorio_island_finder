"""
Stage 1.5 GPU terrain oracle -- a STATELESS classifier over 32-tile LATTICE
EDGES ("segments"), not over 32x32 AREAS ("cells"/"blocks").

THIS IS THE THIRD DESIGN FOR STAGE 1.5, NOT THE SECOND (read this before
trusting anything below)
--------------------------------------------------------------------------
Design 1 (see stage1_5_witness.py's docstring): a bounded-K beam search at
TILE resolution. Abandoned: an UNSOUND CONFIRM certificate, because the
top-K cut can silently evict a real frontier cell forever, so "frontier
empty" no longer means "nowhere left to go" -- it can mean "we threw away
the only way out". (CONFIRM there is still safe only because it is backed
by the mandatory CPU verify; the beam itself does not certify enclosure.)

Design 2 (the version this file REPLACES, git history only -- do not resurrect
it): classified 32x32-tile BLOCKS via a sparse 3x3 (9-point) minimum-elevation
sub-sample per block, on a block graph identical in spirit to a coarse grid.
That design measured a validated 99.4-99.7% REJECT rate on real mainland
batches. It is DISCARDED here not because it was unsound (it wasn't -- REJECT
there was airtight for the same EPSILON_FP reason this design's is) but
because the project owner specified a more precise abstraction: a GRAPH of
lattice-point NODES connected by 1-DIMENSIONAL, fully/gaplessly-sampled EDGES,
not a grid of AREAS sampled at 9 scattered points out of ~1,024 tiles. Do NOT
assume design 2's measured numbers (REJECT rate, cells/sec, timing) carry
over to this design -- they are RE-MEASURED fresh in this module's own
test-run reports (see this project's stage1_5 test-run scripts), because a
segment graph touches nearly 4x the tile-samples per unit of graph explored
(32 tiles/edge here vs 9 samples/block there) and, per this design's own
EPSILON_FP CAVEAT section below, is a *stricter* traversability test that is
expected to walk through less of a real coastline per graph-edge than the
old design's looser 9-point block test did.

THIS MODULE'S ONLY JOB: given a flattened batch of (seed, base_i, base_j,
dir_code) EDGE requests -- possibly from many different "live" seeds'
solvers, all concatenated into one worklist for one round -- return each
edge's classification. It has no notion of "rounds", "frontiers", or "seeds
still active"; all of that state lives on the CPU side (stage1_5_solver.py).
This mirrors CLAUDE.md's target architecture ("GPU is a pure terrain
oracle... CPU threads run the graph solver... where all verdicts are
issued").

LATTICE / GRAPH GEOMETRY
--------------------------------------------------------------------------
NODE_SPACING = 32 tiles. NODES are lattice points at tile coordinates
(NODE_SPACING*i, NODE_SPACING*j) for integer (i, j) -- node (0, 0) is exactly
spawn's own tile (0, 0).

DOMAIN_RADIUS_NODES = 62 -> a 125x125 = 15,625-node domain, nominal radius
62*32 = 1,984 tiles. This matches the project owner's own back-of-envelope
"about a 16,000 node graph" almost exactly (15,625). As with the discarded
block design, this is deliberately bigger than the definitional enclosure
ring (RING_TILES = 1400) to give the exhaustion certificate headroom to run:
REJECT below fires strictly at true Euclidean distance >= 1400 tiles from
spawn; 1,984 is just how far the search domain extends before the CPU
solver's own domain-exhaustion safety valve (DEFER) kicks in.

EDGES ("segments") connect each node to its 4 ORTHOGONAL neighbors ONLY --
node (i, j) to (i+1, j), (i-1, j), (i, j+1), (i, j-1). NO DIAGONAL neighbors:
this project has already found and fixed a real correctness bug elsewhere
(stage2_floodfill.py / stage1_5_witness.py both document it) from allowing
diagonal moves that can cross a single-cell diagonal water chain that a real
4-connected walk could never cross. 4-connectivity here is the same
non-negotiable requirement, for the same reason.

A segment is identified by (base_i, base_j, dir_code) -- dir_code in
{0, 1, 2, 3} meaning +x, -x, +y, -y (see DIR_DI/DIR_DJ below) -- and consists
of the SEGMENT_TILES=32 individual tile coordinates
    (base_i*NODE_SPACING + o*DIR_DI[dir_code], base_j*NODE_SPACING + o*DIR_DJ[dir_code])
for o = 0, 1, ..., 31 (i.e. starting AT the base node's own tile, offset 0,
and running 31 more tiles toward -- but not quite reaching -- the neighbor
node, which sits at offset 32). ALL 32 of those tiles are sampled -- not a
sparse subsample -- which is the entire point of this design over the
discarded block design's sparse 9-of-~1,024 2D coverage: along the exact
1-dimensional line this module tests, there is NO sampling gap whatsoever.

CLASSIFICATION -- BINARY, PER THE SPEC (no AMBIGUOUS/CERTIFIED_WATER split)
--------------------------------------------------------------------------
    EDGE_LAND    : min(32 sampled elevations) >  EPSILON_FP  (+3.0)
    EDGE_BLOCKED : otherwise (min elevation <= EPSILON_FP, i.e. at least one
                   of the 32 tiles is not certified land -- whether that
                   tile is deep water, thinly negative, or just ambiguous
                   makes no difference: an uncertified tile anywhere along
                   the line is a WALL for this graph's purposes, full stop).

Only two states, unlike the discarded block design's 3-way CERTIFIED_LAND /
CERTIFIED_WATER / AMBIGUOUS split: that split existed there purely to report
a diagnostic "closure margin" signal, and this design's own spec calls for
exactly the binary EDGE_LAND/EDGE_BLOCKED framing quoted above. min_elev is
still returned (float) so a caller CAN reconstruct a closure-margin
diagnostic from it if desired -- the module just doesn't pre-bucket it into
a third state.

SOUNDNESS ARGUMENT FOR REJECT (read this before trusting the CPU-backstop-
free REJECT path -- mirrors stage1_5_witness.py's CERTIFIED LAND argument,
restated here for a chain of edges instead of a single point)
--------------------------------------------------------------------------
For a single sampled point: GPU_elevation > EPSILON_FP => CPU_elevation > 0
(real land), because CPU_elevation >= GPU_elevation - EPSILON_FP >
EPSILON_FP - EPSILON_FP = 0, where EPSILON_FP = 3.0 is the empirically
measured worst-case discrepancy between noise_gpu.py's exp2f/log2f-based
elevation approximation and the exact CPU reference (core/noise.cpp) --
~12.24M matched points found max |GPU-CPU| ~= 2.579 in a dedicated
120,000-seed adversarial search for the worst case of the underlying bounded
quantity (see stage1_5_witness.py's MEASURED NUMBERS section for the full
provenance chain; not re-derived here).

For a segment: EDGE_LAND means ALL 32 of its individually-sampled tiles
independently satisfy that inequality, so all 32 are REAL land at REAL,
specific, known tile coordinates -- not "probably land based on a nearby
sample", every single one of the 32 tiles on that exact line.

For a CHAIN of segments: this module's caller (stage1_5_solver.py) only ever
walks from an already-reached node OUTWARD along an edge based at that node
(see NODE REACHABILITY below for exactly why this is enough and why testing
only one direction per candidate node is still safe). Consecutive segments
in a walked chain partition the underlying straight tile-line with NO gaps
and NO overlaps: segment (base=(i,j), dir=+x) covers tiles
[32i, 32i+31] x {32j}; if the walk continues from node (i+1, j) onward via
segment (base=(i+1,j), dir=+x), that one covers [32i+32, 32i+63] x {32j} --
picking up EXACTLY at the next integer tile with nothing skipped, nothing
double-sampled. At an L-shaped turn (continuing in a new axis from a pivot
node), the same holds: the pivot node's own tile is covered by whichever of
its two "outgoing" edges (dir=+x or dir=+y from it) the walk actually takes,
and that edge's turn-corner tile is still 4-connected-adjacent to the
previous edge's last tile (see the worked example in this module's original
design notes / task derivation -- omitted here for length, but it is a
straightforward consequence of both segments meeting at the same lattice
point). So a chain of EDGE_LAND segments from spawn's node out to any
segment whose OWN sampled max_dist (see below) is >= RING_TILES constitutes
one single, real, physical, unbroken 4-connected line of individually
certified-land tiles, all the way out -- exactly the same kind of witness
stage1_5_witness.py's tile-resolution beam produced, just discovered via a
coarser (and therefore much cheaper) lattice-restricted search. This makes
REJECT airtight with NO CPU backstop needed, for exactly the same reason
EPSILON_FP is airtight everywhere else in this pipeline.

`max_dist` here is the max Euclidean tile-distance from spawn among the 32
ACTUALLY-SAMPLED tiles of a segment -- NOT the nominal, possibly-unsampled
neighbor-node coordinate at offset 32. This is deliberate and load-bearing:
REJECT must only ever point at a tile that was itself individually verified
land, never at an inferred/nominal endpoint. Since a segment's 32 samples run
from its base node's own tile out to 31 tiles toward the neighbor (i.e. to
within 1 tile of the full 32-tile hop), this loses essentially none of the
ring-crossing detection power in practice.

NODE REACHABILITY -- WHY TESTING ONLY ONE INCOMING DIRECTION PER NODE IS SAFE
--------------------------------------------------------------------------
stage1_5_solver.py's wave BFS marks a candidate node "touched" (claimed) the
instant it is enqueued as the target of SOME already-reached neighbor's
outgoing edge, and never re-requests it via a different neighbor even if one
exists. This means at most ONE of a node's up-to-4 possible incoming edges is
ever actually tested. This is intentional and SAFE, not an oversight: an
untested or EDGE_BLOCKED incoming direction can only ever cause a node (and
everything beyond it) to be treated as unreached when a real land connection
might in fact exist via a DIFFERENT direction -- i.e. it can only under-
connect the graph, which can only ever cost a missed REJECT opportunity
(falling through to CONFIRM or DEFER, both CPU-backstopped and therefore
safe). It can never manufacture a connection that isn't real, because
REJECT is only ever claimed from segments that were actually tested and
found EDGE_LAND. See stage1_5_solver.py's module docstring for the full
argument and the "EPSILON_FP CAVEAT" section below for a related, distinct
concern about the graph's overall SENSITIVITY (not soundness).

EPSILON_FP CAVEAT -- READ BEFORE ASSUMING THIS RESOLVES MOST MAINLANDS
CHEAPLY (honest self-assessment, not a hedge to be skipped)
--------------------------------------------------------------------------
Every individual sample this module takes is EXACT and GAPLESS along the
exact 1-D line it tests -- there is no sub-tile sampling gap the way the
discarded block design had between its 9 scattered 2D sample points. That
part of this design is airtight and was the explicit point of moving to a
segment graph.

The real, honestly-assessed remaining question is different: does a genuine
land connection between two lattice points 32+ tiles apart NECESSARILY
include at least one of this lattice's straight, axis-aligned, exactly-32-
tile segments as a fully-land subset? My honest answer is NO, not in
general. A real coastline can (and, per this project's own noise field,
plausibly often does) wind diagonally through the interior of a lattice
cell -- e.g. a corridor that snakes from the middle of one cell's south edge
to the middle of the neighboring cell's east edge, touching neither of the
two straight lattice edges bounding that cell along its full 32-tile length.
Such a path is real, 4-connected, physically land the whole way, and would
be found instantly by an unrestricted tile-resolution flood fill -- but this
lattice-restricted graph has NO edge that lies exactly along that path, so
it can neither confirm nor refute it: every lattice edge touching that
region would test a straight line that the winding corridor only partially
overlaps (or misses on one or both ends), so as soon as the corridor drifts
off that straight line for even one tile, the FULL-32-TILE-AND test fails
and the segment reports EDGE_BLOCKED, even though real land connects the two
endpoints just fine a few tiles off-axis.

Why this does not block shipping this design (per the task's own framing,
repeated here because it is the crux of why this is still safe even though
it is admittedly not the most SENSITIVE possible design): this graph's
under-connectivity relative to the true tile-resolution connectivity graph
can only ever cause a MISSED reject opportunity. REJECT is claimed only when
a real chain of fully-certified-land EDGES was actually found -- never
inferred from an assumption that "probably" a nearby corridor exists. A
seed whose only escape corridor snakes diagonally through lattice-cell
interiors the way just described will fail to REJECT here (falling through
to CONFIRM, still CPU-verified, or to DEFER, falling through to stage 2
unchanged) -- it will NEVER be able to cause a false REJECT, because there
is no code path that claims REJECT without a literally-tested, fully-land
32-tile run backing it. So this is a real, honestly-reported SENSITIVITY gap
(this design will very plausibly reject a smaller fraction of real mainlands
than an unrestricted flood fill would, and quite possibly a smaller fraction
than the old, looser 9-point block design did too, since a "some elevation
somewhere in a 32x32 area exceeds the margin at 9 sparse points" test is a
much weaker bar than "every one of 32 SPECIFIC consecutive tiles on an
axis-aligned line exceeds the margin") -- but it is not a SOUNDNESS gap. See
this module's own test-run report for whether this concern is realized in
the measured REJECT rate (spoiler for whoever reads this before running the
measurement: it plausibly is, and the report says so plainly either way).

RING / DISTANCE
--------------------------------------------------------------------------
RING_TILES = 5000 (see the RING_TILES assignment below for the 1400 -> 2000
(rejected) -> 5000 history). This is the project's standing enclosure
boundary -- same constant every stage in this pipeline uses for "the ring".
NOTE: stage2_floodfill.py's own grid (grid_radius_cells(350) *
FLOOD_FILL_STEP(4) = +-1400 tiles) predates this bump and has NOT been
resized to match -- see that module's own STATUS note for why this matters
(it's currently a dormant DEFER backstop, never yet exercised).
"""

import torch

import noise_gpu as ng
import noise_triton_fused as ntf

NODE_SPACING = 32  # tiles between adjacent lattice nodes
SEGMENT_TILES = 32  # samples per segment == NODE_SPACING (see module docstring)
# DOMAIN_RADIUS_NODES = ceil(RING_TILES * 1.417 / NODE_SPACING) -- 1.417 is the
# same headroom ratio the project has used at every prior RING_TILES value
# (62 nodes*32/1400 = 1.417), kept fixed so domain-vs-ring margin doesn't
# silently shrink as RING_TILES changes.
DOMAIN_RADIUS_NODES = 222  # -> 445x445 = 198,025-node domain, nominal radius 7,104 tiles
RING_TILES = 5000.0
# RAISED from 1400 (then a rejected 2000 experiment) to 5000: islands can now
# be long/elongated enough that 1400 (and even 2000) risked a false REJECT on
# a real escape-shaped island (see find_islands.py's --max-radius history for
# the concrete near-miss this was already fixed for on the ray-screen side;
# the certified net walk's own RING_TILES needed the same correction). This
# was previously measured impractical (~404 days) on the pre-Triton unfused
# evaluator -- see gpu/stage1_5_pillar.py's RING_TILES=5000 sweep notes -- but
# the fused/stripe Triton kernels plus prange-parallel apply_round change that
# calculus enough to revisit; re-measure end-to-end before trusting the old
# "impractical" conclusion at face value for THIS evaluator stack.
EPSILON_FP = 3.0  # see stage1_5_witness.py's CERTIFIED LAND section for full provenance

# Direction codes (int64). dir_code -> (delta_i, delta_j) in NODE units.
# 0 = +x, 1 = -x, 2 = +y, 3 = -y. Do not renumber without checking call sites
# in stage1_5_solver.py (its njit apply_round hardcodes the same 4-tuple).
DIR_DI = (1, -1, 0, 0)
DIR_DJ = (0, 0, 1, -1)

# Classification codes (torch.int8). Do not renumber without checking call sites.
EDGE_LAND = 0
EDGE_BLOCKED = 1
CLASS_NAMES = {EDGE_LAND: "EDGE_LAND", EDGE_BLOCKED: "EDGE_BLOCKED"}

# requests per _classify_chunk() call -- a MEMORY knob, not a correctness one.
# Historically the dominant per-request memory cost here was _gather_tables()
# MATERIALIZING a full duplicated permutation-table row per request
# (~96KB/request, independent of how many tile-samples each request takes).
# noise_gpu.py's noise_internal/elevation_nauvis now accept an optional
# seed_idx tensor and fetch each request's specific table entries directly
# out of the small ORIGINAL [N,256]-shaped tables via fused 2D advanced
# indexing (see noise_gpu.py's _gather_table_u8/_gather_grad) -- so this
# module no longer pre-duplicates anything (see classify_edges/_classify_chunk
# below, which pass `tables` and `seed_idx` straight through instead of
# calling a local _gather_tables helper). This constant is still a plain
# memory/latency-granularity knob (how many [R,32] elevation evaluations sit
# in flight at once), just no longer sized against the duplication cost that
# used to dominate; see this module's test-run report for current measured
# peak memory.
DEFAULT_GPU_CHUNK = 50_000


def build_tables(seeds):
    """
    seeds: torch.int64 [N] on target device, values in [0, 2**32) -- the FULL
    set of "live" seeds a stage 1.5 run will ever classify edges for, in a
    fixed order. Returns noise_gpu's per-seed permutation-table bundle, built
    ONCE (see noise_gpu.build_all_tables's own docstring for why rebuilding
    this per round/chunk is a >10x slowdown).

    Callers (stage1_5_solver.py) index into this table set via `seed_idx`
    (position in `seeds`, NOT the raw seed value) on every classify_edges()
    call.
    """
    return ng.build_all_tables(seeds)


_GEOM_CACHE = {}  # device -> (offs[segment_tiles], dir_di[4], dir_dj[4]); the lattice/segment
# geometry is identical for every seed and every request (same NODE_SPACING, same 4 directions,
# same 32 offsets), so it is built once per device and reused, not reconstructed via
# torch.arange/torch.tensor inside every _classify_chunk call (previously up to ~20 chunks/round
# x tens of rounds -- pure wasted kernel-launch/allocation overhead for numbers that never change).


def _get_geom(device, segment_tiles):
    key = (device, segment_tiles)
    cached = _GEOM_CACHE.get(key)
    if cached is None:
        offs = torch.arange(segment_tiles, dtype=torch.float32, device=device)
        dir_di = torch.tensor(DIR_DI, dtype=torch.float32, device=device)
        dir_dj = torch.tensor(DIR_DJ, dtype=torch.float32, device=device)
        cached = (offs, dir_di, dir_dj)
        _GEOM_CACHE[key] = cached
    return cached


BACKEND_PYTORCH = "pytorch"
BACKEND_TRITON_FUSED = "triton_fused"
BACKEND_TRITON_STRIPE = "triton_stripe"
DEFAULT_BACKEND = BACKEND_PYTORCH
# BACKEND_TRITON_FUSED uses noise_triton_fused.elevation_nauvis_triton -- a
# full-chain (all 26 noise_internal calls) fused Triton kernel, measured
# 28.9x over BACKEND_PYTORCH on this exact [R,32]/seed_idx access pattern
# (see gpu/noise_triton_fused.py's module docstring). NOT the default yet:
# switch explicitly via classify_edges(..., backend=BACKEND_TRITON_FUSED)
# until its own ground-truth (CPU) epsilon regression has been run and
# recorded here, per this project's standing rule (CLAUDE.md SS7.4) that any
# evaluator change needs a re-measured epsilon_fp before being trusted beyond
# a throughput prototype.
#
# BACKEND_TRITON_STRIPE uses noise_triton_fused.classify_stripes_triton --
# NO host-side position broadcasting (positions generated from base+dir*offset
# INSIDE the kernel) and NO host-side min/max/threshold reduction (also done
# in-kernel). It ALSO changes what gets sampled: every request explicitly
# includes BOTH its endpoint nodes plus segment_tiles interior tiles
# (NUM_POINTS = segment_tiles + 2), instead of this module's existing
# convention (sample the base node out to segment_tiles-1 tiles, relying on a
# DIFFERENT segment's own test to cover the far endpoint -- see this module's
# docstring). That's a strictly MORE conservative test (checks a superset of
# tiles), so it cannot introduce a false EDGE_LAND -- it can only be pickier
# (more BLOCKED) at the margin. Same "not the default without a soundness
# sign-off" status as BACKEND_TRITON_FUSED.


def _classify_chunk(tables, seed_idx, base_i, base_j, dir_code, node_spacing, segment_tiles, epsilon_fp,
                     backend=DEFAULT_BACKEND):
    device = seed_idx.device

    if backend == BACKEND_TRITON_STRIPE:
        # No host-side position broadcast at all here -- base_x/base_y/
        # dir_code/seed_idx are passed straight through; classify_stripes_triton
        # generates every sample position INSIDE the kernel.
        base_x = base_i.to(torch.float32) * node_spacing
        base_y = base_j.to(torch.float32) * node_spacing
        with torch.no_grad():
            result = ntf.classify_stripes_triton(tables, base_x, base_y, dir_code, seed_idx, epsilon_fp,
                                                  stripe_tiles=segment_tiles)
        return result["classification"], result["min_elev"], result["max_dist"]

    offs, dir_di, dir_dj = _get_geom(device, segment_tiles)  # [32], [4], [4] -- built once per device
    di = dir_di[dir_code]  # [R]
    dj = dir_dj[dir_code]  # [R]

    base_x = base_i.to(torch.float32) * node_spacing  # [R]
    base_y = base_j.to(torch.float32) * node_spacing

    px = base_x.unsqueeze(1) + di.unsqueeze(1) * offs.unsqueeze(0)  # [R, 32]
    py = base_y.unsqueeze(1) + dj.unsqueeze(1) * offs.unsqueeze(0)  # [R, 32]

    # Pass the ORIGINAL (small, [N,256]-shaped) tables straight through, plus
    # seed_idx, instead of pre-gathering a duplicated [R,256]-per-variant
    # table subset by hand -- noise_gpu.elevation_nauvis now does the
    # per-request seed lookup fused into its existing per-corner gathers.
    # This is the fix for this module's documented ~96KB/request duplication
    # cost (see DEFAULT_GPU_CHUNK's comment above).
    with torch.no_grad():
        if backend == BACKEND_TRITON_FUSED:
            r, p = px.shape
            seed_idx_expanded = seed_idx.unsqueeze(1).expand(-1, p).reshape(-1)
            elevation_flat = ntf.elevation_nauvis_triton(tables, px.reshape(-1), py.reshape(-1), seed_idx_expanded)
            elevation = elevation_flat.view(r, p)
        else:
            elevation = ng.elevation_nauvis(tables, px, py, seed_idx=seed_idx)  # [R, 32]

    min_elev = elevation.min(dim=1).values
    dist = torch.sqrt(px * px + py * py)  # [R, 32] -- true dist of each ACTUALLY-SAMPLED tile
    max_dist = dist.max(dim=1).values

    classification = torch.where(min_elev > epsilon_fp,
                                  torch.full_like(min_elev, EDGE_LAND, dtype=torch.int8),
                                  torch.full_like(min_elev, EDGE_BLOCKED, dtype=torch.int8))

    return classification, min_elev, max_dist


def classify_edges(tables, seed_idx, base_i, base_j, dir_code, node_spacing=NODE_SPACING,
                    segment_tiles=SEGMENT_TILES, epsilon_fp=EPSILON_FP, chunk_size=DEFAULT_GPU_CHUNK,
                    backend=DEFAULT_BACKEND):
    """
    The stateless oracle call. One call classifies an entire round's worklist
    (concatenated across every live seed's solver) in one shot.

    tables:    dict from build_tables(seeds) -- built once for the whole run.
    seed_idx:  torch.int64 [R] -- index into the `seeds` batch build_tables
               was called with (NOT a raw seed value). Repeats freely.
    base_i, base_j: torch.int64 [R] -- the ALREADY-REACHED node this segment
               is tested FROM (see module docstring's NODE / GRAPH GEOMETRY).
    dir_code:  torch.int64 [R] -- 0/1/2/3 for +x/-x/+y/-y (DIR_DI/DIR_DJ).
    chunk_size: requests are split into chunks of this size before the
               (memory-heavy, [R,32]-shaped) elevation evaluation -- purely a
               memory-safety knob, not a behavior change.

    Returns dict of torch tensors, all length R, same order as input:
      classification: torch.int8   [R] -- EDGE_LAND / EDGE_BLOCKED
      min_elev:       torch.float32[R] -- min of the 32 sampled elevations
                       (diagnostic "closure margin" signal)
      max_dist:       torch.float32[R] -- max Euclidean tile-distance from
                       spawn among the 32 ACTUALLY-SAMPLED tiles (see module
                       docstring's SOUNDNESS ARGUMENT for why this, not the
                       nominal neighbor-node distance, is what REJECT must
                       use). Only meaningful for REJECT purposes when
                       classification == EDGE_LAND.
    """
    r = seed_idx.shape[0]
    device = seed_idx.device
    classification = torch.empty(r, dtype=torch.int8, device=device)
    min_elev = torch.empty(r, dtype=torch.float32, device=device)
    max_dist = torch.empty(r, dtype=torch.float32, device=device)

    for start in range(0, r, chunk_size):
        end = min(start + chunk_size, r)
        c, m, d = _classify_chunk(tables, seed_idx[start:end], base_i[start:end], base_j[start:end],
                                   dir_code[start:end], node_spacing, segment_tiles, epsilon_fp, backend=backend)
        classification[start:end] = c
        min_elev[start:end] = m
        max_dist[start:end] = d

    return {"classification": classification, "min_elev": min_elev, "max_dist": max_dist}


def classify_edges_for_seeds(seeds, base_i, base_j, dir_code, **kwargs):
    """
    Convenience wrapper for standalone/unit testing: builds tables for
    exactly the given `seeds` tensor and classifies one edge per seed
    (seed_idx = arange). NOT used by the real per-round pipeline
    (stage1_5_solver.py builds tables once for the whole live batch and calls
    classify_edges directly with an explicit seed_idx per request).
    """
    device = seeds.device
    tables = build_tables(seeds)
    seed_idx = torch.arange(seeds.shape[0], dtype=torch.int64, device=device)
    return classify_edges(tables, seed_idx, base_i, base_j, dir_code, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, NODE_SPACING={NODE_SPACING}, SEGMENT_TILES={SEGMENT_TILES}, "
          f"DOMAIN_RADIUS_NODES={DOMAIN_RADIUS_NODES}, EPSILON_FP={EPSILON_FP}")

    ground_truth = [20271579, 3627767453, 1064911508, 1489155600]
    seeds = torch.tensor(ground_truth, dtype=torch.int64, device=device)

    # --- unit test 1: a segment through solid interior land (spawn's own
    # +x edge, i.e. base=(0,0), dir=0) should classify EDGE_LAND for every
    # ground-truth seed (all 4 have spawn_land == True in production, and
    # the very first 32 tiles east of spawn are deep interior land for a
    # real island/mainland of the sizes in this project's ground truth). ---
    zero = torch.zeros(4, dtype=torch.int64, device=device)
    dir0 = torch.zeros(4, dtype=torch.int64, device=device)
    interior_result = classify_edges_for_seeds(seeds, zero, zero, dir0)
    print("\n=== unit test 1: spawn's own +x edge (base=(0,0), dir=+x) ===")
    all_ok = True
    for i, seed in enumerate(ground_truth):
        cls = int(interior_result["classification"][i])
        min_e = float(interior_result["min_elev"][i])
        name = CLASS_NAMES[cls]
        ok = cls == EDGE_LAND
        all_ok &= ok
        print(f"  seed {seed}: edge -> {name} (min_elev={min_e:.4f})  {'OK' if ok else '*** FAIL ***'}")

    # --- unit test 2: the segment(s) touching seed 1064911508's chokepoint
    # tile (-455,-447) (true elevation ~-0.0361, i.e. water under the exact
    # CPU model) must classify EDGE_BLOCKED. The chokepoint tile is not
    # exactly on a lattice line, so we test EVERY one of the (up to 4)
    # lattice segments whose 32-tile sampled line passes through or adjacent
    # to it, by picking the enclosing lattice cell and testing its 4
    # boundary segments -- an honest test of the EPSILON_FP CAVEAT above:
    # report what's actually measured, don't assume the outcome. ---
    choke_x, choke_y = -455.0, -447.0
    import math
    cell_i0 = math.floor(choke_x / NODE_SPACING)
    cell_j0 = math.floor(choke_y / NODE_SPACING)
    print(f"\n=== unit test 2: seed 1064911508 chokepoint tile ({choke_x:.0f},{choke_y:.0f}) ===")
    print(f"  enclosing lattice cell: i in [{cell_i0},{cell_i0+1}], j in [{cell_j0},{cell_j0+1}] "
          f"(node tiles ({cell_i0*NODE_SPACING},{cell_j0*NODE_SPACING}) .. "
          f"({(cell_i0+1)*NODE_SPACING},{(cell_j0+1)*NODE_SPACING}))")
    seed_1064 = torch.tensor([1064911508], dtype=torch.int64, device=device)

    # The 4 boundary segments of the enclosing cell, each as (base_i,base_j,dir):
    #   south edge: base=(cell_i0,   cell_j0),   dir=+x (0)
    #   north edge: base=(cell_i0,   cell_j0+1), dir=+x (0)
    #   west  edge: base=(cell_i0,   cell_j0),   dir=+y (2)
    #   east  edge: base=(cell_i0+1, cell_j0),   dir=+y (2)
    boundary_segments = [
        ("south", cell_i0, cell_j0, 0),
        ("north", cell_i0, cell_j0 + 1, 0),
        ("west", cell_i0, cell_j0, 2),
        ("east", cell_i0 + 1, cell_j0, 2),
    ]
    choke_ok = True
    for name_lbl, bi, bj, d in boundary_segments:
        bi_t = torch.tensor([bi], dtype=torch.int64, device=device)
        bj_t = torch.tensor([bj], dtype=torch.int64, device=device)
        d_t = torch.tensor([d], dtype=torch.int64, device=device)
        res = classify_edges_for_seeds(seed_1064, bi_t, bj_t, d_t)
        cls = int(res["classification"][0])
        min_e = float(res["min_elev"][0])
        cname = CLASS_NAMES[cls]
        # We do not assert any single one of these MUST be EDGE_BLOCKED (the
        # chokepoint tile might not be one of THIS segment's 32 samples at
        # all) -- we report all 4 honestly and separately confirm at least
        # one of them (or a direct sample AT the chokepoint tile) is blocked.
        print(f"  {name_lbl:>5} edge base=({bi},{bj}) dir={d}: {cname} (min_elev={min_e:.4f})")

    # Direct, unambiguous check: sample the exact chokepoint tile itself via
    # noise_gpu directly (bypassing the lattice quantization entirely) to
    # confirm it is indeed NOT certified land under EPSILON_FP -- this is
    # the ground-truth fact the lattice segments above are being compared
    # against.
    tables_1064 = build_tables(seed_1064)
    px = torch.tensor([[choke_x]], dtype=torch.float32, device=device)
    py = torch.tensor([[choke_y]], dtype=torch.float32, device=device)
    with torch.no_grad():
        choke_elev = ng.elevation_nauvis(tables_1064, px, py)[0, 0].item()
    choke_certified_land = choke_elev > EPSILON_FP
    print(f"  direct sample AT chokepoint tile: elevation={choke_elev:.4f}, "
          f"certified_land={choke_certified_land} "
          f"{'OK (correctly not certified land)' if not choke_certified_land else '*** FAIL ***'}")
    all_ok &= not choke_certified_land

    # And: any lattice segment whose 32-tile line happens to pass EXACTLY
    # through the chokepoint tile's integer coordinate must be EDGE_BLOCKED
    # (it cannot be EDGE_LAND while containing a tile we just proved is not
    # certified land) -- the chokepoint tile (-455,-447) is not on any of
    # this cell's 4 boundary lines (both coordinates are off the lattice by
    # 9 and 7 tiles respectively from the nearest multiples of 32: -455 is
    # 9 tiles off i=-15 (*32=-480) i.e. row -455 is not x=-480 or -448, and
    # -447 is 7 tiles off j=-14 (*32=-448)), so none of the 4 boundary
    # segments tested above are expected to directly contain it -- this is
    # itself the EPSILON_FP CAVEAT made concrete: a real 1-tile-wide
    # chokepoint need not sit on this lattice's sampled lines at all. We
    # report this honestly rather than force a same-tile match that doesn't
    # exist for this specific example.
    on_lattice_row = (choke_x % NODE_SPACING) == 0
    on_lattice_col = (choke_y % NODE_SPACING) == 0
    print(f"  chokepoint tile lattice-alignment: x%32={choke_x % NODE_SPACING:.0f}, "
          f"y%32={choke_y % NODE_SPACING:.0f} (on a sampled lattice line only if either is 0) "
          f"-- {'ON-LATTICE' if (on_lattice_row or on_lattice_col) else 'OFF-LATTICE (expected; see EPSILON_FP CAVEAT)'}")

    print("\nALL ORACLE UNIT TESTS PASSED" if all_ok else "\n*** ORACLE UNIT TEST FAILURE ***")
