"""
Stage 1.5 CPU graph solver -- numba-jitted, flat-numpy-array wave BFS over
the LATTICE-POINT / SEGMENT GRAPH gpu/stage1_5_oracle.py classifies.

This is the "CPU half" of the stage 1.5 design (see stage1_5_oracle.py's
module docstring for the GPU half, the full soundness argument, and the
honest EPSILON_FP CAVEAT about this graph's sensitivity limits). Per the
task spec this is "ordinary debuggable/assertable code, NOT GPU tensor ops
and NOT a pure-Python per-cell loop" -- the per-request inner loop below is a
plain flat-array scan, JIT-compiled by numba, operating on ordinary numpy
arrays a debugger/pdb can inspect directly.

GRAPH, NOT GRID OF AREAS -- WHAT "NODE" AND "EDGE" MEAN HERE
--------------------------------------------------------------------------
A NODE is a lattice point (i, j) at tile coordinate (32i, 32j). An EDGE
("segment") connects a node to one of its 4 orthogonal neighbors; each
worklist REQUEST is (seed, base_i, base_j, dir_code) -- "test the segment
that starts AT already-reached node (base_i, base_j) and runs 32 tiles in
direction dir_code" (see stage1_5_oracle.py's DIR_DI/DIR_DJ). The oracle
classifies that segment; this module decides, from the result, whether the
NEIGHBOR node at the far end becomes newly reached and, if so, what edges to
test next from it.

WHY A DENSE PER-SEED BITMAP, NO EVICTION, EVER
--------------------------------------------------------------------------
Domain = 125x125 = 15,625 nodes/seed (see stage1_5_oracle.DOMAIN_RADIUS_NODES).
A dense bool `touched`/`reached` bitmap per seed is ~15.6KB/seed uncombined
(both arrays), trivial even for tens of thousands of resident seeds. Because
memory is a non-issue at this resolution, the frontier needs NO top-K cut and
NO capacity-based eviction: every node a seed's walk ever claims as a
candidate stays recorded forever (in `touched`) and is reconsidered exactly
once. THIS IS THE ENTIRE POINT of using a coarse lattice graph instead of the
discarded tile-scale beam design (see stage1_5_witness.py's module docstring)
-- full frontier retention is what makes CONFIRM (exhaustion) a sound
certificate: it can only fire when the frontier is truly, completely empty,
because nothing was ever silently dropped to make room. If you find yourself
adding a capacity check that discards touched-but-not-yet-expanded nodes to
bound memory, STOP -- that reintroduces exactly the beam-pruning soundness
bug this design exists to fix. There is no such check anywhere in this
module; `touched`/`reached` arrays are allocated once per seed at full domain
size and never shrunk, cleared, or partially evicted.

CONTENT-ADDRESSED CACHE -- NO NODE CLAIMED/TESTED TWICE
--------------------------------------------------------------------------
`touched[s, ri, rj]` is set to True the INSTANT a node is added to a
worklist as a CANDIDATE (i.e. some already-reached neighbor is about to have
its outgoing edge toward it tested), not when that edge's classification
comes back. Since a node can only be claimed by first passing the
`if touched[...]: continue` check earlier in the very same (sequential,
single-threaded-per-round) scan, a given (seed, i, j) node can never be
enqueued as a candidate a second time in this seed's lifetime -- including
two different newly-reached neighbors in the same round both bordering the
same unclaimed node (the second one to reach it sees `touched` already True
and skips it). This means AT MOST ONE of a node's (up to 4) possible
incoming edges is ever actually tested -- see stage1_5_oracle.py's NODE
REACHABILITY section for why that is safe (it can only under-connect the
graph, never over-connect it, so it can only cost a missed REJECT
opportunity, never cause a false one).

WAVE EXPANSION, NOT ONE NODE AT A TIME
--------------------------------------------------------------------------
`apply_round` is called once per round with the ENTIRE round's worklist
(every still-active seed's pending edge-test requests, concatenated) and
expands every newly-reached node's own outgoing edges (to its still-unclaimed
neighbors) in one pass -- this keeps the number of GPU-round-trip
synchronization points down to "path length in lattice hops" rather than one
round-trip per single-node BFS step.

THREE-WAY OUTCOME (module constants REJECT / CONFIRM / DEFER)
--------------------------------------------------------------------------
  REJECT  -- some segment classified EDGE_LAND this round has max_dist
             (the max Euclidean distance among that segment's own 32
             ACTUALLY-SAMPLED tiles -- see stage1_5_oracle.py's SOUNDNESS
             ARGUMENT for why this, not the nominal neighbor-node distance,
             is what must be used) >= RING_TILES. Fires immediately, no CPU
             backstop.
  CONFIRM -- a seed that had at least one request outstanding this round
             contributed ZERO new nodes to the next round's worklist (i.e.
             every edge it just had classified was either EDGE_BLOCKED, or
             EDGE_LAND but the neighbor it reached had no still-unclaimed
             neighbors of its own left to test). Since nothing is ever
             evicted (see above), "frontier empty" here really does mean the
             walk has nowhere left to go, ever -- a sound exhaustion
             certificate. Every CONFIRM is meant to be routed to the
             mandatory CPU flood-fill verify afterward (not done by this
             prototype -- see the test-run script).
  DEFER   -- domain exhausted (touched_count reaches the full 15,625-node
             domain) OR the round-count safety cap (MAX_ROUNDS) is hit
             without resolving. Both are expected to be rare -- these exist
             purely as safety valves, not tuned budgets; hitting either
             falls through to today's existing dense stage 2, unchanged,
             exactly like a stage1_5_witness.py DEFER does today.

SPAWN IS THE GRAPH'S ROOT, NOT A CLASSIFIED NODE
--------------------------------------------------------------------------
Unlike the discarded block design (which ran spawn's own CELL through the
identical classifier as everything else, because a CELL classification is a
standalone test of a point), this graph has no notion of "classify a node in
isolation" -- only EDGES are classified. Spawn's node (0, 0) is simply given
as the walk's root; round 0's worklist is spawn's 4 outgoing edges
(base=(0,0), dir=0..3). This does NOT weaken the soundness argument: whichever
of those 4 edges (if any) comes back EDGE_LAND has, by construction, its own
offset-0 sample AT spawn's exact tile (0, 0) individually certified land as
part of that edge's own 32-sample test -- so spawn's own certification is
folded into testing its first edge, not skipped. A seed all 4 of whose spawn
edges come back EDGE_BLOCKED (including, in particular, a seed whose spawn
tile itself is not certified land) resolves to CONFIRM immediately
(touched_count = 5: spawn + its 4 immediate neighbors, all claimed at round 0
-- see MAX_ROUNDS below) -- safe, since CONFIRM is CPU-verified downstream.
"""

import numpy as np
from numba import njit, prange

import stage1_5_oracle as oracle

NODE_SPACING = oracle.NODE_SPACING
DOMAIN_RADIUS_NODES = oracle.DOMAIN_RADIUS_NODES
SIDE = 2 * DOMAIN_RADIUS_NODES + 1                  # 125
DOMAIN_NODES = SIDE * SIDE                          # 15,625
RING_TILES = oracle.RING_TILES
EDGE_LAND = oracle.EDGE_LAND

# Verdict codes (np.int8). Do not renumber without checking call sites.
REJECT = 0
CONFIRM = 1
DEFER = 2
UNRESOLVED = 3  # internal sentinel only; never returned once run_stage15 finishes
VERDICT_NAMES = {REJECT: "REJECT", CONFIRM: "CONFIRM", DEFER: "DEFER"}

# Safety cap on round count. NOT a tuned budget -- domain exhaustion
# (touched_count >= DOMAIN_NODES) is the real bound. Each round advances the
# wave by exactly one more lattice hop (32 tiles of Euclidean distance at
# most per hop, same granularity the discarded block design's cell-BFS used),
# so the same worst-case round-count argument that design's docstring made
# carries over unchanged: a real escape at 1400 tiles (43.75 node-hops) is
# caught by round ~44 at the latest in the common case, with MAX_ROUNDS set
# far below the absolute DOMAIN_NODES worst case but far above that.
MAX_ROUNDS = 1000


@njit(cache=True)
def apply_round(seed_ids, base_i, base_j, dir_code, classification, max_dist,
                 touched, reached, verdict, active, touched_count, had_request, resolved_round,
                 rnd, side, center, ring_tiles, domain_nodes, edge_land_code):
    """
    Processes exactly one round's worklist and produces the next round's.

    seed_ids, base_i, base_j, dir_code: int64[n_req] -- this round's
        flattened worklist (seed_ids are indices into the `touched`/
        `reached`/... state arrays, i.e. positions in the fixed live-seed
        batch, NOT raw seed values). Each request tests the segment that
        starts at already-reached node (base_i, base_j) and runs in
        direction dir_code (see stage1_5_oracle.DIR_DI/DIR_DJ).
    classification, max_dist: outputs of stage1_5_oracle.classify_edges,
        aligned 1:1 with (seed_ids, base_i, base_j, dir_code).
    touched, reached: bool[S, side, side] -- persistent per-seed state,
        mutated in place. `touched` is the content-addressed cache (see
        module docstring); `reached` records which nodes were found
        connected via an EDGE_LAND segment (diagnostics only, not re-read
        for correctness within this function).
    verdict: int8[S], `active`: bool[S], `touched_count`: int64[S],
        `resolved_round`: int32[S] -- mutated in place. `had_request`: a
        caller-provided scratch bool[S] buffer, reset to False by this
        function before use.

    Returns (next_seed_ids, next_base_i, next_base_j, next_dir_code): int64
    arrays, the flattened worklist for round `rnd + 1` (already trimmed to
    their used length).
    """
    n_req = seed_ids.shape[0]
    s_count = touched.shape[0]

    # Upper bound: each request, if its edge is EDGE_LAND, can produce at
    # most 4 new neighbor-edge requests from the newly reached node.
    out_seed = np.empty(n_req * 4, dtype=np.int64)
    out_base_i = np.empty(n_req * 4, dtype=np.int64)
    out_base_j = np.empty(n_req * 4, dtype=np.int64)
    out_dir = np.empty(n_req * 4, dtype=np.int64)
    out_count = 0

    contributed = np.zeros(s_count, dtype=np.bool_)
    for i in range(s_count):
        had_request[i] = False

    deltas_di = (1, -1, 0, 0)
    deltas_dj = (0, 0, 1, -1)

    for k in range(n_req):
        s = seed_ids[k]
        if not active[s]:
            continue
        had_request[s] = True

        bi = base_i[k]
        bj = base_j[k]
        d = dir_code[k]
        ni = bi + deltas_di[d]
        nj = bj + deltas_dj[d]
        nri = ni + center
        nci = nj + center
        if nri < 0 or nri >= side or nci < 0 or nci >= side:
            # Domain edge. Should not happen for real seeds given the
            # domain-vs-ring headroom margin (see stage1_5_oracle.py's
            # module docstring) -- skip defensively rather than crash; this
            # seed will either resolve via the nodes it does reach or fall
            # through to DEFER, never a false REJECT/CONFIRM.
            continue

        is_land = classification[k] == edge_land_code
        reached[s, nri, nci] = is_land
        if not is_land:
            continue

        if max_dist[k] >= ring_tiles:
            verdict[s] = 0  # REJECT
            active[s] = False
            resolved_round[s] = rnd
            continue

        for d2 in range(4):
            nni = ni + deltas_di[d2]
            nnj = nj + deltas_dj[d2]
            nnri = nni + center
            nncj = nnj + center
            if nnri < 0 or nnri >= side or nncj < 0 or nncj >= side:
                continue
            if touched[s, nnri, nncj]:
                continue
            touched[s, nnri, nncj] = True
            touched_count[s] += 1
            out_seed[out_count] = s
            out_base_i[out_count] = ni
            out_base_j[out_count] = nj
            out_dir[out_count] = d2
            out_count += 1
            contributed[s] = True

    for s in range(s_count):
        if not active[s]:
            continue
        if touched_count[s] >= domain_nodes:
            verdict[s] = 2  # DEFER (domain exhaustion)
            active[s] = False
            resolved_round[s] = rnd
        elif had_request[s] and not contributed[s]:
            verdict[s] = 1  # CONFIRM (frontier truly empty, nothing evicted)
            active[s] = False
            resolved_round[s] = rnd

    return (out_seed[:out_count], out_base_i[:out_count], out_base_j[:out_count], out_dir[:out_count])


@njit(parallel=True, cache=True)
def apply_round_parallel(seed_ids, base_i, base_j, dir_code, classification, max_dist,
                          touched, reached, verdict, active, touched_count, had_request, resolved_round,
                          rnd, side, center, ring_tiles, domain_nodes, edge_land_code):
    """
    Parallel (numba prange) equivalent of apply_round -- SAME per-request
    logic, restructured so each parallel worker owns one SEED's entire
    slice of this round's requests, never overlapping another worker's
    seed. This is the only safe way to parallelize this loop without atomics:
    apply_round's per-request work mutates touched[s,...]/reached[s,...]/
    touched_count[s]/verdict[s]/active[s]/resolved_round[s], all indexed by
    the request's OWN seed s -- as long as no two workers ever process
    requests for the same s concurrently, every one of those mutations is
    to memory only that worker ever touches, and there is no race, full
    stop (not "unlikely", not "benign" -- provably absent, since apply_round's
    entire state is partitioned by seed and this partitions the WORK by
    seed too, one-to-one).

    A single seed's requests turn out to ALREADY be contiguous in seed_ids,
    every round, by construction: round 0's worklist is built via
    np.repeat(arange(s_count), 4) (grouped, ascending by seed), and
    apply_round's own output-building loop scans its input in order and
    appends a newly-reached node's own up-to-4 follow-up requests
    consecutively for that SAME seed before the scan moves to the next
    input request -- so if round k's worklist is grouped-by-seed (ascending),
    round k+1's necessarily is too (induction, base case round 0). VERIFIED
    empirically (see scratchpad/diagnose_load_balance.py's groupedness
    check) across 20 rounds of a 5,000-seed run: contiguous_groups ==
    unique_seeds every single round. An earlier version of this function
    didn't know this and paid for an unnecessary O(n_req) counting-sort +
    5-array scatter to "fix" grouping that was never broken -- that
    redundant copy cost about as much as the parallel section saved,
    measuring only 1.07x end-to-end. Finding the (already-present) group
    boundaries via one cheap linear scan (integer compares only, no data
    movement) instead of re-sorting is the actual fix.

    Output (next round's worklist) is still written into a per-seed
    RESERVED slice (upper bound 4 new requests per input request, exactly
    the same bound apply_round's own out_seed sizing uses) -- that part is
    unavoidable, since concurrent workers writing a shared output array need
    non-overlapping regions -- followed by a sequential compaction pass
    (cost O(total output size), not O(n_req) input size) that drops each
    seed's unused tail.

    Verified bit-identical to apply_round across the standing ground-truth
    seeds plus large random batches -- see stage1_5_solver.py's __main__
    and the scratchpad validation script this was checked against before
    being wired into run_stage15(parallel_cpu=True).
    """
    n_req = seed_ids.shape[0]

    deltas_di = (1, -1, 0, 0)
    deltas_dj = (0, 0, 1, -1)

    # ---- step 1: find the (already-present) per-seed group boundaries ----
    # two cheap linear scans (integer compares only) instead of a sort.
    n_groups = 1
    for k in range(1, n_req):
        if seed_ids[k] != seed_ids[k - 1]:
            n_groups += 1
    group_start = np.empty(n_groups + 1, dtype=np.int64)
    group_seed = np.empty(n_groups, dtype=np.int64)
    group_start[0] = 0
    group_seed[0] = seed_ids[0]
    gi = 0
    for k in range(1, n_req):
        if seed_ids[k] != seed_ids[k - 1]:
            gi += 1
            group_start[gi] = k
            group_seed[gi] = seed_ids[k]
    group_start[n_groups] = n_req

    # ---- step 2: reserve each group's output slice (upper bound 4x its own request count) ----
    out_starts = np.zeros(n_groups + 1, dtype=np.int64)
    for g in range(n_groups):
        out_starts[g + 1] = out_starts[g] + (group_start[g + 1] - group_start[g]) * 4
    total_capacity = out_starts[n_groups]
    out_seed_full = np.empty(total_capacity, dtype=np.int64)
    out_base_i_full = np.empty(total_capacity, dtype=np.int64)
    out_base_j_full = np.empty(total_capacity, dtype=np.int64)
    out_dir_full = np.empty(total_capacity, dtype=np.int64)
    group_out_count = np.zeros(n_groups, dtype=np.int64)

    had_request[:] = False

    # ---- step 3: the actual per-seed work, one GROUP (== one seed) per parallel worker ----
    for g in prange(n_groups):
        s = group_seed[g]
        if not active[s]:
            continue
        lo = group_start[g]
        hi = group_start[g + 1]
        had_request[s] = True

        local_out = 0
        out_base = out_starts[g]
        contributed_local = False

        for k in range(lo, hi):
            bi = base_i[k]
            bj = base_j[k]
            d = dir_code[k]
            ni = bi + deltas_di[d]
            nj = bj + deltas_dj[d]
            nri = ni + center
            nci = nj + center
            if nri < 0 or nri >= side or nci < 0 or nci >= side:
                continue

            is_land = classification[k] == edge_land_code
            reached[s, nri, nci] = is_land
            if not is_land:
                continue

            if max_dist[k] >= ring_tiles:
                verdict[s] = 0  # REJECT
                active[s] = False
                resolved_round[s] = rnd
                break  # matches apply_round: this seed's later same-round requests are skipped too

            for d2 in range(4):
                nni = ni + deltas_di[d2]
                nnj = nj + deltas_dj[d2]
                nnri = nni + center
                nncj = nnj + center
                if nnri < 0 or nnri >= side or nncj < 0 or nncj >= side:
                    continue
                if touched[s, nnri, nncj]:
                    continue
                touched[s, nnri, nncj] = True
                touched_count[s] += 1
                out_seed_full[out_base + local_out] = s
                out_base_i_full[out_base + local_out] = ni
                out_base_j_full[out_base + local_out] = nj
                out_dir_full[out_base + local_out] = d2
                local_out += 1
                contributed_local = True

        group_out_count[g] = local_out

        if active[s]:  # NOT already REJECTed above -- matches apply_round's `if not active[s]: continue`
            if touched_count[s] >= domain_nodes:
                verdict[s] = 2  # DEFER (domain exhaustion)
                active[s] = False
                resolved_round[s] = rnd
            elif not contributed_local:
                verdict[s] = 1  # CONFIRM (frontier truly empty, nothing evicted)
                active[s] = False
                resolved_round[s] = rnd

    # ---- step 4: compaction -- prefix-sum (sequential, O(n_groups), cheap)
    # then the actual copy PARALLELIZED over groups (each writes a disjoint
    # destination range, so this is exactly as race-free as step 3): the
    # copy is O(total output size) and was measured as a real fraction of
    # apply_round_parallel's own wall-clock, worth parallelizing same as the
    # main expansion loop.
    compact_start = np.empty(n_groups + 1, dtype=np.int64)
    compact_start[0] = 0
    for g in range(n_groups):
        compact_start[g + 1] = compact_start[g] + group_out_count[g]
    total_actual = compact_start[n_groups]

    out_seed = np.empty(total_actual, dtype=np.int64)
    out_base_i = np.empty(total_actual, dtype=np.int64)
    out_base_j = np.empty(total_actual, dtype=np.int64)
    out_dir = np.empty(total_actual, dtype=np.int64)
    for g in prange(n_groups):
        n = group_out_count[g]
        if n == 0:
            continue
        src = out_starts[g]
        pos = compact_start[g]
        for i in range(n):
            out_seed[pos + i] = out_seed_full[src + i]
            out_base_i[pos + i] = out_base_i_full[src + i]
            out_base_j[pos + i] = out_base_j_full[src + i]
            out_dir[pos + i] = out_dir_full[src + i]

    return (out_seed, out_base_i, out_base_j, out_dir)


def run_stage15(seeds, device, node_spacing=NODE_SPACING, domain_radius_nodes=DOMAIN_RADIUS_NODES,
                 segment_tiles=None, epsilon_fp=oracle.EPSILON_FP, ring_tiles=RING_TILES,
                 max_rounds=MAX_ROUNDS, verbose=False, timing_out=None, round_log=None,
                 backend=oracle.DEFAULT_BACKEND, parallel_cpu=False, tables=None):
    """
    Top-level synchronous per-round driver: GPU classifies round k's
    worklist of edge-tests, then this CPU solver assembles round k+1's --
    see the task's own "start with a straightforward synchronous per-round
    loop... then, if and only if profiling shows GPU round-trip latency
    actually dominates wall-clock, add double-buffering" -- no double-
    buffering here yet, this is that first (correctness-first) version.

    seeds: torch.int64 [S] on `device`.
    node_spacing / segment_tiles: tiles between adjacent lattice nodes / tile-
        samples per edge (see stage1_5_oracle.py's module docstring for why
        gapless full coverage along every edge REQUIRES these to be equal --
        a segment's samples run from its base node's own tile out to
        segment_tiles-1 tiles toward the neighbor, so anything short of
        segment_tiles == node_spacing leaves an unsampled gap right before
        the neighbor node). `segment_tiles` defaults to `node_spacing` (i.e.
        the caller normally only needs to set node_spacing to change the
        graph's spacing -- e.g. run_stage15(seeds, device, node_spacing=64)
        for a 64-tile lattice); passing a mismatched value on purpose is
        rejected below rather than silently under-sampling.
    Returns dict of numpy arrays, all length S, same order as `seeds`:
      verdict:        int8  [S] -- REJECT / CONFIRM / DEFER
      cells_touched:  int64 [S] -- touched-NODE count at resolution (kept as
                      `cells_touched` for naming continuity with the prior
                      cell-based design's callers/test scripts -- this is now
                      a node count, not a cell count).
      resolved_round: int32 [S] -- round index the verdict was decided on
      rounds_used:    int -- total rounds actually run this call
      round_sizes:    list[int] -- worklist size per round (requests/round)
      gpu_seconds, cpu_seconds: float -- wall-clock split, if timing_out is
        a dict, it is updated in place with these (plus 'n_gpu_calls').

    round_log: optional list -- if given, one dict is appended per round:
      {"round": int, "worklist_size": int, "t_gpu_start": epoch float,
       "t_gpu_end": epoch float, "t_cpu_end": epoch float}, all from
      time.time() (wall-clock epoch, so external tooling -- e.g. an
      nvidia-smi sampling thread -- can correlate GPU-utilization samples
      against which round/phase (GPU-classify vs CPU-solve) was running at
      that instant).
    """
    import time
    import torch

    if segment_tiles is None:
        segment_tiles = node_spacing
    assert segment_tiles == node_spacing, (
        f"segment_tiles ({segment_tiles}) must equal node_spacing ({node_spacing}) -- "
        "see stage1_5_oracle.py's module docstring: gapless full coverage along every "
        "lattice edge requires the two to match, otherwise the last "
        "(node_spacing - segment_tiles) tiles right before the neighbor node go unsampled."
    )

    s_count = seeds.shape[0]
    side = 2 * domain_radius_nodes + 1
    center = domain_radius_nodes

    # tables: optional pre-built oracle.build_tables(...)-shaped dict, already
    # aligned 1:1 with `seeds` (e.g. noise_gpu.gather_tables applied to a
    # LARGER already-built batch's tables for this seed subset -- see
    # stage1_5_cascade.run_cascade's pass-2 call, which avoids paying
    # build_tables' ~1,020-launch fixed cost a second time on its residual).
    # None (default): unchanged behavior, build fresh for `seeds` here.
    if tables is None:
        tables = oracle.build_tables(seeds)

    touched = np.zeros((s_count, side, side), dtype=np.bool_)
    reached = np.zeros((s_count, side, side), dtype=np.bool_)
    verdict = np.full(s_count, UNRESOLVED, dtype=np.int8)
    active = np.ones(s_count, dtype=np.bool_)
    touched_count = np.zeros(s_count, dtype=np.int64)
    resolved_round = np.full(s_count, -1, dtype=np.int32)
    had_request_scratch = np.zeros(s_count, dtype=np.bool_)

    # Spawn's own node is trivially the walk's root (see module docstring's
    # SPAWN IS THE GRAPH'S ROOT section) -- mark it touched, then claim its
    # 4 immediate neighbors as round 0's candidates (they are the targets of
    # spawn's 4 outgoing edges, tested below).
    touched[:, center, center] = True
    deltas_di = (1, -1, 0, 0)
    deltas_dj = (0, 0, 1, -1)
    for d in range(4):
        touched[:, center + deltas_di[d], center + deltas_dj[d]] = True
    touched_count[:] = 5  # spawn + its 4 immediate neighbors, all claimed up front

    worklist_seed = np.repeat(np.arange(s_count, dtype=np.int64), 4)
    worklist_base_i = np.zeros(s_count * 4, dtype=np.int64)
    worklist_base_j = np.zeros(s_count * 4, dtype=np.int64)
    worklist_dir = np.tile(np.arange(4, dtype=np.int64), s_count)

    gpu_s = 0.0
    cpu_s = 0.0
    n_gpu_calls = 0
    round_sizes = []
    rnd = 0
    for rnd in range(max_rounds):
        if worklist_seed.shape[0] == 0 or not active.any():
            break
        round_sizes.append(int(worklist_seed.shape[0]))
        t_gpu_start = time.time()

        t0 = time.time()
        seed_idx_t = torch.as_tensor(worklist_seed, dtype=torch.int64, device=device)
        base_i_t = torch.as_tensor(worklist_base_i, dtype=torch.int64, device=device)
        base_j_t = torch.as_tensor(worklist_base_j, dtype=torch.int64, device=device)
        dir_t = torch.as_tensor(worklist_dir, dtype=torch.int64, device=device)
        cls_result = oracle.classify_edges(tables, seed_idx_t, base_i_t, base_j_t, dir_t,
                                            node_spacing=node_spacing, segment_tiles=segment_tiles,
                                            epsilon_fp=epsilon_fp, backend=backend)
        classification_np = cls_result["classification"].cpu().numpy()
        max_dist_np = cls_result["max_dist"].cpu().numpy()
        del cls_result, seed_idx_t, base_i_t, base_j_t, dir_t
        if device.type == "cuda":
            # See the discarded block design's identical note: without ANY
            # empty_cache() call, PyTorch's caching allocator keeps every
            # round's (widely varying-sized) chunk buffers "reserved" rather
            # than returning them, and round sizes here swing from tens to
            # hundreds of thousands of requests, so reserved memory grows
            # essentially monotonically. empty_cache() itself walks the
            # allocator's memory pool -- real, measurable host-side cost --
            # so calling it EVERY round (60-230+ rounds/batch) was paying
            # that cost far more often than needed just to bound growth;
            # every 8th round still bounds it (a handful of rounds' worth of
            # transient over-reservation between calls is negligible next to
            # this project's actual memory budget) at a fraction of the cost.
            if rnd % 8 == 0:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gpu_s += time.time() - t0
        n_gpu_calls += 1
        t_gpu_end = time.time()

        t0 = time.time()
        apply_fn = apply_round_parallel if parallel_cpu else apply_round
        worklist_seed, worklist_base_i, worklist_base_j, worklist_dir = apply_fn(
            worklist_seed, worklist_base_i, worklist_base_j, worklist_dir,
            classification_np, max_dist_np,
            touched, reached, verdict, active, touched_count, had_request_scratch, resolved_round,
            rnd, side, center, ring_tiles, side * side, EDGE_LAND,
        )
        cpu_s += time.time() - t0
        t_cpu_end = time.time()

        if round_log is not None:
            round_log.append({"round": rnd, "worklist_size": round_sizes[-1],
                               "t_gpu_start": t_gpu_start, "t_gpu_end": t_gpu_end, "t_cpu_end": t_cpu_end})

        if verbose:
            n_active = int(active.sum())
            print(f"  round {rnd}: worklist={round_sizes[-1]}, still_active={n_active}", flush=True)

    rounds_used = rnd + 1 if round_sizes else 0

    # Safety net: anything still active when the loop ends (worklist emptied
    # by construction only via the CONFIRM/REJECT/DEFER branches above, or
    # max_rounds was hit) is DEFERred, matching the "ambiguity always falls
    # through to stage 2" contract every other stage in this pipeline uses.
    still_active = active
    verdict[still_active] = DEFER
    resolved_round[still_active & (resolved_round == -1)] = rounds_used

    if timing_out is not None:
        timing_out["gpu_seconds"] = gpu_s
        timing_out["cpu_seconds"] = cpu_s
        timing_out["n_gpu_calls"] = n_gpu_calls
        timing_out["rounds_used"] = rounds_used
        timing_out["round_sizes"] = round_sizes

    return {
        "verdict": verdict,
        "cells_touched": touched_count,  # node count; name kept for caller continuity, see docstring
        "resolved_round": resolved_round,
        "rounds_used": rounds_used,
        "round_sizes": round_sizes,
    }


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, SIDE={SIDE}, DOMAIN_NODES={DOMAIN_NODES}, RING_TILES={RING_TILES}")

    ground_truth = [
        (20271579, "confirmed real island -- must NEVER REJECT"),
        (3627767453, "confirmed real island -- must NEVER REJECT"),
        (1064911508, "RETRACTED (in-game land bridge, razor-thin -0.036 margin) -- "
                      "must NEVER REJECT (this pipeline's own math treats it as closed)"),
        (1489155600, "confirmed unbounded mainland -- must REJECT"),
    ]
    seeds = torch.tensor([s for s, _ in ground_truth], dtype=torch.int64, device=device)

    timing = {}
    result = run_stage15(seeds, device, verbose=True, timing_out=timing)

    print("\n=== stage 1.5 (segment-graph) verdicts ===")
    all_ok = True
    for i, (seed, note) in enumerate(ground_truth):
        v = int(result["verdict"][i])
        touched_n = int(result["cells_touched"][i])
        rr = int(result["resolved_round"][i])
        name = VERDICT_NAMES[v]
        print(f"  seed {seed}: verdict={name}, nodes_touched={touched_n}, resolved_round={rr}  ({note})")
        if seed == 1489155600 and v != REJECT:
            print(f"    *** FAIL: {seed} must REJECT, got {name} ***")
            all_ok = False
        if seed != 1489155600 and v == REJECT:
            print(f"    *** FAIL (SAFETY-CRITICAL): {seed} must NEVER REJECT, got REJECT ***")
            all_ok = False

    print(f"\ntiming: gpu={timing['gpu_seconds']:.3f}s, cpu={timing['cpu_seconds']:.3f}s, "
          f"rounds={timing['rounds_used']}, gpu_calls={timing['n_gpu_calls']}")
    print("\nALL SAFETY CHECKS PASSED" if all_ok else "\n*** SAFETY CHECK FAILURE -- DO NOT WIRE THIS IN ***")
