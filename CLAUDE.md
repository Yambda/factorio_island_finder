# CLAUDE.md — Factorio Island Seed Census

You are working inside a GPU-accelerated pipeline that searches all 2^32 Factorio seeds
(default map gen, `elevation_nauvis`) for naturally-enclosed spawn islands. The goal is a
**complete census in ~1 day on a single RTX 5090**, with a provable "no island missed" claim.
This is a mature project with measured numbers and a history of subtle correctness bugs.
Read this whole file before proposing or changing anything.

---

## 1. The problem

- "Island" = spawn's 4-connected land component does not reach the definitional ring.
  Production radius: **2,000 tiles** (was 1,400; enclosure is monotone in radius, so
  island@1400 ⇒ island@2000 and one walk at 2,000 yields both verdicts).
- Elevation is NOT one noise call: six independently seeded components
  (`nauvis_hills_plateaus`, `nauvis_bridge_billows`, `nauvis_persistance`,
  `nauvis_detail` [persistence-chained], `nauvis_macro` [2-oct x 1-oct product],
  `starting_lake_noise`) combined via min/max/lerp. ~20–25 gradient-noise evaluations
  per point.
- Permutation and gradient tables are **shuffled per-seed from a per-seed LFSR**.
  There is NO cross-seed structure at the hash/gather level (tested and disproven).
  Do not propose cross-seed batching of gathers or seed inversion; the LFSR is
  effectively a PRF and inversion degenerates to brute force.
- What IS seed-invariant: the amplitude ladder, per-octave value bounds, Lipschitz
  constants, and variances. These are computed once offline and are the basis of all
  certified-evaluation tricks (the "R ladder").

## 2. Architecture (target state)

Ray screen (ADVISOR only) → certified net walk (THE MEASURE, final verdict) →
tile-res refinement / dense CCL (deferral backstop) → async CPU verify (positives only).

1. **Tiered ray advisor** (all seeds). 48+ rays from spawn, tiered octave evaluation
   with hard +R margins. It DECIDES NOTHING. Outputs: (a) bearing prior — the clearest /
   latest-blocking rays point at the land corridor; (b) walk priority ordering.
   Rationale: any screen rejecting on a statistic over subsamples has an irreducible
   geometric false-negative channel (sub-stride moats, one clean ray under an
   all-blocked threshold). We do not let it have the last word.
2. **Certified net walk** (every seed's final verdict). 32-tile grid ("net") over the
   2,000-radius disk (~12.3k cells). Terrain evaluated **lazily** on net LINES only,
   in 32-tile segments, as the search touches them. Two graphs from the same lines:
   - **Reject graph (pessimistic)**: nodes = certified-land line tiles, 4-connected
     along lines. An escape path to the ring is a literal walkable witness → certified
     MAINLAND. Every tile on a used path must be evaluated and clear u0 by margin
     (no gaps, no "mostly land" segments).
   - **Confirm graph (optimistic)**: nodes = cells; edge open iff the shared segment
     has ≥1 land tile. Spawn disconnected from ring even under this over-approximation
     → certified ISLAND.
   - Neither fires → refine frontier cells' interiors at tile resolution and re-ask;
     budget overrun → DEFER to dense stage.
3. **Dense backstop** for deferrals: full mask + GPU CCL (label propagation + pointer
   doubling — NEVER Jump Flooding, see §4).
4. **CPU verify** (the real game binary flood-fill) audits POSITIVES asynchronously,
   off the critical path.

GPU/CPU split: GPU is a pure **terrain oracle** — flattened batch of (seed_id, segment)
requests in, 2-bit-per-tile classifications (certified-land / certified-water /
ambiguous, ε and R already folded in) out, ~8 bytes each way per segment. CPU threads
run the graph solver (compiled core, flat arrays — NOT Python-per-expansion), which is
where all verdicts are issued. Pipeline the two: GPU evaluates round k while CPU solves
round k−1 and assembles k+1. Batch across thousands of seeds per round; expand whole
frontiers per wave; speculatively pre-request segments along the ray bearing (FLOPs are
cheap, latency rounds are not). Segment cache keyed (seed, line, index), never
re-evaluate a segment.

## 3. Correctness invariants — NEVER violate

1. **Hard certificates on every REJECT path; heuristics only on ACCEPT paths.**
   The funnel audits positives (CPU verify) and never negatives. A false reject is
   silent and unrecoverable; a false confirm is caught downstream. Therefore:
   spurious WATER is harmless (blocks search → deferral); spurious LAND is fatal
   (bridges a moat → silent lost island). Any cell used in a reject witness must be
   certified land: full/exact evaluation, clearing u0 by ε_fp. Water-side checks may
   use coarse tiers, soft kσ, reduced precision.
2. **CONFIRM = exhaustion with zero pruning.** "Frontier empty" certifies enclosure
   ONLY if no reachable cell was ever discarded. Any structure that prunes (beam,
   top-K) must poison CONFIRM → DEFER for that seed. This bug already happened
   (see §4). At net scale (~12.3k cells, ~1.5KB bitmap/seed) full frontier retention
   is trivial — there is no reason to prune. Budget N ≈ 13k cells; exceeding budget
   is DEFER, never a classification.
3. **Connectivity duality.** Land is 4-connected ⇒ enclosing water is 8-connected.
   A single-tile purely-diagonal water chain legally encloses spawn. Reject-path
   moves are 4-connected only. Fully-evaluated net lines are exact sensors for both
   dualities (any 4-connected land path crossing a line lands ON it; any 8-connected
   water chain crossing a line has a water tile ON it) — the diagonal trap cannot
   hide from an evaluated line, only inside cell interiors (which only the confirm
   graph reasons about, safely, via over-approximation). GPU stage and CPU verifier
   must match conventions bit-for-bit.
4. **Cheapening axis = octaves, never geometry.** Coarse-octave evaluation with a
   hard +R margin is a certified superset screen: true water (E < u0) always flags
   under Ê < u0 + R, and every screen criterion in use is monotone in water flags.
   Subsampling POINTS (bigger stride, fewer rays) is unsound in the reject direction
   — it can only un-flag water. Tiers coarsen evaluation, not sampling.
5. **ε_fp gating.** Any decision quantity within ε_fp of its gate defers to CPU.
   At 4.29G seeds, 10^-9-rate discrepancies produce wrong verdicts.
6. **Definition is versioned, not perfect.** Census claims are relative to
   (radius cap, w_min if any, connectivity convention). A rerun costs ~1 day, so
   revise the definition via census v2, never by silently changing constants.

## 4. Bugs already hit — do not reintroduce

- **Jump Flooding (JFA)**: confirmed false negative — leaps 2^k cells in SPACE,
  teleporting connectivity across straits. Only label-space acceleration over
  verified adjacent pairs (union-find / label propagation + pointer doubling) or
  monotone dilation is exact. Monotone operators tolerate any chaotic/asynchronous
  schedule (Knaster–Tarski) — block-local Gauss–Seidel is safe.
- **Beam-pruning false CONFIRMs**: tile-scale greedy best-first beam emptied its
  frontier after pruning the true escape corridor → 85–90% of known-mainland seeds
  falsely CONFIRMed. Root cause was pruning itself, not the ranking metric
  ("farthest from spawn" ≡ "closest to ring" with centered spawn). Fixed by net
  scale + full frontier. Any future pruning structure must poison CONFIRM.
- **`exp2f`/`log2f` bias**: ~1.05% bias in `modified_amplitude()` on `nauvis_macro`,
  amplified up to 60x via `3 × macro × starting_macro_multiplier × ELEVATION_MAGNITUDE(20)`.
  Measured worst-case GPU/CPU diff 2.579 over adversarial 120k seeds → **ε_fp = 3.0**.
  NOT floor-boundary lattice flips (ruled out). Open upgrades: bias-correct or use
  exact-rounded exp2 at that one callsite, then re-measure; derive the analytic bound
  (max rel. error × 60x chain × macro value bound) so ε rests on proof, not an
  extrapolated max over 120k ≪ 4.29G seeds.
- **Anti-correlated features**: land density in a fixed box around spawn scored
  islands LOW and mainland HIGH (0/1000 survivors) — and its sign depends on box
  size relative to island size (the 1.47M-tile region reads 100% land in a small
  box). Fixed-radius single-ring water fraction: best real island peaked at 75%
  water on its best ring; no fixed threshold works. Do not resurrect these.
- **`torch.where` is predication, not sparsity**: eager PyTorch evaluates both
  branches. Sparse savings require a two-pass worklist (nonzero → dense eval on the
  flattened worklist → scatter) or a fused branching kernel.

## 5. Measured constants & facts

- Legacy funnel: ray screen shortlists ~1.1%; end-to-end 3,692 seeds/s (13.5x CPU
  baseline 273/s); old dilation stage 2 was 75–78% of wall-clock.
- Census target: 2^32 / 86,400s = **49,710 seeds/s**; design for ~100k/s headroom.
  Budget arithmetic: tiered advisor ~17k gradient evals/seed + directed tiered
  escape ~10–40k ⇒ ~30–60k evals/seed. Unfused evaluator ≈ 1.15e9 evals/s ⇒
  20–35k seeds/s (1.5–2.5 days); fused Triton (register-resident, per-seed tables
  in L1/shared, early-exit branch) ≥3e9 ⇒ one day with margin. Fallback: the
  problem is embarrassingly parallel over seed ranges — GPUs multiply linearly.
- ε_fp = 3.0 elevation units (16% headroom over 2.579 worst-case; mechanism §4).
- Confirmed islands n=2 (948K, 734K tiles) plus one retracted seed whose
  mis-measured "closed" region was 1.47M tiles. Component sizes are HEAVY-TAILED:
  with n this small, the next island exceeds the current max with prob ~1/(n+1).
  Never use a size prior as a rejection criterion — deferral boundary only.
  Chokepoint proof: flipping one ~0.036-margin tile at seed 1064911508 blows
  1.47M → 15.48M tiles (unbounded mainland). This is exactly what ε_fp prevents.
- Rarity ~1 in 1.75M is order-of-magnitude only: exact Poisson CI at n≈3 spans
  1-in-8.5M to 1-in-600k (~14x). Quote it as "order 10^-6"; the census replaces
  it with an exact count. Population estimate: ~2,500 islands in existence
  (95% CI roughly 500–7,200) — the census is an enumeration, not a sample.
- Segment payloads: ~8 B request, 8 B response (2 bits × 32 tiles). PCIe is never
  the bottleneck; synchronous per-expansion round-trips ARE (10–30µs each) — hence
  waves + cross-seed batching + speculation.
- Polar ray geometry is anisotropic: angular gap 0.131·r ⇒ 39 tiles at r=300,
  183 at r=1,400. Ray-adjacency "wrap/connecting" tests are FN-prone as filters;
  use only as island-certificate promoters (closed certified-water ring:
  Ê + R + L·(s'/2 + √2/2) < u0 along a refined loop) unless a gold-block audit
  licenses filtering.

## 6. Verdict ledger (2,000-radius walk)

- Escape to 2,000 ring → MAINLAND @1400 and @2000 (monotonicity).
- Exhaust → ISLAND @2000; component touches 1,400 ring? → new class
  "annulus island" (enclosed @2000, crosses @1400); else ISLAND @1400 too.
- Exhaust with component within ~2 net cells of the 2,000 ring → tag
  `ring-limited`; a later targeted pass (r=4,000, ~49k cells, same machinery)
  walks only those. Every bucket boundary is a certificate boundary.
- Report island count vs. moat radius. Percolation predicts ~exp(−r/ξ_eff) circuit
  decay; a steep falloff by r≈2,000 evidences cap-independence, a flat tail
  evidences truncation. If component sizes look power-law (near-critical), report
  the distribution, not a mean.

## 7. Standing test suite — run before trusting any change

1. **Ground-truth replay**: all seeds with CPU ground truth. Assert: every
   confirmed island CONFIRMs (or defers — never rejects); zero true mainlands
   CONFIRM. Extend with implementation-agreement diffs if a GPU solver is added
   (CPU solver stays the reference).
2. **Superset check** for any evaluation-tier change: coarse+R pass set ⊇ exact
   pass set on the historical shortlist, including all confirmed islands.
3. **Gold-block audit** before any production census: stride-5 / 96-ray / relaxed
   threshold / full-res screen over ~50M seeds; any island found that production
   settings reject is a measured FN. Zero hits still bounds recall statistically.
4. **ε_fp regression**: re-measure GPU/CPU divergence (incl. lattice-boundary and
   adversarial cases) after ANY change to the evaluator, precision, or math libs.
5. **500-known-mainland replay** for walk changes: REJECT rate (expect ~100% with
   full frontier), cells-touched and evals-per-verdict distributions, deferral
   rate, tail percentile.

## 8. Open measurements (highest value first)

- Fused-evaluator microbenchmark (Triton): ≥3e9 evals/s ⇒ single-GPU one-day is
  comfortable; <1.5e9 ⇒ shard or fall back to multi-GPU.
- Tier-A "bite": reject fraction of the coarsest octave subset with hard R on 1M
  random seeds. Kill criterion <50% ⇒ switch that tier to kσ soft margins and
  buy the license via the gold-block audit (state the measured FN bound).
- Escape-cost distribution on RANDOM seeds (not the shortlist — it is 99% weird
  near-islands and mis-estimates the mainland bulk) at r=2,000, ray-seeded,
  net scale. Also: annulus (1,400–2,000) field-statistics check — spawn-forcing
  and starting-lake terms fade with r; re-verify tier rejection and certification
  fractions transfer.
- Net cell-size sweep c ∈ {16, 32, 64} on the historical shortlist:
  decided-at-net fraction vs. evals-per-decision (32 trades certificate fire
  rate against laziness; a moat ≥ ~33 wide guarantees confirm-graph candidates).
- Minimum moat widths of confirmed islands vs. any stride in use.
- `exp2f` fix + analytic ε bound (§4).

## 9. Design heuristics that keep winning here

- Compaction cost = (compacted state size) × (compaction frequency). Compact at
  SEED granularity every K rounds (cheap, proven); never per-point per-level.
  Restructure sparse ideas as: batch-wide flattened worklists, fixed-shape
  adaptive stepping, or dense pre-gather — not ragged per-seed tensors.
- Witness asymmetry: mainland is provable by one path (cheap), island only by
  exhaustion (expensive at tile scale, cheap at net scale). Route accordingly.
- Evaluation is the cost; graph solves are microseconds. Optimize evals touched,
  not graph algorithmics.
- When speed surplus appears, spend it on RECALL (finer advisor geometry, wider
  margins) before wall-clock. The census's value is the completeness claim.
- Reruns cost a day. Prefer versioned definitions + reruns over cleverness that
  makes one run "final".
