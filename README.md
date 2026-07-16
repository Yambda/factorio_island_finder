# Factorio Natural Island Census

A GPU-accelerated search of all 4,294,967,296 possible Factorio seeds (default
map settings, no `island` elevation type) for spawns that land on a landmass
fully enclosed by water. Full writeup on r/factorio: [link].

**Result**: 2,599 confirmed natural islands. Rarity: ~1 in 1,652,000. Largest
found: seed `1925425905`, 7,430,080 tiles².

## Hardware and performance

Ran mainly on a single home PC: **AMD Ryzen 9 9950X3D** (16 cores / 32
threads) + **NVIDIA RTX 5090** (32GB VRAM). Sustained throughput on that
one machine was **~7,300–9,700 seeds/sec** (about 27–36x a single-threaded
CPU-only flood fill, which runs around 273 seeds/sec). At that rate alone,
the full 2³² seed space is roughly a 5–7 day job.

To close it out faster, the last stretch of the range also ran on a
temporary multi-cloud burst — a Google Cloud L4 instance and two rented
Vast.ai boxes (4x and 8x RTX 5090) — pushing the combined rate as high as
**~92,000 seeds/sec** across 14 parallel workers for the final few hours.
None of that infrastructure is part of this repo; it was scaffolding for
one run, not a reusable part of the pipeline.

## Credit where it's due

This project exists because of two things I didn't build myself:

- **[u/int_ua's "Peninsula Seed Finder: First Island seed, 20271579" post](https://www.reddit.com/r/factorio/comments/1twnccz/peninsula_seed_finder_first_island_seed_20271579/)**
  on r/factorio — seeing a single hand-found natural island seed is what
  made me want to know how many of these actually exist across the whole
  seed space, not just find one more by hand.
- **[ness056/fast-factorio-seed-finder](https://github.com/ness056/fast-factorio-seed-finder)**
  — the reverse-engineered Factorio map-generation algorithms this whole
  pipeline builds on (`code/mesh/noise.cpp`/`gradients.cpp` are lightly
  adapted from that repo). Wube gave permission for those algorithms to be
  shared, per that project's own README.

Everything in `code/` beyond that base — the GPU noise reimplementation,
the certified cascade, the multi-machine scale-out, and the census itself
— is new work for this project.

## What's in here

- **`code/mesh/`** — the terrain generation. `noise_gpu.py` (and
  `noise_triton_fused.py`, a fused Triton kernel version) is a from-scratch
  reimplementation of Factorio's actual elevation noise — six independently
  seeded noise components combined via min/max/lerp, matched against the
  reverse-engineered reference C++ in `noise.cpp`/`gradients.cpp` (credit to
  the original reverse-engineering project this built on).
- **`code/floodfill/`** — the enclosure check. `stages.cpp` is the exact,
  ground-truth flood fill (a real BFS from spawn using the same terrain
  code) — this is what every single confirmed result in `data/` was verified
  against. `stage2_floodfill.py` is the GPU-batched approximate version used
  to pre-filter candidates before the exact check.
- **`code/pipeline/`** — the certified "cascade" — a multi-resolution grid
  ("net") over the 5,000-tile search radius, evaluated lazily, only where
  the search actually goes (`stage1_5_cascade.py`/`stage1_5_solver.py`/
  `stage1_5_oracle.py`) — plus the orchestration script (`find_islands.py`)
  that ties it all together and dispatches to the real flood fill for final
  confirmation. (CLAUDE.md describes an optional ray-cast advisor stage too,
  but it was disabled for this run — every seed went straight to the
  cascade.)
- **`data/MASTER_all_fronts.csv`** — every confirmed island: seed, area
  (tiles²), and distance from spawn to the farthest point in the enclosed
  component.
- **`data/ZERO_AREA_SEEDS_excluded.csv`** — see "the zero-area edge case"
  below.
- **`data/island_area_histogram.png`** — a log-tail-then-linear histogram
  of every confirmed island's area. The 6 zero-area-edge-case islands
  (3–32 tiles²) are pulled out as noted outliers so the main population —
  all ≥ 522,416 tiles² — isn't squashed onto a few pixels; median
  955,344 tiles², largest 7,430,080.
- **`images/part1/`, `part2/`, `part3/`** — split into three folders purely
  because GitHub's file browser caps directory listings at 1,000 entries;
  together they're one set, and every file is a `FLOODFILL_PROOF_*.png` —
  a rendered visualization of the actual flood-filled region for one
  confirmed seed (filename encodes area and seed). For net-grid
  visualizations, see `leader_showcase/` below instead.
- **`images/top_400_mosaic.jpg`** — the 400 largest confirmed islands (by
  area, largest first, reading left-to-right/top-to-bottom), each one's
  flood-fill result shrunk into a 150×150 tile and packed into a single
  20×20 grid — one image, ~2.6MB, that gives a sense of scale and shape
  variety across the top of the distribution at a glance.
- **`excluded_zero_area_images/`** — proof images for the 98 seeds excluded
  per the section below (kept for the record, not counted as islands).
- **`leader_showcase/`** — the current record holder (seed `1925425905`,
  7,430,080 tiles²): only the real data the pipeline actually calculated
  at each cascade pass, plus its flood-fill result — nothing synthetic.
  See below.

## Mesh sizes actually used

The cascade runs seeds through up to three resolutions before falling back
to a dense flood fill, coarsest first (all node spacings and domain radii
below are the real production constants from `stage1_5_cascade.py`):

| Pass | Node spacing | Domain radius (nodes) | Nominal radius |
|---|---|---|---|
| 0 (`--stage15-three-tier`) | 128 tiles | 56 | 7,168 tiles |
| 1 | 64 tiles | 111 | 7,104 tiles |
| 2 (residual only) | 32 tiles | 222 | 7,104 tiles |

The definitional enclosure boundary — the radius a seed's land component
has to stay inside to count as an island — is **`RING_TILES = 5000`**
tiles, hardcoded in `stage1_5_oracle.py`. (A separate, smaller 2,000-tile
radius exists in `find_islands.py` too, but that's only the disabled
ray-screen stage's own parameter — unrelated to the actual verdict.)

Positives that survive the cascade get a real, exact flood fill
(`code/floodfill/stages.cpp`) before being counted. That flood fill is
capped at `FLOOD_FILL_MAX_CELLS = 2,000,000` cells, each cell covering
`FLOOD_FILL_STEP² = 16` tiles — a **32,000,000 tile² cap**. A seed that
hits this cap without its search queue emptying is unbounded mainland, not
a real island (mainland never "closes off" a finite frontier); the cap
exists to bound worst-case runtime, not to reject genuinely enclosed
islands, since actual confirmed islands are many times smaller than this.

### `leader_showcase/` — only what the pipeline actually calculated for this seed

The elevation field itself (`elevation_nauvis` in `code/mesh/`) has no
edges — it's a pure function of `(seed, x, y)` computable at any tile
coordinate, exactly like the real game's unbounded world. So the first
version of these images (rendering a densely-colored square around
spawn for every panel) was misleading in a specific way: it painted
every pixel in the frame as if the pipeline had "looked at" all of it,
when in reality the cascade only ever evaluates the exact net lines its
lazy search visits, and the flood fill only ever visits the cells it
actually reaches. These images now show *only* that — real evaluated
data on an otherwise blank canvas, produced by instrumenting the actual
production code for one seed (`1925425905`), not by re-deriving a
look-alike:

- **Panels 2–4 (`2_cascade_pass0_128tile.png`, `3_cascade_pass1_64tile.png`,
  `4_cascade_pass2_32tile.png`)**: every line drawn is one real
  `(base_i, base_j, dir_code)` segment request that `stage1_5_solver.py`
  actually sent to `stage1_5_oracle.classify_edges` for this seed during
  that pass — captured by monkeypatching the oracle call (no source
  changes) and running the real `stage1_5_cascade.run_cascade(...,
  tiers=THREE_TIER_CASCADE)`. Each segment is colored by *its own real
  verdict*: green for `EDGE_LAND`, blue-gray for `EDGE_BLOCKED` — not a
  re-derived color. The short blue-gray "whiskers" sticking off the green
  skeleton are exactly what they look like: the reject-graph's own
  tested-and-blocked probe edges. Segment counts for this seed: pass 0
  (128-tile) 476 segments, pass 1 (64-tile) 1,845, pass 2 (32-tile) 6,943
  — all three passes CONFIRM, matching the official verdict, and nothing
  outside those segments was ever asked about, so nothing outside them is
  drawn.
- **Panel 5 (`5_floodfill_result.png`)**: a real 4-connected BFS from
  spawn over the same elevation field, colored only where the search
  actually dequeued a cell — matching `code/floodfill/stages.cpp`'s
  `cache.visited` semantics exactly (land cells that got kept, plus the
  water cells checked immediately at the boundary and rejected). This
  run's component: 464,380 cells × 16 tiles²/cell = **7,430,080 tiles²**,
  matching the official record exactly. You'll notice small dark patches
  *inside* the green landmass — those are interior lakes more than one
  tile deep from any bordering land. The real algorithm never needs to
  look past the first water tile in a direction to know that direction is
  blocked, so it never queries deeper into a lake than that — and neither
  does this image.
- **Panel 1 (`1_mesh_terrain.png`)** — the mesh has no discrete "step" of
  its own in production (it's a pure function queried on demand by
  everything else), so rather than invent an arbitrary scan region, this
  panel is the union of every tile the pipeline queried across all four
  other panels: the flood-fill's visited region plus every cascade
  segment from all three passes layered on top. It's literally the
  complete, real set of `(seed, x, y)` queries this seed's run made
  against the mesh — nothing more, nothing invented.
- **`composite_1234.png`** — panels 2, 3, 4, and 5 (pass 0 → pass 1 →
  pass 2 → flood fill) in a row across the top, and panel 1 (the mesh
  union) alone underneath, all numbered in their corners — the net's
  growing density pass over pass reads left to right, with the complete
  query union shown separately below it.

All five panels (and the composite) share the same tile-to-pixel mapping
and the same two-color key — island/land `(40,230,60)` green,
water/blocked `(51,83,95)` blue-gray — against a neutral dark background
`(22,24,28)` standing in for "never evaluated." There's no ring or
boundary circle drawn anywhere; the shape of the data itself is the only
thing on the canvas. The crop is tight rather than an arbitrary fixed
window: it's sized to 1.1x the actual bounding box of everything any
panel evaluated (segments and flood-fill visits together), so the data
fills the frame instead of floating in a mostly-empty square.

## The zero-area edge case

105 seeds originally came out of the pipeline with area 0 — a small lake
happens to sit exactly on the spawn tile, so a flood fill starting *at*
spawn finds zero land before ever leaving water. The pipeline counted these
as "enclosed" by the letter of the definition, but the real game would
never actually spawn you in water — it forces you onto the nearest solid
ground. Checking what's actually next to spawn:

- **101 of the 105** turn out to just be ordinary mainland once you flood
  fill from the nearest real land tile instead of the water-classified
  origin — not islands at all, just an unrelated small lake sitting on an
  otherwise normal mainland spawn. These are in
  `data/ZERO_AREA_SEEDS_excluded.csv`, kept for the record but not counted.
- **4 of the 105** are genuinely tiny enclosed islands: `44981135` (6
  tiles²), `1915445000` (3 tiles²), `1931003541` (3 tiles²), and
  `3436407377` (3 tiles²) — confirmed by direct in-game check, not just the
  automated re-analysis. These are included in `MASTER_all_fronts.csv`
  with their real (non-zero) area.

Two more tiny islands worth flagging as genuinely marginal: `2098331609`
(16 tiles²) and `2888143787` (32 tiles²) aren't part of the zero-area
group — spawn itself is on land for both — but they sit right at the
precision limit of this whole approach. Re-deriving their area with a
simple from-scratch script gets meaningfully smaller numbers (9 and 7
tiles² respectively) than the pipeline's own certified result, and tracing
the boundary tiles shows elevation values within ~0.01–0.02 of the exact
water/land threshold — far closer than this project's own measured
worst-case GPU/CPU floating-point divergence (2.579, see `CLAUDE.md`).
That's consistent with a few boundary tiles genuinely disagreeing between
implementations, not a bug in either one. Both are kept in the confirmed
count; treat their exact tile counts as less certain than everything else
in this dataset.

## The method, briefly

1. **Mesh**: generate the elevation field (`code/mesh/`) — the same six
   noise components, combined the same way, as the real game. Fully eager:
   every requested point gets every noise component computed unconditionally.
2. **Cascade**: walk a certified grid over the 5,000-tile disk around
   spawn — lazily evaluated (only the net lines the search frontier
   actually touches get requested from the mesh at all). Escapes to the
   ring boundary confirm mainland; full enclosure of the frontier confirms
   island; anything ambiguous defers to a denser pass or the dense flood
   fill.
3. **Floodfill**: every positive result is re-checked against the real,
   exact flood fill (`code/floodfill/stages.cpp`) before being counted —
   nothing in `data/` is a "probably."

See `CLAUDE.md` for the full design writeup, correctness invariants, and the
project's history of subtle bugs (kept in for anyone curious how a project
like this actually goes, not just the clean final result).
