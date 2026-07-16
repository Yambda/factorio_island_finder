# Factorio Natural Island Census

A GPU-accelerated search of all 4,294,967,296 possible Factorio seeds (default
map settings, no `island` elevation type) for spawns that land on a landmass
fully enclosed by water. Full writeup on r/factorio: [link].

**Result**: 2,599 confirmed natural islands. Rarity: ~1 in 1,652,000. Largest
found: seed `1925425905`, 7,430,080 tiles².

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
- **`images/part1/`, `part2/`, `part3/`** — split into three folders purely
  because GitHub's file browser caps directory listings at 1,000 entries;
  together they're one set. `FLOODFILL_PROOF_*.png` files are rendered
  visualizations of the actual flood-filled region for a given seed
  (filename encodes area and seed). The `*_net.png` files show the
  certified net-walk grid at two resolutions for a few example seeds.
- **`excluded_zero_area_images/`** — proof images for the 98 seeds excluded
  per the section below (kept for the record, not counted as islands).
- **`leader_showcase/`** — the current record holder (seed `1925425905`,
  7,430,080 tiles²) rendered at every mesh scale actually used in
  production, plus its flood-fill result and ring coverage. See below.

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

### `leader_showcase/` — the current record holder at every scale

- `mesh_scale_128tile.png` / `mesh_scale_64tile.png` / `mesh_scale_32tile.png`
  — the same terrain, same island, with each pass's net grid overlaid so
  you can see how the resolution changes across the cascade.
- `floodfill_result.png` — the actual flood-filled result: island in green,
  connected water in blue-gray, everything else (mainland) in red.
- `ring_coverage.png` — the 1,400 / 2,000 / 5,000-tile reference rings, with
  the farthest enclosed tile from spawn marked (this island's farthest
  point is ~3,626 tiles out — well inside the 5,000-tile boundary).

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
