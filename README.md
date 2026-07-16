# Factorio Natural Island Census

A GPU-accelerated search of all 4,294,967,296 possible Factorio seeds (default
map settings, no `island` elevation type) for spawns that land on a landmass
fully enclosed by water. Full writeup on r/factorio: [link].

**Result**: 2,595 confirmed natural islands. Rarity: ~1 in 1,655,000. Largest
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
- **`code/pipeline/`** — the rest of the funnel: ray-cast screening, the
  certified "net" walk (a coarse grid over the search radius, evaluated
  lazily), and the orchestration script (`find_islands.py`) that ties it all
  together and dispatches to the real flood fill for final confirmation.
- **`data/MASTER_all_fronts.csv`** — every confirmed island: seed, area
  (tiles²), and distance from spawn to the farthest point in the enclosed
  component.
- **`images/`** — `FLOODFILL_PROOF_*.png` files are rendered visualizations
  of the actual flood-filled region for a given seed (filename encodes area
  and seed). The `*_net.png` files show the certified net-walk grid at two
  resolutions (64-tile and 32-tile passes) for a few example seeds.

## The method, briefly

1. A cheap ray-cast screen from spawn throws out the ~99% of seeds that
   obviously connect to the mainland.
2. Survivors get walked on a certified grid over a 2,000-tile disk around
   spawn — lazily evaluated, only where the search actually goes.
3. Anything still ambiguous falls back to a dense flood fill.
4. Every positive result is re-checked against the real, exact flood fill
   (`code/floodfill/stages.cpp`) before being counted — nothing in `data/`
   is a "probably."

See `CLAUDE.md` for the full design writeup, correctness invariants, and the
project's history of subtle bugs (kept in for anyone curious how a project
like this actually goes, not just the clean final result).
