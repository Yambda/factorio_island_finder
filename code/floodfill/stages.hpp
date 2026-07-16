#pragma once

#include "finder.hpp"
#include <queue>
#include <unordered_set>
#include <utility>
#include <cstdint>

// Uses the default elevation type (2.0) and default water settings (water_scale/coverage = 1.0):
// unlike the dedicated "island" elevation type, 2.0 doesn't guarantee a water-locked spawn, so most
// seeds are just unbounded mainland. Two cheaper proxies were tried and both failed: land-density in
// a box is anti-correlated with real islands (confirmed: 1000/1000 top-scoring candidates were
// unbounded mainland), and a "ring around spawn is all water" closure check is too strict for
// irregular coastlines (confirmed against a known real island: never exceeded 75% water on any
// tested ring). So this stage does a real connected-component flood fill from spawn directly --
// slower per seed, but it's the one method already proven correct.
// RAISED from 200,000 to 2,000,000 (2026-07-09): the natural-island census (RING_TILES=5000,
// gpu/stage1_5_cascade.py) found a real confirmed island at 3,097,730 tiles^2 = 193,608 cells --
// only 3.2% under the old 200,000-cell cap. With records still climbing during an active run, the
// old cap risked a real bounded island silently reading as "hit cap" (indistinguishable from
// genuine unbounded mainland) and being discarded as a false confirm. 2,000,000 cells / 32,000,000
// tiles^2 gives an order of magnitude of headroom above the current record.
// NOTE the prior warning below (1,000,000 "timed out badly") predates this change and was measured
// against the OLD unordered_set-based visited-tracking at that scale -- re-measure actual capped-case
// timing at 2,000,000 before trusting this doesn't reproduce that slowdown (see cpu_verify.py's own
// distribution-tracking: if the largest confirmed island's cell count exceeds 60% of this constant,
// that's the signal to raise it again, not to run this number up further speculatively).
// ORIGINAL NOTE: 1,000,000 timed out badly: unordered_set gets cache-unfriendly at that scale (each
// of the ~4M neighbor lookups/inserts starts missing cache), making a "give up, unbounded" seed cost
// roughly 1s instead of the ~250us naively expected from raw is_tile_water() cost alone. 200,000 still
// covers everything seen so far (the guaranteed-island-type winner was 2,582,832 tiles^2, under
// this cap's ~3,200,000) while keeping the worst case (most seeds, since most never close off) cheap.
constexpr int32_t FLOOD_FILL_STEP = 4;
constexpr int64_t FLOOD_FILL_MAX_CELLS = 2'000'000; // ~32,000,000 tiles^2 if ever reached

struct PairHash {
    size_t operator()(const std::pair<int32_t,int32_t>& p) const {
        return (uint64_t)(uint32_t)p.first << 32 | (uint32_t)p.second;
    }
};

// Reused across calls (cleared each time) so the hot loop doesn't reallocate a hash set/queue per seed.
struct FloodFillCache {
    std::unordered_set<std::pair<int32_t,int32_t>, PairHash> visited;
    std::queue<std::pair<int32_t,int32_t>> q;
};

constexpr Finder<void>::StageSettings stage1_settings{
    .check_twin_seeds = false,
    .check_water_settings = false,
    .check_elevation_types = false,

    .seed_nb_to_next_stage = 1'000
};

Finder<void>::EvalResult stage1_eval(
    const MapGenSettings&, const NoisePrecompute&, NoiseCache&, uint32_t seed, void*, FloodFillCache&
);
