#include "stages.hpp"

Finder<void>::EvalResult stage1_eval(
    const MapGenSettings& settings, const NoisePrecompute& precompute, NoiseCache&, uint32_t seed, void*,
    FloodFillCache& cache
) {
    Noise noise(seed, true, settings.elevation_type == ELEVATION_2_0);

    cache.visited.clear();
    while (!cache.q.empty()) cache.q.pop();
    cache.q.push({0, 0});
    cache.visited.insert({0, 0});

    int64_t land_cells = 0;
    while (!cache.q.empty() && (int64_t)cache.visited.size() < FLOOD_FILL_MAX_CELLS) {
        auto [x, y] = cache.q.front(); cache.q.pop();
        if (noise.is_tile_water(settings, precompute, { (float)x, (float)y })) continue;
        land_cells++;

        const std::pair<int32_t,int32_t> neighbors[4] = {
            {x + FLOOD_FILL_STEP, y}, {x - FLOOD_FILL_STEP, y}, {x, y + FLOOD_FILL_STEP}, {x, y - FLOOD_FILL_STEP}
        };
        for (auto& n : neighbors) {
            if (cache.visited.insert(n).second) cache.q.push(n);
        }
    }

    // Queue non-empty => we stopped because of the cell cap, not because the flood fill ran out of
    // land to expand into. That means the landmass never closed off -- unbounded mainland, not a
    // real island.
    bool hit_cap = !cache.q.empty();
    return { .eliminate = hit_cap, .score = (float)(land_cells * FLOOD_FILL_STEP * FLOOD_FILL_STEP) };
}
