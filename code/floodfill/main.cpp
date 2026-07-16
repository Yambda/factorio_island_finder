#include "stages.hpp"

int main(int argc, char* argv[]) {
    // Seed finder that looks for naturally-occurring islands (default elevation type 2.0, default
    // water settings) with the largest landmass around spawn, via a real flood fill from spawn
    // (see stages.cpp) -- eliminates any seed whose landmass never closes off (unbounded mainland).

    MapGenSettings settings;

    Finder<void> finder(settings);

    finder.add_stage_with_cache<FloodFillCache>(stage1_eval, stage1_settings);

    return finder.run("largest_island", argc, argv);
}
