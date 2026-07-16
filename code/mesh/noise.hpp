#pragma once

#include "utils.hpp"
#include "constants.hpp"
#include "random.hpp"
#include <algorithm>
#include <cassert>

class NoisePrecompute {
public:
    NoisePrecompute() = delete;
    NoisePrecompute(const MapGenSettings& settings);

    inline const auto& get_starter_penalties() const { return s_starter_penalties; }
    inline const auto& get_regular_penalties() const { return s_regular_penalties; }

    inline const auto& get_base_regular_densities() const { return _base_regular_densities; }
    inline const auto& get_base_regular_quantities() const { return _base_regular_quantities; }

    inline const auto& get_starter_base_densities() const { return _starter_base_densities; }
    inline const auto& get_starter_quantities() const { return _starter_quantities; }
    inline const auto& get_starter_radii() const { return _starter_radii; }

    inline float get_water_level() const { return _water_level; }
    
    inline float get_nauvis_hills_input_scale() const { return _nauvis_hills_input_scale; }
    inline float get_nauvis_hills_cliff_level_input_scale() const { return _nauvis_hills_cliff_level_input_scale; }
    inline float get_starting_macro_multiplier_base() const { return _starting_macro_multiplier_base; }
    inline float get_nauvis_bridge_billows_input_scale() const { return _nauvis_bridge_billows_input_scale; }
    inline float get_nauvis_persistance_input_scale() const { return _nauvis_persistance_input_scale; }
    inline float get_nauvis_offset_x() const { return _nauvis_offset_x; }
    inline float get_nauvis_detail_input_scale() const { return _nauvis_detail_input_scale; }
    inline float get_nauvis_macro_input_scale() const { return _nauvis_macro_input_scale; }
    inline float get_starting_island_multiplier() const { return _starting_island_multiplier; }

private:
    inline static std::array<float, STARTER_NB_SPOTS>* s_starter_penalties = nullptr;
    inline static std::array<float, REGULAR_MAX_NB_SPOTS>* s_regular_penalties = nullptr;

    std::array<float, NB_RESOURCE_TYPE> _starter_base_densities;
    std::array<float, NB_RESOURCE_TYPE> _starter_quantities;
    std::array<float, NB_RESOURCE_TYPE> _starter_radii;

    std::array<float, NB_RESOURCE_TYPE> _base_regular_densities;
    std::array<float, NB_RESOURCE_TYPE> _base_regular_quantities;

    float _water_level;

    float _nauvis_hills_input_scale;
    float _nauvis_hills_cliff_level_input_scale;
    float _starting_macro_multiplier_base;
    float _nauvis_bridge_billows_input_scale;
    float _nauvis_persistance_input_scale;
    float _nauvis_offset_x;
    float _nauvis_detail_input_scale;
    float _nauvis_macro_input_scale;
    float _starting_island_multiplier;
};

PositionI32 starter_lake_position(uint32_t seed);

class Noise {
public:
    Noise() = delete;
    /**
     * quick_multioctave_precompute will make this constructor significantly
     * slower but it is necessary for the quick_multioctave_noise function
     * and the other functions that use quick_multioctave_noise (lake/elevation functions)
     * 
     * Also the stater_patches function requires a quick_multioctave_precompute == true
     * Noise as one of its arguments.
     */
    Noise(uint32_t seed0, bool quick_multioctave_precompute, bool elevation2_0_precompute);

    float noise(uint8_t seed1, PositionF32 pos, float input_scale, float output_scale, float offset_x, float offset_y) const;
    float multioctave_noise(
        uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
        float input_scale, float output_scale, float offset_x, float offset_y
    ) const;
    float persistence_multioctave_noise(
        uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
        float input_scale, float output_scale, float offset_x, float offset_y
    ) const;
    float quick_multioctave_noise(
        uint8_t seed1, PositionF32 pos, uint32_t octaves, float input_scale,
        float output_scale, float offset_x, float offset_y, float octave_input_scale_multiplier,
        float octave_output_scale_multiplier, uint32_t octave_seed0_shift
    ) const;

    // 2.0 default elevation
    float elevation_nauvis(const MapGenSettings&, const NoisePrecompute&, PositionF32 pos) const;
    // 1.1 default elevation, also used by patch generation
    float elevation_lakes(const MapGenSettings&, const NoisePrecompute&, PositionF32 pos) const;
    float elevation_island(const MapGenSettings&, const NoisePrecompute&, PositionF32 pos) const;

    inline float elevation(const MapGenSettings& settings, const NoisePrecompute& precompute, PositionF32 pos) const {
        switch (settings.elevation_type) {
            case ELEVATION_2_0: return elevation_nauvis(settings, precompute, pos);
            case ELEVATION_1_1: return elevation_lakes(settings, precompute, pos);
            case ELEVATION_ISLAND: return elevation_island(settings, precompute, pos);
            default: throw std::runtime_error("Invalid elevation type.");
        }
    }

    inline bool is_tile_water(const MapGenSettings& settings, const NoisePrecompute& precompute, PositionF32 pos) const {
        return elevation(settings, precompute, pos) <= 0;
    }

    inline bool any_water_in_box(const MapGenSettings& settings, const NoisePrecompute& precompute, const BoxI32& box, int32_t sampling_distance) {
        for (int32_t x = box.left_top.x; x < box.right_bottom.x; x += sampling_distance) {
            for (int32_t y = box.left_top.y; y < box.right_bottom.y; y += sampling_distance) {
                if (is_tile_water(settings, precompute, { (float)x, (float)y })) return true;
            }
        }

        return false;
    }

private:
    struct Permutations {
        std::array<uint8_t, 256> p1;
        std::array<uint8_t, 256> p2;
        std::array<uint8_t, 256> p3;
        std::array<std::pair<float, float>, 256> grad;
    };

    void _make_permutations(uint32_t seed0, Permutations& permutations);

    float _noise_internal(const Permutations&, uint8_t seed1, PositionF32 pos, float input_scale, float output_scale, float offset_x, float offset_y) const;
    std::pair<float, float> _gradient(uint8_t x, uint8_t y, uint8_t p1, const Permutations&) const;
    float _multioctave_noise_internal(
        const Permutations&, uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
        float input_scale, float output_scale, float offset_x, float offset_y
    ) const;
    float _persistence_multioctave_noise_internal(
        const Permutations&, uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
        float input_scale, float output_scale, float offset_x, float offset_y
    ) const;

    float _make_0_12like_lakes(const NoisePrecompute&, PositionF32 pos, float bias, float seg_multiplier) const;
    float _finish_elevation(const NoisePrecompute&, float elevation, PositionF32 pos, float seg_multiplier) const;

    float _nauvis_hills_plateaus(const MapGenSettings&, const NoisePrecompute&, PositionF32 pos) const;
    float _elevation_nauvis_function(const MapGenSettings&, const NoisePrecompute&, PositionF32 pos, float added_cliff_elevation) const;

    bool _quick_multioctave_precompute;
    std::array<Permutations, QUICK_MULTIOCTAVE_MAX_SEED_OFFSET> _permutations;
    bool _elevation2_0_precompute;
    std::array<Permutations, (size_t)Seed0CustomOffsets::NB> _custom_offset_permutations;

    PositionI32 _starter_lake;
};

struct Candidate {
    PositionI32 pos;
    union {
        float density;      // used by regular patches
        float favorability; // used by starter patches
        float radius;       // used by biters
    };
    float quantity;
};

template<int NbCandidates>
struct CandidateChunk {
    std::array<PositionI32, NbCandidates> data;
    uint32_t id = 0;
    size_t count = 0;

    inline void insert(PositionI32 pos) {
        assert(count < data.size());
        data[count] = pos;
        count++;
    }

    inline void clear() {
        count = 0;
    }

    inline auto begin() {
        return data.begin();
    }

    inline auto end() {
        return data.begin() + count;
    }
};

struct Patch {
    PositionI32 pos;
    float radius;
    float quantity;
};

class PatchArray {
public:
    inline void insert(PositionI32 pos, float radius, float quantity) {
        assert(_count < _data.size());
        _data[_count] = {pos, radius, quantity};
        _count++;
    }

    inline void insert(const Patch& patch) {
        assert(_count < _data.size());
        _data[_count] = patch;
        _count++;
    }

    inline void remove(size_t idx) {
        assert(_count > 0);
        while (idx + 1 < _count) {
            _data[idx] = _data[idx + 1];
            idx++;
        }
        _count--;
    }

    inline void clear() {
        _count = 0;
    }

    inline auto begin() {
        return _data.begin();
    }

    inline auto begin() const {
        return _data.begin();
    }

    inline auto end() {
        return _data.begin() + _count;
    }

    inline auto end() const {
        return _data.begin() + _count;
    }

    inline size_t size() const {
        return _count;
    }

    inline Patch& at(size_t i) {
        assert(i < _count);
        return _data[i];
    }

    inline const Patch& at(size_t i) const {
        assert(i < _count);
        return _data[i];
    }

    inline Patch& operator[](size_t i) {
        return at(i);
    }

    inline const Patch& operator[](size_t i) const {
        return at(i);
    }

private:
    std::array<Patch, MAX_NB_SPOTS> _data;
    size_t _count = 0;
};

// NOT thread safe.
class NoiseCache {
public:
    NoiseCache() {
        _starter_chunks = new CandidateChunk<NB_CANDIDATES[STARTER]>[CHUNK_COUNTS[STARTER]*CHUNK_COUNTS[STARTER]];
        _regular_chunks = new CandidateChunk<NB_CANDIDATES[REGULAR]>[CHUNK_COUNTS[REGULAR]*CHUNK_COUNTS[REGULAR]];
    }

    ~NoiseCache() {
        delete[] _starter_chunks;
        delete[] _regular_chunks;
    }

    inline uint32_t get_id() { return ++_id; }
    template<PatchType Type>
    inline auto& get_chunks() {
        if constexpr (Type == STARTER) return _starter_chunks;
        else return _regular_chunks;
    }

private:
    uint32_t _id = 0;
    CandidateChunk<NB_CANDIDATES[STARTER]>* _starter_chunks;
    CandidateChunk<NB_CANDIDATES[REGULAR]>* _regular_chunks;
};

using Patches = std::array<PatchArray, NB_RESOURCE_TYPE>;

inline void iterate_patches(Patches& patches, std::function<void(Patch&, ResourceType, size_t i)> callback) {
    for (ResourceType type = IRON; type < NB_RESOURCE_TYPE; type++) {
        PatchArray& array = patches[type];
        for (size_t i = 0; i < array.size(); i++) {
            callback(array[i], type, i);
        }
    }
}

inline void iterate_patches(const Patches& patches, std::function<void(const Patch&, ResourceType, size_t i)> callback) {
    for (ResourceType type = IRON; type < NB_RESOURCE_TYPE; type++) {
        const PatchArray& array = patches[type];
        for (size_t i = 0; i < array.size(); i++) {
            callback(array[i], type, i);
        }
    }
}

// Noise& must have quick_multioctave_precompute == true
Patches starter_patches(const MapGenSettings&, const NoisePrecompute&, const Noise&, NoiseCache&, uint32_t seed);
Patches regular_patches(const NoisePrecompute&, NoiseCache&, uint32_t seed, PositionI32 region);
PatchArray enemy_bases(const MapGenSettings&, const NoisePrecompute&, uint32_t seed, PositionI32 region);