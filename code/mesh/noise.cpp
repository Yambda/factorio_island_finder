#define _USE_MATH_DEFINES
#include "noise.hpp"
#include "gradients.hpp"
#include <numeric>
#include <cmath>
#include <print>
#include "math.hpp"

NoisePrecompute::NoisePrecompute(const MapGenSettings& settings) {
    if (!s_starter_penalties) {
        s_starter_penalties = new std::array<float, STARTER_NB_SPOTS>[REGION_SIZES[STARTER]*REGION_SIZES[STARTER]];
        s_regular_penalties = new std::array<float, REGULAR_MAX_NB_SPOTS>[REGION_SIZES[REGULAR]*REGION_SIZES[REGULAR]];

        for (int32_t i = 0; i < REGION_SIZES[STARTER]; i++) {
            for (int32_t j = 0; j < REGION_SIZES[STARTER]; j++) {
                int32_t x = i - (REGION_SIZES[STARTER] / 2);
                int32_t y = j - (REGION_SIZES[STARTER] / 2);
    
                Random::random_penalty_at<STARTER_NB_SPOTS>(
                    s_starter_penalties[i + j*REGION_SIZES[STARTER]], x, y, 0.5f, 1
                );
            }
        }
    
        for (int32_t i = 0; i < REGION_SIZES[REGULAR]; i++) {
            for (int32_t j = 0; j < REGION_SIZES[REGULAR]; j++) {
                int32_t x = i - (REGION_SIZES[REGULAR] / 2);
                int32_t y = j - (REGION_SIZES[REGULAR] / 2);
    
                Random::random_penalty_between<REGULAR_MAX_NB_SPOTS>(
                    s_regular_penalties[i + j*REGION_SIZES[REGULAR]], x, y, RANDOM_SPOT_SIZE_MINIMUM, RANDOM_SPOT_SIZE_MAXIMUM, 1
                );
            }
        }
    }

    for (ResourceType type = IRON; type < NB_RESOURCE_TYPE; type++) {
        /*
        size_effective_distance_at = if(has_starting_area_placement == -1, distance, distance - regular_patch_fade_in_distance)

        regular_density_at(distance) = "base_density * frequency_multiplier * size_multiplier * \z
                        if(has_starting_area_placement == -1, 1, clamp((distance - starting_resource_placement_radius) / regular_patch_fade_in_distance, 0, 1)) * \z
                        (1 + clamp(size_effective_distance_at(distance) / double_density_distance, 0, 1))"

        regular_spot_quantity_base_at = "1000000 / base_spots_per_km2 / frequency_multiplier * regular_density_at(distance)"

        regular_spot_quantity_expression = "random_penalty_between(random_spot_size_minimum, random_spot_size_maximum, 1) * \z
                                            regular_spot_quantity_base_at(distance)"
        */
        _base_regular_densities[type] = BASE_DENSITIES[type] * settings.frequencies[type] * settings.sizes[type];
        _base_regular_quantities[type] = 1000000.f / BASES_SPOTS_PER_KM2[type] / settings.frequencies[type];

        /*
        density_expression = starting_amount / (pi * starting_resource_placement_radius * starting_resource_placement_radius) * \z
            starting_modulation,\z

        spot_quantity_expression = starting_area_spot_quantity,\z

        spot_radius_expression = starting_rq_factor * starting_area_spot_quantity ^ (1/3),\z

        spot_favorability_expression = clamp((elevation_lakes - 1) / 10, 0, 1) * starting_modulation * 2 - \z
            distance / starting_resource_placement_radius + random_penalty_at(0.5, 1),\z

        starting_amount = "20000 * base_density * (frequency_multiplier + 1) * size_multiplier",

        starting_area_spot_quantity = "starting_amount / starting_patches_split / frequency_multiplier",

        starting_modulation = "starting_resource_placement_radius > distance",
        */
        const float starting_amount = 20000.f * BASE_DENSITIES[type] * (settings.frequencies[type] + 1.f) * settings.sizes[type];
        const float density_expression = starting_amount /
            ((float)M_PI * STARTING_RESOURCE_PLACEMENT_RADIUS*STARTING_RESOURCE_PLACEMENT_RADIUS);
        _starter_base_densities[type] = REGION_SIZES[STARTER]*REGION_SIZES[STARTER] * density_expression / STARTER_NB_SPOTS;

        _starter_quantities[type] = starting_amount / STARTING_PATCHES_SPLIT / settings.frequencies[type];
        _starter_radii[type] = RQ_FACTORS[STARTER][type] * Math::cbrtf(_starter_quantities[type]);
    }

    const float water_frequency = 1.f / settings.water_scale;
    const float water_size = settings.water_coverage;

    _water_level = 10 * Math::log2f(water_size);

    const float nauvis_segmentation_multiplier = 1.5f * water_frequency;
    _nauvis_hills_input_scale = nauvis_segmentation_multiplier / 90.f;
    _nauvis_hills_cliff_level_input_scale = nauvis_segmentation_multiplier / 500.f;
    _starting_macro_multiplier_base = nauvis_segmentation_multiplier / 2000.f;
    _nauvis_bridge_billows_input_scale = nauvis_segmentation_multiplier / 150.f;
    _nauvis_persistance_input_scale = nauvis_segmentation_multiplier / 2.f;
    _nauvis_offset_x = 10000.f / nauvis_segmentation_multiplier;
    _nauvis_detail_input_scale = nauvis_segmentation_multiplier / 14.f;
    _nauvis_macro_input_scale = nauvis_segmentation_multiplier / 1600.f;
    _starting_island_multiplier = water_frequency / 200.f;
}

PositionI32 starter_lake_position(uint32_t seed) {
    Random random(seed);

    constexpr double d = 75.f;
    float angle = (float)(random.random() * (2*M_PI) * std::exp2(-32.));

    return { (int32_t)std::trunc(d*Math::cos(angle)), (int32_t)std::trunc(d*Math::sin(angle)) };
}

std::pair<float, float> Noise::_gradient(uint8_t x, uint8_t y, uint8_t p1, const Permutations& p) const {
    uint8_t y_permutation = p1 ^ p.p2[y];
    uint8_t xy_permutation = y_permutation ^ p.p3[x];
    return p.grad[xy_permutation];
}

void Noise::_make_permutations(uint32_t seed0, Permutations& p) {
    Random random(seed0);

    std::iota(p.p1.begin(), p.p1.end(), (uint8_t)0);
    std::iota(p.p2.begin(), p.p2.end(), (uint8_t)0);
    std::iota(p.p3.begin(), p.p3.end(), (uint8_t)0);
    p.grad = default_gradients();

    random.shuffle(p.p1);
    random.shuffle(p.p2);
    random.shuffle(p.p3);
    random.shuffle(p.grad);
}

Noise::Noise(uint32_t seed0, bool quick_multioctave_precompute, bool elevation2_0_precompute) {
    _quick_multioctave_precompute = quick_multioctave_precompute;
    _elevation2_0_precompute = elevation2_0_precompute;

    const uint32_t max = quick_multioctave_precompute ? QUICK_MULTIOCTAVE_MAX_SEED_OFFSET : 1;
    for (uint32_t i = 0; i < max; i++) {
        _make_permutations(seed0 + i, _permutations[i]);
    }

    if (elevation2_0_precompute) {
        for (Seed0CustomOffsets i = (Seed0CustomOffsets)0; i < Seed0CustomOffsets::NB; i++) {
            _make_permutations(seed0 + 7 * (SEED0_CUSTOM_OFFSETS[(size_t)i] >> 8), _custom_offset_permutations[(size_t)i]);
        }
    }

    _starter_lake = starter_lake_position(seed0);
}

float Noise::_noise_internal(
    const Permutations& p, uint8_t seed1, PositionF32 pos, float input_scale, float output_scale, float offset_x, float offset_y
) const {
    const uint8_t p1 = p.p1[seed1];
    
    const float x_scaled = (pos.x + offset_x) * input_scale;
    const float y_scaled = (pos.y + offset_y) * input_scale;

    const float x_floor = std::floorf(x_scaled);
    const float y_floor = std::floorf(y_scaled);
    const float x_ceil = x_floor + 1.f;
    const float y_ceil = y_floor + 1.f;
    const float x_frac = x_scaled - x_floor;
    const float y_frac = y_scaled - y_floor;

    const uint8_t int_x_floor = (uint8_t)x_floor;
    const uint8_t int_y_floor = (uint8_t)y_floor;
    const uint8_t int_x_ceil = (uint8_t)x_ceil;
    const uint8_t int_y_ceil = (uint8_t)y_ceil;
    const std::array<std::pair<uint8_t, uint8_t>, 4> points{
        std::make_pair( int_x_floor, int_y_floor ),
        std::make_pair( int_x_ceil, int_y_floor ),
        std::make_pair( int_x_floor, int_y_ceil ),
        std::make_pair( int_x_ceil, int_y_ceil )
    };

    const std::array<std::pair<float, float>, 4> dcba{
        std::make_pair( 0.f, 0.f ),
        std::make_pair( 1.f, 0.f ),
        std::make_pair( 0.f, 1.f ),
        std::make_pair( 1.f, 1.f )
    };

    float sum1 = 0.f;
    float sum2 = 0.f;

    for (int i = 0; i < 2; i++) {
        const auto gradient = _gradient(points[i].first, points[i].second, p1, p);

        const float x_frac_offset = x_frac - dcba[i].first;
        const float y_frac_offset = y_frac - dcba[i].second;

        const float d2 = 1.f - std::min(1.f, x_frac_offset*x_frac_offset + y_frac_offset*y_frac_offset);
        const float d2_3 = d2*d2*d2;

        sum1 += (
            (x_frac_offset * gradient.first) +
            (y_frac_offset * gradient.second)
        ) * d2_3;
    }

    for (int i = 2; i < 4; i++) {
        const auto gradient = _gradient(points[i].first, points[i].second, p1, p);

        const float x_frac_offset = x_frac - dcba[i].first;
        const float y_frac_offset = y_frac - dcba[i].second;

        const float d2 = 1.f - std::min(1.f, x_frac_offset*x_frac_offset + y_frac_offset*y_frac_offset);
        const float d2_3 = d2*d2*d2;

        sum2 += (
            (x_frac_offset * gradient.first) +
            (y_frac_offset * gradient.second)
        ) * d2_3;
    }

    return (sum1 + sum2) * output_scale;
}

float Noise::noise(uint8_t seed1, PositionF32 pos, float input_scale, float output_scale, float offset_x, float offset_y) const {
    return _noise_internal(_permutations[0], seed1, pos, input_scale, output_scale, offset_x, offset_y);
}

static float modified_amplitude(float output_scale, uint32_t octaves, float persistence) {
    if (persistence == 1.f) {
        return (float)(output_scale / std::sqrt((double)octaves));
    } else if (persistence == 0.f) {
        return output_scale;
    } else {
        const float persistence_2 = persistence*persistence;
        const float whatever_this_is = Math::exp2f(Math::log2f(persistence_2) * (float)octaves);
        const float whatever_that_is = (persistence_2 - 1.f) / (whatever_this_is - 1.f);
        return std::sqrtf(whatever_that_is) * output_scale;
    }
}

float Noise::_multioctave_noise_internal(
    const Permutations& p, uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
    float input_scale, float output_scale, float offset_x, float offset_y
) const {
    const float inv_persistence = 1.f / persistence;
    output_scale = modified_amplitude(output_scale, octaves, inv_persistence);\
    float sum = 0.f;

    for (uint32_t current_octave = 0; current_octave < octaves; current_octave++) {
        const PositionF32 scaled_pos(
            (float)((double)(input_scale * pos.x) + (17.17 * current_octave)),
            input_scale * pos.y
        );

        sum += _noise_internal(p, seed1, scaled_pos, 1.f, output_scale, offset_x, offset_y);

        input_scale *= 0.5f;
        output_scale *= inv_persistence;
    }

    return sum;
}

float Noise::multioctave_noise(
    uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
    float input_scale, float output_scale, float offset_x, float offset_y
) const {
    return _multioctave_noise_internal(
        _permutations[0], seed1, pos, persistence, octaves, input_scale, output_scale, offset_x, offset_y
    );
}

float Noise::_persistence_multioctave_noise_internal(const Permutations& p, uint8_t seed1, PositionF32 pos,
    float persistence, uint32_t octaves, float input_scale, float output_scale, float offset_x, float offset_y
) const {
    input_scale *= 0.5f;
    output_scale *= Math::powf(2.f, (float)octaves);

    float sum = 0.f;
    for (uint32_t i = 1; i < octaves; i++) {
        sum += _noise_internal(p, seed1, pos, input_scale, 1.f, offset_x, offset_y);
        sum *= persistence;

        input_scale *= 0.5f;
    }

    sum += _noise_internal(p, seed1, pos, input_scale, 1.f, offset_x, offset_y);

    return sum * output_scale;
}

float Noise::persistence_multioctave_noise(
    uint8_t seed1, PositionF32 pos, float persistence, uint32_t octaves,
    float input_scale, float output_scale, float offset_x, float offset_y
) const {
    return _persistence_multioctave_noise_internal(
        _permutations[0], seed1, pos, persistence, octaves, input_scale, output_scale, offset_x, offset_y
    );
}

float Noise::quick_multioctave_noise(
    uint8_t seed1, PositionF32 pos, uint32_t octaves, float input_scale,
    float output_scale, float offset_x, float offset_y, float octave_input_scale_multiplier,
    float octave_output_scale_multiplier, uint32_t octave_seed0_shift
) const {
    assert(_quick_multioctave_precompute);
    assert(octave_seed0_shift * octaves <= QUICK_MULTIOCTAVE_MAX_SEED_OFFSET);

    float sum = 0.f;

    for (uint32_t i = 0; i < octaves; i++) {
        sum += _noise_internal(_permutations[i*octave_seed0_shift], seed1, pos, input_scale, output_scale, offset_x, offset_y);

        input_scale *= octave_input_scale_multiplier;
        output_scale *= octave_output_scale_multiplier;
    }

    return sum;
}

float Noise::_nauvis_hills_plateaus(const MapGenSettings&, const NoisePrecompute& precompute, PositionF32 pos) const {
    const float nauvis_hills = std::abs(_multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_HILLS],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_HILLS] % 256),
        pos,
        0.5f, 4,
        precompute.get_nauvis_hills_input_scale(), 1.f,
        0.f, 0.f
    ));

    const float nauvis_hills_cliff_level = std::clamp(0.65f + _noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_HILLS_CLIFF_LEVEL],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_HILLS_CLIFF_LEVEL] % 256),
        pos,
        precompute.get_nauvis_hills_cliff_level_input_scale(), 0.6f,
        0.f, 0.f
    ), 0.15f, 1.15f);

    const float nauvis_plateaus = 0.5f + std::clamp((nauvis_hills - nauvis_hills_cliff_level) * 10.f, -0.5f, 0.5f);

    return 0.1f * nauvis_hills + 0.8f * nauvis_plateaus;
}

float Noise::_elevation_nauvis_function(const MapGenSettings&, const NoisePrecompute& precompute, PositionF32 pos, float added_cliff_elevation) const {
    const float distance_from_spawn = pos.length();
    const float starting_lake_distance = std::min(PositionF32::distance(pos, _starter_lake), 1024.f);

    const float starting_macro_multiplier = std::clamp(distance_from_spawn * precompute.get_starting_macro_multiplier_base(), 0.f, 1.f);

    const float nauvis_bridge_billows = std::abs(_multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_BRIDGE_BILLOWS],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_BRIDGE_BILLOWS] % 256),
        pos,
        0.5f, 4,
        precompute.get_nauvis_bridge_billows_input_scale(), 1.f,
        0.f, 0.f
    ));

    const float nauvis_persistance = std::clamp(_persistence_multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_PERSISTANCE],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_PERSISTANCE] % 256),
        pos,
        0.7f, 5,
        precompute.get_nauvis_persistance_input_scale(), NAUVIS_PERSISTANCE_OUTPUT_SCALE,
        precompute.get_nauvis_offset_x(), 0.f
    ) + 0.55f, 0.5f, 0.65f);

    const float nauvis_detail = _persistence_multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_DETAIL],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_DETAIL] % 256),
        pos,
        nauvis_persistance, 5,
        precompute.get_nauvis_detail_input_scale(), 0.03f,
        precompute.get_nauvis_offset_x(), 0.f
    );

    const float nauvis_macro = _multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_MACRO_1],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_MACRO_1] % 256),
        pos,
        0.6f, 2,
        precompute.get_nauvis_macro_input_scale(), 1.f,
        0.f, 0.f
    ) * std::max(0.f, _multioctave_noise_internal(
        _custom_offset_permutations[(size_t)Seed0CustomOffsets::NAUVIS_MACRO_2],
        (uint8_t)(SEED0_CUSTOM_OFFSETS[(size_t)Seed0CustomOffsets::NAUVIS_MACRO_2] % 256),
        pos,
        0.6f, 1,
        precompute.get_nauvis_macro_input_scale(), 1.f,
        0.f, 0.f
    ));

    const float starting_lake_noise = quick_multioctave_noise(
        14,
        pos,
        4,
        STARTING_LAKE_NOISE_INPUT_SCALE, STARTING_LAKE_NOISE_OUTPUT_SCALE,
        0.f, 0.f,
        2.f, 0.68f, 1
    );

    const float nauvis_bridges = 1.f - 0.1f * nauvis_bridge_billows - 0.9f * std::max(0.f, -0.1f + nauvis_bridge_billows);

    const float nauvis_main = 
        ELEVATION_MAGNITUDE * (
            Math::lerp(
                0.5f * added_cliff_elevation - 0.6f,
                1.9f * added_cliff_elevation + 1.6f,
                0.1f + 0.5f * nauvis_bridges
            ) +
            0.25f * nauvis_detail +
            3.f * nauvis_macro * starting_macro_multiplier
        );

    const float starting_island = nauvis_main + ELEVATION_MAGNITUDE * (2.5f - distance_from_spawn * precompute.get_starting_island_multiplier());
    const float starting_lake = ELEVATION_MAGNITUDE * (-3.f + (starting_lake_distance + starting_lake_noise) / 8.f) / 8.f;
    
    const float wlc_elevation = std::max(nauvis_main - precompute.get_water_level() * WLC_AMPLITUDE, starting_island);

    return std::min(wlc_elevation, starting_lake);
}

float Noise::elevation_nauvis(const MapGenSettings& settings, const NoisePrecompute& precompute, PositionF32 pos) const {
    assert(_elevation2_0_precompute);
    const float nauvis_hills_plateaus = _nauvis_hills_plateaus(settings, precompute, pos);
    return _elevation_nauvis_function(settings, precompute, pos, nauvis_hills_plateaus);
}

float Noise::_make_0_12like_lakes(const NoisePrecompute& precompute, PositionF32 pos, float bias, float seg_multiplier) const {
    const float input_scale = seg_multiplier / 2.f;
    const float offset_x = 10000.f / seg_multiplier;
    
    const float persistence = std::clamp(
        0.3f + persistence_multioctave_noise(
            MAKE_0_12LIKE_LAKES_PERSISTENCE_SEED1, pos, MAKE_0_12LIKE_LAKES_PERSISTENCE_PERSISTENCE,
            MAKE_0_12LIKE_LAKES_PERSISTENCE_OCTAVES, input_scale, MAKE_0_12LIKE_LAKES_PERSISTENCE_OUTPUT_SCALE,
            offset_x, 0
        ),
        0.1f, 0.9f
    );

    const float distance_from_spawn = pos.length();

    const float a = bias + persistence_multioctave_noise(
        MAKE_0_12LIKE_LAKES_BIAS_SEED1A, pos, persistence, MAKE_0_12LIKE_LAKES_OCTAVES_A,
        input_scale, MAKE_0_12LIKE_LAKES_OUTPUT_SCALE, offset_x, 0
    );

    const float b = 20.f + precompute.get_water_level() -
        0.1f * seg_multiplier * distance_from_spawn +
        persistence_multioctave_noise(
            MAKE_0_12LIKE_LAKES_BIAS_SEED1B, pos, persistence, MAKE_0_12LIKE_LAKES_OCTAVES_B,
            input_scale, MAKE_0_12LIKE_LAKES_OUTPUT_SCALE, offset_x, 0
        );
    
    return std::max(a, b);
}

float Noise::_finish_elevation(const NoisePrecompute& precompute, float elevation, PositionF32 pos, float seg_multiplier) const {
    const float starting_lake_distance = std::min(PositionF32::distance(pos, _starter_lake), 1024.f);

    const float starting_lake_noise = quick_multioctave_noise(
        STARTER_LAKE_SEED1, pos, STARTER_LAKE_OCTAVES, STARTER_LAKE_INPUT_SCALE2,
        STARTER_LAKE_OUTPUT_SCALE2, 0.f, 0.f, STARTER_LAKE_OCTAVE_INPUT_SCALE_MULTIPLIER2,
        STARTER_LAKE_PERSISTENCE, 1
    );

    const float a = std::min(
        (elevation - precompute.get_water_level()) / seg_multiplier,
        noise(FINISH_ELEVATION_NOISE_SEED1, pos, 1.f/8, 1.5f, 0.f, 0.f) + starting_lake_distance / 4.f - 4.f
    );

    const float b = std::min(
        -1 + (starting_lake_distance + starting_lake_noise) / 16.f,
        std::max(2.f, 2.f + starting_lake_distance / 16.f + starting_lake_noise / 2.f)
    );

    return std::min(a, b);
}

float Noise::elevation_lakes(const MapGenSettings& settings, const NoisePrecompute& precompute, PositionF32 pos) const {
    float seg_multiplier = 1.f / settings.water_scale;
    float lakes = _make_0_12like_lakes(precompute, pos, MAKE_0_12LIKE_LAKES_BIAS[ELEVATION_1_1], seg_multiplier);
    return _finish_elevation(precompute, lakes, pos, seg_multiplier);
}

float Noise::elevation_island(const MapGenSettings& settings, const NoisePrecompute& precompute, PositionF32 pos) const {
    float seg_multiplier = 0.25f / settings.water_scale;
    float lakes = _make_0_12like_lakes(precompute, pos, MAKE_0_12LIKE_LAKES_BIAS[ELEVATION_ISLAND], seg_multiplier);
    return _finish_elevation(precompute, lakes, pos, seg_multiplier);
}

template<PatchType Type>
static std::array<Candidate, NB_CANDIDATES[Type]> generate_candidates(
    NoiseCache& cache, const uint32_t seed0, PositionI32 region
) {
    static_assert(Type == STARTER || Type == REGULAR);

    std::array<Candidate, NB_CANDIDATES[Type]> candidates;

    const uint32_t id = cache.get_id();
    auto& chunks = cache.get_chunks<Type>();
    constexpr auto CHUNK_COUNT = CHUNK_COUNTS[Type];
    constexpr auto CHUNK_SIZE = CHUNK_SIZES[Type];

    const uint32_t seed = (region.y * 0x1ee3 + region.x * 0x1eef + SEEDS1[Type] * 0x1ef7 + 0x3fbe2c) ^ seed0;
    Random random(seed);

    constexpr float c_suggested_distance2 = SUGGESTED_DISTANCES[Type] * SUGGESTED_DISTANCES[Type];
    float suggested_distance2 = c_suggested_distance2;

    constexpr int32_t HALF_REGION_SIZE = REGION_SIZES[Type] / 2;
    const int32_t offset_x = REGION_SIZES[Type] * region.x - HALF_REGION_SIZE;
    const int32_t offset_y = REGION_SIZES[Type] * region.y - HALF_REGION_SIZE;

    int i = 0;
    while (i < NB_CANDIDATES[Type]) {
        int32_t x = random.random() % REGION_SIZES[Type];
        int32_t y = random.random() % REGION_SIZES[Type];
        const int32_t chunk_x = x / CHUNK_SIZE;
        const int32_t chunk_y = y / CHUNK_SIZE;
        
        x += offset_x;
        y += offset_y;

        const int other_chunk_y_min = std::max(0, chunk_y - 1);
        const int other_chunk_x_max = std::min(CHUNK_COUNT - 1, chunk_x + 1);
        const int other_chunk_y_max = std::min(CHUNK_COUNT - 1, chunk_y + 1);

        bool valid = true;

        for (int other_chunk_x = std::max(0, chunk_x - 1); other_chunk_x <= other_chunk_x_max; other_chunk_x++) {
            for (int other_chunk_y = other_chunk_y_min; other_chunk_y <= other_chunk_y_max; other_chunk_y++) {
                auto& other_array = chunks[other_chunk_x + other_chunk_y * CHUNK_COUNT];
                if (other_array.id != id) continue;

                for (auto& other : other_array) {
                    int32_t dx = x - other.x;
                    int32_t dy = y - other.y;
                    if (dx*dx + dy*dy < suggested_distance2) {
                        suggested_distance2 *= 0.9375f;
                        valid = false;
                        goto invalid;
                    }
                }
            }
        }
        invalid:;

        if (valid) {
            auto& candidate = candidates[i];
            candidate.pos.x = x;
            candidate.pos.y = y;

            auto& other_array = chunks[chunk_x + chunk_y * CHUNK_COUNT];
            if (other_array.id != id) {
                other_array.clear();
                other_array.id = id;
            }
            other_array.insert({ x, y });
            i++;
        }
    }

    return candidates;
}

static std::array<Candidate, NB_CANDIDATES[BITER]> generate_candidates_biters(const uint32_t seed0, PositionI32 region) {
    std::array<Candidate, NB_CANDIDATES[BITER]> candidates;

    const uint32_t seed = (region.y * 0x1ee3 + region.y * 0x1eef + SEEDS1[BITER] * 0x1ef7 + 0x3fbe2c) ^ seed0;
    Random random(seed);

    constexpr int32_t HALF_REGION_SIZE = REGION_SIZES[BITER] / 2;
    const int32_t offset_x = REGION_SIZES[BITER] * region.x - HALF_REGION_SIZE;
    const int32_t offset_y = REGION_SIZES[BITER] * region.y - HALF_REGION_SIZE;

    for (auto& c : candidates) {
        c.pos.x = random.random() % REGION_SIZES[BITER] + offset_x;
        c.pos.y = random.random() % REGION_SIZES[BITER] + offset_y;
    }

    return candidates;
}

Patches starter_patches(
    const MapGenSettings& settings, const NoisePrecompute& precompute, const Noise& noise, NoiseCache& cache,
    const uint32_t seed0
) {
    Patches patches;

    auto candidates = generate_candidates<STARTER>(cache, seed0, { 0, 0 });

    for (ResourceType type = IRON; type <= STONE; type++) {
        const auto offset = OFFSETS[type];

        const auto& penalties = precompute.get_starter_penalties()[candidates[offset].pos.x+120 + (candidates[offset].pos.y+120)*240];
        std::array<Candidate*, STARTER_NB_SPOTS> candidates_ptr;

        const float base_density = precompute.get_starter_base_densities()[type];
        float total_density = 0.f;

        for (uint32_t i = 0; i < STARTER_NB_SPOTS; i++) {
            auto &candidate = candidates[i*SPANS[STARTER] + offset];
            candidates_ptr[i] = &candidate;

            const float distance_from_spawn = candidate.pos.length();

            float elevation = 0.f;
            if (STARTING_RESOURCE_PLACEMENT_RADIUS > distance_from_spawn) {
                elevation = std::clamp((noise.elevation_lakes(settings, precompute, candidate.pos) - 1.f) / 10.f, 0.f, 1.f) * 2.f;
                total_density += base_density;
            }
            candidate.favorability = elevation - distance_from_spawn / STARTING_RESOURCE_PLACEMENT_RADIUS + penalties[i];
        }

        struct CandidateCompare {
            bool operator()(const Candidate* a, const Candidate* b) const {
                return a->favorability > b->favorability;
            }
        };

        std::stable_sort(candidates_ptr.begin(), candidates_ptr.begin() + STARTER_NB_SPOTS, CandidateCompare());
        const float quantity = precompute.get_starter_quantities()[type];
        const float radius = precompute.get_starter_radii()[type];
        const float nb_patches = total_density / quantity;
        const float nb_patches_ceil = std::ceil(nb_patches);
        const int last = (int)nb_patches_ceil - 1;

        for (int i = 0; i < last; i++) {
            const auto &candidate = *candidates_ptr[i];
            patches[type].insert(candidate.pos, radius, quantity);
        }

        const float frac = 1 - nb_patches_ceil + nb_patches;
        const float last_quantity = quantity * frac;
        const float last_radius = radius * Math::powf(last_quantity / quantity, 1.f/3);
        const auto &candidate = *candidates_ptr[last];
        patches[type].insert(candidate.pos, last_radius, last_quantity);
    }

    return patches;
}

static float regular_density_modifier(PositionI32 pos) {
    const float d = pos.length();

    const float closeness_penalty = std::clamp((d - STARTING_RESOURCE_PLACEMENT_RADIUS) / REGULAR_PATCH_FADE_IN_DISTANCE, 0.f, 1.f);

    const float size_effective_distance_at = d - REGULAR_PATCH_FADE_IN_DISTANCE;
    const float distance_bonus = 1 + std::clamp(size_effective_distance_at / DOUBLE_DENSITY_DISTANCE, 0.f, 1.f);

    return closeness_penalty * distance_bonus;
}

static float regular_quantity(const float base_quantity, const float density, const float penalty) {
    return base_quantity * density * penalty;
}

static float regular_radius(const float rq_factor, const float quantity) {
    /*
    spot_radius_expression = min(32, regular_rq_factor * regular_spot_quantity_expression ^ (1/3))
    */

    return std::min(32.f, rq_factor * std::cbrtf(quantity));
}

Patches regular_patches(const NoisePrecompute& precompute, NoiseCache& cache, const uint32_t seed0, const PositionI32 region) {
    Patches patches;

    auto candidates = generate_candidates<REGULAR>(cache, seed0, region);

    for (ResourceType type = IRON; type < NB_RESOURCE_TYPE; type++) {
        const float base_density = precompute.get_base_regular_densities()[type];
        const float base_quantity = precompute.get_base_regular_quantities()[type];
        const float rq_factor = RQ_FACTORS[REGULAR][type];

        float total_density = 0.f;
        const auto nb_spots = REGULAR_NB_SPOTS[type];
        const auto offset = OFFSETS[type];

        for (int i = 0; i < nb_spots; i++) {
            auto &candidate = candidates[i*SPANS[REGULAR] + offset];
            const float density = base_density * regular_density_modifier(candidate.pos);

            candidate.density = density;
            total_density += density;
        }

        int penalties_offset = REGULAR_MAX_NB_SPOTS - nb_spots;
        const auto& penalties = precompute.get_regular_penalties()[candidates[offset].pos.x+512 + (candidates[offset].pos.y+512)*1024];

        total_density = REGION_SIZES[REGULAR]*REGION_SIZES[REGULAR] * total_density / nb_spots;
        float total_quantity = 0.f;

        for (int i = 0; i < nb_spots; i++) {
            const auto &candidate = candidates[i*SPANS[REGULAR] + offset];
            const float quantity = regular_quantity(base_quantity, candidate.density, type == OIL ? 1.f : penalties[i + penalties_offset]);
            const float radius = regular_radius(rq_factor, quantity);

            patches[type].insert(candidate.pos, radius, quantity);
            total_quantity += quantity;

            if (total_quantity >= total_density) break;
        }
    }

    return patches;
}

PatchArray enemy_bases(const MapGenSettings& settings, const NoisePrecompute&, uint32_t seed, const PositionI32 region) {
    PatchArray patches;

    auto candidates = generate_candidates_biters(seed, region);

    const float frequency = settings.biter_frequency;
    const float size = settings.biter_size;
    const float size_sq = std::sqrt(size);

    float total_density = 0.f;

    for (auto& c : candidates) {
        float distance_from_spawn = c.pos.length();
        float intensity = std::clamp(distance_from_spawn, 0.f, 2400.f) / 325.f;

        c.radius = std::max(0.f, size_sq * (15.f + 4.f * intensity));
        constexpr float scaled_pi = (float)M_PI / 90.f;
        c.quantity = scaled_pi * Math::powf(c.radius, 3);

        const float enemy_base_frequency = std::max(0.f, (0.00001f + 0.000003f * intensity) * frequency);
        const float density = c.quantity * enemy_base_frequency;

        total_density += density;
    }
    
    constexpr float base_density = (float)REGION_SIZES[BITER]*REGION_SIZES[BITER] / NB_CANDIDATES[BITER];
    total_density = base_density * total_density;
    float total_quantity = 0.f;

    for (auto& c : candidates) {
        patches.insert(c.pos, c.radius, c.quantity);
        total_quantity += c.quantity;

        if (total_quantity >= total_density) break;
    }

    return patches;
}
