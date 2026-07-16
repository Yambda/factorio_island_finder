"""
noise_gpu.py -- Batched PyTorch reimplementation of Factorio's elevation_nauvis
noise function (default ELEVATION_2_0 map type, default water_scale/
water_coverage = 1.0), vectorized over many seeds and many query points at once.

This is the numerical core of the GPU island-screening pipeline (see
stage1_screen.py, stage2_floodfill.py, find_islands.py). It reimplements, in
PyTorch, the same algorithm the project's C++ code implements in
core/random.hpp (class Random), core/noise.cpp (class Noise) and
core/gradients.cpp (default_gradients()).

WHAT IS EXACT vs APPROXIMATED (read before trusting results)
--------------------------------------------------------------
EXACT (bit-for-bit, or numerically indistinguishable from the C++ reference):
  - The 3-lane LFSR PRNG (Random::random()) and the Fisher-Yates shuffle
    (Random::shuffle<T>()) used to build permutation tables p1/p2/p3/grad,
    for both the "family A" (Noise::_permutations[0..4]) and "family B"
    (Noise::_custom_offset_permutations[0..6]) table sets.
  - The 256 default gradient vectors (verbatim float32 bit patterns, taken
    from core/gradients.cpp's default_gradients()).
  - Corner-coordinate lattice indexing: floor/ceil, float->int truncation,
    uint8 wraparound (x & 0xFF), and the _gradient() XOR-index composition.
  - The falloff/dot-product math inside _noise_internal (elementary float
    arithmetic, no hand-rolled approximations involved there).
  - Math::powf(2, integer_octaves) in _persistence_multioctave_noise_internal
    -- implemented as plain 2.0**octaves, exact for integer exponents.
  - starter_lake_position(): Math::cos/Math::sin reimplemented bit-for-bit
    from core/math.hpp's double-precision folded-angle polynomial (this runs
    once per SEED, not per tile, so there's no perf reason to approximate it,
    and the eventual int32 trunc() makes a naive real cos/sin a discrete-error
    risk -- it can flip the starter-lake tile by +-1).
  - All the fixed "one-time" scale constants from core/constants.hpp
    (NAUVIS_PERSISTANCE_OUTPUT_SCALE, STARTING_LAKE_NOISE_* etc.) are
    hardcoded literals matching the spec.

APPROXIMATED (deliberately -- this is the entire point of a GPU port that
trades bit-exactness for throughput):
  - Math::exp2f / Math::log2f, as used inside modified_amplitude()
    (core/noise.cpp), which feeds _multioctave_noise_internal's persistence-
    normalization scalar. There are exactly 3 DISTINCT modified_amplitude()
    evaluations in the nauvis chain -- AMP_HILLS_BILLOWS (shared by
    nauvis_hills and nauvis_bridge_billows, persistence=0.5/octaves=4),
    AMP_MACRO1 (nauvis_macro_1, persistence=0.6/octaves=2), and AMP_MACRO2
    (nauvis_macro_2, persistence=0.6/octaves=1) -- and for every one of them
    the (persistence, octaves) pair is a fixed Python-level constant, NOT
    seed- or position-dependent. So this was never even a per-tile GPU cost:
    modified_amplitude's output for each call is precomputed exactly once,
    in plain Python, at import time.
      * AMP_MACRO1 is now a BIT-EXACT port of Math::exp2f/log2f (see
        `exact_math_exp2f`/`exact_math_log2f`/`_modified_amplitude_exact`
        below), because a prior measurement pass identified it as the
        dominant driver of GPU-vs-CPU elevation discrepancy: a ~1.053%
        relative bias in the real-math substitution, amplified up to 60x by
        `3 * ELEVATION_MAGNITUDE(20) * starting_macro_multiplier(<=1)` in
        nauvis_main, producing observed GPU-vs-CPU elevation diffs up to
        ~2.58. Since this call site is evaluated exactly ONCE per process
        (a scalar Python constant, not a per-tile tensor op), an exact port
        costs nothing at runtime -- there was no reason to keep the
        approximation once its bias was identified as load-bearing. Verified
        bit-exact against a real C++ dump of Math::exp2f/Math::log2f/
        modified_amplitude (5,000,000 random points plus the exact 3
        production (persistence,octaves) constants): 0 mismatches.
      * AMP_HILLS_BILLOWS keeps the real-math (math.log2/math.exp2)
        approximation: persistence=2.0 (post-inversion) is an exact power of
        2, so real log2(4.0)=2.0 exactly and the two implementations agree
        completely for this input -- 0% bias, nothing to fix.
      * AMP_MACRO2 also keeps the real-math approximation: measured bias is
        0.0007%, three orders of magnitude below AMP_MACRO1's, and was
        judged negligible rather than worth the (small but nonzero) extra
        code-path complexity of a second exact-port call site. See
        `gpu/README.md` / the epsilon_fp measurement notes for the
        measured/derived residual this leaves.
    Because every GPU-flagged candidate is re-verified against the exact CPU
    flood fill before being reported (see cpu_verify.py / find_islands.py),
    any remaining bias here can at most shift a seed's rank within the
    shortlist -- it can never turn into a wrongly-reported final result.

Everything else (the PRNG, lattice/gradient indexing, octave-sum control
flow, and all additive/multiplicative glue in _nauvis_hills_plateaus and
_elevation_nauvis_function) follows core/noise.cpp literally, operating on
float32 tensors throughout (matching the C++ `float` type), with the two
documented double-precision exceptions (the per-octave x nudge in
_multioctave_noise_internal, and starter_lake_position) done in float64
exactly as the C++ source does.

Validated during development against a real CPU dump of
Noise::elevation_nauvis (built directly against core/noise.cpp/gradients.cpp)
at ~1000 sample points across multiple seeds, including both ground-truth
seeds used throughout this project (20271579, a confirmed real natural
island, and 1489155600, confirmed unbounded mainland): 100% land/water sign
agreement, with the (expected, documented) small numerical bias from the
exp2f/log2f approximation above. Both seeds are re-checked end to end by
find_islands.py as a standing regression test.

PER-POINT EVALUATION COST -- TWO IDEAS INVESTIGATED AND REJECTED, DO NOT
RE-ATTEMPT WITHOUT NEW EVIDENCE (this noise formula is the per-point cost
bottleneck of stage 1; both ideas below tried to cheapen it and both failed
on correctness or on a structural ceiling, not on implementation quality):

  - Octave truncation (drop high octaves of nauvis_bridge_billows and/or
    nauvis_detail, on the theory that they're "fine coastline detail" and
    barely affect the land/water sign): HARD REJECTED, this is a correctness
    bug, not a speed/accuracy tradeoff. Measured directly, component isolated
    one at a time: dropping ONE octave from nauvis_bridge_billows alone
    (4->3, detail held at full 5) has NO effect on close_score. Dropping ONE
    octave from nauvis_detail alone (5->4, bridge_billows held at full 4) is
    what actually drops real ground-truth island 1064911508's close_score
    from exactly 1.0 to 0.9792 -- below stage1_screen.py's 0.98 threshold,
    i.e. the island is silently dropped from the shortlist. (An earlier
    version of this note attributed this to bridge_billows -- that was wrong,
    confirmed by isolating each component's truncation independently; the
    load-bearing component is nauvis_detail.) At-scale random-batch testing
    (confirmed at both 20,000 and 100,000 seeds, consistent false-negative
    counts across both) at truncation configs (detail=4,bridge=3) ["mild"]
    and (detail=3,bridge=2) ["harsher"] showed real known islands becoming
    false negatives and the shortlist simultaneously INFLATING 4.68x-19.3x
    (hundreds of false negatives out of ~1,000-1,100 true positives, plus
    thousands of new false positives) -- i.e. this would make stage 2 slower
    too, not just less correct. Root cause: nauvis_detail is a direct 0.25x
    additive term in nauvis_main, structurally load-bearing at the
    exact-closure decision boundary, not merely cosmetic high-frequency
    texture, and close_score's exact-1.0 quantization (see
    stage1_screen.py's THRESHOLD CHOICE) leaves zero slack for any biased
    approximation of the field. Do not revisit any truncation of
    nauvis_detail's octave count without re-deriving why a single octave has
    this outsized effect; bridge_billows truncation alone was not shown to
    be harmful in isolation but was not tested at scale on its own either.

  - Mip-pyramid / interval-arithmetic decidability bound (precompute a
    coarse grid of Lipschitz-bounded cells per component so cells whose
    bound doesn't straddle zero can skip full-resolution evaluation): sound
    (0 sign-flip violations across 648,000 dense-resample checks against all
    4 ground-truth seeds) but not useful. Even for nauvis_macro -- the
    lowest-frequency, best-behaved of the six components, and the only one
    genuinely signed rather than provably non-negative -- the decidable
    fraction of cells collapses from ~26% at 2-tile cells to ~5% at 8 tiles
    (2x native stage-2 resolution) to ~0% by 32 tiles. nauvis_persistance's
    own natural cell size (where its Lipschitz-bound amplitude/slope ratio
    would justify coarsening) is ~5 tiles -- AT OR BELOW stage 2's native
    4-tile grid -- and nauvis_detail's persistence parameter is itself a
    per-tile tensor fed from nauvis_persistance, breaking the linear-octave-
    sum decomposition a bound needs. A sound joint bound needs all six
    components simultaneously decidable in a cell, so it is capped by the
    worst of them: near-native evaluation almost everywhere. Rejected;
    do not build a general multi-level pyramid for this noise stack without
    a fundamentally different (non-Lipschitz) bound on nauvis_persistance/
    nauvis_detail specifically.

  (A related idea -- using only the cheap nauvis_macro sub-term to predict
  the full elevation sign and skip the rest -- was also investigated and
  rejected; see stage1_screen.py's docstring, since that idea is about
  compacting the stage-1 ray batch rather than the noise formula itself.)
"""

import math
import os
import struct

import numpy as np
import torch

MASK32 = 0xFFFFFFFF

# Kill switch for the CUDA-graph-accelerated shuffle path (see its use in
# make_permutation_tables) -- OFF by default after it crashed live production
# via a stream-capture conflict with --double-buffer's background thread.
# Set NOISE_GPU_ENABLE_SHUFFLE_GRAPH=1 to re-enable once that's fixed.
_SHUFFLE_GRAPH_DISABLED = os.environ.get("NOISE_GPU_ENABLE_SHUFFLE_GRAPH") != "1"

# ============================================================================
# Section 1: PRNG + permutation-table construction
# (core/random.hpp Random, core/noise.cpp Noise::_make_permutations)
# ============================================================================

# 256 default gradient vectors, extracted verbatim (bit-for-bit, via
# struct.unpack('<f', struct.pack('<I', hexval))) from the `gradients[512]`
# uint32 table in core/gradients.cpp (default_gradients()). Flat list of
# 512 floats: [x0, y0, x1, y1, ..., x255, y255].
_DEFAULT_GRADIENTS_FLAT = [
    4.199999809265137, 0.0, 4.198735237121582, 0.10307316482067108, 4.19494104385376, 0.2060842365026474, 4.188620090484619, 0.3089711666107178,
    4.179775714874268, 0.4116719961166382, 4.168414115905762, 0.5141248106956482, 4.154541492462158, 0.61626797914505, 4.1381659507751465, 0.7180399298667908,
    4.119297981262207, 0.8193793892860413, 4.097949028015137, 0.920225203037262, 4.074131488800049, 1.0205167531967163, 4.0478596687316895, 1.120193600654602,
    4.019149303436279, 1.2191956043243408, 3.988018274307251, 1.3174633979797363, 3.9544851779937744, 1.4149373769760132, 3.918569803237915, 1.5115591287612915,
    3.88029408454895, 1.6072704792022705, 3.8396809101104736, 1.7020134925842285, 3.796755075454712, 1.7957314252853394, 3.751542091369629, 1.8883676528930664,
    3.7040693759918213, 1.9798662662506104, 3.654365301132202, 2.0701723098754883, 3.6024601459503174, 2.159231424331665, 3.54838490486145, 2.2469899654388428,
    3.4921722412109375, 2.333395004272461, 3.433856248855591, 2.4183945655822754, 3.37347149848938, 2.5019371509552, 3.3110549449920654, 2.583972692489624,
    3.2466440200805664, 2.664451837539673, 3.180277109146118, 2.74332594871521, 3.111994743347168, 2.820547580718994, 3.041837692260742, 2.8960702419281006,
    2.969848394393921, 2.9698486328125, 2.8960704803466797, 3.041837692260742, 2.820547580718994, 3.111994743347168, 2.74332594871521, 3.180277109146118,
    2.664451837539673, 3.2466437816619873, 2.583972692489624, 3.3110549449920654, 2.501936912536621, 3.373471736907959, 2.4183945655822754, 3.4338560104370117,
    2.333395004272461, 3.4921722412109375, 2.2469899654388428, 3.54838490486145, 2.159231662750244, 3.6024599075317383, 2.0701723098754883, 3.654365301132202,
    1.9798663854599, 3.704069137573242, 1.8883674144744873, 3.751542091369629, 1.7957314252853394, 3.796755075454712, 1.702013373374939, 3.8396811485290527,
    1.607270359992981, 3.88029408454895, 1.5115593671798706, 3.918569803237915, 1.4149372577667236, 3.9544851779937744, 1.3174633979797363, 3.988018274307251,
    1.2191954851150513, 4.019149303436279, 1.120193600654602, 4.0478596687316895, 1.0205169916152954, 4.074131011962891, 0.9202251434326172, 4.097949028015137,
    0.8193795084953308, 4.119297981262207, 0.7180398106575012, 4.1381659507751465, 0.6162680387496948, 4.154541492462158, 0.5141246318817139, 4.168414115905762,
    0.4116719663143158, 4.179775714874268, 0.3089713454246521, 4.188620090484619, 0.20608413219451904, 4.19494104385376, 0.10307326912879944, 4.198735237121582,
    -1.8358784359406854e-07, 4.199999809265137, -0.1030731350183487, 4.198735237121582, -0.2060839980840683, 4.19494104385376, -0.30897122621536255, 4.188620090484619,
    -0.41167184710502625, 4.179775714874268, -0.5141249895095825, 4.168414115905762, -0.6162679195404053, 4.154541492462158, -0.7180401682853699, 4.1381659507751465,
    -0.8193793892860413, 4.119297981262207, -0.9202250242233276, 4.097949028015137, -1.0205168724060059, 4.074131488800049, -1.1201934814453125, 4.0478596687316895,
    -1.21919584274292, 4.019149303436279, -1.3174632787704468, 3.988018274307251, -1.414937138557434, 3.9544851779937744, -1.511559247970581, 3.918569803237915,
    -1.6072702407836914, 3.88029408454895, -1.702013611793518, 3.8396809101104736, -1.7957313060760498, 3.796755075454712, -1.888367772102356, 3.75154185295105,
    -1.9798659086227417, 3.7040696144104004, -2.0701723098754883, 3.6543655395507812, -2.159231662750244, 3.6024601459503174, -2.246990442276001, 3.548384666442871,
    -2.333394765853882, 3.4921724796295166, -2.4183943271636963, 3.433856248855591, -2.5019373893737793, 3.37347149848938, -2.583972454071045, 3.3110551834106445,
    -2.6644515991210938, 3.2466440200805664, -2.74332594871521, 3.180277109146118, -2.8205480575561523, 3.111994504928589, -2.8960700035095215, 3.0418379306793213,
    -2.969848394393921, 2.969848394393921, -3.0418379306793213, 2.8960700035095215, -3.111994504928589, 2.8205478191375732, -3.180277109146118, 2.74332594871521,
    -3.2466440200805664, 2.6644515991210938, -3.3110551834106445, 2.583972215652466, -3.37347149848938, 2.5019371509552, -3.433856248855591, 2.4183943271636963,
    -3.4921724796295166, 2.333394765853882, -3.548384666442871, 2.246990203857422, -3.6024601459503174, 2.159231662750244, -3.6543655395507812, 2.0701723098754883,
    -3.704069137573242, 1.9798667430877686, -3.751542091369629, 1.888367772102356, -3.796755075454712, 1.7957313060760498, -3.8396811485290527, 1.7020131349563599,
    -3.880293846130371, 1.6072707176208496, -3.918569803237915, 1.5115591287612915, -3.9544851779937744, 1.414937138557434, -3.988018274307251, 1.317463755607605,
    -4.019149303436279, 1.2191957235336304, -4.0478596687316895, 1.1201934814453125, -4.074131488800049, 1.020516276359558, -4.097949028015137, 0.9202254414558411,
    -4.119297981262207, 0.8193793296813965, -4.1381659507751465, 0.7180396318435669, -4.154541492462158, 0.6162683963775635, -4.168414115905762, 0.5141249299049377,
    -4.179775714874268, 0.41167178750038147, -4.188620090484619, 0.30897068977355957, -4.19494104385376, 0.2060844451189041, -4.198735237121582, 0.10307308286428452,
    -4.199999809265137, -3.671756871881371e-07, -4.198735237121582, -0.10307281464338303, -4.19494104385376, -0.20608417689800262, -4.188620090484619, -0.3089714050292969,
    -4.179775714874268, -0.41167151927948, -4.168414115905762, -0.5141246318817139, -4.154541492462158, -0.6162680983543396, -4.1381659507751465, -0.7180403470993042,
    -4.119298458099365, -0.8193790316581726, -4.097949028015137, -0.920225203037262, -4.074131011962891, -1.0205169916152954, -4.0478596687316895, -1.1201931238174438,
    -4.019149303436279, -1.2191954851150513, -3.988018274307251, -1.3174635171890259, -3.9544849395751953, -1.4149378538131714, -3.918569803237915, -1.5115588903427124,
    -3.88029408454895, -1.6072704792022705, -3.8396809101104736, -1.7020138502120972, -3.796755075454712, -1.7957310676574707, -3.751542091369629, -1.8883675336837769,
    -3.704069137573242, -1.9798665046691895, -3.654365062713623, -2.0701727867126465, -3.6024601459503174, -2.159231424331665, -3.54838490486145, -2.246990203857422,
    -3.4921722412109375, -2.33339524269104, -3.43385648727417, -2.418394088745117, -3.373471736907959, -2.5019371509552, -3.3110547065734863, -2.583972930908203,
    -3.2466442584991455, -2.6644513607025146, -3.1802773475646973, -2.743325710296631, -3.111994743347168, -2.8205478191375732, -3.041837453842163, -2.8960704803466797,
    -2.9698486328125, -2.969848155975342, -2.8960702419281006, -3.041837692260742, -2.820547342300415, -3.111994981765747, -2.743326187133789, -3.180276870727539,
    -2.664451837539673, -3.2466437816619873, -2.583972454071045, -3.3110551834106445, -2.501936674118042, -3.373471975326538, -2.418393850326538, -3.433856725692749,
    -2.3333957195281982, -3.4921717643737793, -2.24699068069458, -3.548384666442871, -2.1592319011688232, -3.6024599075317383, -2.0701725482940674, -3.654365301132202,
    -1.9798661470413208, -3.7040693759918213, -1.8883671760559082, -3.751542329788208, -1.795730710029602, -3.796755313873291, -1.7020143270492554, -3.8396806716918945,
    -1.6072709560394287, -3.880293846130371, -1.5115594863891602, -3.918569564819336, -1.4149373769760132, -3.9544851779937744, -1.3174630403518677, -3.98801851272583,
    -1.2191951274871826, -4.0191497802734375, -1.1201927661895752, -4.0478596687316895, -1.0205175876617432, -4.074131011962891, -0.9202257990837097, -4.097949028015137,
    -0.8193796277046204, -4.119297981262207, -0.7180399298667908, -4.1381659507751465, -0.6162676811218262, -4.154541492462158, -0.5141242742538452, -4.168414115905762,
    -0.41167110204696655, -4.179775714874268, -0.30897200107574463, -4.188619613647461, -0.20608475804328918, -4.19494104385376, -0.10307340323925018, -4.198735237121582,
    5.008449832644146e-08, -4.199999809265137, 0.10307350009679794, -4.198735237121582, 0.20608486235141754, -4.19494104385376, 0.3089720904827118, -4.188619613647461,
    0.4116712212562561, -4.179775714874268, 0.51412433385849, -4.168414115905762, 0.6162678003311157, -4.154541492462158, 0.7180400490760803, -4.1381659507751465,
    0.8193797469139099, -4.119297981262207, 0.9202258586883545, -4.097949028015137, 1.0205177068710327, -4.074131011962891, 1.1201928853988647, -4.0478596687316895,
    1.2191952466964722, -4.0191497802734375, 1.3174631595611572, -3.98801851272583, 1.4149374961853027, -3.9544849395751953, 1.5115596055984497, -3.918569564819336,
    1.6072710752487183, -3.880293846130371, 1.7020126581192017, -3.839681386947632, 1.7957308292388916, -3.796755313873291, 1.8883671760559082, -3.751542329788208,
    1.9798662662506104, -3.7040693759918213, 2.0701725482940674, -3.654365301132202, 2.1592319011688232, -3.6024599075317383, 2.24699068069458, -3.548384428024292,
    2.3333942890167236, -3.492172956466675, 2.418393850326538, -3.43385648727417, 2.501936912536621, -3.373471736907959, 2.583972692489624, -3.3110549449920654,
    2.664452075958252, -3.2466437816619873, 2.743326425552368, -3.180276870727539, 2.8205482959747314, -3.1119942665100098, 2.8960695266723633, -3.0418384075164795,
    2.969848155975342, -2.969848871231079, 3.041837453842163, -2.8960704803466797, 3.111994743347168, -2.820547580718994, 3.1802773475646973, -2.743325710296631,
    3.2466442584991455, -2.6644513607025146, 3.3110554218292236, -2.5839719772338867, 3.3734710216522217, -2.5019378662109375, 3.4338557720184326, -2.4183948040008545,
    3.4921722412109375, -2.33339524269104, 3.54838490486145, -2.2469899654388428, 3.6024603843688965, -2.159231185913086, 3.6543655395507812, -2.07017183303833,
    3.7040696144104004, -1.979865550994873, 3.7515416145324707, -1.8883683681488037, 3.796754837036133, -1.7957319021224976, 3.8396809101104736, -1.7020137310028076,
    3.88029408454895, -1.607270359992981, 3.918569803237915, -1.5115587711334229, 3.9544854164123535, -1.4149367809295654, 3.988018751144409, -1.31746244430542,
    4.019149303436279, -1.2191964387893677, 4.047859191894531, -1.1201940774917603, 4.074131011962891, -1.0205169916152954, 4.097949028015137, -0.9202250838279724,
    4.119298458099365, -0.8193789720535278, 4.138166427612305, -0.7180392742156982, 4.154541492462158, -0.6162670254707336, 4.168414115905762, -0.5141255259513855,
    4.179775714874268, -0.4116724133491516, 4.188620090484619, -0.3089713156223297, 4.19494104385376, -0.20608408749103546, 4.198735237121582, -0.10307271778583527,
]


_default_gradients_cache = {}  # (device, dtype) -> [256, 2] tensor


def _default_gradients_tensor(device, dtype=torch.float32):
    """Returns the 256 default gradient vectors as a [256, 2] tensor on `device`.
    Cached per (device, dtype): was reallocating from CPU data on every call
    (harmless but wasteful in the eager path; a hard error under CUDA graph
    capture, which forbids unpinned host->device copies mid-capture -- see
    _shuffle_graph_replay)."""
    key = (device, dtype)
    cached = _default_gradients_cache.get(key)
    if cached is None:
        flat = torch.tensor(_DEFAULT_GRADIENTS_FLAT, dtype=dtype, device=device)
        cached = flat.view(256, 2)
        _default_gradients_cache[key] = cached
    return cached


def _random_step(a, b, c):
    """
    One call to Random::random() from core/random.hpp, vectorized over a batch.

    a, b, c: torch.int64 tensors of shape [N], each holding a 32-bit LFSR
    state value (already masked to [0, 2**32)).

    Returns (a_new, b_new, c_new, output) where output = a_new ^ b_new ^ c_new
    (all masked to 32 bits), matching:
        _a = ((_a & 0xfffffffe) << 0xc) | (((_a << 0xd) ^ _a) >> 0x13);
        _b = ((_b & 0xffffff8)  << 0x4) | (((_b *  4) ^ _b) >> 0x19);
        _c = ((_c & 0xfffffff0) << 0x11)| (((_c *  8) ^ _c) >> 0xb);
        return _a ^ _b ^ _c;

    NOTE the C++ mask on _b is 0x0FFFFFF8 (only 7 hex digits, i.e. 28 bits
    set) -- NOT 0xFFFFFFF8. This looks like it could be a typo in the
    original reversed algorithm, but it is reproduced exactly here since it
    is part of Factorio's actual (reverse-engineered) map-gen PRNG behavior.
    """
    a_part1 = ((a & 0xfffffffe) << 0xc) & MASK32
    a_tmp = (((a << 0xd) & MASK32) ^ a) >> 0x13
    a_new = (a_part1 | a_tmp) & MASK32

    b_part1 = ((b & 0x0ffffff8) << 0x4) & MASK32
    b_tmp = (((b * 4) & MASK32) ^ b) >> 0x19
    b_new = (b_part1 | b_tmp) & MASK32

    c_part1 = ((c & 0xfffffff0) << 0x11) & MASK32
    c_tmp = (((c * 8) & MASK32) ^ c) >> 0xb
    c_new = (c_part1 | c_tmp) & MASK32

    output = (a_new ^ b_new ^ c_new) & MASK32
    return a_new, b_new, c_new, output


def _init_state(seeds):
    """
    Batched equivalent of Random::Random(uint32_t seed) in core/random.hpp:
        seed = max(341, seed); _a = _b = _c = seed;

    seeds: torch.int64 tensor [N], values in [0, 2**32).
    Returns (a, b, c): three torch.int64 tensors [N], each = max(341, seeds).
    """
    seeds = seeds.to(torch.int64)
    clamped = torch.clamp(seeds, min=341) & MASK32
    return clamped.clone(), clamped.clone(), clamped.clone()


def _shuffle_indices(a_state, b_state, c_state, n):
    """
    Batched Fisher-Yates shuffle of the identity permutation [0..255], one
    independent shuffle per batch row, using (and advancing) the shared LFSR
    state (a_state, b_state, c_state). Exactly mirrors
    Random::shuffle<T>(std::array<T,256>&) in core/random.hpp:

        for i in range(255, 0, -1):   # C++: for (i = 255; i >= 0; i--)
            j = random() % (i + 1)    # C++: i == 0 sets j = 0, no random() call
            swap(a[i], a[j])

    This only permutes *indices*, so it's dtype-independent -- used both to
    build p1/p2/p3 (permuting uint8 values) and the permutation applied to
    the default gradient table (permuting std::pair<float,float> values).

    Returns (perm, a_state, b_state, c_state):
        perm: torch.int64 [N, 256], perm[k] = the shuffled index array for
              batch row k (a permutation of 0..255).
        a_state, b_state, c_state: advanced LFSR state, to feed into the next
              call (e.g. shuffling p2 right after p1).
    """
    device = a_state.device
    perm = torch.arange(256, dtype=torch.int64, device=device).unsqueeze(0).repeat(n, 1)
    row_idx = torch.arange(n, dtype=torch.int64, device=device)

    for i in range(255, 0, -1):
        a_state, b_state, c_state, draw = _random_step(a_state, b_state, c_state)
        j = draw % (i + 1)  # [N], elementwise mod matching `random() % (i + 1)`

        val_i = perm[:, i].clone()
        val_j = perm[row_idx, j]
        perm[:, i] = val_j
        perm[row_idx, j] = val_i  # safe even when j == i (val_i == val_j)

    return perm, a_state, b_state, c_state


def make_permutation_tables(seeds):
    """
    Batched, bit-exact reimplementation of Noise::_make_permutations for a
    whole batch of seed0 values at once.

    seeds: torch.int64 tensor, shape [N], values in [0, 2**32). This is the
           `seed0` argument to Noise::_make_permutations (caller is
           responsible for adding any octave offset beforehand, e.g.
           seed0 + i, if reproducing multi-octave permutation chains).

    Returns a dict:
        {
          "p1":   torch.int64   [N, 256]     values in [0, 255]
          "p2":   torch.int64   [N, 256]     values in [0, 255]
          "p3":   torch.int64   [N, 256]     values in [0, 255]
          "grad": torch.float32 [N, 256, 2]  (x, y) gradient components
        }

    All output tensors live on seeds.device. p1/p2/p3 are independent
    permutations of [0..255] (NOT globally identical across seeds); grad is
    the default gradient table permuted per-seed by a 4th shuffle call
    sharing the same running LFSR state, matching the C++ order exactly:
        p1   = shuffle(identity)
        p2   = shuffle(identity)     # continues LFSR state from p1's shuffle
        p3   = shuffle(identity)     # continues LFSR state from p2's shuffle
        grad = default_gradients[shuffle(identity)]  # continues from p3

    PERFORMANCE NOTE: the Fisher-Yates shuffle above is an inherently
    sequential, 255-step-per-shuffle Python loop (4 shuffles chained =~1020
    sequential tiny GPU kernel launches), whose wall-clock cost is dominated
    by fixed per-step launch overhead almost independent of the batch size N.
    Because of that, this function should be called ONCE on the largest
    batch you have, never in a per-chunk loop -- see build_all_tables()
    below, which additionally concatenates all 12 permutation-table variants
    a seed needs into a SINGLE call to this function (one combined [N*12]
    shuffle pass) instead of calling it 12 separate times.
    """
    seeds = seeds.to(torch.int64)
    n = seeds.shape[0]
    device = seeds.device

    key = (device, n)
    if _SHUFFLE_GRAPH_DISABLED:
        # KILL SWITCH (2026-07-10): CUDA graph capture of this shuffle chain
        # crashed live production -- torch.AcceleratorError:
        # cudaErrorStreamCaptureUnsupported. Root cause: --double-buffer's
        # background cpu_tail thread (island_extent.farthest_tile_distance,
        # called whenever a new island is confirmed) issues CUDA work on the
        # same default stream the main thread was mid-capturing on, which
        # CUDA does not tolerate -- it corrupts the stream permanently until
        # process restart (every subsequent CUDA call fails the same way,
        # even unrelated ones). Needs proper dedicated-stream isolation
        # (capture AND replay on a stream nothing else ever touches, with
        # explicit event-based sync against the default stream) before this
        # is safe to re-enable, since --double-buffer is a load-bearing,
        # separately-validated feature that must keep working concurrently.
        # Set NOISE_GPU_ENABLE_SHUFFLE_GRAPH=1 to re-enable while fixing this.
        pass
    elif device.type == "cuda" and _shuffle_seen.get(key, 0) >= 2:
        # This exact batch size has recurred (the common case: build_all_tables
        # is always called on the full round batch, N*12, every round) --
        # replay a captured CUDA graph instead of re-dispatching ~1020
        # sequential tiny kernels from the Python interpreter each time. See
        # bench_cudagraph_shuffle.py for the isolated benchmark (1.28x on this
        # function alone, bit-exact match vs eager, verified before this was
        # wired in). N values that DON'T recur (e.g. a tier's residual
        # seed-count, different every round) never reach this path -- they
        # always take fewer than 2 hits on the same exact N, so no graph is
        # ever built for them, avoiding paying capture overhead for a graph
        # that would never get reused.
        return _shuffle_graph_replay(seeds, n, device)

    _shuffle_seen[key] = _shuffle_seen.get(key, 0) + 1

    a_state, b_state, c_state = _init_state(seeds)

    p1, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    p2, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    p3, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    grad_perm, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)

    default_grad = _default_gradients_tensor(device)  # [256, 2]
    grad = default_grad[grad_perm]  # [N, 256] gather -> [N, 256, 2]

    return {"p1": p1, "p2": p2, "p3": p3, "grad": grad}


_shuffle_seen = {}  # (device, N) -> times this exact batch size has been seen (eager path)
_shuffle_graph_cache = {}  # (device, N) -> {"graph", "static_seeds", "static_p1/p2/p3/grad"}


def _shuffle_chain_eager(seeds, n, device):
    """Same body as make_permutation_tables's eager path, factored out so the
    CUDA graph capture below and the normal eager call share one implementation."""
    a_state, b_state, c_state = _init_state(seeds)
    p1, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    p2, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    p3, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    grad_perm, a_state, b_state, c_state = _shuffle_indices(a_state, b_state, c_state, n)
    default_grad = _default_gradients_tensor(device)
    grad = default_grad[grad_perm]
    return {"p1": p1, "p2": p2, "p3": p3, "grad": grad}


def _shuffle_graph_replay(seeds, n, device):
    key = (device, n)
    entry = _shuffle_graph_cache.get(key)
    if entry is None:
        static_seeds = torch.empty(n, dtype=torch.int64, device=device)
        static_seeds.copy_(seeds)

        # Required warmup on a side stream before capture (torch.cuda.graph docs).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _shuffle_chain_eager(static_seeds, n, device)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _shuffle_chain_eager(static_seeds, n, device)
        torch.cuda.synchronize()

        entry = {"graph": graph, "static_seeds": static_seeds, "static_out": static_out}
        _shuffle_graph_cache[key] = entry

    entry["static_seeds"].copy_(seeds)
    entry["graph"].replay()
    # Clone before returning: the graph's output buffers get overwritten
    # in-place on the NEXT replay (of this or another call reusing the same
    # graph), so a caller holding an un-cloned reference would see its table
    # silently change under it later -- exactly the kind of silent
    # correctness bug this project's whole culture is about avoiding.
    out = entry["static_out"]
    return {"p1": out["p1"].clone(), "p2": out["p2"].clone(),
            "p3": out["p3"].clone(), "grad": out["grad"].clone()}


# ============================================================================
# Section 2: exact Math::cos / Math::sin (core/math.hpp)
# ============================================================================

def _bits_to_double(h):
    return struct.unpack(">d", bytes.fromhex(format(h, "016x")))[0]


_C_2500 = _bits_to_double(0x3FD0000000000000)
_C_INV_2PI = _bits_to_double(0x3FC45F306DC9C883)
_C_76_56 = _bits_to_double(0x405324687A27A35E)
_C_81_60 = _bits_to_double(0x405466876B29F494)
_C_41_34 = _bits_to_double(0x4044ABBC02329376)
_C_2PI = _bits_to_double(0x401921FB51BF1614)
_C_39_96 = _bits_to_double(0x4043D4243780214B)


def math_sin(x_f32):
    """Exact port of Math::sin (core/math.hpp:79), float32 in, float32 out."""
    d = x_f32.double()
    scaled_d = d * _C_INV_2PI + (-_C_2500)
    scaled_d = _C_2500 - torch.abs(scaled_d - torch.round(scaled_d))
    d2 = scaled_d * scaled_d
    d4 = d2 * d2
    val = (d4 * d4 * _C_39_96 + (d2 * (-_C_76_56) + _C_81_60) * d4 + d2 * (-_C_41_34) + _C_2PI) * scaled_d
    return val.float()


def math_cos(x_f32):
    """Exact port of Math::cos (core/math.hpp:97)."""
    d = x_f32.double()
    scaled_d = d * _C_INV_2PI
    scaled_d = _C_2500 - torch.abs(scaled_d - torch.round(scaled_d))
    d2 = scaled_d * scaled_d
    d4 = d2 * d2
    val = (d4 * d4 * _C_39_96 + (d2 * (-_C_76_56) + _C_81_60) * d4 + d2 * (-_C_41_34) + _C_2PI) * scaled_d
    return val.float()


# ============================================================================
# Section 3: fixed constants (core/constants.hpp) + the documented
# exp2f/log2f approximation, both computed once at import time.
# ============================================================================

NAUVIS_PERSISTANCE_OUTPUT_SCALE = (1.0 - 0.7) / (2.0 ** 5) / (1.0 - 0.7 ** 5) * 0.5  # ~0.0056344885
STARTING_LAKE_NOISE_INPUT_SCALE = (1.0 / 8.0) * (0.5 ** 3)  # 1/64 = 0.015625
STARTING_LAKE_NOISE_OUTPUT_SCALE = 0.8 * (2.0 ** 3)  # 6.4

ELEVATION_MAGNITUDE = 20.0
WLC_AMPLITUDE = 2.0

# Default-settings (water_scale=1.0, water_coverage=1.0) precompute scalars
# (core/noise.cpp) -- identical for every seed under default settings.
WATER_LEVEL = 10.0 * math.log2(1.0)  # exactly 0.0 (real log2 -- negligible vs Math::log2f, see module docstring)
NAUVIS_SEG_MULT = 1.5 * 1.0
NAUVIS_HILLS_INPUT_SCALE = NAUVIS_SEG_MULT / 90.0
NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE = NAUVIS_SEG_MULT / 500.0
STARTING_MACRO_MULTIPLIER_BASE = NAUVIS_SEG_MULT / 2000.0
NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE = NAUVIS_SEG_MULT / 150.0
NAUVIS_PERSISTANCE_INPUT_SCALE = NAUVIS_SEG_MULT / 2.0
NAUVIS_OFFSET_X = 10000.0 / NAUVIS_SEG_MULT
NAUVIS_DETAIL_INPUT_SCALE = NAUVIS_SEG_MULT / 14.0
NAUVIS_MACRO_INPUT_SCALE = NAUVIS_SEG_MULT / 1600.0
STARTING_ISLAND_MULTIPLIER = 1.0 / 200.0


def _modified_amplitude(output_scale, octaves, persistence):
    """modified_amplitude() (core/noise.cpp) -- see APPROXIMATED note above:
    real math.log2/math.exp2 instead of Math::log2f/exp2f, evaluated once
    per fixed (persistence, octaves) call site, not per seed/tile."""
    if persistence == 1.0:
        return output_scale / math.sqrt(octaves)
    if persistence == 0.0:
        return output_scale
    p2 = persistence * persistence
    whatever_this_is = math.exp2(math.log2(p2) * octaves)
    whatever_that_is = (p2 - 1.0) / (whatever_this_is - 1.0)
    return math.sqrt(whatever_that_is) * output_scale


# ----------------------------------------------------------------------------
# Bit-exact port of Math::exp2f / Math::log2f (core/math.hpp), used ONLY to
# compute AMP_MACRO1 exactly (see APPROXIMATED note in the module docstring
# for why only this one of the 3 modified_amplitude() call sites needs it).
#
# This is deliberately plain-Python/numpy scalar code, NOT a torch tensor/
# bit-view kernel: modified_amplitude() at this call site is evaluated
# exactly ONCE, at import time, for a fixed (persistence=0.6, octaves=2)
# pair -- it is a Python-level constant baked into AMP_MACRO1 below, never a
# per-tile GPU op. A scalar exact port fully eliminates the error mechanism
# at zero runtime cost; there is no FLOP-bound hot path here to justify a
# torch/CUDA bit-trick implementation instead.
#
# Verified bit-exact against a real build of core/math.hpp's Math::exp2f/
# Math::log2f (see the epsilon_fp measurement notes): 5,000,000 random points
# in the safe (non-UB) domain plus the 3 exact production (persistence,
# octaves) constants, 0 mismatches.
# ----------------------------------------------------------------------------

def _round_half_away_from_zero_f32(x):
    """float32 round-to-nearest-integer, ties away from zero -- matches C++
    std::round(float) (NOT numpy's default round-half-to-even)."""
    x = np.float32(x)
    ax = np.abs(x)
    r = np.floor(np.float32(ax + np.float32(0.5))).astype(np.float32)
    return np.float32(np.copysign(r, x))


def _f32_bits_to_u32(f):
    return np.array([np.float32(f)], dtype=np.float32).view(np.uint32)[0]


def _f32_bits_to_i32(f):
    return np.array([np.float32(f)], dtype=np.float32).view(np.int32)[0]


def _i32_bits_to_f32(i):
    return np.array([np.int32(i)], dtype=np.int32).view(np.float32)[0]


def exact_math_exp2f(exp_val):
    """Bit-exact port of Math::exp2f (core/math.hpp). float32 in, float32 out.

    Transcribed straight from the C++ bit-hack (a classic fast-pow2 trick:
    build an approximate log-linear correction, scale into the IEEE-754
    exponent field via the 2**23 mantissa-width constant, then reinterpret
    the resulting int32 bit pattern as a float). Every intermediate value in
    the C++ source is already float32-typed (float op float => float, no
    double promotion happens in that expression), so straightforward
    np.float32 arithmetic at each step reproduces it exactly.
    """
    exp_val = np.float32(exp_val)
    sign = np.float32(1.0) if exp_val < np.float32(0.0) else np.float32(0.0)
    clamped_exp = np.float32(max(np.float32(-126.0), exp_val))
    frac = np.float32(clamped_exp - _round_half_away_from_zero_f32(clamped_exp))
    t = np.float32(frac + sign)
    a = np.float32(np.float32(4.8425255) - t)
    b = np.float32(np.float32(27.728024) / a)
    c = np.float32(clamped_exp + np.float32(121.27406))
    d = np.float32(b + c)
    e = np.float32(t * np.float32(1.4901291))
    g = np.float32(d - e)
    scaled = np.float32(g * np.float32(8388608.0))  # 2**23
    # C++ `(int)` cast on a float truncates toward zero. float32->float64
    # widening is exact, so int(...) (Python truncates toward zero) matches
    # -- valid as long as `scaled` is in-range for int32, which it always is
    # for the actual (small-magnitude) exp2f arguments this project ever
    # evaluates (see module docstring: exp2f has no UPPER clamp on its
    # input, so huge |exp| is genuine C++ UB territory this port does not
    # need to chase).
    truncated = np.int32(int(np.float64(scaled)))
    return _i32_bits_to_f32(truncated)


def exact_math_log2f(v):
    """Bit-exact port of Math::log2f (core/math.hpp). float32 in, float32 out.

    One step ("1.72588 / (mantissa + 0.35208872f)") is genuinely double-
    precision in the C++ source (the "1.72588" literal has no 'f' suffix,
    which promotes the whole division to double before the final float
    cast) -- reproduced faithfully below; every other step is float32-only,
    same reasoning as exact_math_exp2f above.
    """
    v = np.float32(v)
    bits_u = int(_f32_bits_to_u32(v))
    bits_i = int(_f32_bits_to_i32(v))

    mantissa_bits = np.uint32((bits_u & 0x7FFFFF) | 0x3F000000)
    mantissa_f = _i32_bits_to_f32(mantissa_bits.astype(np.int32))

    term1 = np.float32(np.float32(bits_i) * np.float32(0.00000011920929))
    term1b = np.float32(term1 - np.float32(124.22552))
    term2 = np.float32(mantissa_f * np.float32(1.4980303))
    term3 = np.float32(term1b - term2)

    denom_f32 = np.float32(mantissa_f + np.float32(0.35208872))
    term4_double = 1.72588 / float(denom_f32)  # genuinely double precision
    term4 = np.float32(term4_double)

    return np.float32(term3 - term4)


def _modified_amplitude_exact(output_scale, octaves, raw_persistence):
    """Bit-exact port of modified_amplitude() (core/noise.cpp), used only for
    AMP_MACRO1. Unlike `_modified_amplitude` above, `raw_persistence` here is
    the RAW octave persistence (e.g. 0.6), matching the real call chain
    (Noise::_multioctave_noise_internal computes `inv_persistence = 1.f /
    persistence` in float32 and passes THAT to modified_amplitude) -- doing
    the inversion here, in float32, instead of accepting a pre-inverted
    Python double, avoids reproducing a second, unrelated rounding
    discrepancy on top of the exp2f/log2f one this function exists to fix.
    """
    persistence = np.float32(np.float32(1.0) / np.float32(raw_persistence))  # inv_persistence, float32
    if persistence == np.float32(1.0):
        return output_scale / math.sqrt(octaves)
    if persistence == np.float32(0.0):
        return output_scale
    p2 = np.float32(persistence * persistence)
    log2_p2 = exact_math_log2f(p2)
    whatever_this_is = exact_math_exp2f(np.float32(log2_p2 * np.float32(octaves)))
    whatever_that_is = np.float32(
        np.float32(p2 - np.float32(1.0)) / np.float32(whatever_this_is - np.float32(1.0)))
    return float(np.float32(np.sqrt(whatever_that_is) * np.float32(output_scale)))


# nauvis_hills / nauvis_bridge_billows: persistence=0.5, octaves=4 -> inv_persistence=2.0
# Real-math approximation kept: inv_persistence=2.0 exactly, log2(4.0)=2.0
# exactly under IEEE-754 too, so real math.log2/exp2 agree with the bit-hack
# exp2f/log2f completely for this input -- 0% bias, nothing to fix.
AMP_HILLS_BILLOWS = _modified_amplitude(1.0, 4, 1.0 / 0.5)
# nauvis_macro_1: persistence=0.6, octaves=2. BIT-EXACT PORT (see block
# above) -- this was the ~1.05%-biased call site the exp2f/log2f real-math
# substitution mattered for; see module docstring.
AMP_MACRO1 = _modified_amplitude_exact(1.0, 2, 0.6)
# nauvis_macro_2: persistence=0.6, octaves=1 -> inv=1/0.6. Real-math
# approximation kept: measured bias 0.0007%, judged negligible (see module
# docstring).
AMP_MACRO2 = _modified_amplitude(1.0, 1, 1.0 / 0.6)

# Seed0CustomOffsets: (offset_added_to_seed0, seed1_byte). The seed1_byte is
# baked directly into the noise_internal() call sites below (it selects a
# column of the p1 table); only the offset matters for table construction.
CUSTOM_OFFSETS = {
    "NAUVIS_HILLS": (21, 132),
    "NAUVIS_HILLS_CLIFF_LEVEL": (2723, 0),
    "NAUVIS_BRIDGE_BILLOWS": (14, 188),
    "NAUVIS_PERSISTANCE": (7, 244),
    "NAUVIS_DETAIL": (14, 88),
    "NAUVIS_MACRO_1": (21, 232),
    "NAUVIS_MACRO_2": (28, 76),
}

_FAMILY_A_COUNT = 5
_FAMILY_B_NAMES = list(CUSTOM_OFFSETS.keys())
_ALL_VARIANT_OFFSETS = list(range(_FAMILY_A_COUNT)) + [CUSTOM_OFFSETS[name][0] for name in _FAMILY_B_NAMES]
_N_VARIANTS = len(_ALL_VARIANT_OFFSETS)  # 5 family-A + 7 family-B = 12


# ============================================================================
# Section 4: per-seed table construction (batched over the whole input)
# ============================================================================

def build_all_tables(seeds):
    """
    seeds: torch.int64 [N] on target device, values in [0, 2**32).

    Builds all 12 permutation-table variants a seed needs for elevation_nauvis
    (5 "family A" tables at seed0+0..+4, 7 "family B" tables at seed0+offset
    for each of the 7 nauvis custom offsets) as a SINGLE combined batched
    Fisher-Yates shuffle pass over an [N*12] flattened seed-variant tensor,
    instead of calling make_permutation_tables() 12 separate times.

    WHY THIS MATTERS (this is the single biggest throughput lever found while
    prototyping this pipeline): the shuffle is an inherently sequential loop
    of ~1020 tiny GPU kernel launches (255 steps x 4 shuffles), and that
    per-call overhead is nearly independent of the batch size. Calling
    make_permutation_tables() 12 times means paying that ~1020-launch
    sequential chain 12 times (~12,240 launches total); concatenating all 12
    seed variants into one [N*12]-row batch and shuffling once pays that same
    ~1020-launch chain exactly ONCE, with each individual launch simply
    operating on a 12x wider batch (which GPUs absorb far more cheaply than
    12x the sequential launches). This is a pure win with no accuracy
    tradeoff -- it changes only how the (bit-exact) shuffle work is batched.

    Returns dict:
      family_a: list of 5 permutation-table dicts (p1,p2,p3,grad), for
                seed0+0 .. seed0+4 (mod 2**32) -- core/noise.cpp _permutations[0..4]
      family_b: dict name -> permutation-table dict, for the 7 nauvis custom
                offsets -- core/noise.cpp _custom_offset_permutations[]
      starter_lake: torch.float32 [N, 2] (x, y) starter lake position
    """
    seeds = seeds.to(torch.int64)
    device = seeds.device
    n = seeds.shape[0]

    offsets_t = torch.tensor(_ALL_VARIANT_OFFSETS, dtype=torch.int64, device=device)  # [12]
    variant_seeds = (seeds.unsqueeze(1) + offsets_t.unsqueeze(0)) & MASK32  # [N, 12]
    flat_seeds = variant_seeds.reshape(-1)  # [N*12]

    flat_tables = make_permutation_tables(flat_seeds)  # ONE combined shuffle pass

    def _unflatten(x):
        return x.view(n, _N_VARIANTS, *x.shape[1:])

    p1 = _unflatten(flat_tables["p1"])
    p2 = _unflatten(flat_tables["p2"])
    p3 = _unflatten(flat_tables["p3"])
    grad = _unflatten(flat_tables["grad"])

    def _variant(i):
        return {"p1": p1[:, i], "p2": p2[:, i], "p3": p3[:, i], "grad": grad[:, i]}

    family_a = [_variant(i) for i in range(_FAMILY_A_COUNT)]
    family_b = {name: _variant(_FAMILY_A_COUNT + j) for j, name in enumerate(_FAMILY_B_NAMES)}

    starter_lake = starter_lake_position(seeds)

    return {"family_a": family_a, "family_b": family_b, "starter_lake": starter_lake}


def slice_tables(tables, start, end):
    """
    Slice a build_all_tables(...) result along the seed (batch) dimension to
    [start:end], without rebuilding any permutation tables. Use this to chunk
    the (memory-heavy) per-point elevation evaluation over a subset of an
    already-built batch, instead of re-running the (launch-overhead-heavy,
    batch-size-insensitive) shuffle-based table construction per chunk --
    doing the latter was a measured >10x slowdown during development.
    """
    def _slice_pt(pt):
        return {"p1": pt["p1"][start:end], "p2": pt["p2"][start:end],
                "p3": pt["p3"][start:end], "grad": pt["grad"][start:end]}

    return {
        "family_a": [_slice_pt(t) for t in tables["family_a"]],
        "family_b": {name: _slice_pt(t) for name, t in tables["family_b"].items()},
        "starter_lake": tables["starter_lake"][start:end],
    }


def gather_tables(tables, idx):
    """
    Same idea as slice_tables, but for an arbitrary (non-contiguous) set of
    seed rows, given as idx: torch.int64 [M] indices into tables' seed
    dimension. Use this instead of re-running build_all_tables on a residual
    subset of an already-built batch (e.g. stage1_5_cascade.py's pass-2
    residual): a fresh build_all_tables call costs the same ~1,020-launch
    fixed overhead REGARDLESS of how few seeds it's called on (measured
    ~230ms for 13 seeds vs ~580ms for 50,000 -- the shuffle sequence's cost
    is batch-size-insensitive, see build_all_tables's own docstring), while
    gathering rows already computed is a single indexing op (measured
    <1ms for the same 13-row case) -- a straightforward >500x win with zero
    accuracy cost (it's the exact same table rows, not recomputed).
    """
    def _gather_pt(pt):
        return {"p1": pt["p1"][idx], "p2": pt["p2"][idx], "p3": pt["p3"][idx], "grad": pt["grad"][idx]}

    return {
        "family_a": [_gather_pt(t) for t in tables["family_a"]],
        "family_b": {name: _gather_pt(t) for name, t in tables["family_b"].items()},
        "starter_lake": tables["starter_lake"][idx],
    }


def starter_lake_position(seeds):
    """Exact port of starter_lake_position(seed0) (core/noise.cpp).

    One Random() draw per seed (fresh, unrelated to any +offset table seed).
    Returns torch.float32 [N, 2] (x, y) -- the C++ truncates to int32 when
    constructing the PositionI32, so we replicate that truncation exactly;
    the result is exact int-valued floats.
    """
    a, b, c = _init_state(seeds)
    a, b, c, draw = _random_step(a, b, c)
    draw_d = draw.double()
    angle_f32 = (draw_d * (2.0 * math.pi) * math.pow(2.0, -32.0)).float()

    cos_a = math_cos(angle_f32)
    sin_a = math_sin(angle_f32)

    d = 75.0
    x = torch.trunc(d * cos_a.double())
    y = torch.trunc(d * sin_a.double())
    return torch.stack([x.float(), y.float()], dim=-1)


# ============================================================================
# Section 5: core gradient-noise evaluation
# (core/noise.cpp Noise::_noise_internal and the octave-combination chains)
# ============================================================================

def _gather_table_u8(table_u8, idx, seed_idx=None):
    """table_u8: [N, 256] int64. idx: [M, P] int64 in [0,255]. -> [M, P] int64.

    seed_idx: optional torch.int64 [M], M possibly != N. When None (default,
    the stage1_screen.py/stage2_floodfill.py call pattern), behaves exactly
    as before -- a per-row torch.gather assuming table_u8's row i already
    belongs to request row i (M == N, identity mapping), with NO extra
    indexing overhead. When given, does a single fused 2D advanced-index
    gather `table_u8[seed_idx[:, None], idx]`, pulling row seed_idx[m] of the
    ORIGINAL [N, 256] table for each request m -- equivalent to (but without
    materializing the intermediate of) first duplicating table_u8[seed_idx]
    into an [M, 256] tensor and then torch.gather-ing idx out of THAT, which
    is what the caller used to do by hand before this parameter existed.
    """
    if seed_idx is None:
        return torch.gather(table_u8, 1, idx)
    return table_u8[seed_idx.unsqueeze(1), idx]


def _gather_grad(grad, idx, seed_idx=None):
    """grad: [N, 256, 2] float32. idx: [M, P] int64. -> [M, P, 2] float32.

    seed_idx: see _gather_table_u8 -- same None-is-identical-to-before
    contract, same fused-gather behavior when given.
    """
    if seed_idx is None:
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(grad, 1, idx_exp)
    return grad[seed_idx.unsqueeze(1), idx]


def noise_internal(table, seed1, pos_x, pos_y, input_scale, output_scale, offset_x, offset_y,
                    seed_idx=None):
    """
    Exact port of Noise::_noise_internal (core/noise.cpp), vectorized.

    table: dict with p1,p2,p3 [N,256] int64, grad [N,256,2] float32.
    seed1: python int (0..255) -- a compile-time constant at every call site.
    pos_x, pos_y: torch.float32 [M, P] (M == N when seed_idx is None).
    input_scale, output_scale, offset_x, offset_y: python floats (scalar).
    seed_idx: optional torch.int64 [M] -- see module docstring / this
        function's PERFORMANCE NOTE below. Default None: IDENTICAL behavior
        and cost to before this parameter existed (table row i is assumed to
        already belong to request row i, exactly stage1_screen.py's and
        stage2_floodfill.py's call pattern -- untouched by this change).

    PERFORMANCE NOTE: when seed_idx is given (stage1_5_oracle.py's call
    pattern: many requests referencing few distinct seeds' tables, with
    heavy duplication in seed_idx), this looks up each request's specific
    table row DIRECTLY out of the small original [N,256]/[N,256,2] tables via
    2D advanced indexing, instead of requiring the caller to first gather a
    fully duplicated [M,256]/[M,256,2] copy of the whole table (one row per
    request, mostly-duplicate rows) before calling this function. This is
    the fix for the "_gather_tables materializes ~96KB/request" cost
    documented in stage1_5_oracle.py -- it doesn't change any arithmetic,
    only which tensor the per-corner gathers read from.

    Returns torch.float32 [M, P].
    """
    if seed_idx is None:
        p1_val = table["p1"][:, seed1].unsqueeze(1)  # [N, 1]
    else:
        p1_val = table["p1"][seed_idx, seed1].unsqueeze(1)  # [M, 1]

    x_scaled = (pos_x + offset_x) * input_scale
    y_scaled = (pos_y + offset_y) * input_scale

    x_floor = torch.floor(x_scaled)
    y_floor = torch.floor(y_scaled)
    x_frac = x_scaled - x_floor
    y_frac = y_scaled - y_floor

    # truncate-toward-zero (already-integer-valued float) then wrap to uint8
    # via two's-complement AND on signed int64 -- matches (int)trunc(x) & 0xFF.
    ix_floor = x_floor.to(torch.int64) & 0xFF
    iy_floor = y_floor.to(torch.int64) & 0xFF
    ix_ceil = (x_floor + 1.0).to(torch.int64) & 0xFF
    iy_ceil = (y_floor + 1.0).to(torch.int64) & 0xFF

    corners = [
        (ix_floor, iy_floor, 0.0, 0.0),
        (ix_ceil, iy_floor, 1.0, 0.0),
        (ix_floor, iy_ceil, 0.0, 1.0),
        (ix_ceil, iy_ceil, 1.0, 1.0),
    ]

    contribs = []
    for cx, cy, dx, dy in corners:
        y_perm = p1_val ^ _gather_table_u8(table["p2"], cy, seed_idx)
        xy_perm = y_perm ^ _gather_table_u8(table["p3"], cx, seed_idx)
        grad = _gather_grad(table["grad"], xy_perm, seed_idx)  # [M, P, 2]

        xo = x_frac - dx
        yo = y_frac - dy
        d2 = 1.0 - torch.clamp(xo * xo + yo * yo, max=1.0)
        d2_3 = d2 * d2 * d2
        contribs.append((xo * grad[..., 0] + yo * grad[..., 1]) * d2_3)

    # preserve C++ grouping: (c0+c1) + (c2+c3), then * output_scale
    total = (contribs[0] + contribs[1]) + (contribs[2] + contribs[3])
    return total * output_scale


def multioctave_noise_internal(table, seed1, pos_x, pos_y, amplitude_scalar, octaves,
                                input_scale, inv_persistence, offset_x, offset_y,
                                seed_idx=None):
    """
    Port of _multioctave_noise_internal (core/noise.cpp).

    amplitude_scalar: precomputed modified_amplitude(output_scale=1.0,
        octaves, inv_persistence) -- see module-level AMP_* constants
        (APPROXIMATED via real math.log2/exp2, see module docstring).
    inv_persistence: python float, = 1/persistence (matches the C++ call
        convention: modified_amplitude is fed 1/persistence as its own
        "persistence" argument).
    seed_idx: optional torch.int64 [M] -- see noise_internal's docstring.
        Passed straight through; None (default) is fully unchanged behavior.
    """
    in_scale = input_scale
    out_scale = amplitude_scalar
    total = torch.zeros_like(pos_x)
    for oct_i in range(octaves):
        scaled_x = ((in_scale * pos_x).double() + 17.17 * oct_i).float()
        scaled_y = in_scale * pos_y
        total = total + noise_internal(table, seed1, scaled_x, scaled_y, 1.0, out_scale, offset_x, offset_y,
                                        seed_idx=seed_idx)
        in_scale *= 0.5
        out_scale *= inv_persistence
    return total


def persistence_multioctave_noise_internal(table, seed1, pos_x, pos_y, persistence, octaves,
                                            input_scale, output_scale, offset_x, offset_y,
                                            seed_idx=None):
    """
    Port of _persistence_multioctave_noise_internal (core/noise.cpp).

    persistence: python float OR torch.float32 tensor [N, P] (dynamic case:
        nauvis_detail's persistence is nauvis_persistance's per-tile value).
    output_scale: python float scalar (Math::powf(2,octaves) is exact for
        integer octaves -- implemented directly as 2.0**octaves, no approx).
    seed_idx: optional torch.int64 [M] -- see noise_internal's docstring.
        Passed straight through; None (default) is fully unchanged behavior.
    """
    in_scale = input_scale * 0.5
    out_scale = output_scale * (2.0 ** octaves)

    total = None
    for i in range(1, octaves):
        term = noise_internal(table, seed1, pos_x, pos_y, in_scale, 1.0, offset_x, offset_y, seed_idx=seed_idx)
        total = term if total is None else (total + term)
        total = total * persistence
        in_scale *= 0.5
    term = noise_internal(table, seed1, pos_x, pos_y, in_scale, 1.0, offset_x, offset_y, seed_idx=seed_idx)
    total = term if total is None else (total + term)
    return total * out_scale


def quick_multioctave_noise(family_a, seed1, pos_x, pos_y, octaves, input_scale, output_scale,
                             offset_x, offset_y, in_mult, out_mult, seed0_shift=1, seed_idx=None):
    """Port of quick_multioctave_noise (core/noise.cpp).

    seed_idx: optional torch.int64 [M] -- see noise_internal's docstring.
        Passed straight through; None (default) is fully unchanged behavior.
    """
    in_scale = input_scale
    out_scale = output_scale
    total = torch.zeros_like(pos_x)
    for i in range(octaves):
        table = family_a[i * seed0_shift]
        total = total + noise_internal(table, seed1, pos_x, pos_y, in_scale, out_scale, offset_x, offset_y,
                                        seed_idx=seed_idx)
        in_scale *= in_mult
        out_scale *= out_mult
    return total


def nauvis_hills_plateaus(tables, pos_x, pos_y, seed_idx=None):
    """Port of Noise::_nauvis_hills_plateaus (core/noise.cpp).

    seed_idx: optional torch.int64 [M] -- see noise_internal's docstring.
        Passed straight through; None (default) is fully unchanged behavior
        (stage1_screen.py's/stage2_floodfill.py's call pattern).
    """
    fb = tables["family_b"]

    nauvis_hills = torch.abs(multioctave_noise_internal(
        fb["NAUVIS_HILLS"], 132, pos_x, pos_y, AMP_HILLS_BILLOWS, 4,
        NAUVIS_HILLS_INPUT_SCALE, 2.0, 0.0, 0.0, seed_idx=seed_idx))

    nauvis_hills_cliff_level = torch.clamp(
        0.65 + noise_internal(fb["NAUVIS_HILLS_CLIFF_LEVEL"], 0, pos_x, pos_y,
                               NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE, 0.6, 0.0, 0.0, seed_idx=seed_idx),
        0.15, 1.15)

    nauvis_plateaus = 0.5 + torch.clamp((nauvis_hills - nauvis_hills_cliff_level) * 10.0, -0.5, 0.5)

    return 0.1 * nauvis_hills + 0.8 * nauvis_plateaus


def elevation_nauvis_function(tables, pos_x, pos_y, added_cliff_elevation, seed_idx=None):
    """Port of Noise::_elevation_nauvis_function (core/noise.cpp).

    seed_idx: optional torch.int64 [M] -- see noise_internal's docstring.
        Passed straight through; None (default) is fully unchanged behavior
        (stage1_screen.py's/stage2_floodfill.py's call pattern, where
        tables["starter_lake"] already has one row per request).
    """
    fb = tables["family_b"]
    fa = tables["family_a"]
    starter_lake = tables["starter_lake"] if seed_idx is None else tables["starter_lake"][seed_idx]  # [M, 2]

    distance_from_spawn = torch.sqrt(pos_x * pos_x + pos_y * pos_y)

    slx = starter_lake[:, 0:1] - pos_x
    sly = starter_lake[:, 1:2] - pos_y
    starting_lake_distance = torch.clamp(torch.sqrt(slx * slx + sly * sly), max=1024.0)

    starting_macro_multiplier = torch.clamp(distance_from_spawn * STARTING_MACRO_MULTIPLIER_BASE, 0.0, 1.0)

    nauvis_bridge_billows = torch.abs(multioctave_noise_internal(
        fb["NAUVIS_BRIDGE_BILLOWS"], 188, pos_x, pos_y, AMP_HILLS_BILLOWS, 4,
        NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE, 2.0, 0.0, 0.0, seed_idx=seed_idx))

    nauvis_persistance = torch.clamp(
        persistence_multioctave_noise_internal(
            fb["NAUVIS_PERSISTANCE"], 244, pos_x, pos_y, 0.7, 5,
            NAUVIS_PERSISTANCE_INPUT_SCALE, NAUVIS_PERSISTANCE_OUTPUT_SCALE,
            NAUVIS_OFFSET_X, 0.0, seed_idx=seed_idx) + 0.55,
        0.5, 0.65)

    nauvis_detail = persistence_multioctave_noise_internal(
        fb["NAUVIS_DETAIL"], 88, pos_x, pos_y, nauvis_persistance, 5,
        NAUVIS_DETAIL_INPUT_SCALE, 0.03, NAUVIS_OFFSET_X, 0.0, seed_idx=seed_idx)

    nauvis_macro = multioctave_noise_internal(
        fb["NAUVIS_MACRO_1"], 232, pos_x, pos_y, AMP_MACRO1, 2,
        NAUVIS_MACRO_INPUT_SCALE, 1.0 / 0.6, 0.0, 0.0, seed_idx=seed_idx
    ) * torch.clamp(multioctave_noise_internal(
        fb["NAUVIS_MACRO_2"], 76, pos_x, pos_y, AMP_MACRO2, 1,
        NAUVIS_MACRO_INPUT_SCALE, 1.0 / 0.6, 0.0, 0.0, seed_idx=seed_idx
    ), min=0.0)

    starting_lake_noise = quick_multioctave_noise(
        fa, 14, pos_x, pos_y, 4,
        STARTING_LAKE_NOISE_INPUT_SCALE, STARTING_LAKE_NOISE_OUTPUT_SCALE,
        0.0, 0.0, 2.0, 0.68, seed0_shift=1, seed_idx=seed_idx)

    nauvis_bridges = 1.0 - 0.1 * nauvis_bridge_billows - 0.9 * torch.clamp(-0.1 + nauvis_bridge_billows, min=0.0)

    lerp_alpha = 0.1 + 0.5 * nauvis_bridges
    lerp_a = 0.5 * added_cliff_elevation - 0.6
    lerp_b = 1.9 * added_cliff_elevation + 1.6
    lerp_val = lerp_a + (lerp_b - lerp_a) * lerp_alpha

    nauvis_main = ELEVATION_MAGNITUDE * (
        lerp_val + 0.25 * nauvis_detail + 3.0 * nauvis_macro * starting_macro_multiplier
    )

    starting_island = nauvis_main + ELEVATION_MAGNITUDE * (2.5 - distance_from_spawn * STARTING_ISLAND_MULTIPLIER)
    starting_lake = ELEVATION_MAGNITUDE * (-3.0 + (starting_lake_distance + starting_lake_noise) / 8.0) / 8.0

    wlc_elevation = torch.maximum(nauvis_main - WATER_LEVEL * WLC_AMPLITUDE, starting_island)

    return torch.minimum(wlc_elevation, starting_lake)


def elevation_nauvis(tables, pos_x, pos_y, seed_idx=None):
    """
    Top-level port of Noise::elevation_nauvis. pos_x, pos_y: torch.float32 [M, P]
    (M == N, tables' seed-batch size, when seed_idx is None).
    Returns torch.float32 [M, P]. Land iff > 0.0, water iff <= 0.0 (matches
    Noise::is_tile_water's use of elevation_nauvis under default settings).

    seed_idx: optional torch.int64 [M], values in [0, N) indexing into
        `tables`' seed-batch dimension -- one entry per (pos_x, pos_y) request
        row, values may repeat. When given, row m of the request is answered
        using seed-table row seed_idx[m], fetched directly out of the small
        original per-seed tables (no full-row duplication anywhere) -- this
        is what lets stage1_5_oracle.py's classify_edges() pass a
        many-requests-few-distinct-seeds batch straight through without
        first materializing an [M,256]-per-variant duplicated copy of the
        whole permutation-table set (previously done by hand in
        stage1_5_oracle._gather_tables). When None (default), M must equal
        N and row i is assumed to already belong to seed i -- IDENTICAL
        behavior/cost to before this parameter existed; this is
        stage1_screen.py's and stage2_floodfill.py's call pattern and is
        untouched by this parameter's existence.
    """
    added_cliff_elevation = nauvis_hills_plateaus(tables, pos_x, pos_y, seed_idx=seed_idx)
    return elevation_nauvis_function(tables, pos_x, pos_y, added_cliff_elevation, seed_idx=seed_idx)
