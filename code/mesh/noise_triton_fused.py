"""
Full-chain fused Triton kernel for Noise::elevation_nauvis (core/noise.cpp),
covering ALL 26 noise_internal() calls across the 8 named sub-terms
(nauvis_hills, nauvis_hills_cliff_level, nauvis_bridge_billows,
nauvis_persistance, nauvis_detail, nauvis_macro_1, nauvis_macro_2,
starting_lake_noise) plus the min/max/lerp combination logic, in ONE kernel
launch per batch of points.

This is the follow-on to noise_triton.py (which fused a single noise_internal
call and measured 8.93x over the unfused PyTorch primitive). That result
only ever applied to 1/26th of the real per-point cost; this module is the
thing that actually determines classify_edges' end-to-end throughput.

STRUCTURE: matches noise_gpu.py's call graph exactly --
    elevation_nauvis = elevation_nauvis_function(nauvis_hills_plateaus(...))
See noise_gpu.py's docstrings for the full derivation of every scale
constant and the double-precision x-nudge in multioctave_noise_internal
(reproduced here via explicit .to(tl.float64) arithmetic, NOT skipped --
this project's octave-loop x-nudge is a real, measured-load-bearing detail,
not an approximation opportunity).

CORRECTNESS: same grouping, same operation order as noise_gpu.py at every
step (float addition/multiplication is not associative, so grouping is part
of the bit-exactness contract, same as noise_triton.py). Expect the same
class of benign FMA-rounding divergence noise_triton.py measured for a
single noise_internal call (max abs diff ~2.68e-4), likely accumulated
somewhat across 26 chained calls -- MUST be re-measured end-to-end (see
validate_fused.py in the scratchpad) before this is trusted for anything
beyond a throughput prototype. Do not assume the single-call error bound
scales linearly; measure it.

TABLE LAYOUT: expects the same [N, 12, 256] / [N, 12, 256, 2] layout
build_all_tables() produces before unflattening into family_a/family_b (see
noise_gpu.build_all_tables). Variant index within the 12-wide dimension:
    0..4   = family_a[0..4]      (used by starting_lake_noise, seed1=14)
    5      = NAUVIS_HILLS                 seed1=132
    6      = NAUVIS_HILLS_CLIFF_LEVEL      seed1=0
    7      = NAUVIS_BRIDGE_BILLOWS         seed1=188
    8      = NAUVIS_PERSISTANCE            seed1=244
    9      = NAUVIS_DETAIL                 seed1=88
    10     = NAUVIS_MACRO_1                seed1=232
    11     = NAUVIS_MACRO_2                seed1=76
This order is exactly noise_gpu._FAMILY_B_NAMES (dict insertion order of
CUSTOM_OFFSETS) prefixed by the 5 family-A slots -- do not reorder without
updating both sides.
"""
import torch
import triton
import triton.language as tl

import noise_gpu as ng

# Triton kernels can only see module-level globals if they're wrapped as
# triton.language.constexpr -- plain Python ints are invisible inside @jit
# functions (TRITON_ALLOW_NON_CONSTEXPR_GLOBALS is explicitly "not promised
# forever" per Triton's own error message, so don't rely on it).
V_HILLS = tl.constexpr(5)
V_HILLS_CLIFF = tl.constexpr(6)
V_BRIDGE_BILLOWS = tl.constexpr(7)
V_PERSISTANCE = tl.constexpr(8)
V_DETAIL = tl.constexpr(9)
V_MACRO_1 = tl.constexpr(10)
V_MACRO_2 = tl.constexpr(11)
N_VARIANTS = tl.constexpr(12)

SEED1_HILLS = tl.constexpr(132)
SEED1_HILLS_CLIFF = tl.constexpr(0)
SEED1_BRIDGE_BILLOWS = tl.constexpr(188)
SEED1_PERSISTANCE = tl.constexpr(244)
SEED1_DETAIL = tl.constexpr(88)
SEED1_MACRO_1 = tl.constexpr(232)
SEED1_MACRO_2 = tl.constexpr(76)
SEED1_STARTING_LAKE = tl.constexpr(14)

# Direction codes, matching stage1_5_oracle.py's DIR_DI/DIR_DJ exactly (0=+x,
# 1=-x, 2=+y, 3=-y) -- do not renumber without checking that module.
DIR_DX = (1.0, -1.0, 0.0, 0.0)
DIR_DY = (0.0, 0.0, 1.0, -1.0)

_PACKED_CACHE_KEY = "_noise_triton_fused_packed_cache"


def _pack_tables(tables):
    """
    Re-pack a build_all_tables(...) result into the contiguous [N,12,256]/
    [N,12,256,2] flat layout this module's kernels expect (5 family_a + 7
    family_b, in noise_gpu._FAMILY_B_NAMES order), memoized ON THE `tables`
    DICT ITSELF (under _PACKED_CACHE_KEY), not in an external id()-keyed map.

    WHY ON THE DICT, NOT AN EXTERNAL id()-KEYED CACHE: an earlier version of
    this cached by id(tables) in a module-level dict. That is UNSAFE and
    caused a real, reproducible "CUDA illegal memory access" crash in
    production: callers like stage1_5_cascade.run_cascade create a FRESH
    tables dict per tier (noise_gpu.gather_tables's return value), used once
    and then dropped -- once Python garbage-collects that dict, CPython is
    free to reuse its memory address (id()) for the NEXT tier's brand-new
    (different-sized!) tables dict. The external cache would then return the
    PREVIOUS, now-stale, WRONG-SIZED packed tensors for a completely
    unrelated tables object with the same recycled id -- silently feeding
    out-of-range seed indices into the kernel's flat table gathers. Caching
    on the dict itself ties the cached data's lifetime exactly to that
    SPECIFIC object's lifetime: no id can ever collide with a live cache
    entry for a different object, and the cache entry is garbage-collected
    together with the tables dict it belongs to (no leak either -- the
    previous external-dict version also leaked, since entries were never
    evicted when their tables object died).
    """
    cached = tables.get(_PACKED_CACHE_KEY)
    if cached is not None:
        return cached

    fa = tables["family_a"]
    fb = tables["family_b"]
    fb_order = list(ng._FAMILY_B_NAMES)

    p1_stack = torch.stack([fa[i]["p1"] for i in range(5)] + [fb[name]["p1"] for name in fb_order], dim=1)
    p2_stack = torch.stack([fa[i]["p2"] for i in range(5)] + [fb[name]["p2"] for name in fb_order], dim=1)
    p3_stack = torch.stack([fa[i]["p3"] for i in range(5)] + [fb[name]["p3"] for name in fb_order], dim=1)
    grad_stack = torch.stack([fa[i]["grad"] for i in range(5)] + [fb[name]["grad"] for name in fb_order], dim=1)

    p1_flat = p1_stack.contiguous().to(torch.int32).view(-1)
    p2_flat = p2_stack.contiguous().to(torch.int32).view(-1)
    p3_flat = p3_stack.contiguous().to(torch.int32).view(-1)
    gx_flat = grad_stack[..., 0].contiguous().view(-1)
    gy_flat = grad_stack[..., 1].contiguous().view(-1)

    starter_lake = tables["starter_lake"]
    starter_x = starter_lake[:, 0].contiguous()
    starter_y = starter_lake[:, 1].contiguous()

    packed = (p1_flat, p2_flat, p3_flat, gx_flat, gy_flat, starter_x, starter_y)
    tables[_PACKED_CACHE_KEY] = packed
    return packed


@triton.jit
def _corners(x_scaled, y_scaled, p1_val, p2_ptr, p3_ptr, gx_ptr, gy_ptr, mask):
    """Raw (unscaled) sum over the 4 lattice corners -- (c0+c1)+(c2+c3), the
    same grouping/arithmetic as noise_gpu.noise_internal's corner loop and
    noise_triton.py's single-call kernel (already measured against PyTorch).
    Caller applies output_scale. p2_ptr/p3_ptr/gx_ptr/gy_ptr must already be
    offset to this (seed, variant)'s row base (row_base*256)."""
    x_floor = tl.floor(x_scaled)
    y_floor = tl.floor(y_scaled)
    x_frac = x_scaled - x_floor
    y_frac = y_scaled - y_floor

    ix_floor = x_floor.to(tl.int32) & 0xFF
    iy_floor = y_floor.to(tl.int32) & 0xFF
    ix_ceil = (x_floor + 1.0).to(tl.int32) & 0xFF
    iy_ceil = (y_floor + 1.0).to(tl.int32) & 0xFF

    # corner 0: (ix_floor, iy_floor), (dx,dy)=(0,0)
    p2v = tl.load(p2_ptr + iy_floor, mask=mask, other=0).to(tl.int32)
    yperm = p1_val ^ p2v
    p3v = tl.load(p3_ptr + ix_floor, mask=mask, other=0).to(tl.int32)
    xyperm = yperm ^ p3v
    gx = tl.load(gx_ptr + xyperm, mask=mask, other=0.0)
    gy = tl.load(gy_ptr + xyperm, mask=mask, other=0.0)
    xo = x_frac
    yo = y_frac
    d2 = 1.0 - tl.minimum(xo * xo + yo * yo, 1.0)
    d2_3 = d2 * d2 * d2
    c0 = (xo * gx + yo * gy) * d2_3

    # corner 1: (ix_ceil, iy_floor), (dx,dy)=(1,0)
    p2v = tl.load(p2_ptr + iy_floor, mask=mask, other=0).to(tl.int32)
    yperm = p1_val ^ p2v
    p3v = tl.load(p3_ptr + ix_ceil, mask=mask, other=0).to(tl.int32)
    xyperm = yperm ^ p3v
    gx = tl.load(gx_ptr + xyperm, mask=mask, other=0.0)
    gy = tl.load(gy_ptr + xyperm, mask=mask, other=0.0)
    xo = x_frac - 1.0
    yo = y_frac
    d2 = 1.0 - tl.minimum(xo * xo + yo * yo, 1.0)
    d2_3 = d2 * d2 * d2
    c1 = (xo * gx + yo * gy) * d2_3

    # corner 2: (ix_floor, iy_ceil), (dx,dy)=(0,1)
    p2v = tl.load(p2_ptr + iy_ceil, mask=mask, other=0).to(tl.int32)
    yperm = p1_val ^ p2v
    p3v = tl.load(p3_ptr + ix_floor, mask=mask, other=0).to(tl.int32)
    xyperm = yperm ^ p3v
    gx = tl.load(gx_ptr + xyperm, mask=mask, other=0.0)
    gy = tl.load(gy_ptr + xyperm, mask=mask, other=0.0)
    xo = x_frac
    yo = y_frac - 1.0
    d2 = 1.0 - tl.minimum(xo * xo + yo * yo, 1.0)
    d2_3 = d2 * d2 * d2
    c2 = (xo * gx + yo * gy) * d2_3

    # corner 3: (ix_ceil, iy_ceil), (dx,dy)=(1,1)
    p2v = tl.load(p2_ptr + iy_ceil, mask=mask, other=0).to(tl.int32)
    yperm = p1_val ^ p2v
    p3v = tl.load(p3_ptr + ix_ceil, mask=mask, other=0).to(tl.int32)
    xyperm = yperm ^ p3v
    gx = tl.load(gx_ptr + xyperm, mask=mask, other=0.0)
    gy = tl.load(gy_ptr + xyperm, mask=mask, other=0.0)
    xo = x_frac - 1.0
    yo = y_frac - 1.0
    d2 = 1.0 - tl.minimum(xo * xo + yo * yo, 1.0)
    d2_3 = d2 * d2 * d2
    c3 = (xo * gx + yo * gy) * d2_3

    return (c0 + c1) + (c2 + c3)


@triton.jit
def _elevation_nauvis_point(
    pos_x, pos_y, seed_idx,
    p1_ptr, p2_ptr, p3_ptr, gx_ptr, gy_ptr,
    starter_x_ptr, starter_y_ptr,
    mask,
    AMP_HILLS_BILLOWS: tl.constexpr,
    AMP_MACRO1: tl.constexpr,
    AMP_MACRO2: tl.constexpr,
    NAUVIS_HILLS_INPUT_SCALE: tl.constexpr,
    NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE: tl.constexpr,
    NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_OUTPUT_SCALE: tl.constexpr,
    NAUVIS_DETAIL_INPUT_SCALE: tl.constexpr,
    NAUVIS_MACRO_INPUT_SCALE: tl.constexpr,
    NAUVIS_OFFSET_X: tl.constexpr,
    STARTING_LAKE_NOISE_INPUT_SCALE: tl.constexpr,
    STARTING_LAKE_NOISE_OUTPUT_SCALE: tl.constexpr,
    STARTING_MACRO_MULTIPLIER_BASE: tl.constexpr,
    STARTING_ISLAND_MULTIPLIER: tl.constexpr,
    ELEVATION_MAGNITUDE: tl.constexpr,
    WLC_AMPLITUDE: tl.constexpr,
    WATER_LEVEL: tl.constexpr,
):
    """Shared per-point elevation_nauvis body (all 26 noise_internal calls +
    combination math), factored out of _elevation_nauvis_kernel so both the
    plain per-point kernel AND the stripe kernels (_stripe_eval_kernel) call
    the EXACT same code -- one source of truth, no duplicated 26-call chain.
    seed_idx here is already int64; row_base is recomputed internally (cheap,
    a single multiply) since callers may derive it from a per-REQUEST index,
    not a per-POINT one (see _stripe_eval_kernel)."""
    row_base = seed_idx * N_VARIANTS  # int64, avoids overflow for large N

    # ---------------- nauvis_hills (4 octaves, double-nudge) ----------------
    rb = (row_base + V_HILLS) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_HILLS, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    in_scale = NAUVIS_HILLS_INPUT_SCALE
    out_scale = AMP_HILLS_BILLOWS
    total = tl.zeros_like(pos_x)
    for oct_i in range(4):
        x_pre = in_scale * pos_x
        x_scaled = (x_pre.to(tl.float64) + 17.17 * oct_i).to(tl.float32)
        y_scaled = in_scale * pos_y
        total += out_scale * _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        in_scale *= 0.5
        out_scale *= 2.0
    nauvis_hills = tl.abs(total)

    # ---------------- nauvis_hills_cliff_level (1 octave) -------------------
    rb = (row_base + V_HILLS_CLIFF) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_HILLS_CLIFF, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb
    x_scaled = pos_x * NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE
    y_scaled = pos_y * NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE
    raw = _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask) * 0.6
    nauvis_hills_cliff_level = tl.minimum(tl.maximum(0.65 + raw, 0.15), 1.15)

    nauvis_plateaus = 0.5 + tl.minimum(tl.maximum((nauvis_hills - nauvis_hills_cliff_level) * 10.0, -0.5), 0.5)
    added_cliff_elevation = 0.1 * nauvis_hills + 0.8 * nauvis_plateaus

    # ---------------- nauvis_bridge_billows (4 octaves, double-nudge) -------
    rb = (row_base + V_BRIDGE_BILLOWS) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_BRIDGE_BILLOWS, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    in_scale = NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE
    out_scale = AMP_HILLS_BILLOWS
    total = tl.zeros_like(pos_x)
    for oct_i in range(4):
        x_pre = in_scale * pos_x
        x_scaled = (x_pre.to(tl.float64) + 17.17 * oct_i).to(tl.float32)
        y_scaled = in_scale * pos_y
        total += out_scale * _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        in_scale *= 0.5
        out_scale *= 2.0
    nauvis_bridge_billows = tl.abs(total)

    # ---------------- nauvis_persistance (5 octaves, static persistence) ----
    rb = (row_base + V_PERSISTANCE) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_PERSISTANCE, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    in_scale = NAUVIS_PERSISTANCE_INPUT_SCALE * 0.5
    total = tl.zeros_like(pos_x)
    for i in range(1, 5):
        x_scaled = (pos_x + NAUVIS_OFFSET_X) * in_scale
        y_scaled = pos_y * in_scale
        term = _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        total = total + term
        total = total * 0.7
        in_scale *= 0.5
    x_scaled = (pos_x + NAUVIS_OFFSET_X) * in_scale
    y_scaled = pos_y * in_scale
    term = _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
    total = total + term
    nauvis_persistance_raw = total * (NAUVIS_PERSISTANCE_OUTPUT_SCALE * 32.0)
    nauvis_persistance = tl.minimum(tl.maximum(nauvis_persistance_raw + 0.55, 0.5), 0.65)

    # ---------------- nauvis_detail (5 octaves, DYNAMIC persistence) --------
    rb = (row_base + V_DETAIL) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_DETAIL, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    in_scale = NAUVIS_DETAIL_INPUT_SCALE * 0.5
    total = tl.zeros_like(pos_x)
    for i in range(1, 5):
        x_scaled = (pos_x + NAUVIS_OFFSET_X) * in_scale
        y_scaled = pos_y * in_scale
        term = _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        total = total + term
        total = total * nauvis_persistance  # per-point dynamic tensor multiply
        in_scale *= 0.5
    x_scaled = (pos_x + NAUVIS_OFFSET_X) * in_scale
    y_scaled = pos_y * in_scale
    term = _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
    total = total + term
    nauvis_detail = total * (0.03 * 32.0)

    # ---------------- nauvis_macro_1 (2 octaves, double-nudge) --------------
    rb = (row_base + V_MACRO_1) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_MACRO_1, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    in_scale = NAUVIS_MACRO_INPUT_SCALE
    out_scale = AMP_MACRO1
    macro1 = tl.zeros_like(pos_x)
    for oct_i in range(2):
        x_pre = in_scale * pos_x
        x_scaled = (x_pre.to(tl.float64) + 17.17 * oct_i).to(tl.float32)
        y_scaled = in_scale * pos_y
        macro1 += out_scale * _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        in_scale *= 0.5
        out_scale *= (1.0 / 0.6)

    # ---------------- nauvis_macro_2 (1 octave, double-nudge) ---------------
    rb = (row_base + V_MACRO_2) * 256
    p1v = tl.load(p1_ptr + rb + SEED1_MACRO_2, mask=mask, other=0).to(tl.int32)
    p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb

    x_pre = NAUVIS_MACRO_INPUT_SCALE * pos_x
    x_scaled = (x_pre.to(tl.float64) + 0.0).to(tl.float32)
    y_scaled = NAUVIS_MACRO_INPUT_SCALE * pos_y
    macro2 = AMP_MACRO2 * _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)

    nauvis_macro = macro1 * tl.maximum(macro2, 0.0)

    # ---------------- starting_lake_noise (4 octaves, family_a[0..3]) -------
    in_scale = STARTING_LAKE_NOISE_INPUT_SCALE
    out_scale = STARTING_LAKE_NOISE_OUTPUT_SCALE
    starting_lake_noise = tl.zeros_like(pos_x)
    for i in range(4):
        rb = (row_base + i) * 256
        p1v = tl.load(p1_ptr + rb + SEED1_STARTING_LAKE, mask=mask, other=0).to(tl.int32)
        p2b, p3b, gxb, gyb = p2_ptr + rb, p3_ptr + rb, gx_ptr + rb, gy_ptr + rb
        x_scaled = pos_x * in_scale
        y_scaled = pos_y * in_scale
        starting_lake_noise += out_scale * _corners(x_scaled, y_scaled, p1v, p2b, p3b, gxb, gyb, mask)
        in_scale *= 2.0
        out_scale *= 0.68

    # ---------------- final combination --------------------------------
    distance_from_spawn = tl.sqrt(pos_x * pos_x + pos_y * pos_y)

    starter_x = tl.load(starter_x_ptr + seed_idx, mask=mask, other=0.0)
    starter_y = tl.load(starter_y_ptr + seed_idx, mask=mask, other=0.0)
    slx = starter_x - pos_x
    sly = starter_y - pos_y
    starting_lake_distance = tl.minimum(tl.sqrt(slx * slx + sly * sly), 1024.0)

    starting_macro_multiplier = tl.minimum(tl.maximum(distance_from_spawn * STARTING_MACRO_MULTIPLIER_BASE, 0.0), 1.0)

    nauvis_bridges = 1.0 - 0.1 * nauvis_bridge_billows - 0.9 * tl.maximum(-0.1 + nauvis_bridge_billows, 0.0)
    lerp_alpha = 0.1 + 0.5 * nauvis_bridges
    lerp_a = 0.5 * added_cliff_elevation - 0.6
    lerp_b = 1.9 * added_cliff_elevation + 1.6
    lerp_val = lerp_a + (lerp_b - lerp_a) * lerp_alpha

    nauvis_main = ELEVATION_MAGNITUDE * (lerp_val + 0.25 * nauvis_detail + 3.0 * nauvis_macro * starting_macro_multiplier)
    starting_island = nauvis_main + ELEVATION_MAGNITUDE * (2.5 - distance_from_spawn * STARTING_ISLAND_MULTIPLIER)
    starting_lake = ELEVATION_MAGNITUDE * (-3.0 + (starting_lake_distance + starting_lake_noise) / 8.0) / 8.0

    wlc_elevation = tl.maximum(nauvis_main - WATER_LEVEL * WLC_AMPLITUDE, starting_island)
    elevation = tl.minimum(wlc_elevation, starting_lake)
    return elevation


_SCALE_KWARG_NAMES = (
    "AMP_HILLS_BILLOWS", "AMP_MACRO1", "AMP_MACRO2",
    "NAUVIS_HILLS_INPUT_SCALE", "NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE",
    "NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE", "NAUVIS_PERSISTANCE_INPUT_SCALE",
    "NAUVIS_PERSISTANCE_OUTPUT_SCALE", "NAUVIS_DETAIL_INPUT_SCALE",
    "NAUVIS_MACRO_INPUT_SCALE", "NAUVIS_OFFSET_X",
    "STARTING_LAKE_NOISE_INPUT_SCALE", "STARTING_LAKE_NOISE_OUTPUT_SCALE",
    "STARTING_MACRO_MULTIPLIER_BASE", "STARTING_ISLAND_MULTIPLIER",
    "ELEVATION_MAGNITUDE", "WLC_AMPLITUDE", "WATER_LEVEL",
)


def _scale_kwargs():
    """The 17 fixed elevation_nauvis scale constants, as float kwargs -- same
    values every call site needs, built once here instead of copy-pasted at
    every kernel-launch call site."""
    return {
        "AMP_HILLS_BILLOWS": float(ng.AMP_HILLS_BILLOWS),
        "AMP_MACRO1": float(ng.AMP_MACRO1),
        "AMP_MACRO2": float(ng.AMP_MACRO2),
        "NAUVIS_HILLS_INPUT_SCALE": float(ng.NAUVIS_HILLS_INPUT_SCALE),
        "NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE": float(ng.NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE),
        "NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE": float(ng.NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE),
        "NAUVIS_PERSISTANCE_INPUT_SCALE": float(ng.NAUVIS_PERSISTANCE_INPUT_SCALE),
        "NAUVIS_PERSISTANCE_OUTPUT_SCALE": float(ng.NAUVIS_PERSISTANCE_OUTPUT_SCALE),
        "NAUVIS_DETAIL_INPUT_SCALE": float(ng.NAUVIS_DETAIL_INPUT_SCALE),
        "NAUVIS_MACRO_INPUT_SCALE": float(ng.NAUVIS_MACRO_INPUT_SCALE),
        "NAUVIS_OFFSET_X": float(ng.NAUVIS_OFFSET_X),
        "STARTING_LAKE_NOISE_INPUT_SCALE": float(ng.STARTING_LAKE_NOISE_INPUT_SCALE),
        "STARTING_LAKE_NOISE_OUTPUT_SCALE": float(ng.STARTING_LAKE_NOISE_OUTPUT_SCALE),
        "STARTING_MACRO_MULTIPLIER_BASE": float(ng.STARTING_MACRO_MULTIPLIER_BASE),
        "STARTING_ISLAND_MULTIPLIER": float(ng.STARTING_ISLAND_MULTIPLIER),
        "ELEVATION_MAGNITUDE": float(ng.ELEVATION_MAGNITUDE),
        "WLC_AMPLITUDE": float(ng.WLC_AMPLITUDE),
        "WATER_LEVEL": float(ng.WATER_LEVEL),
    }


@triton.jit
def _elevation_nauvis_kernel(
    pos_x_ptr, pos_y_ptr, seed_idx_ptr,
    p1_ptr, p2_ptr, p3_ptr, gx_ptr, gy_ptr,
    starter_x_ptr, starter_y_ptr,
    out_ptr,
    M,
    AMP_HILLS_BILLOWS: tl.constexpr,
    AMP_MACRO1: tl.constexpr,
    AMP_MACRO2: tl.constexpr,
    NAUVIS_HILLS_INPUT_SCALE: tl.constexpr,
    NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE: tl.constexpr,
    NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_OUTPUT_SCALE: tl.constexpr,
    NAUVIS_DETAIL_INPUT_SCALE: tl.constexpr,
    NAUVIS_MACRO_INPUT_SCALE: tl.constexpr,
    NAUVIS_OFFSET_X: tl.constexpr,
    STARTING_LAKE_NOISE_INPUT_SCALE: tl.constexpr,
    STARTING_LAKE_NOISE_OUTPUT_SCALE: tl.constexpr,
    STARTING_MACRO_MULTIPLIER_BASE: tl.constexpr,
    STARTING_ISLAND_MULTIPLIER: tl.constexpr,
    ELEVATION_MAGNITUDE: tl.constexpr,
    WLC_AMPLITUDE: tl.constexpr,
    WATER_LEVEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    pos_x = tl.load(pos_x_ptr + offs, mask=mask, other=0.0)
    pos_y = tl.load(pos_y_ptr + offs, mask=mask, other=0.0)
    seed_idx = tl.load(seed_idx_ptr + offs, mask=mask, other=0).to(tl.int64)

    elevation = _elevation_nauvis_point(
        pos_x, pos_y, seed_idx,
        p1_ptr, p2_ptr, p3_ptr, gx_ptr, gy_ptr,
        starter_x_ptr, starter_y_ptr,
        mask,
        AMP_HILLS_BILLOWS, AMP_MACRO1, AMP_MACRO2,
        NAUVIS_HILLS_INPUT_SCALE, NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE,
        NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE, NAUVIS_PERSISTANCE_INPUT_SCALE,
        NAUVIS_PERSISTANCE_OUTPUT_SCALE, NAUVIS_DETAIL_INPUT_SCALE,
        NAUVIS_MACRO_INPUT_SCALE, NAUVIS_OFFSET_X,
        STARTING_LAKE_NOISE_INPUT_SCALE, STARTING_LAKE_NOISE_OUTPUT_SCALE,
        STARTING_MACRO_MULTIPLIER_BASE, STARTING_ISLAND_MULTIPLIER,
        ELEVATION_MAGNITUDE, WLC_AMPLITUDE, WATER_LEVEL,
    )
    tl.store(out_ptr + offs, elevation, mask=mask)


def elevation_nauvis_triton(tables, pos_x, pos_y, seed_idx, block_size=256):
    """
    Drop-in equivalent of noise_gpu.elevation_nauvis(tables, pos_x, pos_y,
    seed_idx=seed_idx) -- REQUIRES seed_idx (flattened [M] request pattern,
    matching stage1_5_oracle.py's classify_edges call convention), unlike
    noise_gpu's version which also supports the M==N identity-mapping case.

    tables: a build_all_tables(...) result (family_a list of 5, family_b dict
        of 7, starter_lake [N,2]) -- re-packed here into the flat [N,12,256]/
        [N,12,256,2] layout this kernel expects.
    pos_x, pos_y: torch.float32 [M] (flattened -- caller reshapes/flattens as
        needed; noise_gpu's [M,P] 2D convention isn't supported here, flatten
        first).
    seed_idx: torch.int64 [M].
    Returns torch.float32 [M].
    """
    device = pos_x.device
    M = pos_x.shape[0]

    p1_flat, p2_flat, p3_flat, gx_flat, gy_flat, starter_x, starter_y = _pack_tables(tables)

    pos_x = pos_x.contiguous().to(torch.float32)
    pos_y = pos_y.contiguous().to(torch.float32)
    seed_idx_i64 = seed_idx.contiguous().to(torch.int64)

    out = torch.empty(M, dtype=torch.float32, device=device)

    grid = (triton.cdiv(M, block_size),)
    _elevation_nauvis_kernel[grid](
        pos_x, pos_y, seed_idx_i64,
        p1_flat, p2_flat, p3_flat, gx_flat, gy_flat,
        starter_x, starter_y,
        out,
        M,
        BLOCK_SIZE=block_size,
        **_scale_kwargs(),
    )
    return out


# ============================================================================
# Stripe classification: base_x/base_y/dir_code/seed_idx in, LAND/BLOCKED +
# min_elev + max_dist out -- NO PyTorch-side position broadcasting (positions
# are generated from base+dir*offset INSIDE the kernel) and NO PyTorch-side
# min/max/threshold reduction (also done in-kernel, by _reduce_classify_kernel).
#
# WHY TWO KERNELS, NOT ONE: the tempting fully-fused design is one kernel
# where each thread owns a whole stripe (not a point) and loops over all
# NUM_POINTS internally, never writing per-point elevations to global memory
# at all. Rejected -- that nests a NUM_POINTS-iteration Python loop AROUND
# the entire 26-noise_internal-call body (itself already unrolled across
# 1-5 octaves x 4 corners per call), so Triton's trace-time unrolling would
# multiply the already-large kernel body by NUM_POINTS (18x or 66x), risking
# very slow compilation and/or register-file spilling. Splitting into (1) a
# point-eval kernel -- same per-point body as _elevation_nauvis_kernel,
# UNCHANGED size, just R*NUM_POINTS lanes instead of M lanes, computing each
# lane's (x,y) from (base,dir,offset) instead of reading precomputed
# pos_x/pos_y arrays -- and (2) a cheap reduction kernel (R lanes, each
# looping NUM_POINTS times over trivial loads/compares, negligible cost
# regardless of NUM_POINTS) gets the same "no PyTorch assembly, no PyTorch
# reduction" result without that blowup risk.
# ============================================================================

EDGE_LAND = 0
EDGE_BLOCKED = 1


@triton.jit
def _stripe_eval_kernel(
    base_x_ptr, base_y_ptr, dir_code_ptr, seed_idx_ptr,
    p1_ptr, p2_ptr, p3_ptr, gx_ptr, gy_ptr,
    starter_x_ptr, starter_y_ptr,
    elevation_out_ptr,
    R,
    NUM_POINTS: tl.constexpr,
    DIR_DX_0: tl.constexpr, DIR_DX_1: tl.constexpr, DIR_DX_2: tl.constexpr, DIR_DX_3: tl.constexpr,
    DIR_DY_0: tl.constexpr, DIR_DY_1: tl.constexpr, DIR_DY_2: tl.constexpr, DIR_DY_3: tl.constexpr,
    AMP_HILLS_BILLOWS: tl.constexpr,
    AMP_MACRO1: tl.constexpr,
    AMP_MACRO2: tl.constexpr,
    NAUVIS_HILLS_INPUT_SCALE: tl.constexpr,
    NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE: tl.constexpr,
    NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_INPUT_SCALE: tl.constexpr,
    NAUVIS_PERSISTANCE_OUTPUT_SCALE: tl.constexpr,
    NAUVIS_DETAIL_INPUT_SCALE: tl.constexpr,
    NAUVIS_MACRO_INPUT_SCALE: tl.constexpr,
    NAUVIS_OFFSET_X: tl.constexpr,
    STARTING_LAKE_NOISE_INPUT_SCALE: tl.constexpr,
    STARTING_LAKE_NOISE_OUTPUT_SCALE: tl.constexpr,
    STARTING_MACRO_MULTIPLIER_BASE: tl.constexpr,
    STARTING_ISLAND_MULTIPLIER: tl.constexpr,
    ELEVATION_MAGNITUDE: tl.constexpr,
    WLC_AMPLITUDE: tl.constexpr,
    WATER_LEVEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One lane = one STRIPE POINT (R*NUM_POINTS lanes total, flat index m =
    r*NUM_POINTS + o). Computes (x,y) from (base_x[r], base_y[r], dir[r],
    offset o) internally -- the host never builds an [R,NUM_POINTS] position
    array."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    M = R * NUM_POINTS
    mask = offs < M

    r = offs // NUM_POINTS
    o = (offs % NUM_POINTS).to(tl.float32)

    base_x = tl.load(base_x_ptr + r, mask=mask, other=0.0)
    base_y = tl.load(base_y_ptr + r, mask=mask, other=0.0)
    dir_code = tl.load(dir_code_ptr + r, mask=mask, other=0).to(tl.int32)
    seed_idx = tl.load(seed_idx_ptr + r, mask=mask, other=0).to(tl.int64)

    dir_dx = tl.where(dir_code == 0, DIR_DX_0, tl.where(dir_code == 1, DIR_DX_1,
              tl.where(dir_code == 2, DIR_DX_2, DIR_DX_3)))
    dir_dy = tl.where(dir_code == 0, DIR_DY_0, tl.where(dir_code == 1, DIR_DY_1,
              tl.where(dir_code == 2, DIR_DY_2, DIR_DY_3)))

    pos_x = base_x + dir_dx * o
    pos_y = base_y + dir_dy * o

    elevation = _elevation_nauvis_point(
        pos_x, pos_y, seed_idx,
        p1_ptr, p2_ptr, p3_ptr, gx_ptr, gy_ptr,
        starter_x_ptr, starter_y_ptr,
        mask,
        AMP_HILLS_BILLOWS, AMP_MACRO1, AMP_MACRO2,
        NAUVIS_HILLS_INPUT_SCALE, NAUVIS_HILLS_CLIFF_LEVEL_INPUT_SCALE,
        NAUVIS_BRIDGE_BILLOWS_INPUT_SCALE, NAUVIS_PERSISTANCE_INPUT_SCALE,
        NAUVIS_PERSISTANCE_OUTPUT_SCALE, NAUVIS_DETAIL_INPUT_SCALE,
        NAUVIS_MACRO_INPUT_SCALE, NAUVIS_OFFSET_X,
        STARTING_LAKE_NOISE_INPUT_SCALE, STARTING_LAKE_NOISE_OUTPUT_SCALE,
        STARTING_MACRO_MULTIPLIER_BASE, STARTING_ISLAND_MULTIPLIER,
        ELEVATION_MAGNITUDE, WLC_AMPLITUDE, WATER_LEVEL,
    )
    tl.store(elevation_out_ptr + offs, elevation, mask=mask)


@triton.jit
def _reduce_classify_kernel(
    elevation_ptr, base_x_ptr, base_y_ptr, dir_code_ptr,
    classification_out_ptr, min_elev_out_ptr, max_dist_out_ptr,
    R, epsilon_fp,
    NUM_POINTS: tl.constexpr,
    DIR_DX_0: tl.constexpr, DIR_DX_1: tl.constexpr, DIR_DX_2: tl.constexpr, DIR_DX_3: tl.constexpr,
    DIR_DY_0: tl.constexpr, DIR_DY_1: tl.constexpr, DIR_DY_2: tl.constexpr, DIR_DY_3: tl.constexpr,
    EDGE_LAND: tl.constexpr, EDGE_BLOCKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One lane = one REQUEST (stripe). Reduces its NUM_POINTS elevations
    (elevation_ptr, laid out [R, NUM_POINTS] row-major, i.e. flat index
    r*NUM_POINTS+o -- matching _stripe_eval_kernel's output layout exactly)
    to a single min, and separately derives max_dist from the two stripe
    ENDPOINTS ONLY (o=0 and o=NUM_POINTS-1): distance-from-spawn along a
    straight line is a convex function of the step index (its second
    difference is the constant dx^2+dy^2 > 0), so its max over the sampled
    range is always at an endpoint, never in the interior -- exact, not an
    approximation, and cheaper than tracking a running max over all
    NUM_POINTS iterations."""
    pid = tl.program_id(0)
    r = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = r < R

    base_x = tl.load(base_x_ptr + r, mask=mask, other=0.0)
    base_y = tl.load(base_y_ptr + r, mask=mask, other=0.0)
    dir_code = tl.load(dir_code_ptr + r, mask=mask, other=0).to(tl.int32)
    dir_dx = tl.where(dir_code == 0, DIR_DX_0, tl.where(dir_code == 1, DIR_DX_1,
              tl.where(dir_code == 2, DIR_DX_2, DIR_DX_3)))
    dir_dy = tl.where(dir_code == 0, DIR_DY_0, tl.where(dir_code == 1, DIR_DY_1,
              tl.where(dir_code == 2, DIR_DY_2, DIR_DY_3)))

    min_elev = tl.full(r.shape, float("inf"), dtype=tl.float32)
    row_base = r * NUM_POINTS
    for o in range(NUM_POINTS):
        e = tl.load(elevation_ptr + row_base + o, mask=mask, other=float("inf"))
        min_elev = tl.minimum(min_elev, e)

    far_o = NUM_POINTS - 1
    far_x = base_x + dir_dx * far_o
    far_y = base_y + dir_dy * far_o
    dist_near = tl.sqrt(base_x * base_x + base_y * base_y)
    dist_far = tl.sqrt(far_x * far_x + far_y * far_y)
    max_dist = tl.maximum(dist_near, dist_far)

    classification = tl.where(min_elev > epsilon_fp, EDGE_LAND, EDGE_BLOCKED).to(tl.int8)

    tl.store(classification_out_ptr + r, classification, mask=mask)
    tl.store(min_elev_out_ptr + r, min_elev, mask=mask)
    tl.store(max_dist_out_ptr + r, max_dist, mask=mask)


def classify_stripes_triton(tables, base_x, base_y, dir_code, seed_idx, epsilon_fp,
                             stripe_tiles=16, block_size=256):
    """
    Self-contained stripe classifier: ONE request per (base_x, base_y,
    dir_code, seed_idx) row, each covering BOTH endpoint nodes plus
    `stripe_tiles` interior tiles (NUM_POINTS = stripe_tiles + 2 total sample
    points) -- unlike stage1_5_oracle.py's current segment convention (which
    samples only `node_spacing` tiles, from its own base node up to but NOT
    including the far endpoint, relying on a DIFFERENT segment's test to
    cover it), every request here is independently self-contained.

    No PyTorch-side position broadcasting (positions are generated from
    base+dir*offset INSIDE _stripe_eval_kernel) and no PyTorch-side
    min/max/threshold reduction (done in-kernel by _reduce_classify_kernel)
    -- host inputs are just 4 flat [R] arrays, host outputs just 3 flat [R]
    arrays.

    base_x, base_y: torch.float32 [R] -- the stripe's NEAR endpoint (world
        tile coordinates).
    dir_code: torch.int64 [R] -- 0/1/2/3 for +x/-x/+y/-y, matching
        stage1_5_oracle.DIR_DI/DIR_DJ exactly.
    seed_idx: torch.int64 [R].
    epsilon_fp: python float.
    stripe_tiles: python int -- interior tile count; NUM_POINTS = stripe_tiles+2.
        Different values (e.g. 16 vs 64) each get their own compiled kernel
        variant automatically (Triton specializes per tl.constexpr value) --
        no separate Python-level kernel needed per stripe length.

    Returns dict: classification (torch.int8 [R]), min_elev (torch.float32
    [R]), max_dist (torch.float32 [R]) -- same contract as
    stage1_5_oracle.classify_edges's per-chunk result.
    """
    device = base_x.device
    R = base_x.shape[0]
    num_points = stripe_tiles + 2

    p1_flat, p2_flat, p3_flat, gx_flat, gy_flat, starter_x, starter_y = _pack_tables(tables)

    base_x = base_x.contiguous().to(torch.float32)
    base_y = base_y.contiguous().to(torch.float32)
    dir_code_i64 = dir_code.contiguous().to(torch.int64)
    seed_idx_i64 = seed_idx.contiguous().to(torch.int64)

    elevation = torch.empty(R * num_points, dtype=torch.float32, device=device)
    grid1 = (triton.cdiv(R * num_points, block_size),)
    _stripe_eval_kernel[grid1](
        base_x, base_y, dir_code_i64, seed_idx_i64,
        p1_flat, p2_flat, p3_flat, gx_flat, gy_flat,
        starter_x, starter_y,
        elevation,
        R,
        NUM_POINTS=num_points,
        DIR_DX_0=DIR_DX[0], DIR_DX_1=DIR_DX[1], DIR_DX_2=DIR_DX[2], DIR_DX_3=DIR_DX[3],
        DIR_DY_0=DIR_DY[0], DIR_DY_1=DIR_DY[1], DIR_DY_2=DIR_DY[2], DIR_DY_3=DIR_DY[3],
        BLOCK_SIZE=block_size,
        **_scale_kwargs(),
    )

    classification = torch.empty(R, dtype=torch.int8, device=device)
    min_elev = torch.empty(R, dtype=torch.float32, device=device)
    max_dist = torch.empty(R, dtype=torch.float32, device=device)
    grid2 = (triton.cdiv(R, block_size),)
    _reduce_classify_kernel[grid2](
        elevation, base_x, base_y, dir_code_i64,
        classification, min_elev, max_dist,
        R, float(epsilon_fp),
        NUM_POINTS=num_points,
        DIR_DX_0=DIR_DX[0], DIR_DX_1=DIR_DX[1], DIR_DX_2=DIR_DX[2], DIR_DX_3=DIR_DX[3],
        DIR_DY_0=DIR_DY[0], DIR_DY_1=DIR_DY[1], DIR_DY_2=DIR_DY[2], DIR_DY_3=DIR_DY[3],
        EDGE_LAND=EDGE_LAND, EDGE_BLOCKED=EDGE_BLOCKED,
        BLOCK_SIZE=block_size,
    )

    return {"classification": classification, "min_elev": min_elev, "max_dist": max_dist}
