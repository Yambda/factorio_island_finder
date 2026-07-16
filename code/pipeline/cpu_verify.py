"""
Final-authority CPU verification: runs the project's real `largest_island`
binary (exact BFS flood fill, seed_finders/largest_island/stages.cpp) on one
seed at a time via --first-seed S --last-seed S+2, and parses its CSV output.

*** THIS IS THE NON-NEGOTIABLE CORRECTNESS GATE OF THE WHOLE PIPELINE. ***
Per project convention, ANY seed the GPU pipeline flags as a candidate island
MUST be re-checked here before being reported as a real result -- the GPU
pipeline (stage1_screen.py + stage2_floodfill.py) is an approximate,
high-throughput pre-filter only. It trades bit-exactness for speed, so on
its own it is never sufficient evidence of a real natural island.
"""
import os
import subprocess
import tempfile

BINARY = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "build", "seed_finders", "largest_island", "largest_island",
))

# Must mirror seed_finders/largest_island/stages.hpp's FLOOD_FILL_MAX_CELLS
# exactly -- there is no shared source of truth between the C++ binary and
# this Python module, so if one changes, change the other too. RAISED
# 2026-07-09 from 200,000 to 2,000,000 after a real confirmed island
# (3,097,730 tiles^2 = 193,608 cells) came within 3.2% of the old cap.
FLOOD_FILL_MAX_CELLS = 2_000_000
FLOOD_FILL_STEP = 4

# Distribution tracking: warn if the largest REAL (bounded) island found so
# far has climbed past this fraction of the current cap -- that's the signal
# to raise FLOOD_FILL_MAX_CELLS again (in BOTH stages.hpp and here), not a
# reason to guess a bigger cap speculatively up front (see the measured
# ~5.8x per-call slowdown a too-large cap costs on every UNBOUNDED/capped
# candidate, which is the common case -- stages.hpp's own comments).
DISTRIBUTION_WARN_FRACTION = 0.60
_max_confirmed_cells_seen = 0  # process-lifetime high-water mark


def cpu_verify_seed(seed, binary=BINARY, timeout=60):
    """
    Runs the exact CPU flood fill for a single seed.

    Returns dict: {"is_island": bool, "area_tiles2": float or None}
    is_island=True iff the CPU flood fill did NOT hit the FLOOD_FILL_MAX_CELLS
    cap (i.e. the landmass closed off -- a real, bounded natural island).

    Side effect: tracks the largest confirmed-real-island cell count seen by
    this process and prints a warning if it exceeds DISTRIBUTION_WARN_FRACTION
    of the current cap -- see module docstring above.
    """
    global _max_confirmed_cells_seen

    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"CPU verifier binary not found at {binary!r}. Build it first, e.g.:\n"
            f"  cd {os.path.dirname(os.path.dirname(binary))} && "
            f"cmake --build build --config Release --target largest_island"
        )

    with tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as tf:
        out_path = tf.name
    try:
        subprocess.run(
            [binary, "--output", out_path, "--threads", "1",
             "--first-seed", str(seed), "--last-seed", str(seed + 2)],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        with open(out_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        # header line: rank;seed;score;water scale;water coverage;elevation type
        # a row only appears if the seed SURVIVED (i.e. is a real bounded island)
        for line in lines[1:]:
            parts = line.split(";")
            if len(parts) >= 3 and int(parts[1]) == seed:
                # area==0 is the rare (~1-in-several-hundred) case of the
                # starting_lake_noise component placing the starting lake
                # exactly at spawn's own tile -- a real, verified game state
                # (spawn on water), not a pipeline bug. Kept, not filtered.
                area_tiles2 = float(parts[2])
                cells = area_tiles2 / (FLOOD_FILL_STEP * FLOOD_FILL_STEP)
                if cells > _max_confirmed_cells_seen:
                    _max_confirmed_cells_seen = cells
                    frac = cells / FLOOD_FILL_MAX_CELLS
                    if frac > DISTRIBUTION_WARN_FRACTION:
                        print(f"*** CAP WARNING: seed {seed}'s confirmed area ({area_tiles2:,.0f} "
                              f"tiles^2 = {cells:,.0f} cells) is {frac*100:.1f}% of "
                              f"FLOOD_FILL_MAX_CELLS ({FLOOD_FILL_MAX_CELLS:,}) -- exceeds the "
                              f"{DISTRIBUTION_WARN_FRACTION*100:.0f}% threshold. Raise the cap in "
                              f"BOTH seed_finders/largest_island/stages.hpp and this module's "
                              f"FLOOD_FILL_MAX_CELLS, then rebuild largest_island, before this "
                              f"climbs further. ***", flush=True)
                return {"is_island": True, "area_tiles2": area_tiles2}
        return {"is_island": False, "area_tiles2": None}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    for s in (20271579, 1489155600):
        print(s, cpu_verify_seed(s))
