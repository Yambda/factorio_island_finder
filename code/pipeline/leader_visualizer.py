"""
Generates a per-stage visualization (terrain, stage-1 ray screen, pillar
strip, pass-1 64-tile net, pass-2 32-tile net) for one seed -- used to render
a fresh set of images every time a new largest-confirmed-island record is
set, so there's always an up-to-date visual explanation of how the current
leader looks to each stage of the pipeline. Adapted from the one-off
scratchpad script used earlier in this project's history; kept here as an
importable, in-process module (reuses the caller's already-initialized CUDA
context instead of spawning a new process per call).
"""
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import noise_gpu as ng
import stage1_screen as stage1
import stage1_5_pillar as pillar

STEP = 4
MARGIN_TILES = 300  # extra margin beyond the island's own extent / RING_TILES


def _terrain_and_island(seed, half_extent, device):
    seeds_t = torch.tensor([seed], dtype=torch.int64, device=device)
    tables = ng.build_all_tables(seeds_t)

    dim = (2 * half_extent) // STEP
    xs = torch.arange(-half_extent, half_extent, STEP, dtype=torch.float32, device=device)
    ys = torch.arange(-half_extent, half_extent, STEP, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    px = gx.reshape(1, -1)
    py = gy.reshape(1, -1)

    CHUNK = 400_000
    elev_chunks = []
    with torch.no_grad():
        for start in range(0, px.shape[1], CHUNK):
            end = min(start + CHUNK, px.shape[1])
            elev_chunks.append(ng.elevation_nauvis(tables, px[:, start:end], py[:, start:end]))
    elevation = torch.cat(elev_chunks, dim=1).reshape(dim, dim)
    is_water = (elevation <= 0.0).cpu().numpy()

    def w2i(x, y):
        return int(round((y + half_extent) / STEP)), int(round((x + half_extent) / STEP))

    sr, sc = w2i(0, 0)
    island = np.zeros((dim, dim), dtype=bool)
    visited = np.zeros((dim, dim), dtype=bool)
    stack = [(sr, sc)]
    visited[sr, sc] = True
    hit_border = False
    while stack:
        r, c = stack.pop()
        if is_water[r, c]:
            continue
        island[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= dim or nc < 0 or nc >= dim:
                hit_border = True
                continue
            if not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))

    return dim, is_water, island, hit_border, tables


def generate_leader_visualization(seed, area_tiles2, out_dir, device, verbose=True):
    """
    Renders 0_terrain / 1_stage1_ray_screen / 2_pass1_64tile_net /
    3_pass2_32tile_net / 4_pillar_strip / overview_all_stages.png into
    out_dir (created if missing). Best-effort: any failure is caught and
    logged, never raised, so this can never take down the main pipeline.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Size the canvas from the island's own extent (via a first pass at a
        # generous guess), falling back to a bigger guess if it hit the border.
        # Snapped to a multiple of STEP -- torch.arange's actual element count
        # for a non-multiple range doesn't match (2*half_extent)//STEP (floor
        # division), which silently mismatches the reshape below.
        def snap(v):
            return ((v + STEP - 1) // STEP) * STEP

        half_extent = snap(max(2200, int(math.sqrt(max(area_tiles2, 0)) * 1.2) + MARGIN_TILES))
        dim, is_water, island, hit_border, tables = _terrain_and_island(seed, half_extent, device)
        if hit_border:
            half_extent = snap(int(half_extent * 1.8))
            dim, is_water, island, hit_border, tables = _terrain_and_island(seed, half_extent, device)

        def to_px(x, y):
            return ((x + half_extent) / STEP, (y + half_extent) / STEP)

        def base_image():
            img = np.zeros((dim, dim, 3), dtype=np.uint8)
            img[island] = (40, 230, 60)
            img[~island & is_water] = (51, 83, 95)
            img[~island & ~is_water] = (180, 60, 40)
            return Image.fromarray(img, mode="RGB").convert("RGBA")

        def draw_spawn(draw):
            sx, sy = to_px(0, 0)
            draw.line([(sx - 6, sy), (sx + 6, sy)], fill=(255, 255, 255, 255), width=2)
            draw.line([(sx, sy - 6), (sx, sy + 6)], fill=(255, 255, 255, 255), width=2)

        def draw_ring(draw, radius, color):
            bbox = [to_px(-radius, -radius), to_px(radius, radius)]
            draw.ellipse(bbox, outline=color, width=2)

        def draw_net(draw, spacing, radius_nodes, color, width):
            n = radius_nodes
            lim = n * spacing
            for i in range(-n, n + 1):
                x = i * spacing
                if -half_extent <= x <= half_extent:
                    y0, y1 = max(-lim, -half_extent), min(lim, half_extent)
                    draw.line([to_px(x, y0), to_px(x, y1)], fill=color, width=1)
            for j in range(-n, n + 1):
                y = j * spacing
                if -half_extent <= y <= half_extent:
                    x0, x1 = max(-lim, -half_extent), min(lim, half_extent)
                    draw.line([to_px(x0, y), to_px(x1, y)], fill=color, width=1)

        STAGE1_MAX_RADIUS = 2000.0  # matches find_islands.py's --max-radius default

        seeds_t = torch.tensor([seed], dtype=torch.int64, device=device)
        s1 = stage1.screen_seeds(seeds_t, max_radius=STAGE1_MAX_RADIUS)
        close_score = float(s1["close_score"][0].item())
        hit_radius = s1["hit_radius"][0].cpu().numpy()
        n_angles = hit_radius.shape[0]

        # ---- 0: terrain ----
        img0 = base_image()
        d0 = ImageDraw.Draw(img0, "RGBA")
        draw_spawn(d0)
        img0.convert("RGB").save(os.path.join(out_dir, "0_terrain.png"))

        # ---- 1: stage-1 ray screen ----
        img1 = base_image()
        d1 = ImageDraw.Draw(img1, "RGBA")
        for a in range(n_angles):
            theta = a * (2.0 * math.pi / n_angles)
            r = float(hit_radius[a])
            hit = r < STAGE1_MAX_RADIUS
            ex, ey = r * math.cos(theta), r * math.sin(theta)
            color = (255, 220, 40, 255) if hit else (255, 30, 30, 255)
            d1.line([to_px(0, 0), to_px(ex, ey)], fill=color, width=2)
            if hit:
                px_, py_ = to_px(ex, ey)
                d1.ellipse([px_ - 3, py_ - 3, px_ + 3, py_ + 3], fill=(255, 220, 40, 255))
        draw_ring(d1, STAGE1_MAX_RADIUS, (255, 255, 255, 200))
        draw_spawn(d1)
        img1.convert("RGB").save(os.path.join(out_dir, "1_stage1_ray_screen.png"))

        # ---- 2: pass1 64-tile net ----
        img2 = base_image()
        d2 = ImageDraw.Draw(img2, "RGBA")
        draw_net(d2, 64, 31, (255, 210, 60, 160), 1)
        draw_ring(d2, 1400, (255, 255, 255, 200))
        draw_spawn(d2)
        img2.convert("RGB").save(os.path.join(out_dir, "2_pass1_64tile_net.png"))

        # ---- 3: pass2 32-tile net ----
        img3 = base_image()
        d3 = ImageDraw.Draw(img3, "RGBA")
        draw_net(d3, 32, 62, (140, 200, 255, 110), 1)
        draw_ring(d3, 1400, (255, 255, 255, 200))
        draw_spawn(d3)
        img3.convert("RGB").save(os.path.join(out_dir, "3_pass2_32tile_net.png"))

        # ---- 4: pillar strip (East) ----
        img4 = base_image()
        d4 = ImageDraw.Draw(img4, "RGBA")
        n = pillar.WIDTH_NODES
        for i in range(-pillar.BACK, pillar.L + 1):
            x = i * pillar.SEGMENT_TILES
            if -half_extent <= x <= half_extent:
                y0, y1 = max(-n * pillar.SEGMENT_TILES, -half_extent), min(n * pillar.SEGMENT_TILES, half_extent)
                d4.line([to_px(x, y0), to_px(x, y1)], fill=(255, 210, 60, 200), width=1)
        for j in range(-n, n + 1):
            y = j * pillar.SEGMENT_TILES
            x0 = max(-pillar.BACK * pillar.SEGMENT_TILES, -half_extent)
            x1 = min(pillar.L * pillar.SEGMENT_TILES, half_extent)
            d4.line([to_px(x0, y), to_px(x1, y)], fill=(255, 210, 60, 200), width=1)
        draw_ring(d4, 1400, (255, 255, 255, 200))
        draw_spawn(d4)
        img4.convert("RGB").save(os.path.join(out_dir, "4_pillar_strip.png"))

        # ---- 5: RING_TILES=5000 coverage (current definitional boundary) ----
        # The other panels above are sized to the island's own (usually much
        # smaller) extent, so they can't show the full 5000-tile ring in
        # frame -- this one uses its own, larger half_extent specifically so
        # the ring is always fully visible, regardless of island size.
        half_extent_5k = snap(max(half_extent, 5500))
        dim5, is_water5, island5, hit_border5, _ = _terrain_and_island(seed, half_extent_5k, device)

        def to_px5(x, y):
            return ((x + half_extent_5k) / STEP, (y + half_extent_5k) / STEP)

        farthest_dist5, farthest_rc5 = 0.0, (dim5 // 2, dim5 // 2)
        island_rc = np.argwhere(island5)
        if island_rc.size > 0:
            wx = island_rc[:, 1] * STEP - half_extent_5k
            wy = island_rc[:, 0] * STEP - half_extent_5k
            d = np.hypot(wx, wy)
            i_max = int(np.argmax(d))
            farthest_dist5 = float(d[i_max])
            farthest_rc5 = (int(island_rc[i_max, 0]), int(island_rc[i_max, 1]))

        img5 = np.zeros((dim5, dim5, 3), dtype=np.uint8)
        img5[island5] = (40, 230, 60)
        img5[~island5 & is_water5] = (51, 83, 95)
        img5[~island5 & ~is_water5] = (180, 60, 40)
        im5 = Image.fromarray(img5, mode="RGB").convert("RGBA")
        d5 = ImageDraw.Draw(im5, "RGBA")
        for radius, color in [(1400, (255, 200, 0, 220)), (2000, (255, 140, 0, 220)),
                               (5000, (255, 40, 40, 255))]:
            bbox = [to_px5(-radius, -radius), to_px5(radius, radius)]
            d5.ellipse(bbox, outline=color, width=3)
        sx5, sy5 = to_px5(0, 0)
        fr5, fc5 = farthest_rc5
        d5.line([(sx5, sy5), (fc5, fr5)], fill=(0, 255, 255, 255), width=2)
        d5.ellipse([fc5 - 6, fr5 - 6, fc5 + 6, fr5 + 6], outline=(0, 255, 255, 255), width=3)
        draw_spawn(d5)
        d5.text((10, 10), f"farthest tile: {farthest_dist5:,.0f} -- margin vs RING_TILES=5000: "
                           f"{5000 - farthest_dist5:,.0f}", fill=(255, 255, 255, 255))
        img5_rgb = im5.convert("RGB")
        img5_rgb.save(os.path.join(out_dir, "5_ring5000_coverage.png"))

        # ---- overview ----
        pad, label_h = 20, 32
        tile = dim
        grid_w = tile * 2 + pad * 3
        grid_h = (tile + label_h) * 3 + pad * 4
        combo = Image.new("RGB", (grid_w, grid_h), (18, 18, 20))
        cd = ImageDraw.Draw(combo)
        panels = [
            (img0, "Terrain (ground truth)"),
            (img1, f"Stage 1 ray screen (close_score={close_score:.3f})"),
            (img2, "Pass 1 -- 64-tile net"),
            (img3, "Pass 2 -- 32-tile net"),
            (img4, "Pillar -- East strip"),
            (img5_rgb.resize((tile, tile)), "RING_TILES=5000 coverage"),
        ]
        for idx, (panel_img, label) in enumerate(panels):
            col, row = idx % 2, idx // 2
            x = pad + col * (tile + pad)
            y = pad + row * (tile + label_h + pad)
            cd.text((x, y), label, fill=(230, 230, 230))
            combo.paste(panel_img.convert("RGB"), (x, y + label_h))
        combo.save(os.path.join(out_dir, "overview_all_stages.png"))

        if verbose:
            print(f"leader_visualizer: wrote 7 images to {out_dir} for seed {seed} "
                  f"(area={area_tiles2:,.0f} tiles^2)")
    except Exception as e:
        import sys
        print(f"WARNING: leader_visualizer failed for seed {seed} ({e!r}), continuing without it.",
              file=sys.stderr)
