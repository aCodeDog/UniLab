"""Rasterize the G1 stepping-stone box terrain into a MuJoCo height field (PNG).

The IsaacLab vision ablation feeds the policy a terrain height scan.  UniLab's
height scanner samples a MuJoCo ``hfield`` geom, but our stepping-stone scene
collides against explicit box geoms (ported from the deploy big-map XML).  This
script rasterizes those boxes' top surfaces into a 16-bit grayscale heightmap so
a *scan-only* (contact-disabled) hfield geom can be added to the scene.

All boxes in the big-map are yaw-rotated only (verified: quat qx==qy==0), so each
box top surface is horizontal and a 2.5-D heightmap is an exact representation.

The PNG/`<hfield size>` convention matches ``unilab.terrains.terrain_generator``:
- pixel ``[row, col]`` maps to world ``(x, y)`` with
  ``col = (x + size_x/2) / hscale`` and ``row = (-y + size_y/2) / hscale``;
- stored value is ``(z - z_min) / (z_max - z_min)`` scaled to uint16;
- ``<hfield size="rx ry z_extent z_base">`` with ``rx = size_x/2`` etc., geom
  placed at ``pos=(0, 0, z_min)``.

Usage::

    uv run python scripts/motion/build_stepping_stone_hfield.py --deploy-xml DEPLOY.xml \
        --resolution 0.05
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

DEFAULT_OUTPUT = (
    ASSETS_ROOT_PATH / "robots" / "g1" / "assets" / "hfields" / "stepping_stone_terrain.png"
)


def _parse_boxes(deploy_xml: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (half_sizes[N,3], positions[N,3], yaw[N]) for every box geom."""
    worldbody = ET.parse(deploy_xml).getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in {deploy_xml}")
    sizes: list[list[float]] = []
    positions: list[list[float]] = []
    yaws: list[float] = []
    for geom in worldbody.findall("geom"):
        if geom.get("type") != "box":
            continue
        sizes.append([float(v) for v in geom.get("size").split()])
        positions.append([float(v) for v in geom.get("pos").split()])
        quat = geom.get("quat")
        if quat is None:
            yaws.append(0.0)
        else:
            w, x, y, z = (float(v) for v in quat.split())
            # Yaw-only quaternion (x==y==0): yaw = 2*atan2(z, w).
            yaws.append(2.0 * np.arctan2(z, w))
    if not sizes:
        raise ValueError(f"No box geoms found in {deploy_xml}")
    return (
        np.asarray(sizes, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(yaws, dtype=np.float64),
    )


def rasterize_heightmap(
    half_sizes: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    *,
    resolution: float,
    floor_z: float = 0.0,
) -> tuple[np.ndarray, float, float, float, float]:
    """Rasterize box tops to a height grid.

    Returns ``(heights_yx, size_x, size_y, z_min, z_max)`` where ``heights_yx`` is
    indexed ``[row(y), col(x)]`` following the MuJoCo/terrain_generator convention.
    """
    # World extent from box footprints (axis-aligned bound including yaw).
    # For a yaw-rotated box, the AABB half-extent is |hx*cos|+|hy*sin| etc.
    cos = np.abs(np.cos(yaws))
    sin = np.abs(np.sin(yaws))
    aabb_hx = half_sizes[:, 0] * cos + half_sizes[:, 1] * sin
    aabb_hy = half_sizes[:, 0] * sin + half_sizes[:, 1] * cos
    x_min = float((positions[:, 0] - aabb_hx).min())
    x_max = float((positions[:, 0] + aabb_hx).max())
    y_min = float((positions[:, 1] - aabb_hy).min())
    y_max = float((positions[:, 1] + aabb_hy).max())
    # Symmetric extent centered on origin (terrain is centered at 0).
    rx = max(abs(x_min), abs(x_max))
    ry = max(abs(y_min), abs(y_max))
    size_x = 2.0 * rx
    size_y = 2.0 * ry

    cols = int(np.ceil(size_x / resolution)) + 1
    rows = int(np.ceil(size_y / resolution)) + 1

    # Grid cell centers in world coords.
    xs = -rx + np.arange(cols) * resolution  # col -> x
    ys = ry - np.arange(rows) * resolution  # row -> y  (row 0 = +y, matches generator)
    heights = np.full((rows, cols), floor_z, dtype=np.float64)

    box_top = positions[:, 2] + half_sizes[:, 2]

    # For each box, mark grid cells whose center lies inside the (yaw-rotated)
    # footprint and raise the height to the box top (max over overlapping boxes).
    for i in range(half_sizes.shape[0]):
        px, py = positions[i, 0], positions[i, 1]
        hx, hy = half_sizes[i, 0], half_sizes[i, 1]
        c, s = np.cos(yaws[i]), np.sin(yaws[i])
        # Coarse AABB to limit the cell window.
        ahx, ahy = aabb_hx[i], aabb_hy[i]
        c0 = max(int(np.floor((px - ahx + rx) / resolution)), 0)
        c1 = min(int(np.ceil((px + ahx + rx) / resolution)), cols - 1)
        r0 = max(int(np.floor((ry - (py + ahy)) / resolution)), 0)
        r1 = min(int(np.ceil((ry - (py - ahy)) / resolution)), rows - 1)
        if c1 < c0 or r1 < r0:
            continue
        gx = xs[c0 : c1 + 1][None, :]  # (1, w)
        gy = ys[r0 : r1 + 1][:, None]  # (h, 1)
        dx = gx - px
        dy = gy - py
        # Rotate into box-local frame (inverse yaw).
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
        sub = heights[r0 : r1 + 1, c0 : c1 + 1]
        np.maximum(sub, np.where(inside, box_top[i], sub), out=sub)

    z_min = floor_z
    z_max = float(box_top.max())
    return heights, size_x, size_y, z_min, z_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=float, default=0.05)
    args = parser.parse_args()

    import imageio.v3 as iio

    half_sizes, positions, yaws = _parse_boxes(args.deploy_xml)
    heights, size_x, size_y, z_min, z_max = rasterize_heightmap(
        half_sizes, positions, yaws, resolution=args.resolution
    )

    span = max(z_max - z_min, 1e-6)
    normalized = np.clip((heights - z_min) / span, 0.0, 1.0)
    grid_u16 = np.rint(normalized * np.iinfo(np.uint16).max).astype(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, grid_u16)

    rows, cols = heights.shape
    print(f"Wrote {cols}x{rows} hfield ({args.resolution} m) -> {args.output}")
    print(f"  world size_x={size_x:.3f} size_y={size_y:.3f}")
    print(f"  z_min={z_min:.4f} z_max={z_max:.4f} z_extent={span:.4f}")
    print("  <hfield> size attr (rx ry z_extent z_base):")
    print(f"    {size_x / 2:.6g} {size_y / 2:.6g} {span:.6g} 0.05")
    print(f"  geom pos: 0 0 {z_min:.6g}")


if __name__ == "__main__":
    main()
