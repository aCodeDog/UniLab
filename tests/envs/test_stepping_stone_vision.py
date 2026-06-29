"""Tests for the G1 vision stepping-stone task (raw-reference actor + height scan).

Fast tests cover config wiring, the scan-hfield scene (contact isolation), the
hfield resolution-fidelity measurement (USER-REQUIRED), and dual-loader pairing.
The ``slow`` test builds the env and runs reset + step end to end.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from unilab.base.registry import ensure_registries

_DEPLOY_XML = (
    "/home/zifan_wang/Mondo/whole_body_tracking/scripts/mujoco_deploy/"
    "g1_robot/g1_29dof_big_map_urdf.xml"
)


def test_stepping_stone_vision_cfg_registers():
    ensure_registries()
    from unilab.base import registry
    from unilab.envs.motion_tracking.g1.stepping_stone_vision import G1SteppingStoneVisionEnvCfg

    assert registry.contains("G1SteppingStoneVision")
    cfg = G1SteppingStoneVisionEnvCfg()
    assert cfg.scene.model_file.endswith("scene_stepping_stone_vision.xml")
    assert cfg.scan_frame_body == "torso_link"
    assert cfg.height_scan.geom_name == "scan_hfield_geom"
    # 17 x 11 = 187 rays (matches IsaacLab GridPatternCfg(0.1, (1.6, 1.0))).
    assert len(cfg.height_scan.measured_points_x) == 17
    assert len(cfg.height_scan.measured_points_y) == 11
    # Inherits stepping-stone timing/thresholds.
    assert cfg.sim_substeps == 4
    assert cfg.anchor_pos_z_threshold == pytest.approx(0.35)


def test_vision_scene_scan_hfield_is_contact_disabled():
    """The scan hfield must survive discardvisual yet collide with nothing."""
    pytest.importorskip("mujoco")
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend.mujoco.xml import create_discardvisual_xml

    scene = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone_vision.xml"
    if not scene.exists():
        pytest.skip("vision scene not generated")

    # After discardvisual (what physics actually uses) the geom must remain.
    model = mujoco.MjModel.from_xml_path(create_discardvisual_xml(str(scene)))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scan_hfield_geom")
    assert gid >= 0, "scan_hfield_geom was discarded by discardvisual"

    contype = int(model.geom_contype[gid])
    conaff = int(model.geom_conaffinity[gid])
    # Must not collide with the floor/robot/terrain (all contype=1, conaffinity=1).
    assert (contype & 1) == 0 and (1 & conaff) == 0, "scan hfield can collide with terrain"
    assert model.nhfield >= 1


@pytest.mark.skipif(not Path(_DEPLOY_XML).exists(), reason="deploy big-map XML not present")
def test_hfield_resolution_fidelity():
    """USER-REQUIRED: measure heightmap fidelity vs the analytic box terrain.

    Compares the rasterized hfield against the exact box top-surface height at
    random query points (interior) and reports edge-band aliasing separately.
    """
    pytest.importorskip("mujoco")
    import imageio.v3 as iio

    from unilab.assets import ASSETS_ROOT_PATH

    png = ASSETS_ROOT_PATH / "robots" / "g1" / "assets" / "hfields" / "stepping_stone_terrain.png"
    if not png.exists():
        pytest.skip("hfield PNG not generated")

    # Parse box terrain (yaw-rotated boxes).
    xml = Path(_DEPLOY_XML).read_text()
    geoms = re.findall(
        r'<geom name="[^"]*" type="box" size="([-\d. ]+)" pos="([-\d. ]+)" quat="([-\d. ]+)"',
        xml,
    )
    hs = np.array([[float(v) for v in s.split()] for s, _, _ in geoms])
    pos = np.array([[float(v) for v in p.split()] for _, p, _ in geoms])
    quat = np.array([[float(v) for v in q.split()] for _, _, q in geoms])
    yaw = 2.0 * np.arctan2(quat[:, 3], quat[:, 0])
    box_top = pos[:, 2] + hs[:, 2]

    def analytic_height(x, y):
        dx = x - pos[:, 0]
        dy = y - pos[:, 1]
        c, s = np.cos(yaw), np.sin(yaw)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        inside = (np.abs(lx) <= hs[:, 0]) & (np.abs(ly) <= hs[:, 1])
        return box_top[inside].max() if inside.any() else 0.0

    def edge_dist(x, y):
        """Distance to the nearest box edge (small => height discontinuity band)."""
        dx = x - pos[:, 0]
        dy = y - pos[:, 1]
        c, s = np.cos(yaw), np.sin(yaw)
        lx = np.abs(c * dx + s * dy)
        ly = np.abs(-s * dx + c * dy)
        # Distance from each box's edge contour; min over all boxes.
        return float(np.min(np.maximum(lx - hs[:, 0], ly - hs[:, 1])))

    grid = iio.imread(png).astype(np.float64)
    rows, cols = grid.shape
    res = 0.05
    rx = cols * res / 2.0
    ry = rows * res / 2.0
    z_min, z_extent = 0.0, 0.4

    def png_height(x, y):
        col = int(round((x + rx) / res))
        row = int(round((ry - y) / res))
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        return z_min + grid[row, col] / 65535.0 * z_extent

    # Nearest-cell sampling can legitimately alias within ~2 cells of a step edge
    # (the height is discontinuous there); classify those points as "edge band".
    edge_band = 2.0 * res
    rng = np.random.default_rng(0)
    interior_err, edge_err = [], []
    for _ in range(5000):
        x = rng.uniform(-33.0, 33.0)
        y = rng.uniform(-33.0, 33.0)
        err = abs(analytic_height(x, y) - png_height(x, y))
        if abs(edge_dist(x, y)) < edge_band:
            edge_err.append(err)
        else:
            interior_err.append(err)

    interior_err = np.array(interior_err)
    # Interior fidelity: away from step discontinuities the heightmap is exact to
    # within one quantization step (0.4 m / 65535 ~ 6e-6 m).
    assert interior_err.max() < 0.01, f"interior max error {interior_err.max():.4f} >= 0.01"
    assert interior_err.mean() < 1e-3, f"interior mean error {interior_err.mean():.5f} >= 1e-3"
    # Edge-band aliasing is expected and bounded by the tallest step (<= z_extent).
    if edge_err:
        assert np.array(edge_err).max() <= z_extent + 1e-3


def test_dual_loader_pairing_validation():
    """raw and optimized loaders must be frame-aligned; mismatch must fail closed."""
    import glob
    import os

    from unilab.envs.motion_tracking.g1.motion_loader import MotionLoader
    from unilab.envs.motion_tracking.g1.stepping_stone import STEPPING_STONE_MOTION_DIR
    from unilab.envs.motion_tracking.g1.stepping_stone_vision import (
        STEPPING_STONE_RAW_MOTION_DIR,
        _validate_paired_loaders,
    )

    opt = sorted(glob.glob(os.path.join(STEPPING_STONE_MOTION_DIR, "*.npz")))[:3]
    raw = sorted(glob.glob(os.path.join(STEPPING_STONE_RAW_MOTION_DIR, "*.npz")))[:3]
    if not opt or not raw:
        pytest.skip("motion files not present")

    opt_loader = MotionLoader(opt)
    raw_loader = MotionLoader(raw)
    # Matched pair validates OK.
    _validate_paired_loaders(opt_loader, raw_loader)
    # Mismatched count fails closed.
    with pytest.raises(ValueError):
        _validate_paired_loaders(opt_loader, MotionLoader(raw[:2]))


@pytest.mark.slow
def test_stepping_stone_vision_env_builds_resets_steps():
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")

    import glob
    import os

    ensure_registries()
    from unilab.base import registry
    from unilab.envs.motion_tracking.g1.stepping_stone import STEPPING_STONE_MOTION_DIR
    from unilab.envs.motion_tracking.g1.stepping_stone_vision import STEPPING_STONE_RAW_MOTION_DIR

    opt = sorted(glob.glob(os.path.join(STEPPING_STONE_MOTION_DIR, "*.npz")))[:2]
    raw = sorted(glob.glob(os.path.join(STEPPING_STONE_RAW_MOTION_DIR, "*.npz")))[:2]
    if not opt or not raw:
        pytest.skip("motion files not present")

    env = registry.make(
        "G1SteppingStoneVision",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "motion_file": opt,
            "raw_motion_file": raw,
            "sampling_mode": "start",
        },
    )
    try:
        spec = env.obs_groups_spec
        assert sum(spec.values()) == env.observation_space.shape[0]
        # Actor = command(58) + anchor_pos(3) + anchor_ori(6) + gyro(3) + joints/vel/act(87)
        # = 157 (base_lin_vel dropped as deploy-incompatible) + 187 height scan.
        assert spec["obs"] == 157 + 187

        state = env.init_state()
        assert state.obs["obs"].shape == (2, spec["obs"])
        scan = state.obs["obs"][:, -187:]
        assert np.all(np.isfinite(scan))

        state = env.step(np.zeros((2, env.action_space.shape[0])))
        assert np.all(np.isfinite(state.reward))
    finally:
        env.close()
