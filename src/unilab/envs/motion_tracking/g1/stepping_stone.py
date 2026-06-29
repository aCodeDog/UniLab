"""G1 stepping-stone motion tracking - terrain-anchored multi-motion imitation.

Reproduces the IsaacLab ``RGMT-SteppingStone-G1-v0`` task (MLP teacher variant) by
reusing :class:`G1MotionTrackingEnv` on a fixed big-map terrain scene.  The
reference clips are authored in the terrain's world frame, so every environment
shares the same world (UniLab's MuJoCo backend already runs a single shared model
with no per-env origin offset, matching IsaacLab's ``env_spacing=0.0``) and the
reset uses the motion root position directly without any offset.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg

from .tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    PoseRandomization,
    VelocityRandomization,
)

# Terrain-anchored, smoothed, 50 fps reference clips for the mocaphouse big-map.
# These are the IsaacLab WBT clips converted to UniLab body-id layout (a leading
# zero ``world`` body row) via ``scripts/motion/wbt_to_unilab_npz.py``.
STEPPING_STONE_MOTION_DIR = (
    "/data/mondo-training-dataset/whole_body_tracking_motions/motions/terrain/"
    "terrain_mocaphouse/walk_dance1sub2start/optimized_smoothed_w9_s2_unilab"
)


def _discover_stepping_stone_motions() -> list[str]:
    """Return the sorted list of stepping-stone motion NPZ files."""
    files = sorted(glob.glob(os.path.join(STEPPING_STONE_MOTION_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No stepping-stone motion files found in {STEPPING_STONE_MOTION_DIR}"
        )
    return files


@dataclass
class G1SteppingStoneCfg(G1MotionTrackingCfg):
    """Config profile for terrain-anchored multi-motion tracking on the big map."""

    # IsaacLab uses dt=0.005 with decimation=4 (200 Hz sim / 50 Hz control).
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone.xml")
        )
    )
    motion_file: str | list[str] = field(default_factory=_discover_stepping_stone_motions)
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"

    # The clips already carry absolute world pose; do not perturb the reset state
    # (IsaacLab stepping-stone sets pose_range={} / velocity_range={}).
    pose_randomization: PoseRandomization = field(
        default_factory=lambda: PoseRandomization(
            x=(0.0, 0.0),
            y=(0.0, 0.0),
            z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
    )
    velocity_randomization: VelocityRandomization = field(
        default_factory=lambda: VelocityRandomization(
            x=(0.0, 0.0),
            y=(0.0, 0.0),
            z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
    )
    joint_position_range: tuple[float, float] = (0.0, 0.0)

    # Tighter terrain-following thresholds matching the IsaacLab stepping-stone task.
    anchor_pos_z_threshold: float = 0.35
    ee_body_pos_z_threshold: float = 0.35


@registry.envcfg("G1SteppingStone")
@dataclass
class G1SteppingStoneEnvCfg(G1SteppingStoneCfg):
    """Registered configuration for G1 stepping-stone tracking."""

    pass


@registry.env("G1SteppingStone", sim_backend="mujoco")
@registry.env("G1SteppingStone", sim_backend="motrix")
class G1SteppingStoneEnv(G1MotionTrackingEnv):
    """G1 stepping-stone tracking environment; reuses all motion-tracking logic."""

    _cfg: G1SteppingStoneCfg
