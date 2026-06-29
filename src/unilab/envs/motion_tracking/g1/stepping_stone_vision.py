"""G1 vision stepping-stone tracking - raw-reference actor with a terrain height scan.

Reproduces the IsaacLab ``...VisionTeacher...FlatMLP`` ablation (PPO finetune):

* the **actor** is driven by the **raw** (un-optimized) motion reference plus a
  terrain **height scan**, so the policy must use terrain perception to bridge the
  gap between the raw reference and the physically-feasible motion;
* the **critic / rewards / reset** use the **optimized** motion (privileged).

Implementation notes (see ``docs/plans/g1_stepping_stone_vision_plan.md``):

* Two ``MotionLoader`` instances share ONE ``MotionSampler``: the optimized loader
  drives reset/reward/critic; the raw loader is indexed by the primary sampler's
  ``current_frames`` (synced-student semantics).  A fail-closed cross-loader check
  guarantees the pair is frame-aligned.
* A contact-disabled ``hfield`` geom (``scan_hfield_geom``) is sampled by UniLab's
  height scanner; physics still collides against the box terrain.
* v1 uses a single-frame height scan (187 heights, no validity mask) mounted on
  ``torso_link``.  This is NOT IsaacLab-equivalent (the reference uses a 10-frame
  history); documented as a follow-up.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.height_scan import (
    DEFAULT_SCAN_POINTS_X,
    DEFAULT_SCAN_POINTS_Y,
    HeightScanConfig,
    init_height_scan_sensor,
    raw_height_scan_obs,
)

from .motion_loader import MotionLoader
from .stepping_stone import G1SteppingStoneCfg
from .tracking import G1MotionTrackingEnv, _write_motion_anchor_transform

# Raw (un-optimized) clips, converted to UniLab body-id layout via
# ``scripts/motion/wbt_to_unilab_npz.py`` (paired 1:1 with the optimized clips).
STEPPING_STONE_RAW_MOTION_DIR = (
    "/data/mondo-training-dataset/whole_body_tracking_motions/motions/terrain/"
    "terrain_mocaphouse/walk_dance1sub2start/raw_unilab"
)


def _discover_raw_motions() -> list[str]:
    """Return the sorted list of raw stepping-stone motion NPZ files."""
    files = sorted(glob.glob(os.path.join(STEPPING_STONE_RAW_MOTION_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No raw stepping-stone motion files found in {STEPPING_STONE_RAW_MOTION_DIR}"
        )
    return files


@dataclass
class G1SteppingStoneVisionCfg(G1SteppingStoneCfg):
    """Config for the raw-reference + height-scan FlatMLP vision ablation."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone_vision.xml")
        )
    )
    # The actor's reference command/anchor come from these raw clips; the optimized
    # clips (inherited ``motion_file``) drive reward/critic/reset.
    raw_motion_file: str | list[str] = field(default_factory=_discover_raw_motions)
    # Yaw-aligned height scan on torso_link (IsaacLab mounts the scanner there).
    # Default grid is 17x11 = 187 rays spanning +/-0.8m (x) / +/-0.5m (y), matching
    # IsaacLab GridPatternCfg(resolution=0.1, size=(1.6, 1.0)).
    height_scan: HeightScanConfig = field(
        default_factory=lambda: HeightScanConfig(
            enabled=True,
            geom_name="scan_hfield_geom",
            measured_points_x=list(DEFAULT_SCAN_POINTS_X),
            measured_points_y=list(DEFAULT_SCAN_POINTS_Y),
        )
    )
    scan_frame_body: str = "torso_link"


@registry.envcfg("G1SteppingStoneVision")
@dataclass
class G1SteppingStoneVisionEnvCfg(G1SteppingStoneVisionCfg):
    """Registered configuration for G1 vision stepping-stone tracking."""

    pass


def _validate_paired_loaders(optimized: MotionLoader, raw: MotionLoader) -> None:
    """Fail-closed: the raw loader must be frame-aligned with the optimized one.

    Both loaders are constructed with the same ``body_indices`` in the env, so the
    body axis is already aligned; this validates the frame/clip/fps contract.
    """
    if optimized.num_clips != raw.num_clips:
        raise ValueError(
            f"raw/optimized clip count mismatch: {raw.num_clips} vs {optimized.num_clips}"
        )
    if optimized.fps != raw.fps:
        raise ValueError(f"raw/optimized fps mismatch: {raw.fps} vs {optimized.fps}")
    if optimized.num_joints != raw.num_joints or optimized.num_bodies != raw.num_bodies:
        raise ValueError("raw/optimized joint/body dimension mismatch")
    if not np.array_equal(optimized.clip_lengths, raw.clip_lengths):
        raise ValueError("raw/optimized clip-length arrays differ (frames not aligned)")
    if not np.array_equal(optimized.clip_offsets, raw.clip_offsets):
        raise ValueError("raw/optimized clip-offset arrays differ (frames not aligned)")
    opt_names = [os.path.basename(p).replace("_optimized", "") for p in optimized.motion_files]
    raw_names = [os.path.basename(p).replace("_raw", "") for p in raw.motion_files]
    if opt_names != raw_names:
        raise ValueError("raw/optimized clips are not the same sorted basenames")


@registry.env("G1SteppingStoneVision", sim_backend="mujoco")
@registry.env("G1SteppingStoneVision", sim_backend="motrix")
class G1SteppingStoneVisionEnv(G1MotionTrackingEnv):
    """Raw-reference + height-scan FlatMLP vision env."""

    _cfg: G1SteppingStoneVisionCfg
    _height_scan_dim: int = 0

    def __init__(self, cfg: G1SteppingStoneVisionCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

        # Secondary RAW loader, indexed by the primary (optimized) sampler frames.
        motion_body_ids = self._backend.get_motion_body_ids(cfg.body_names)
        self.raw_motion_loader = MotionLoader(cfg.raw_motion_file, body_indices=motion_body_ids)
        _validate_paired_loaders(self.motion_loader, self.raw_motion_loader)
        self._raw_motion_buffer = self.raw_motion_loader.make_motion_data_buffer(num_envs)

        # Height-scan sensor on torso_link, sampling the contact-disabled hfield.
        init_height_scan_sensor(self, cfg.height_scan, cfg.scan_frame_body)
        self._scan_dim = int(self._height_scan_dim)

        # Reusable scratch for the raw actor anchor transform.
        dtype = get_global_dtype()
        self._raw_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
        self._raw_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)

    # ------------------------------------------------------------------ #
    # Observation overrides                                              #
    # ------------------------------------------------------------------ #
    def _actor_obs_dim(self, n: int) -> int:
        # Deploy-realistic actor: drop base_lin_vel (matches IsaacLab's
        # _remove_deploy_incompatible_policy_terms). Layout becomes
        # command(2n) + anchor_pos(3) + anchor_ori(6) + gyro(3) + 3n proprio.
        return 3 + 6 + 3 + n * 5

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        # Same as the base builder but WITHOUT base_lin_vel (deploy-incompatible).
        num_envs = command.shape[0]
        n_action = noisy_joint_pos_rel.shape[1]
        actor_obs = np.empty((num_envs, self._actor_obs_dim(n_action)), dtype=get_global_dtype())
        offset = 0
        actor_obs[:, offset : offset + command.shape[1]] = command
        offset += command.shape[1]
        actor_obs[:, offset : offset + 3] = motion_anchor_pos_b
        offset += 3
        actor_obs[:, offset : offset + 6] = motion_anchor_ori_b
        offset += 6
        actor_obs[:, offset : offset + 3] = noisy_gyro
        offset += 3
        actor_obs[:, offset : offset + n_action] = noisy_joint_pos_rel
        offset += n_action
        actor_obs[:, offset : offset + n_action] = noisy_dof_vel
        offset += n_action
        actor_obs[:, offset : offset + n_action] = last_actions
        return actor_obs

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base = super().obs_groups_spec
        # Actor gains the height scan; critic is unchanged (optimized, privileged).
        return {"obs": base["obs"] + self._scan_dim, "critic": base["critic"]}

    def _height_scan_obs(self, num_envs: int) -> np.ndarray:
        """Raw relative height (sensor_z - terrain_z), IsaacLab-style (no clip/scale)."""
        raw_heights, base_pos = raw_height_scan_obs(self, num_envs)
        if raw_heights is None or base_pos is None:
            return np.zeros((num_envs, self._scan_dim), dtype=get_global_dtype())
        scan = base_pos[:, 2:3] - raw_heights
        return np.asarray(scan, dtype=get_global_dtype())

    def _raw_motion_at(self, env_ids: np.ndarray | None, num_envs: int):
        """Raw-clip motion at the primary sampler's current frames."""
        frames = self.motion_sampler.current_frames
        if env_ids is not None:
            frames = frames[env_ids]
        if num_envs == self._num_envs and env_ids is None:
            return self.raw_motion_loader.get_motion_at_frame(frames, out=self._raw_motion_buffer)
        return self.raw_motion_loader.get_motion_at_frame(frames)

    def _compute_obs(
        self,
        info: dict,
        motion_data,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Build the base (optimized) obs; critic stays optimized/privileged.
        obs = super()._compute_obs(
            info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )
        num_envs = linvel.shape[0]
        n_action = dof_pos.shape[1]
        dtype = get_global_dtype()

        # Overwrite the actor's reference columns with the RAW motion.
        env_ids = info.get("env_ids")
        raw_motion = self._raw_motion_at(env_ids, num_envs)

        raw_anchor_pos_w = raw_motion.body_pos_w[:, self.anchor_body_idx]
        raw_anchor_quat_w = raw_motion.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        if num_envs == self._num_envs and env_ids is None:
            raw_anchor_pos_b = self._raw_anchor_pos_b
            raw_anchor_ori_b = self._raw_anchor_ori_b
        else:
            raw_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
            raw_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
        _write_motion_anchor_transform(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            raw_anchor_pos_w,
            raw_anchor_quat_w,
            raw_anchor_pos_b,
            raw_anchor_ori_b,
        )

        actor = obs["obs"]
        # Actor layout (see _build_actor_obs): command(2n) + anchor_pos(3) + anchor_ori(6) + ...
        actor[:, :n_action] = raw_motion.joint_pos
        actor[:, n_action : 2 * n_action] = raw_motion.joint_vel
        offset = 2 * n_action
        actor[:, offset : offset + 3] = raw_anchor_pos_b
        offset += 3
        actor[:, offset : offset + 6] = raw_anchor_ori_b

        # Append the terrain height scan.
        scan = self._height_scan_obs(num_envs)
        obs["obs"] = np.concatenate([actor, scan], axis=1)
        return obs
