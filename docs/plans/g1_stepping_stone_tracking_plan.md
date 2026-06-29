# Plan: Reproduce IsaacLab `RGMT-SteppingStone-G1-v0` (MLP) in UniLab

Status: DRAFT (pending Claude + Codex review)
Target task name: `G1SteppingStone`
Algorithm: PPO with MLP actor-critic `[512, 512, 256, 128]`

## 0. Goal & Scope
Reproduce the IsaacLab G1 multi-motion, terrain-anchored motion-tracking task as an
**MLP PPO** task inside UniLab. We reproduce the *base MLP teacher*
(`G1SteppingStoneMultiMotionEnvCfg`), NOT the RGMT transformer / vision variant.

Key idea: reuse the existing `G1MotionTracking` env wholesale; the ONLY substantive
differences are (a) the motion set, (b) the terrain scene, and (c) `env_spacing` /
world-frame handling. Everything else (rewards, events, obs, PD gains, body list) is
reused as-is and tuned via the task YAML.

## 1. Verified Facts (do not re-litigate)
- Motion npz keys EXACTLY match `MotionLoader`: `fps, joint_pos[N,29], joint_vel[N,29],
  body_pos_w[N,30,3], body_quat_w[N,30,4], body_lin_vel_w, body_ang_vel_w`.
- Chosen data dir `.../walk_dance1sub2start/optimized_smoothed_w9_s2`: 500 clips,
  **fps = 50.0**, smoothed. 50fps matches UniLab control (50Hz, ctrl_dt=0.02), and the
  sampler advances exactly one motion frame per env step (`motion_loader.py:447`) — same
  as IsaacLab `time_steps += 1` (`commands.py:271`). NO timebase problem.
- `MotionLoader` accepts `list[str]` of npz, concatenates clips, tracks clip offsets
  (`motion_loader.py:120-134`). Multi-motion = supported.
- **env_spacing / world position**: UniLab reset sets `qpos[:,0:3] = motion body_pos_w[:,0]`
  DIRECTLY with NO env-origin offset (`tracking.py:211, 231, 265`). UniLab MuJoCo backend
  runs ONE shared `MjModel`, `nbatch=num_envs`, with NO per-env translation
  (`backend.py:301, 569`). This is exactly IsaacLab's `env_spacing=0.0` case, where
  IsaacLab adds `env_origins`(=0) to motion body pos (`commands.py:134, 150`). => Motions
  already carry absolute world position; we use them as-is. Match confirmed.
- UniLab `g1.xml` and IsaacLab `g1_29dof.xml`: identical 29-joint / 30-body order.
- PD gains already match (UniLab `g1.xml` kp/kv == IsaacLab `g1.py` stiffness/damping
  within rounding). See verification table in section 6.
- `G1MotionTrackingCfg` already uses the same `body_names`, `anchor_body_name="torso_link"`,
  `ee_body_names`, thresholds as IsaacLab's stepping-stone TRACKED/END_EFFECTOR lists.

## 2. Corrected UniLab API facts (from review of draft)
- `@registry.envcfg("Name")` decorates a `@dataclass` CLASS (see `tracking.py:189`).
- Motion field: `motion_file: str | list[str]` (NOT `motion_paths`).
- Reward config: env-cfg field `reward_config: RewardConfig` (`tracking.py:170`); the task
  YAML `reward:` block maps to it (`scales{}` + `std_*`). See existing
  `conf/ppo/task/g1_motion_tracking/mujoco.yaml`.
- obs_groups YAML convention (proven working): `algo.obs_groups.actor: [actor]`.
- Termination fields: `anchor_pos_z_threshold`, `anchor_ori_threshold`,
  `ee_body_pos_z_threshold` (`tracking.py:176-178`).
- Asset paths use `ASSETS_ROOT_PATH / "robots" / "g1" / "..."` (slash-joined parts).

## 3. New Files

### 3.1 `src/unilab/assets/robots/g1/scene_stepping_stone.xml`
- `<include file="g1.xml"/>` (same dir; backend writes temp XML beside source so relative
  includes resolve — `xml.py:169`).
- Worldbody: keep a `floor` plane + port the terrain `box` geoms from
  `/home/zifan_wang/Mondo/whole_body_tracking/scripts/mujoco_deploy/g1_robot/g1_29dof_big_map_urdf.xml`
  (all `C***_*` box geoms with their pos/quat/size). Use a generator script (section 5)
  to copy the `<geom ... type="box" .../>` lines verbatim into a `<body name="terrain">`.
- Keep the existing G1 keyframe block from scene_flat.xml so backend base pose init works.
- Rationale for boxes over the 1.5MB STL: identical collision geometry to IsaacLab's
  baked deploy mesh, faster load, deterministic.

### 3.2 `src/unilab/envs/motion_tracking/g1/stepping_stone.py`
```python
from __future__ import annotations
import glob, os
from dataclasses import dataclass, field
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from .tracking import G1MotionTrackingCfg, PoseRandomization, VelocityRandomization

_MOTION_DIR = (
    "/data/mondo-training-dataset/whole_body_tracking_motions/motions/terrain/"
    "terrain_mocaphouse/walk_dance1sub2start/optimized_smoothed_w9_s2"
)

def _stepping_stone_motions() -> list[str]:
    files = sorted(glob.glob(os.path.join(_MOTION_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No stepping-stone motions in {_MOTION_DIR}")
    return files

@registry.envcfg("G1SteppingStone")
@dataclass
class G1SteppingStoneEnvCfg(G1MotionTrackingCfg):
    """G1 multi-motion terrain-anchored tracking (MLP teacher reproduction)."""
    # IsaacLab dt=0.005, decimation=4 => sim 200Hz / ctrl 50Hz.
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone.xml")))
    motion_file: list[str] = field(default_factory=_stepping_stone_motions)
    sampling_mode: str = "adaptive"
    # Terrain-anchored motions carry absolute world pose: do NOT perturb root xy.
    # IsaacLab stepping-stone uses pose_range={} / velocity_range={} (zeroed).
    pose_randomization: PoseRandomization = field(default_factory=lambda: PoseRandomization(
        x=(0.0, 0.0), y=(0.0, 0.0), z=(0.0, 0.0),
        roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)))
    velocity_randomization: VelocityRandomization = field(default_factory=lambda: VelocityRandomization(
        x=(0.0, 0.0), y=(0.0, 0.0), z=(0.0, 0.0),
        roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)))
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    # IsaacLab stepping-stone thresholds.
    anchor_pos_z_threshold: float = 0.35
    ee_body_pos_z_threshold: float = 0.35
```
REGISTRY REQUIREMENT (verified): a task needs BOTH `@registry.envcfg("Name")` (cfg) AND
`@registry.env("Name", sim_backend=...)` (env class). See `tracking.py:446-448`,
`registry.py:70,108`. Registering only the cfg is NOT enough. So we MUST add a thin env
subclass bound to the same name:
```python
from .tracking import G1MotionTrackingEnv

@registry.env("G1SteppingStone", sim_backend="mujoco")
@registry.env("G1SteppingStone", sim_backend="motrix")
class G1SteppingStoneEnv(G1MotionTrackingEnv):
    """Terrain-anchored multi-motion tracking; reuses all G1MotionTrackingEnv logic."""
    _cfg: G1SteppingStoneEnvCfg
```
No method overrides needed for v1 — all reward/obs/reset logic is inherited.

### 3.3 `conf/ppo/task/g1_stepping_stone/mujoco.yaml`
```yaml
# @package _global_
training:
  task_name: G1SteppingStone
  sim_backend: mujoco
  play_steps: 1000
algo:
  num_envs: 4096
  max_iterations: 15000
  save_interval: 500
  obs_groups:
    actor:
      - actor
  policy:
    actor_hidden_dims: [512, 512, 256, 128]
    critic_hidden_dims: [512, 512, 256, 128]
    activation: elu
    class_name: ActorCritic
  algorithm:
    entropy_coef: 0.005
reward:
  scales:
    motion_global_root_pos: 0.5
    motion_global_root_ori: 0.5
    motion_body_pos: 1.0
    motion_body_ori: 1.0
    motion_body_lin_vel: 1.0
    motion_body_ang_vel: 1.0
    motion_joint_pos: 0.0
    motion_joint_vel: 0.0
    action_rate_l2: -0.1
    joint_limit: -10.0
  std_root_pos: 0.3
  std_root_ori: 0.4
  std_body_pos: 0.3
  std_body_ori: 0.4
  std_body_lin_vel: 1.0
  std_body_ang_vel: 3.14
  std_joint_pos: 0.2
  std_joint_vel: 1.0
```
- Mirror the proven `g1_motion_tracking/mujoco.yaml`; only override hidden dims + task name.
- Optionally add `conf/appo/task/g1_stepping_stone/motrix.yaml` later.

## 4. Edits to Existing Files
- `src/unilab/envs/motion_tracking/g1/__init__.py`: import + `__all__` add BOTH
  `G1SteppingStoneEnvCfg` AND `G1SteppingStoneEnv` (importing the module also runs the
  `@registry.env` / `@registry.envcfg` decorators, so the imports must execute).
- `src/unilab/envs/motion_tracking/__init__.py`: re-export same two symbols. The parent
  `__unilab_registry_modules__` already includes `unilab.envs.motion_tracking.g1`, so the
  new classes register on import.

## 5. Terrain port helper (one-shot)
Add `scripts/motion/build_stepping_stone_scene.py` (or do inline once): read the deploy
XML, extract every `<geom ... type="box" .../>` under worldbody, emit
`scene_stepping_stone.xml` with `<include file="g1.xml"/>`, floor plane, keyframe, and a
`<body name="terrain" pos="0 0 0">` containing the boxes. Verify it loads with
`mujoco.MjModel.from_xml_path`.

## 6. Known-gap handling (concrete)
| Gap | Decision |
|-----|----------|
| Control rate | Override `sim_dt=0.005, ctrl_dt=0.02` in cfg (matches IsaacLab 200/50Hz). |
| Motion fps | 50fps data chosen; 1 frame/step. No resample needed. |
| Action scale | Keep UniLab scalar 0.25 (MLP teacher). Per-joint G1_ACTION_SCALE optional later. |
| Armature | UniLab uniform 0.01 vs IsaacLab per-actuator. Accept for v1; note as fidelity risk. |
| Obs history | UniLab single-frame vs IsaacLab history_length=10. Accept for MLP v1. |
| Foot/lateral rewards | Drop for v1 (base tracking rewards train fine); implement later if needed. |
| Pose/vel randomization | ZERO out (terrain-anchored). Critical for world-frame correctness. |
| Thresholds | anchor/ee z = 0.35 (IsaacLab stepping-stone). |

PD gain verification (IsaacLab g1.py -> UniLab g1.xml):
- hip_pitch/yaw, waist_yaw: 40.305/2.53 vs 40.179/2.558 OK
- hip_roll, knee: 99.341/6.30 vs 99.098/6.309 OK
- ankle, waist_r/p: 28.527/1.813 vs 28.501/1.814 OK
- shoulder/elbow/wrist_roll: 14.264/0.907 vs 14.251/0.907 OK
- wrist_pitch/yaw: 16.850/1.068 vs 16.778/1.068 OK

## 7. Test Plan
### 7.1 Data/unit (fast, no sim)
- `tests/envs/motion_tracking/test_stepping_stone_motions.py`:
  - glob finds 500 npz; load first/last -> assert `fps==50`, joints==29, bodies==30.
  - `MotionLoader(files)` -> `num_clips==len(files)`, `clip_offsets` strictly increasing,
    `num_frames == sum(clip_lengths)`.
### 7.2 Config/build smoke
- Extend `tests/envs/test_env_configs.py` pattern: instantiate `G1SteppingStoneEnvCfg`,
  assert it exposes every attr the env reads; build env `num_envs=2`:
  - `obs_groups_spec["obs"] > 0`, action dim == 29.
  - `reset()` then 5x `step(zero_action)` -> all finite, no exception.
  - assert reset root xy equals motion body_pos_w[:,0] xy (NO offset) for a known clip.
### 7.3 Scene/terrain
- Load `scene_stepping_stone.xml` via mujoco; assert nbody includes terrain boxes and
  robot pelvis spawns above local terrain height (no deep penetration at frame 0).
### 7.4 Training sanity (short)
- `python scripts/train_ppo.py task=g1_stepping_stone/mujoco algo.num_envs=64 algo.max_iterations=20`
  -> runs, reward finite, mean episode length > 1, checkpoint written.

## 8. Risks / Open Questions
1. Registry binding: confirm cfg-name `G1SteppingStone` resolves to `G1MotionTrackingEnv`
   without a dedicated env subclass. (Mitigation: add thin subclass.)
2. Terrain origin: motions reference the mocaphouse/big_map terrain; the box terrain we
   port MUST be the SAME terrain the motions were optimized against, else feet float/clip.
   Verify the deploy XML terrain == terrain used to generate optimized_smoothed_w9_s2.
3. Self-collision: IsaacLab enables self-collisions; verify UniLab g1.xml contact settings
   don't cause spurious resets on this terrain.
4. 500 clips * 1500 frames in memory: check RAM (MotionLoader concatenates all). ~ a few
   hundred MB float32; acceptable, but consider a subset for sanity runs.
5. Armature/action-scale fidelity gaps may cause sim2sim drift vs IsaacLab policy.

## 9. Review Outcomes (Claude + Codex + direct verification)

Reviewed by: Claude agent (plan audit) + Codex agent (feasibility) + direct code checks.

### Verdict: GO-WITH-FIXES — all fixes already folded into sections above.

### Confirmed CORRECT
- Multi-motion list[str] support is end-to-end (loader + sampler + reset). [codex]
- Shared-world / env_spacing=0 equivalence; reset uses motion `body_pos_w[:,0]` with NO
  offset (`tracking.py:211,231,265`); backend single MjModel, `nbatch=num_envs`
  (`backend.py:301,569`). [codex + direct]
- YAML `reward:` block + `algo.obs_groups.actor:[actor]` convention matches existing
  working task yaml. [claude]
- Asset path style `ASSETS_ROOT_PATH / "robots" / "g1" / ...`. [claude]
- Chosen motions are fps=50 (verified by loading npz), 500 clips. No timebase issue;
  `sim_substeps = round(ctrl_dt/sim_dt) = round(0.02/0.005) = 4` == IsaacLab decimation=4
  (`base.py:54`). [direct]
- PD gains match within rounding. [direct]

### Fixed errors found in the draft
1. [claude] Registering only the cfg is NOT enough — need BOTH `@registry.envcfg` and
   `@registry.env(name, sim_backend=...)`. FIX: added thin `G1SteppingStoneEnv` subclass
   (section 3.2). VERIFIED against `registry.py:70,108` + `tracking.py:446`.
2. [claude] Earlier draft used `motion_paths`; corrected to `motion_file: str|list[str]`.

### Reviewer claim that was itself WRONG (rejected after verification)
- Claude reviewer claimed "G1BaseCfg has no `ctrl_dt` field, only `sim_dt`". FALSE:
  `ctrl_dt` exists at `locomotion/g1/base.py:51` and `locomotion/common/base.py:52`.
  The `sim_dt`/`ctrl_dt` override in section 3.2 is VALID. (Kept as-is.)

### Still-open items to check during implementation (not blockers)
- O1 (HIGH): Terrain-identity — the ported box terrain MUST be the exact terrain the
  `optimized_smoothed_w9_s2` clips were optimized against, else feet float/penetrate.
  Action: confirm `g1_29dof_big_map_urdf.xml` terrain == mocaphouse terrain for these clips.
- O2 (MED): RAM for 500 clips concatenated (~ a few hundred MB f32). Use a subset for the
  fast sanity run (`motion_file` override) if needed.
- O3 (MED): self-collision settings in UniLab g1.xml vs IsaacLab (spurious resets).
- O4 (LOW): per-joint action scale + per-actuator armature for higher sim2sim fidelity.
