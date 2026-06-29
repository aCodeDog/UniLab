# Plan: G1 Vision Stepping-Stone (FlatMLP) — IsaacLab ablation reproduction

Status: AGREED (Claude planner A + Codex reviewer B + reconciliation; verified independently)
Target task: `G1SteppingStoneVision`
Reference: IsaacLab `PPOFinetune-Ablation-RGMT-VisionTeacher-SteppingStone-G1-LatentAnchor-FlatMLP-v0`
(`RGMTG1SteppingStoneVisionTeacherLatentAnchorAblationEnvCfg`, FlatMLP runner).

## 0. Concept
Extend the existing `G1SteppingStone` task: the **policy/actor** is driven by the **RAW**
(un-optimized) motion command + a **terrain height scan**; the **rewards + critic + reset**
use the **OPTIMIZED** motion. FlatMLP actor (no transformer / latent-anchor / distillation).
This forces the policy to use terrain perception to bridge raw->optimized reference.

## 1. AGREED DECISIONS
1. **Terrain/hfield (contact-disabled scan-only):** physics keeps the BOX terrain (accurate
   contacts). Add a SEPARATE `<hfield>` asset + `<geom type="hfield" contype="0" conaffinity="0">`
   used ONLY by the scanner. Verified: `sample_hfield_height` bilinearly interpolates hfield
   GEOMETRY and does NOT require collision (`backend.py:53,66`); scanner targets an explicit
   geom id (`height_scan.py:100`), not "the floor". The existing procedural materialize path
   (`xml.py:488`) makes the hfield a contact geom — we DON'T use it; we inject our own
   contact-disabled hfield geom into the scene XML.
2. **Resolution = 0.05 m, FULL map (78x78m -> 1560x1560, ~4 MB int16).** Justification: smallest
   terrain feature is 0.3 m; at 0.1 m that's only 3 cells, at 0.05 m it's 6 cells. Full map avoids
   crop/boundary origin risk. (Codex B preferred; A preferred 0.075 — reconciled to 0.05 for
   fidelity headroom; the resolution-fidelity TEST gates this.)
3. **Dual MotionLoader, single sampler:** keep primary optimized `MotionLoader` + `MotionSampler`
   (reset/reward/critic). Add a SECOND raw `MotionLoader`, indexed by the SAME
   `motion_sampler.current_frames` (stateless `get_motion_at_frame`, `motion_loader.py:177`).
   No second sampler. Cross-loader fail-closed validation required (see 5).
4. **Single-frame obs for v1** (documented as NOT IsaacLab-equivalent; ref uses history_length=10).
   Add actor history as a follow-up if benchmark parity is needed.
5. **Scope:** FlatMLP only. Drop transformer, latent-anchor machinery, distillation.

## 2. Files to CREATE
- `scripts/motion/build_stepping_stone_hfield.py` — rasterize the deploy big-map box geoms into
  a 1560x1560 16-bit PNG heightmap (top-surface z per xy cell), write the PNG + print
  `(size_x, size_y, z_max, z_min)` for the `<hfield size=...>` attr. Validates by loading in mujoco.
- `src/unilab/assets/robots/g1/hfields/stepping_stone_terrain.png` — generated heightmap.
- `src/unilab/assets/robots/g1/scene_stepping_stone_vision.xml` — copy of scene_stepping_stone.xml
  + `<asset><hfield name="scan_hfield" file="hfields/stepping_stone_terrain.png" size="..."/></asset>`
  + a worldbody `<geom name="scan_hfield_geom" type="hfield" hfield="scan_hfield"
  contype="0" conaffinity="0" group="3" pos="cx cy 0"/>` (pos = map center so hfield frame aligns
  with world). Box terrain + floor unchanged.
- `src/unilab/envs/motion_tracking/g1/stepping_stone_vision.py`:
  - `G1SteppingStoneVisionCfg(G1SteppingStoneCfg)`: scene -> vision XML; add `raw_motion_file`
    field (list of raw npz, converted to UniLab layout); `height_scan: HeightScanConfig` with
    `geom_name="scan_hfield_geom"`, `measured_points_x/y` = IsaacLab grid (x: 16 pts over 1.6 m,
    y: 11 pts over 1.0 m, res 0.1 -> 176 rays); `append_validity_mask=True` (new cfg flag).
  - `@registry.envcfg("G1SteppingStoneVision")` `G1SteppingStoneVisionEnvCfg`.
  - `@registry.env("G1SteppingStoneVision", "mujoco"/"motrix")`
    `G1SteppingStoneVisionEnv(G1MotionTrackingEnv)`: override `__init__` (build raw loader +
    `init_height_scan_sensor`), `obs_groups_spec`, and the obs builder to: actor = raw command +
    proprio + height scan(+mask); critic = optimized (privileged). Reset/rewards inherit (optimized).
- `conf/ppo/task/g1_stepping_stone_vision/mujoco.yaml` — FlatMLP: actor_hidden_dims
  [1024,512,256,128], critic_hidden_dims [512,256,128]; obs_groups actor:[actor], critic:[critic];
  reward block identical to g1_stepping_stone.
- `tests/envs/test_stepping_stone_vision.py` — see Test Plan.

## 3. Files to EDIT
- `src/unilab/envs/motion_tracking/g1/__init__.py` + `motion_tracking/__init__.py`: export
  `G1SteppingStoneVisionCfg/Env/EnvCfg` (runs `@registry` on import).
- Reuse `scripts/motion/wbt_to_unilab_npz.py` to convert the RAW clips (BFS->DFS + world row) into
  `.../walk_dance1sub2start/raw_unilab/`. (Optimized already converted in prior task.)
- DO NOT change `height_scan_obs` signature (used by go2w/go1). Build the validity mask in the new
  env using `raw_height_scan_obs` (returns raw heights) + finite-check; keep shared helpers intact.

## 4. Observation layout (G1 = 29 joints, n=29)
Actor (`obs`): raw_command (2n=58) + motion_anchor_pos_b(3) + motion_anchor_ori6(6) + gyro(3) +
joint_pos(29) + joint_vel(29) + last_action(29) + height_scan(176) + validity_mask(176).
NOTE: raw command/anchor computed against the RAW loader's motion at current frame.
(We keep the same base proprio terms as `_build_actor_obs`; the deploy-incompatible privileged
terms — body pos/ori, base_lin_vel — stay OUT of the actor, matching
`_remove_deploy_incompatible_policy_terms`.)
Critic: the existing critic obs (optimized command + anchor + body pos/ori 9*nbody + proprio).
Exact dims asserted in tests against `obs_groups_spec` (single source of truth, `tracking.py:689`).

## 5. Dual-loader + cross-validation
- In `__init__`: `self.raw_motion_loader = MotionLoader(raw_files, body_indices=motion_body_ids)`.
- Validate (fail-closed) raw vs optimized: identical `sorted(basenames)`, `num_clips`,
  `clip_lengths`, `clip_offsets`, `fps`, `num_joints`, `num_bodies`, and same `body_indices`.
- Per step/reset, fetch raw pose via
  `self.raw_motion_loader.get_motion_at_frame(self.motion_sampler.current_frames)`.
- Reset state still built from OPTIMIZED motion (unchanged), so the robot starts on the optimized
  reference; the actor just SEES the raw reference.

## 6. TEST PLAN
1. **hfield resolution-fidelity (USER-REQUIRED):**
   - Build ground-truth: analytic top-surface z of the box terrain at query (x,y) (max over boxes
     covering the point, else floor 0).
   - INTERIOR case: 800 pts strictly inside box footprints (and on floor gaps) -> sample the
     hfield (bilinear, via the backend scanner OR the PNG sampler) -> assert
     `max_abs_error < 0.05 m` (== one cell), `mean_abs_error < 0.01 m`.
   - EDGE-BAND case: 200 pts within one cell of a box boundary -> assert
     `<= ~half max step (0.2 m)` and DOCUMENT that discontinuity aliasing is expected (report,
     don't hard-fail tight). Guards against gross coordinate/scale errors only.
   - Y-ROW convention: explicitly check a known asymmetric feature so row ordering
     (`terrain_generator.py:157`) matches world +y. (resolution measurement is the core ask.)
2. **Contact isolation:** load vision scene; assert the `scan_hfield_geom` has
   contype==0 & conaffinity==0; drop-test: robot resting z on box vs on vision scene identical
   (hfield must not alter contacts).
3. **Cross-loader sync:** construct env; over 100 random frame ids assert raw & optimized return
   the SAME clip index / frame and the validation passes; mismatched dir raises.
4. **Obs dims:** `sum(obs_groups_spec.values()) == observation_space`; actor includes 176+176
   scan dims; finite after reset + 5 steps.
5. **Env smoke (slow):** build num_envs=2, reset, 5 steps zero-action -> finite reward, base xy ==
   optimized pelvis xy (reset uses optimized); height scan non-NaN, varies with terrain.
6. **Sanity train:** `task=g1_stepping_stone_vision/mujoco algo.num_envs=64 max_iterations=20
   training.no_play=true` -> reward finite & rising.

## 7. Ranked risks (from review)
1. Scan hfield accidentally participates in contacts -> corrupts physics. MITIGATION: contype/
   conaffinity 0 + Test #2.
2. Raw/optimized clip desync -> wrong supervision. MITIGATION: cross-loader validation + Test #3.
3. Hfield coord/scale/y-row mismatch -> wrong scan. MITIGATION: Test #1 fidelity + y-row check.
4. Obs-dim drift vs `obs_groups_spec` -> learner shape error. MITIGATION: Test #4.
5. Fidelity at 0.05 m insufficient for thin steps. MITIGATION: Test #1 gates; bump to 0.025 if it
   fails (8 MB still cheap).

## 8. Open follow-ups (not v1)
- Actor observation history (length 10) for IsaacLab parity.
- Per-joint action scale / per-actuator armature parity.

---

## 9. Misunderstandings, problems & corrections (post-review verification)

These were found by checking the plan's assumptions against the actual code/data. Apply
these corrections during implementation.

### CORRECTED — factual errors in earlier sections
- **C1. Height-scan ray count is 187, not 176.** IsaacLab `GridPatternCfg(resolution=0.1,
  size=(1.6,1.0))` yields `arange(-0.8,0.8,0.1)=17` x-pts and `arange(-0.5,0.5,0.1)=11` y-pts
  => **17x11 = 187 rays** (verified). UniLab's `DEFAULT_SCAN_POINTS_X/Y` are ALSO 17x11=187 and
  span exactly ±0.8 / ±0.5 — i.e. UniLab's default grid already matches IsaacLab. So obs scan
  dims are **187 heights (+187 validity = 374)**, not 176/352. Fix all obs-dim arithmetic in §4.
- **C2. Actor obs dim in §4 is illustrative, not authoritative.** Do NOT hardcode 356/418.
  `obs_groups_spec` (`tracking.py:689`) is the single source of truth; tests assert against it.
- **C3. G1 has 29 joints** (already fixed in §4; the reconciler's 37 was wrong).

### PROBLEM — design points that need an explicit decision/extra work
- **P1. Scan frame body: torso_link vs pelvis.** IsaacLab mounts the height_scanner on
  `torso_link` (`stepping_stone_env_cfg.py:241`). UniLab's `init_height_scan_sensor` takes a
  `base_body_name` and go2w passes `cfg.asset.base_name` (= **pelvis** for G1, `g1/base.py`).
  For parity we must pass **torso_link** explicitly (the tracking anchor body), NOT the default
  pelvis. Action: call `init_height_scan_sensor(self, scan_cfg, "torso_link")`.
- **P2. Height-scan output convention differs.** UniLab `height_scan_obs` returns
  `clip(base_z - vertical_offset - terrain_z, -1, 1) * scale` (scale=5) using the scanner's
  `output="height"` (world terrain z). IsaacLab `height_scan_for_vision` returns
  `sensor_height - hit_z - offset` (raw relative height, NO ±1 clip, scale=1), optionally
  +noise/clip, +validity. These are NOT the same normalization. DECISION: for v1, build the
  scan obs in the new env directly from `raw_height_scan_obs` (raw terrain z + base_pos) to
  reproduce IsaacLab's `sensor_height - terrain_z` WITHOUT the go2 ±1 clip/×5 scale, so the
  semantics match the reference. Do not reuse `height_scan_obs` verbatim.
- **P3. Validity mask.** The hfield scanner clamps out-of-domain samples to the border and
  returns finite values — it does NOT produce NaNs like a real raycaster missing a hit. So a
  "validity mask" from finiteness would be all-ones and useless. DECISION: either (a) drop the
  validity-mask channel for v1 (simplest, since UniLab hfield scan is always valid), or (b)
  synthesize validity = inside-hfield-domain test. Recommend **(a) drop mask for v1** and note
  the obs schema differs from the deploy transformer (acceptable for a FlatMLP teacher). This
  changes scan dims to **187 (heights only)**.
- **P4. hfield `size` attr & world placement.** MuJoCo `<hfield size="rx ry zt zb">` (radius_x,
  radius_y, top_z, bottom_z) and the geom is centered at its `pos`. Our terrain spans
  x[-39,39], y[-39,39] centered at origin already, so `pos="0 0 0"`, `rx=ry=39`,
  `zt=0.40` (max top), `zb=0.05` (base thickness). The PNG must be oriented so row/col map to
  world (y,x) per MuJoCo hfield convention — verified against `terrain_generator.py:157`
  (`rows = (-y + size_y/2)/hscale`, `cols = (x + size_x/2)/hscale`). The builder MUST use the
  same convention or the scan is mirrored.
- **P5. Reset-vs-actor reference mismatch is intentional but must be validated.** Robot resets
  on the OPTIMIZED pose; actor SEES the RAW pose. At frame 0 raw≈optimized (small offset), but
  mid-clip they diverge. This is the whole point (policy uses vision to correct), but the
  fidelity of the raw clip's world-frame alignment to the terrain must hold — verify raw clips
  are in the SAME world frame as optimized (they are: both carry absolute world pose, fps=50,
  equal lengths — verified 20/20 pairs, 0 mismatches).

### VERIFIED OK (no action)
- Raw/optimized clips are paired 1:1 by basename, identical frame counts (20/20 checked),
  fps=50, same 30-body IsaacLab layout. Synced indexing is safe after BFS->DFS conversion.
- `sample_hfield_height` interpolates geometry only (no collision dependency) — contact-disabled
  scan hfield is valid.

### NET EFFECT ON THE PLAN
- Scan obs = **187 height values** (drop validity mask for v1) mounted on **torso_link**, using
  **raw-relative height (sensor_z − terrain_z)** semantics — built in-env from
  `raw_height_scan_obs`, not the go2 `height_scan_obs`.
- All concrete obs dims come from `obs_groups_spec`; tests assert, code does not hardcode.
