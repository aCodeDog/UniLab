"""Tests for the G1 stepping-stone terrain-anchored tracking task.

Fast tests cover config wiring, the generated terrain scene, and the WBT->UniLab
motion converter permutation logic.  The ``slow`` test builds the env via the
registry and runs reset + step (requires the MuJoCo runtime and motion assets).
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.base.registry import ensure_registries


def test_stepping_stone_cfg_registers_and_overrides_defaults():
    ensure_registries()
    from unilab.base import registry
    from unilab.envs.motion_tracking.g1.stepping_stone import G1SteppingStoneEnvCfg

    assert registry.contains("G1SteppingStone")

    cfg = G1SteppingStoneEnvCfg()
    # IsaacLab dt=0.005 / decimation=4 -> 4 sim substeps per control step.
    assert cfg.sim_dt == pytest.approx(0.005)
    assert cfg.ctrl_dt == pytest.approx(0.02)
    assert cfg.sim_substeps == 4
    # Terrain-anchored clips carry absolute world pose: reset must not perturb it.
    assert cfg.pose_randomization.x == (0.0, 0.0)
    assert cfg.pose_randomization.yaw == (0.0, 0.0)
    assert cfg.velocity_randomization.x == (0.0, 0.0)
    assert cfg.joint_position_range == (0.0, 0.0)
    # Stepping-stone termination thresholds.
    assert cfg.anchor_pos_z_threshold == pytest.approx(0.35)
    assert cfg.ee_body_pos_z_threshold == pytest.approx(0.35)
    # Scene points at the generated stepping-stone XML.
    assert cfg.scene.model_file.endswith("scene_stepping_stone.xml")


def test_stepping_stone_scene_loads_in_mujoco():
    pytest.importorskip("mujoco")
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH

    scene = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone.xml"
    if not scene.exists():
        pytest.skip("scene_stepping_stone.xml not generated")
    model = mujoco.MjModel.from_xml_path(str(scene))
    # 7 root DoF + 29 joints.
    assert model.nq == 36
    # Robot bodies (world + 30) plus a terrain body with many box geoms.
    assert model.nbody >= 32
    assert model.ngeom > 1000


def test_wbt_converter_permutations_roundtrip_named_bodies():
    """The BFS->DFS body/joint permutation must place each body at its MuJoCo id."""
    pytest.importorskip("mujoco")
    import mujoco
    from scripts.motion.wbt_to_unilab_npz import _bfs_orders, _build_permutations, _parse_tree

    from unilab.assets import ASSETS_ROOT_PATH

    g1_xml = ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"
    dfs_body, children, joint_of = _parse_tree(g1_xml)
    bfs_body, _ = _bfs_orders(dfs_body, children, joint_of)
    body_perm, joint_perm = _build_permutations(g1_xml)

    model = mujoco.MjModel.from_xml_path(str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"))

    # Each IsaacLab BFS body row maps to the matching MuJoCo body id.
    for k, name in enumerate(bfs_body):
        expected_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_perm[k] == expected_id

    assert body_perm.shape[0] == 30  # IsaacLab body count
    assert joint_perm.shape[0] == 29
    # Permutation is a bijection onto MuJoCo ids 1..30 / joint slots 0..28.
    assert sorted(body_perm.tolist()) == list(range(1, 31))
    assert sorted(joint_perm.tolist()) == list(range(29))


def test_wbt_converter_file_roundtrip_with_sentinels(tmp_path):
    """``convert_file`` must scatter each named body/joint to its UniLab slot."""
    pytest.importorskip("mujoco")
    import mujoco
    from scripts.motion.wbt_to_unilab_npz import (
        _bfs_orders,
        _build_permutations,
        _parse_tree,
        convert_file,
    )

    from unilab.assets import ASSETS_ROOT_PATH

    g1_xml = ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"
    dfs_body, children, joint_of = _parse_tree(g1_xml)
    bfs_body, bfs_joint = _bfs_orders(dfs_body, children, joint_of)
    body_perm, joint_perm = _build_permutations(g1_xml)

    # Sentinel source: each body/joint row carries its BFS index as the value.
    num_bodies = len(bfs_body)
    num_joints = len(bfs_joint)
    joint_sentinel = np.arange(num_joints, dtype=np.float32)[None, :]
    body_sentinel = np.tile(np.arange(num_bodies, dtype=np.float32)[None, :, None], (1, 1, 3))
    src = tmp_path / "src.npz"
    np.savez(
        src,
        fps=np.array([50.0], dtype=np.float32),
        joint_pos=joint_sentinel,
        joint_vel=joint_sentinel,
        body_pos_w=body_sentinel,
        body_quat_w=np.tile(body_sentinel[..., :1], (1, 1, 4)),
        body_lin_vel_w=body_sentinel,
        body_ang_vel_w=body_sentinel,
    )
    dst = tmp_path / "dst.npz"
    convert_file(src, dst, body_perm, joint_perm)

    out = np.load(dst)
    model = mujoco.MjModel.from_xml_path(str(g1_xml))

    # Each body's UniLab row must hold its original BFS sentinel value.
    for bfs_idx, name in enumerate(bfs_body):
        uni_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert out["body_pos_w"][0, uni_id, 0] == pytest.approx(bfs_idx)

    # Joint sentinel must land at the MuJoCo (DFS) joint index for that joint.
    dfs_joints = [joint_of[n] for n in dfs_body if n in joint_of]
    for bfs_idx, jname in enumerate(bfs_joint):
        dfs_idx = dfs_joints.index(jname)
        assert out["joint_pos"][0, dfs_idx] == pytest.approx(bfs_idx)


@pytest.mark.slow
def test_stepping_stone_env_builds_resets_steps():
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

    subset = sorted(glob.glob(os.path.join(STEPPING_STONE_MOTION_DIR, "*.npz")))[:2]
    if not subset:
        pytest.skip(f"no motion files in {STEPPING_STONE_MOTION_DIR}")

    env = registry.make(
        "G1SteppingStone",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"motion_file": subset, "sampling_mode": "start"},
    )
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        assert act_space.shape[0] == 29

        spec = env.obs_groups_spec
        assert sum(spec.values()) == obs_space.shape[0]

        state = env.init_state()
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)

        # Reset must place the base at the motion's absolute world pelvis pose
        # (no env-origin offset for terrain-anchored clips).
        clip = np.load(subset[0])
        pelvis_xy = clip["body_pos_w"][0, 1, :2]  # body id 1 == pelvis (world is 0)
        base_xy = env._backend.get_base_pos()[:, :2]
        np.testing.assert_allclose(base_xy[0], pelvis_xy, atol=1e-3)

        state = env.step(np.zeros((2, act_space.shape[0])))
        assert np.all(np.isfinite(state.reward))
        assert state.reward.shape == (2,)
    finally:
        env.close()
