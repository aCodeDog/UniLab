"""Convert IsaacLab whole-body-tracking motion NPZ files to UniLab layout.

The IsaacLab ``whole_body_tracking`` exporter stores arrays ordered by the Isaac
Lab *articulation* (breadth-first) body/joint order:

- ``joint_pos`` / ``joint_vel`` : ``(T, 29)`` in IsaacLab joint (BFS) order.
- ``body_*_w``                  : ``(T, 30, *)`` in IsaacLab body (BFS) order,
  with ``pelvis`` first and no ``world`` body.

UniLab's :class:`MotionLoader` instead expects:

- ``joint_*`` : ``(T, 29)`` in MuJoCo joint-id (declaration / depth-first) order.
- ``body_*_w`` : ``(T, 31, *)`` in MuJoCo body-id order, where id 0 is ``world``.

Both the BFS (IsaacLab) and DFS (MuJoCo) orders are recovered from the kinematic
tree of the G1 MuJoCo XML, so the conversion is a deterministic per-axis
permutation plus a leading zero ``world`` body row.

Usage::

    uv run python scripts/motion/wbt_to_unilab_npz.py --src-dir SRC --dst-dir DST
"""

from __future__ import annotations

import argparse
import collections
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

# G1 MuJoCo source whose kinematic tree defines both orderings.  UniLab's own
# ``g1.xml`` shares the identical body/joint names and tree, so either works.
DEFAULT_G1_XML = ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"

_BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _parse_tree(g1_xml: Path) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Return (dfs_body_order, children, joint_of_body) from the G1 worldbody."""
    worldbody = ET.parse(g1_xml).getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in {g1_xml}")
    children: dict[str, list[str]] = collections.defaultdict(list)
    joint_of: dict[str, str] = {}
    dfs_order: list[str] = []

    def walk(elem: ET.Element, parent: str | None) -> None:
        for body in elem.findall("body"):
            name = body.get("name")
            if name is None:
                continue
            dfs_order.append(name)
            if parent is not None:
                children[parent].append(name)
            joint = body.find("joint")
            if joint is not None and joint.get("name"):
                joint_of[name] = joint.get("name")
            walk(body, name)

    walk(worldbody, None)
    return dfs_order, children, joint_of


def _bfs_orders(
    dfs_body_order: list[str],
    children: dict[str, list[str]],
    joint_of: dict[str, str],
    root_body: str = "pelvis",
) -> tuple[list[str], list[str]]:
    """Return (bfs_body_order, bfs_joint_order) matching IsaacLab articulation."""
    bfs_body: list[str] = []
    bfs_joint: list[str] = []
    queue: deque[str] = deque([root_body])
    while queue:
        name = queue.popleft()
        bfs_body.append(name)
        if name in joint_of:
            bfs_joint.append(joint_of[name])
        queue.extend(children[name])
    return bfs_body, bfs_joint


def _build_permutations(g1_xml: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (body_perm, joint_perm) mapping IsaacLab rows -> UniLab rows.

    ``body_perm`` has length 30 (the IsaacLab body count); ``body_perm[k]`` is the
    UniLab MuJoCo body id (1..30) for IsaacLab body row ``k``.  ``joint_perm[k]``
    is the UniLab joint index (0..28) for IsaacLab joint row ``k``.
    """
    dfs_body, children, joint_of = _parse_tree(g1_xml)
    bfs_body, bfs_joint = _bfs_orders(dfs_body, children, joint_of)

    # UniLab MuJoCo body id: id 0 = world, then declaration (DFS) order + 1.
    uni_body_id = {name: idx + 1 for idx, name in enumerate(dfs_body)}
    # UniLab joint index: DFS joint order (skip the freejoint / root).
    dfs_joints = [joint_of[name] for name in dfs_body if name in joint_of]
    uni_joint_idx = {name: idx for idx, name in enumerate(dfs_joints)}

    body_perm = np.array([uni_body_id[name] for name in bfs_body], dtype=np.int64)
    joint_perm = np.array([uni_joint_idx[name] for name in bfs_joint], dtype=np.int64)
    return body_perm, joint_perm


def convert_file(src: Path, dst: Path, body_perm: np.ndarray, joint_perm: np.ndarray) -> None:
    """Convert a single IsaacLab WBT NPZ to UniLab body-id / joint-id layout."""
    with np.load(src) as data:
        num_frames = data["joint_pos"].shape[0]
        num_uni_bodies = body_perm.shape[0] + 1  # +1 for the world row.

        out: dict[str, np.ndarray] = {"fps": data["fps"]}

        # Joints: BFS -> DFS permutation.
        inv_joint = np.empty_like(joint_perm)
        inv_joint[joint_perm] = np.arange(joint_perm.shape[0])
        out["joint_pos"] = data["joint_pos"][:, inv_joint].astype(np.float32)
        out["joint_vel"] = data["joint_vel"][:, inv_joint].astype(np.float32)

        # Bodies: scatter BFS rows into UniLab body-id slots (id 0 = world).
        for key in _BODY_KEYS:
            src_arr = data[key].astype(np.float32)
            width = src_arr.shape[2]
            dst_arr = np.zeros((num_frames, num_uni_bodies, width), dtype=np.float32)
            if key == "body_quat_w":
                dst_arr[:, :, 0] = 1.0  # valid identity quaternions everywhere.
            dst_arr[:, body_perm, :] = src_arr
            out[key] = dst_arr

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--pattern", default="*.npz")
    args = parser.parse_args()

    src_files = sorted(args.src_dir.glob(args.pattern))
    if not src_files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.src_dir}")

    body_perm, joint_perm = _build_permutations(args.g1_xml)
    for src in src_files:
        convert_file(src, args.dst_dir / src.name, body_perm, joint_perm)
    print(f"Converted {len(src_files)} files -> {args.dst_dir}")


if __name__ == "__main__":
    main()
