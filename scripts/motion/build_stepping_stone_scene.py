"""Generate the G1 stepping-stone MuJoCo scene from a baked-terrain deploy XML.

The IsaacLab ``RGMT-SteppingStone-G1-v0`` task anchors its motions to a fixed
big-map terrain.  The deploy export
``g1_29dof_big_map_urdf.xml`` already bakes that terrain as a flat list of
``<geom type="box" .../>`` primitives in its ``<worldbody>``.  This script copies
those box geoms verbatim into a new UniLab scene that ``<include>``s the standard
``g1.xml`` robot, so the terrain-anchored motions stay in the same world frame.

Usage::

    uv run python scripts/motion/build_stepping_stone_scene.py --deploy-xml DEPLOY.xml

The output is written to ``src/unilab/assets/robots/g1/scene_stepping_stone.xml``.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from unilab.assets import ASSETS_ROOT_PATH

DEFAULT_OUTPUT = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_stepping_stone.xml"

# Standard G1 standing keyframe (copied from scene_flat.xml) so the backend has a
# valid default base pose.  qpos = [root_pos(3), root_quat(4), 29 joints].
_STAND_QPOS = (
    "0 0 0.754 "
    "1 0 0 0 "
    "-0.312 0.0 0.0 0.669 -0.363 0.0 "
    "-0.312 0.0 0.0 0.669 -0.363 0.0 "
    "0.0 0.0 0.0 "
    "0.2  0.2 0.0 0.6 0.0 0.0 0.0 "
    "0.2 -0.2 0.0 0.6 0.0 0.0 0.0"
)
_STAND_CTRL = (
    "-0.312 0.0 0.0 0.669 -0.363 0.0 "
    "-0.312 0.0 0.0 0.669 -0.363 0.0 "
    "0.0 0.0 0.0 "
    "0.2  0.2 0.0 0.6 0.0 0.0 0.0 "
    "0.2 -0.2 0.0 0.6 0.0 0.0 0.0"
)


def _extract_terrain_boxes(deploy_xml: Path) -> list[ET.Element]:
    """Return all top-level ``<geom type="box">`` elements from the deploy worldbody."""
    tree = ET.parse(deploy_xml)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in {deploy_xml}")
    boxes = [geom for geom in worldbody.findall("geom") if geom.get("type") == "box"]
    if not boxes:
        raise ValueError(f"No box terrain geoms found in {deploy_xml}")
    return boxes


def _build_scene(boxes: list[ET.Element]) -> str:
    """Assemble the scene XML string from the ported terrain box geoms."""
    terrain_lines = []
    for geom in boxes:
        attrs = " ".join(f'{k}="{v}"' for k, v in geom.attrib.items())
        terrain_lines.append(f"      <geom {attrs}/>")
    terrain_block = "\n".join(terrain_lines)

    return f"""<mujoco model="g1_29dof stepping stone scene">
  <include file="g1.xml"/>

  <statistic center="0 0 0.9" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="terrain" pos="0 0 0">
{terrain_block}
    </body>
  </worldbody>

  <keyframe>
    <key name="stand"
      qpos="{_STAND_QPOS}"
      ctrl="{_STAND_CTRL}"/>
  </keyframe>

</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy-xml",
        type=Path,
        required=True,
        help="Baked-terrain deploy XML whose worldbody box geoms define the terrain.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    boxes = _extract_terrain_boxes(args.deploy_xml)
    scene_xml = _build_scene(boxes)
    args.output.write_text(scene_xml)
    print(f"Wrote {len(boxes)} terrain box geoms to {args.output}")

    # Validate that the generated scene loads in MuJoCo.
    model = mujoco.MjModel.from_xml_path(str(args.output))
    print(f"Loaded OK: nbody={model.nbody}, ngeom={model.ngeom}, nq={model.nq}")


if __name__ == "__main__":
    main()
