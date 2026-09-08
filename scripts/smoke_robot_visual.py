#!/usr/bin/env python3
"""Validate the detailed BlueROV2 visual and primitive-only collision setup."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from pxr import Usd, UsdGeom, UsdPhysics

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("Isaac-Underwater-PointNav-Perception-v0", device=args.device, num_envs=1)
    env = gym.make("Isaac-Underwater-PointNav-Perception-v0", cfg=cfg)
    try:
        env.reset()
        env.step(torch.zeros(1, 4, device=env.unwrapped.device))

        stage = omni.usd.get_context().get_stage()
        robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        visual = stage.GetPrimAtPath("/World/envs/env_0/Robot/Visual")
        assert robot.IsValid()
        assert visual.IsValid()

        def descendants(root: Usd.Prim) -> list[Usd.Prim]:
            result: list[Usd.Prim] = []
            pending = [root]
            while pending:
                prim = pending.pop()
                result.append(prim)
                pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
            return result

        meshes = [prim for prim in descendants(visual) if prim.IsA(UsdGeom.Mesh)]
        collisions = [prim for prim in descendants(robot) if prim.HasAPI(UsdPhysics.CollisionAPI)]
        visual_collisions = [prim for prim in collisions if "/Visual/" in str(prim.GetPath())]
        primitive = stage.GetPrimAtPath("/World/envs/env_0/Robot/geometry/mesh")

        assert meshes, "BlueROV2 visual contains no composed meshes"
        assert not visual_collisions, f"Detailed visual unexpectedly has collision: {visual_collisions}"
        assert primitive.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(primitive).ComputeVisibility() == UsdGeom.Tokens.invisible
        print(f"visual_meshes={len(meshes)} collision_prims={len(collisions)}", flush=True)
        print("robot_visual_smoke: PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
