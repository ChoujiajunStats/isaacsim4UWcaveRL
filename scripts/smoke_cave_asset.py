#!/usr/bin/env python3
"""Load a converted cave in every cloned environment and inspect its prims."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="worlds/porth_yr_ogof_sump9.yaml")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--usd-root", default=None)
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
from isaac_underwater.worlds import load_cave_world_cfg, resolve_cave_asset
from isaaclab_tasks.utils import parse_env_cfg


def descendants(root: Usd.Prim) -> list[Usd.Prim]:
    result: list[Usd.Prim] = []
    pending = [root]
    while pending:
        prim = pending.pop()
        result.append(prim)
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    return result


def main() -> None:
    assets = resolve_cave_asset(args.config)
    cfg = parse_env_cfg("Isaac-Underwater-PointNav-Perception-v0", device=args.device, num_envs=args.num_envs)
    cfg.world = load_cave_world_cfg(args.config, args.usd_root)
    env = gym.make("Isaac-Underwater-PointNav-Perception-v0", cfg=cfg)
    try:
        env.reset()
        relative_spawn = env.unwrapped._robot.data.root_pos_w - env.unwrapped.scene.env_origins
        if env.unwrapped._cave_centerline is not None:
            nearest = torch.cdist(relative_spawn, env.unwrapped._cave_centerline).amin(dim=1)
            assert torch.all(nearest < 1.0), f"spawn is not near cave centerline: {nearest}"
            assert not torch.any(env.unwrapped._out_of_bounds), "cave spawn is outside navigation workspace"
        for _ in range(args.steps):
            env.step(torch.zeros(args.num_envs, 4, device=env.unwrapped.device))
        stage = omni.usd.get_context().get_stage()
        visual = stage.GetPrimAtPath("/World/envs/env_0/Cave")
        collision = stage.GetPrimAtPath("/World/envs/env_0/CaveCollision")
        assert visual.IsValid(), "visual cave prim missing"
        assert collision.IsValid(), "collision cave prim missing"
        visual_meshes = [prim for prim in descendants(visual) if prim.IsA(UsdGeom.Mesh)]
        collision_meshes = [prim for prim in descendants(collision) if prim.IsA(UsdGeom.Mesh)]
        collision_api = [prim for prim in collision_meshes if prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert visual_meshes, "visual cave contains no meshes"
        assert collision_meshes, "collision cave contains no meshes"
        assert collision_api, "collision cave has no collision API"
        for index in range(args.num_envs):
            assert stage.GetPrimAtPath(f"/World/envs/env_{index}/Cave").IsValid()
            assert stage.GetPrimAtPath(f"/World/envs/env_{index}/CaveCollision").IsValid()
        print(
            f"asset={assets.name} envs={args.num_envs} visual_meshes={len(visual_meshes)} "
            f"collision_meshes={len(collision_meshes)} collision_api={len(collision_api)}",
            flush=True,
        )
        print("cave_asset_smoke: PASS", flush=True)
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
