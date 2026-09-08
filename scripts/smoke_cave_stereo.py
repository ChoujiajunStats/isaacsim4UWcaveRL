#!/usr/bin/env python3
"""Validate the real cave visual/collision scene with stereo and active lights."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=4)
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
from isaac_underwater.worlds import load_cave_world_cfg
from isaaclab_tasks.utils import parse_env_cfg


def descendants(root: Usd.Prim) -> list[Usd.Prim]:
    result: list[Usd.Prim] = []
    pending = [root]
    while pending:
        prim = pending.pop()
        result.append(prim)
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    return result


def _intensity(prim: Usd.Prim) -> float:
    for name in ("inputs:intensity", "intensity"):
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            return float(attr.Get())
    raise AssertionError(f"No intensity attribute on {prim.GetPath()}")


def main() -> None:
    task = "Isaac-Underwater-Cave-Stereo-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    if args.usd_root is not None:
        cfg.world = load_cave_world_cfg("worlds/porth_yr_ogof_sump9.yaml", args.usd_root)
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        actions = torch.zeros(args.num_envs, 6, device=env.unwrapped.device)
        actions[:, 4] = -1.0
        actions[:, 5] = 1.0
        for _ in range(args.steps):
            _, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(reward).all()
        packets = env.unwrapped.build_sensor_packets()
        assert len(packets) == args.num_envs
        assert all(packet.rgb_left is not None and packet.rgb_right is not None for packet in packets)
        assert all(packet.depth_left is not None and packet.depth_right is not None for packet in packets)
        assert all(torch.isfinite(packet.depth_left.float()).any() for packet in packets)
        assert all(packet.sensor_metadata["stereo"] for packet in packets)
        assert all(abs(packet.sensor_metadata["stereo_near_clip_m"] - 0.25) < 1.0e-6 for packet in packets)

        stage = omni.usd.get_context().get_stage()
        visual = stage.GetPrimAtPath("/World/envs/env_0/Cave")
        collision = stage.GetPrimAtPath("/World/envs/env_0/CaveCollision")
        assert visual.IsValid(), "visual cave prim missing"
        assert collision.IsValid(), "collision cave prim missing"
        visual_meshes = [prim for prim in descendants(visual) if prim.IsA(UsdGeom.Mesh)]
        collision_meshes = [prim for prim in descendants(collision) if prim.IsA(UsdGeom.Mesh)]
        collision_api = [prim for prim in collision_meshes if prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert visual_meshes and collision_meshes and collision_api
        for index in range(args.num_envs):
            assert stage.GetPrimAtPath(f"/World/envs/env_{index}/Cave").IsValid()
            assert stage.GetPrimAtPath(f"/World/envs/env_{index}/CaveCollision").IsValid()
        left = stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightLeft")
        right = stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightRight")
        assert left.IsValid() and right.IsValid() and _intensity(left) < _intensity(right)
        print(
            f"envs={args.num_envs} steps={args.steps} visual_meshes={len(visual_meshes)} "
            f"collision_meshes={len(collision_meshes)} baseline_m={packets[0].sensor_metadata['stereo_baseline_m']} "
            f"near_clip_m={packets[0].sensor_metadata['stereo_near_clip_m']}",
            flush=True,
        )
        print("cave_stereo_smoke: PASS", flush=True)
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
