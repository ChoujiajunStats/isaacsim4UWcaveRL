"""Sensor-suite contracts and optional Isaac Lab sensor factories.

The dataclasses in this module are simulator-independent.  The Isaac Lab
configuration factories are imported lazily so that dynamics-only tests never
initialize the renderer or require an Isaac Sim application.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SensorSuiteCfg:
    rgb_enabled: bool = False
    depth_enabled: bool = False
    imu_enabled: bool = True
    pressure_enabled: bool = True
    ground_truth_enabled: bool = True
    camera_width: int = 320
    camera_height: int = 240
    camera_rate_hz: float = 15.0
    camera_near_clip_m: float = 0.05
    camera_far_clip_m: float = 30.0
    imu_rate_hz: float = 100.0
    timestamp_tolerance_s: float = 1.0e-4
    # Source-derived candidate from assets/bluerov2/source/bluerov2.scn.
    # The scene also contains a conflicting ``baseline=-60.5`` attribute;
    # that value is intentionally not used as Isaac geometry ground truth.
    stereo_baseline_m: float = 0.145
    stereo_left_pos_b_m: tuple[float, float, float] = (0.16, -0.0725, 0.15)
    stereo_right_pos_b_m: tuple[float, float, float] = (0.16, 0.0725, 0.15)
    stereo_rot_wxyz: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)

    def camera_cfg(self, prim_path: str) -> Any:
        """Build a tiled RGB/depth camera config for a robot-mounted sensor."""
        if not (self.rgb_enabled or self.depth_enabled):
            raise ValueError("camera_cfg requires rgb_enabled or depth_enabled")
        from isaaclab import sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        data_types = []
        if self.rgb_enabled:
            data_types.append("rgb")
        if self.depth_enabled:
            data_types.append("depth")
        return TiledCameraCfg(
            prim_path=prim_path,
            update_period=1.0 / self.camera_rate_hz,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.36, 0.0, 0.02),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=4.0,
                horizontal_aperture=20.955,
                clipping_range=(self.camera_near_clip_m, self.camera_far_clip_m),
            ),
            width=self.camera_width,
            height=self.camera_height,
        )

    def stereo_camera_cfgs(self, left_prim_path: str, right_prim_path: str) -> tuple[Any, Any]:
        """Build synchronized left/right cameras for the BlueROV candidate.

        The two cameras are separate ``TiledCamera`` instances because Isaac
        Lab's tiled sensor represents one calibrated view per prim path.  Both
        use identical intrinsics and data types; only the body-frame position
        differs by the source-derived 145 mm baseline.
        """
        if not (self.rgb_enabled or self.depth_enabled):
            raise ValueError("stereo_camera_cfgs requires rgb_enabled or depth_enabled")
        from isaaclab import sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        data_types = []
        if self.rgb_enabled:
            data_types.append("rgb")
        if self.depth_enabled:
            data_types.append("depth")

        def build(prim_path: str, position: tuple[float, float, float]) -> Any:
            return TiledCameraCfg(
                prim_path=prim_path,
                update_period=1.0 / self.camera_rate_hz,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=position,
                    rot=self.stereo_rot_wxyz,
                    convention="ros",
                ),
                data_types=data_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=4.0,
                    horizontal_aperture=20.955,
                    clipping_range=(self.camera_near_clip_m, self.camera_far_clip_m),
                ),
                width=self.camera_width,
                height=self.camera_height,
            )

        return build(left_prim_path, self.stereo_left_pos_b_m), build(
            right_prim_path, self.stereo_right_pos_b_m
        )

    def imu_cfg(self, prim_path: str) -> Any:
        """Build an Isaac Lab IMU config attached to the robot body."""
        if not self.imu_enabled:
            raise ValueError("imu_cfg requires imu_enabled")
        from isaaclab.sensors import ImuCfg

        return ImuCfg(
            prim_path=prim_path,
            update_period=1.0 / self.imu_rate_hz,
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )


def robot_namespace(robot_index: int) -> str:
    if robot_index < 0:
        raise ValueError("robot_index must be non-negative")
    return f"robot_{robot_index:03d}"
