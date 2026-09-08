#!/usr/bin/env python3
"""Convert the vendored Stonefish BlueROV2 meshes into a composed USD visual."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Regenerate component USD files even when cached")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "bluerov2"
SOURCE_DIR = ASSET_ROOT / "source"
USD_DIR = ASSET_ROOT / "usd"
OUTPUT_PATH = ASSET_ROOT / "bluerov2_visual.usda"


def convert_component(name: str, source_name: str) -> Path:
    output_dir = USD_DIR / name
    output_path = output_dir / f"{name}.usd"
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(SOURCE_DIR / source_name),
            usd_dir=str(output_dir),
            usd_file_name=output_path.name,
            force_usd_conversion=args.force,
            make_instanceable=False,
            collision_props=None,
            mesh_collision_props=None,
            mass_props=None,
            rigid_props=None,
        )
    )
    print(f"converted {source_name} -> {converter.usd_path}", flush=True)
    return Path(converter.usd_path)


def define_material(
    stage: Usd.Stage,
    path: str,
    *,
    color: tuple[float, float, float] | None = None,
    texture: str | None = None,
    roughness: float = 0.45,
    metallic: float = 0.0,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if texture is None:
        assert color is not None
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        return material

    st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    st_input.Set("st")
    st_reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_input)

    sampler = UsdShade.Shader.Define(stage, f"{path}/DiffuseTexture")
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        sampler.ConnectableAPI(), "rgb"
    )
    return material


def add_component(
    stage: Usd.Stage,
    root_path: str,
    name: str,
    asset_path: Path,
    material: UsdShade.Material,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    prim = stage.DefinePrim(f"{root_path}/{name}", "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_deg))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # Keep the reference on a child so its converter-authored xformOps cannot
    # collide with the placement xformOps above.
    asset_prim = stage.DefinePrim(f"{root_path}/{name}/Asset", "Xform")
    relative_asset = os.path.relpath(asset_path, OUTPUT_PATH.parent)
    asset_prim.GetReferences().AddReference(relative_asset)
    asset_prim.SetInstanceable(True)


def compose_visual(components: dict[str, Path]) -> None:
    stage = Usd.Stage.CreateNew(str(OUTPUT_PATH))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = "/BlueROV2Visual"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)

    hull_material = define_material(stage, f"{root_path}/Looks/Hull", texture="source/br2.png")
    frame_material = define_material(
        stage, f"{root_path}/Looks/Frame", color=(0.025, 0.035, 0.045), roughness=0.32, metallic=0.15
    )
    propeller_material = define_material(
        stage, f"{root_path}/Looks/Propeller", color=(0.0, 0.42, 0.68), roughness=0.4
    )

    add_component(stage, root_path, "Hull", components["hull"], hull_material)
    add_component(stage, root_path, "HeavyFrame", components["wings"], frame_material)

    propellers = (
        ("FrontRight", "ccw", (0.1355, 0.1, 0.0725), (0.0, 0.0, -45.0)),
        ("FrontLeft", "ccw", (0.1355, -0.1, 0.0725), (0.0, 0.0, 45.0)),
        ("BackRight", "cw", (-0.1475, 0.1, 0.0725), (0.0, 0.0, -135.0)),
        ("BackLeft", "cw", (-0.1475, -0.1, 0.0725), (0.0, 0.0, 135.0)),
        ("DiveFrontRight", "cw", (0.12, 0.218, 0.0), (0.0, -90.0, 0.0)),
        ("DiveFrontLeft", "ccw", (0.12, -0.218, 0.0), (0.0, -90.0, 0.0)),
        ("DiveBackRight", "ccw", (-0.12, 0.218, 0.0), (0.0, -90.0, 0.0)),
        ("DiveBackLeft", "cw", (-0.12, -0.218, 0.0), (0.0, -90.0, 0.0)),
    )
    for name, handedness, translation, rotation_deg in propellers:
        add_component(
            stage,
            root_path,
            name,
            components[handedness],
            propeller_material,
            translation=translation,
            rotation_deg=rotation_deg,
        )

    stage.GetRootLayer().customLayerData = {
        "source": "https://github.com/bvibhav/stonefish_bluerov2",
        "sourceCommit": "6448383af6b7ef6083b0eac2c08102660591e318",
        "license": "Apache-2.0",
    }
    stage.GetRootLayer().Save()
    print(f"composed BlueROV2 visual -> {OUTPUT_PATH}", flush=True)


def main() -> None:
    required = {
        "hull": "bluerov2.obj",
        "wings": "bluerov2_wings.obj",
        "ccw": "ccw.obj",
        "cw": "cw.obj",
    }
    missing = [str(SOURCE_DIR / filename) for filename in required.values() if not (SOURCE_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BlueROV2 source assets: {missing}")
    components = {name: convert_component(name, filename) for name, filename in required.items()}
    compose_visual(components)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
