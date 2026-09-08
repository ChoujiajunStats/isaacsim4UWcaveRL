# BlueROV2 Visual Asset

The source meshes, texture, URDF, and Stonefish scene in `source/` were
retrieved from:

- Repository: <https://github.com/bvibhav/stonefish_bluerov2>
- Commit: `6448383af6b7ef6083b0eac2c08102660591e318`
- Copyright: 2024 Vibhav Bharti
- License: Apache License 2.0; see `LICENSE` in this directory

The generated USD files are mechanical conversions of those source meshes.
`bluerov2_visual.usda` composes the hull, heavy-frame visual, and eight
propellers using transforms recorded in the upstream Stonefish scenario. The
Isaac Lab integration adds no mesh collision: the project-owned primitive
rigid body remains the collision and dynamics representation.
