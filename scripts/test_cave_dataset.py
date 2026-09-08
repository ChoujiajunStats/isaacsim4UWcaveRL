#!/usr/bin/env python3
"""Pure-Python contracts for the multi-cave manifest and route loader."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from isaac_underwater.navigation import balanced_scene_assignment, load_cave_route
from isaac_underwater.worlds import (
    coerce_cave_scene_cfg,
    load_cave_dataset_world_cfg,
    resolve_cave_dataset,
)


def main() -> None:
    config = "worlds/caves_difficulty_v01.yaml"
    dataset = resolve_cave_dataset(config, "train_all", require_sources=True)
    assert [scene.key for scene in dataset.scenes] == ["easy", "medium", "hard"]
    assert dataset.license_status == "not_provided_in_archive"
    assert all(scene.visual_source == scene.collision_source for scene in dataset.scenes)
    assert all(scene.collision_status == "high_poly_visual_mesh_fallback" for scene in dataset.scenes)

    checksum_path = dataset.checksum_manifest_path
    assert checksum_path is not None and checksum_path.is_file()
    expected_checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    assert isinstance(expected_checksums, dict) and expected_checksums
    checksum_root = checksum_path.parent.resolve()
    for relative_path, expected_digest in expected_checksums.items():
        candidate = (checksum_root / relative_path).resolve()
        assert candidate.is_relative_to(checksum_root), f"Unsafe checksum path: {relative_path}"
        assert candidate.is_file(), f"Missing checksummed asset: {candidate}"
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected_digest, f"Checksum mismatch: {candidate}"

    lengths = []
    for scene in dataset.scenes:
        assert scene.navigation_path is not None
        route = load_cave_route(scene.navigation_path)
        assert route.points_m[0] == route.start_m
        assert route.points_m[-1] == route.goal_m
        assert route.length_m > 1.0
        lengths.append(route.length_m)
    assert lengths[0] < lengths[1] < lengths[2]

    assignment = balanced_scene_assignment(8, 3)
    counts = Counter(assignment)
    assert set(counts) == {0, 1, 2}
    assert max(counts.values()) - min(counts.values()) <= 1
    try:
        balanced_scene_assignment(2, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Partial scene coverage must be rejected")

    world = load_cave_dataset_world_cfg(config, "train_all", require_converted=False)
    assert world.dataset_name == "caves_difficulty_v01"
    assert tuple(scene.key for scene in world.scene_variants) == ("easy", "medium", "hard")
    restored = coerce_cave_scene_cfg(world.scene_variants[0].__dict__)
    assert restored == world.scene_variants[0]
    print(
        "cave_dataset_contract: PASS "
        f"scenes={len(dataset.scenes)} checksums={len(expected_checksums)} "
        f"route_lengths_m={[round(value, 2) for value in lengths]}"
    )


if __name__ == "__main__":
    main()
