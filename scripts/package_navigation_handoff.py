#!/usr/bin/env python3
"""Package private local navigation assets/checkpoints, never upload them.

The archive preserves repository-relative logs/outputs paths. External cave
sources go under transfer/assets; set ISAAC_UNDERWATER_ASSET_ROOT accordingly
after extraction and reconvert USDs on the receiving machine. No environment,
credentials, unrelated downloads or generated absolute-path USDs are included.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def stream_digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return stream_digest(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair_directory", type=Path, required=True)
    parser.add_argument("--ppo_baseline", type=Path, required=True)
    parser.add_argument("--flash_checkpoint", type=Path, required=True)
    parser.add_argument("--asset_root", type=Path, required=True)
    parser.add_argument("--report_directory", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if any(path.exists() for path in (output, checksum_path, manifest_path)):
        parser.error("Refusing to overwrite a transfer archive or its manifest/checksum")
    repair = args.repair_directory.resolve()
    status = json.loads((repair / "status.json").read_text())
    if status.get("state") != "complete":
        parser.error("Finish repair training and evaluation before packaging")
    tuned_checkpoint = Path(status["checkpoint"]).resolve()
    dataset = args.asset_root.resolve() / "caves_difficulty_v01"
    if not dataset.is_dir() or not (dataset / "checksums.json").is_file():
        parser.error("asset_root must contain the verified caves_difficulty_v01 directory")
    entries = {}

    def add_file(source, destination=None):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular, existing transfer file: {source}")
        source = source.resolve()
        name = str(destination if destination is not None else source.relative_to(ROOT))
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe archive path: {name}")
        if name in entries and entries[name] != source:
            raise ValueError(f"Duplicate archive destination: {name}")
        entries[name] = source

    for checkpoint in (args.ppo_baseline.resolve(), tuned_checkpoint):
        add_file(checkpoint)
        add_file(checkpoint.parent / "evaluation_full.json")
        for name in ("agent.yaml", "env.yaml"):
            add_file(checkpoint.parent / "params" / name)
    flash = args.flash_checkpoint.resolve()
    # Include native optimizer, reward-normalizer and replay state, not just actor weights.
    metadata = json.loads((flash / "metadata.json").read_text())
    if not metadata.get("replay_saved") or not (flash / "replay_buffer.pt").is_file():
        raise ValueError("A resumable FlashSAC transfer requires its saved replay")
    for name in ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "agent_state.pt",
                 "reward_normalizer.pt", "replay_buffer.pt", "metadata.json"):
        add_file(flash / name)
    for name in ("training_summary.json", "evaluation_full.json"):
        add_file(flash.parent / name)
    for directory in [repair, *(path.resolve() for path in args.report_directory)]:
        # Keep small evidence/provenance files; omit large console logs and unrelated artifacts.
        directory.relative_to(ROOT / "outputs")
        for source in sorted(directory.glob("*.json")):
            add_file(source)
    for source in sorted(dataset.rglob("*")):
        if source.is_file():
            add_file(source, Path("transfer/assets/caves_difficulty_v01") / source.relative_to(dataset))
    manifest = {
        "version": 1, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "policy_accepted": status.get("accepted", False),
        "ppo_tuned_checkpoint": str(tuned_checkpoint.relative_to(ROOT)),
        "ppo_baseline_checkpoint": str(args.ppo_baseline.resolve().relative_to(ROOT)),
        "flash_checkpoint": str(flash.relative_to(ROOT)),
        "flash_upstream_commit": metadata["upstream_commit"],
        "asset_root_after_extraction": "transfer/assets",
        "asset_redistribution": "Private transfer between the user's machines; dataset license not supplied; do not publish",
        "usd_conversion": "Reconvert on destination; absolute-path USD caches intentionally omitted",
        "files": {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(entries.items())},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz", compresslevel=1) as archive:
        for name, source in sorted(entries.items()):
            archive.add(source, arcname=name, recursive=False)
        member = tarfile.TarInfo("transfer/navigation_handoff_manifest.json")
        member.size, member.mode = len(manifest_bytes), 0o644
        archive.addfile(member, io.BytesIO(manifest_bytes))
    # Check the archive structure without extracting/overwriting anything.
    with tarfile.open(output, "r:gz") as archive:
        actual = archive.getmembers()
        expected = set(entries) | {"transfer/navigation_handoff_manifest.json"}
        if {member.name for member in actual} != expected or any(not member.isfile() for member in actual):
            raise RuntimeError("Transfer archive membership verification failed")
        for member in actual:
            if member.name in entries:
                with archive.extractfile(member) as stream:
                    if stream_digest(stream) != manifest["files"][member.name]["sha256"]:
                        raise RuntimeError(f"Transfer payload checksum mismatch: {member.name}")
    manifest_path.write_bytes(manifest_bytes)
    checksum_path.write_text(f"{digest(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "files": len(entries),
                      "checksum": str(checksum_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
