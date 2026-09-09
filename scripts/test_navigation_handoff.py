#!/usr/bin/env python3
"""Small temporary fixtures for private transfer packaging and safety checks."""
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import package_navigation_handoff as package


class HandoffPackageTest(unittest.TestCase):
    def test_complete_roundtrip_excludes_unrelated_and_refuses_overwrite(self):
        with TemporaryDirectory(prefix="cave-handoff-") as directory:
            root = Path(directory)

            def put(name, value="fixture"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
                return path

            for run in ("baseline", "tuned"):
                for name in ("model.pt", "evaluation_full.json", "params/agent.yaml", "params/env.yaml"):
                    put(f"logs/{run}/{name}")
            for name in ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "agent_state.pt",
                         "reward_normalizer.pt", "replay_buffer.pt"):
                put(f"logs/flash/step/{name}")
            put("logs/flash/step/metadata.json", json.dumps({"replay_saved": True, "upstream_commit": "pinned"}))
            put("logs/flash/step/unrelated.secret", "must not package")
            put("logs/flash/evaluation_full.json")
            put("logs/flash/training_summary.json")
            put("source_assets/caves_difficulty_v01/checksums.json", "{}")
            put("source_assets/caves_difficulty_v01/easy/cave.obj")
            put("source_assets/unrelated.txt", "must not package")
            status = put("outputs/repair/status.json", json.dumps({"state": "complete", "accepted": False,
                                                                  "checkpoint": str(root / "logs/tuned/model.pt")}))
            put("outputs/repair/large.log", "must not package")
            output = root / "outputs/transfer/run.tar.gz"
            argv = ["package", "--repair_directory", str(status.parent), "--ppo_baseline", str(root / "logs/baseline/model.pt"),
                    "--flash_checkpoint", str(root / "logs/flash/step"), "--asset_root", str(root / "source_assets"),
                    "--output", str(output)]
            with patch.object(package, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.object(package.subprocess, "check_output", side_effect=["abc123\n", ""]):
                package.main()
            manifest = json.loads(output.with_suffix(".gz.manifest.json").read_text())
            self.assertFalse(manifest["policy_accepted"])
            self.assertFalse(manifest["code_worktree_dirty"])
            self.assertEqual(manifest["code_commit"], "abc123")
            with tarfile.open(output) as archive:
                self.assertFalse(any("unrelated" in name or name.endswith("large.log") for name in archive.getnames()))
                self.assertIn("transfer/assets/caves_difficulty_v01/easy/cave.obj", archive.getnames())
                for name, expected in manifest["files"].items():
                    with archive.extractfile(name) as stream:
                        self.assertEqual(package.stream_digest(stream), expected["sha256"])
            self.assertEqual(output.with_suffix(".gz.sha256").read_text().split()[0], package.digest(output))
            with patch.object(package, "ROOT", root), patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                package.main()


if __name__ == "__main__":
    unittest.main()
