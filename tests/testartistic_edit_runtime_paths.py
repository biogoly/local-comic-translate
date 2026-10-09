import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from modules.artistic_edit.contracts import encode_json_line
from modules.artistic_edit.runtime_paths import (
    WORKER_SOURCE_FILES,
    is_application_executable,
    lora_root,
    worker_source_root,
)


class ArtisticEditRuntimePathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = Path(__file__).resolve().parents[1]

    def copy_worker_tree(self):
        tree = self.root / "flux-worker"
        for relative in WORKER_SOURCE_FILES:
            destination = tree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.repository / relative, destination)
        return tree

    def test_source_paths_remain_repository_relative(self):
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(worker_source_root(), self.repository)
            self.assertEqual(lora_root(), self.repository / "models/loras/flux2-klein")
            self.assertFalse(is_application_executable(sys.executable))

    def test_frozen_worker_uses_only_the_verified_shipped_source_tree(self):
        tree = self.copy_worker_tree()
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "_MEIPASS", str(self.root), create=True),
        ):
            self.assertEqual(worker_source_root(), tree)
            (tree / "modules/artistic_edit/contracts.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "contracts.py"):
                worker_source_root()

    def test_frozen_loras_use_profile_without_writing_to_installation(self):
        profile = self.root / "profile"
        with (
            patch.object(sys, "frozen", True, create=True),
            patch("modules.artistic_edit.runtime_paths.get_user_data_dir", return_value=str(profile)),
        ):
            self.assertEqual(lora_root(), profile / "models/loras/flux2-klein")
            self.assertFalse(profile.exists())
            self.assertTrue(is_application_executable(sys.executable))
            self.assertFalse(is_application_executable(self.root / "optional/python.exe"))

    def test_shipped_worker_source_can_ping_from_outside_the_checkout(self):
        tree = self.copy_worker_tree()
        ping_id, shutdown_id = str(uuid4()), str(uuid4())
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        # Preserve explicit dependency locations used by the test runner, but
        # never let the checkout supply a missing worker module.
        if "PYTHONPATH" in environment:
            environment["PYTHONPATH"] = os.pathsep.join(
                entry for entry in environment["PYTHONPATH"].split(os.pathsep)
                if entry and Path(entry).resolve(strict=False) != self.repository
            )
        completed = subprocess.run(
            [sys.executable, "-B", "-u", "-m", "modules.artistic_edit.flux_worker",
             "--temp-root", str(self.root), "--device-policy", "cpu"],
            cwd=tree,
            env=environment,
            input=(encode_json_line({"id": ping_id, "op": "ping"})
                   + encode_json_line({"id": shutdown_id, "op": "shutdown"})),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([event["id"] for event in events], [ping_id, shutdown_id])
        self.assertTrue(all(event["event"] == "ready" for event in events))
        self.assertEqual(list(tree.rglob("__pycache__")), [])


if __name__ == "__main__":
    unittest.main()
