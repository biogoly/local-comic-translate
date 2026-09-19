from pathlib import Path
import unittest
from packaging.version import Version

from app.update_checker import UpdateChecker
from app.version import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]


class UpdateIdentityTests(unittest.TestCase):
    def test_updater_targets_the_fork(self):
        self.assertEqual(UpdateChecker.REPO_OWNER, "biogoly")
        self.assertEqual(UpdateChecker.REPO_NAME, "local-comic-translate")

    def test_displayed_version_is_valid_for_update_comparison(self):
        # Source installs create their own ignored uv project metadata. The
        # public app version must not depend on that machine-local file.
        self.assertEqual(str(Version(__version__)), __version__)

    def test_startup_does_not_automatically_check_for_updates(self):
        controller_source = (REPO_ROOT / "controller.py").read_text(encoding="utf-8")
        self.assertNotIn("check_for_updates(is_background=True)", controller_source)


if __name__ == "__main__":
    unittest.main()
