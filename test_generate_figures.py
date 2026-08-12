import unittest
import subprocess
import sys
import os
import glob

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Scripts that read the image data rather than only the committed CSVs. The data
# is too large to commit; see ARTIFACTS.md and scripts/fetch_artifacts.sh.
NEEDS_DATA = {
    "generate_figure_histogram_busi.py": "data/BUSI-processed",
}


class TestGenerateFigures(unittest.TestCase):
    def test_generate_figure_scripts(self):
        # Adjust the pattern to match your script naming convention and directory
        script_pattern = os.path.join("toskipornot/analyze/generate_figure_*.py")
        scripts = glob.glob(script_pattern)
        self.assertTrue(scripts, f"No scripts found matching pattern: {script_pattern}")
        # Some scripts import `toskipornot`, so the repo root has to be importable
        # whether or not the package itself was installed into the environment.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        for script in scripts:
            print(f"Testing script: {script}")
            with self.subTest(script=script):
                needs = NEEDS_DATA.get(os.path.basename(script))
                if needs and not os.path.isdir(os.path.join(REPO_ROOT, needs)):
                    self.skipTest(
                        f"{script} needs {needs}, which is not committed; "
                        "fetch it with scripts/fetch_artifacts.sh"
                    )
                result = subprocess.run(
                    [sys.executable, script], capture_output=True, text=True, env=env
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{script} failed with error:\n{result.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
