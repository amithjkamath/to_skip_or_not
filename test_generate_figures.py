"""Smoke-test the figure and table generation scripts.

Large artifacts are not committed (see ARTIFACTS.md), so a script whose inputs
are absent is skipped rather than failed. Fetch the inputs first to exercise
them for real:

    WHAT=results scripts/fetch_artifacts.sh
"""

import glob
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from toskipornot.config import DATA_DIR, RESULTS_DIR  # noqa: E402

# Each script's inputs live in the artifact archive. Map every script to one
# representative directory it needs; absent means skip.
REQUIRES = {
    "generate_figure_for_clinical_robustness.py": RESULTS_DIR / "BUSI-results",
    "generate_figure_for_clinical_robustness_consolidated.py": (
        RESULTS_DIR / "BUSI-results"
    ),
    "generate_figure_table_1_robustness_rankings.py": RESULTS_DIR / "BUSI-results",
    "generate_figure_for_synthetic_robustness.py": RESULTS_DIR / "background-results",
    "generate_figure_4_synthetic_performance_MICCAI.py": (
        RESULTS_DIR / "background-results"
    ),
    "generate_figure_table_3_train_time_analysis.py": RESULTS_DIR
    / "train-time-results",
    "generate_figure_histogram_busi.py": DATA_DIR / "BUSI-processed",
}


class TestGenerateFigures(unittest.TestCase):
    def test_generate_figure_scripts(self):
        scripts = glob.glob(os.path.join("toskipornot/analyze/generate_figure_*.py"))
        self.assertTrue(scripts, "no generate_figure_* scripts found")

        # Every script must be accounted for, so adding one without declaring its
        # inputs is caught here rather than silently passing or failing in CI.
        undeclared = sorted(
            os.path.basename(s) for s in scripts if os.path.basename(s) not in REQUIRES
        )
        self.assertFalse(
            undeclared,
            f"add these to REQUIRES so the test knows what they need: {undeclared}",
        )

        # The scripts import toskipornot, which has to work whether or not the
        # package was installed into the environment.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )

        ran = 0
        for script in sorted(scripts):
            name = os.path.basename(script)
            with self.subTest(script=name):
                needed = REQUIRES[name]
                if not os.path.isdir(needed):
                    self.skipTest(
                        f"{name} needs {needed}, which is not committed; "
                        "fetch it with scripts/fetch_artifacts.sh"
                    )
                print(f"Testing script: {script}")
                result = subprocess.run(
                    [sys.executable, script], capture_output=True, text=True, env=env
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{script} failed with error:\n{result.stderr}",
                )
                ran += 1
        print(f"ran {ran} of {len(scripts)} scripts; the rest lacked inputs")


if __name__ == "__main__":
    unittest.main()
