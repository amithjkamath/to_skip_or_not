# -*- coding: utf-8 -*-
"""Path configuration for the whole project.

Nothing in this repository hardcodes a machine-specific path. Every script
resolves its inputs and outputs through this module, which reads (in order of
precedence):

1. environment variables,
2. a ``.env`` file in the repository root,
3. defaults relative to the repository root.

Recognised settings — see ``.env.example``:

==========================  =======================================  ===========
Variable                    Meaning                                  Default
==========================  =======================================  ===========
``TOSKIPORNOT_DATA``        Images, masks, perturbed test sets        ``<repo>/data``
``TOSKIPORNOT_CHECKPOINTS`` Trained model weights                    ``<repo>/checkpoints``
``TOSKIPORNOT_RESULTS``     Metrics CSVs, figures, videos            ``<repo>/results``
``TOSKIPORNOT_ARCHIVE``     Zip archive to fetch data/weights from   unset
==========================  =======================================  ===========

So a collaborator who receives the zip files only has to write, for example::

    TOSKIPORNOT_ARCHIVE=/Volumes/Shared/2024-11-CIBM/artifacts

and run ``scripts/fetch_artifacts.sh``; or, if they unpacked the archives
somewhere else already::

    TOSKIPORNOT_DATA=/mnt/big-disk/toskipornot/data
    TOSKIPORNOT_CHECKPOINTS=/mnt/big-disk/toskipornot/checkpoints

Usage::

    from toskipornot.config import DATA_DIR, CHECKPOINTS_DIR, data_path

    img_dir = data_path("BUSI-experiment", "in-domain", "test", "image")
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # optional, and only used to populate os.environ
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv is declared but keep this usable
    pass


def _resolve(var, default):
    value = os.environ.get(var)
    if not value:
        return default
    return Path(value).expanduser().resolve()


DATA_DIR = _resolve("TOSKIPORNOT_DATA", REPO_ROOT / "data")
CHECKPOINTS_DIR = _resolve("TOSKIPORNOT_CHECKPOINTS", REPO_ROOT / "checkpoints")
RESULTS_DIR = _resolve("TOSKIPORNOT_RESULTS", REPO_ROOT / "results")

# Only needed by scripts/fetch_artifacts.sh; None when unset.
ARCHIVE_DIR = _resolve("TOSKIPORNOT_ARCHIVE", None)


def data_path(*parts):
    """Path under the data directory."""
    return DATA_DIR.joinpath(*parts)


def checkpoint_path(*parts):
    """Path under the checkpoints directory."""
    return CHECKPOINTS_DIR.joinpath(*parts)


def results_path(*parts):
    """Path under the results directory."""
    return RESULTS_DIR.joinpath(*parts)


def require(path, hint="see ARTIFACTS.md and scripts/fetch_artifacts.sh"):
    """Fail with an actionable message instead of an obscure empty-glob error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing required path: {p}\n  {hint}")
    return p


def describe():
    """One line per configured location, for logs and troubleshooting."""
    return "\n".join(
        [
            f"repo        {REPO_ROOT}",
            f"data        {DATA_DIR}",
            f"checkpoints {CHECKPOINTS_DIR}",
            f"results     {RESULTS_DIR}",
            f"archive     {ARCHIVE_DIR if ARCHIVE_DIR else '(unset)'}",
        ]
    )


if __name__ == "__main__":
    print(describe())
