#!/usr/bin/env bash
# Unpack the inputs and checkpoints needed to reproduce the demonstration videos
# from the artifact archive, then release the archive copies again so local disk
# is not left full.
#
# Configure with environment variables or a .env file in the repository root
# (see .env.example); nothing is hardcoded:
#
#   TOSKIPORNOT_ARCHIVE      where the zip files are            (required here)
#   TOSKIPORNOT_DATA         where to unpack images/masks       (default <repo>/data)
#   TOSKIPORNOT_CHECKPOINTS  where to unpack model weights      (default <repo>/checkpoints)
#
# Example:
#   TOSKIPORNOT_ARCHIVE=/Volumes/Share/2024-11-CIBM/artifacts scripts/fetch_artifacts.sh
#
# See ARTIFACTS.md for the archive layout and what each zip contains.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

# Single source of truth for every location: toskipornot/config.py.
eval "$("$PY" - <<'EOF'
from toskipornot import config
print(f'ARCHIVE={config.ARCHIVE_DIR or ""}')
print(f'DATA_DIR="{config.DATA_DIR}"')
print(f'CKPT_DIR="{config.CHECKPOINTS_DIR}"')
print(f'RESULTS_DIR="{config.RESULTS_DIR}"')
EOF
)"

# What to fetch. "results" is cheap (~16 MB) and is what the figure and table
# scripts read; "inputs" and "checkpoints" are large and only needed to re-run
# inference or rebuild the videos.
WHAT="${WHAT:-all}"
case "$WHAT" in
  all|results|models) ;;
  *) echo "error: WHAT must be one of: all, results, models" >&2; exit 1 ;;
esac

# Allow the shorter ARCHIVE=... spelling on the command line too.
ARCHIVE="${ARCHIVE:-}"
if [[ -z "$ARCHIVE" ]]; then
  echo "error: no artifact archive configured." >&2
  echo "  set TOSKIPORNOT_ARCHIVE (or ARCHIVE) to the directory holding" >&2
  echo "  inputs/ and checkpoints/, e.g." >&2
  echo "    TOSKIPORNOT_ARCHIVE=~/Documents/.../2024-11-CIBM/artifacts $0" >&2
  echo "  or copy .env.example to .env and edit it. See ARTIFACTS.md." >&2
  exit 1
fi
if [[ ! -d "$ARCHIVE" ]]; then
  echo "error: archive directory not found: $ARCHIVE" >&2
  exit 1
fi

echo "archive     $ARCHIVE"
echo "data        $DATA_DIR"
echo "checkpoints $CKPT_DIR"
echo

# Files synced by a cloud provider may be placeholders; materialise on demand and
# release afterwards. Harmless no-ops for a plain local directory.
ON_CLOUD=0
command -v brctl >/dev/null 2>&1 && ON_CLOUD=1

materialise() {
  local f="$1"
  [[ -f "$f" ]] || { echo "error: missing archive file: $f" >&2; exit 1; }
  [[ $ON_CLOUD -eq 1 ]] || return 0
  ls -lO "$f" 2>/dev/null | grep -q dataless || return 0
  echo "  downloading $(basename "$f") ..."
  brctl download "$f"
  for _ in $(seq 1 240); do
    ls -lO "$f" | grep -q dataless || return 0
    sleep 5
  done
  echo "error: timed out downloading $f" >&2
  exit 1
}

release() {
  [[ $ON_CLOUD -eq 1 ]] || return 0
  brctl evict "$1" >/dev/null 2>&1 || true
}

echo "==> results (per-image metrics for every scenario, plus rendered figures)"
for part in metrics-medical metrics-synthetic metrics-train-time figures; do
  materialise "$ARCHIVE/results/$part.zip"
  ZIP="$ARCHIVE/results/$part.zip" DEST="$RESULTS_DIR" "$PY" - <<'EOF'
import os, zipfile
dest = os.environ["DEST"]
z = zipfile.ZipFile(os.environ["ZIP"])
names = [n for n in z.namelist() if not n.endswith("/")]
for n in names:
    target = os.path.join(dest, n)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with z.open(n) as src, open(target, "wb") as out:
        out.write(src.read())
print(f"  {os.path.basename(os.environ['ZIP'])}: {len(names)} files")
EOF
  release "$ARCHIVE/results/$part.zip"
done

if [[ "$WHAT" == "results" ]]; then
  echo
  echo "done (results only). Re-run with WHAT=all for inputs and checkpoints."
  exit 0
fi

echo "==> inputs (test images, masks and the five perturbed sets per dataset)"
materialise "$ARCHIVE/inputs/all-datasets.zip"
ZIP="$ARCHIVE/inputs/all-datasets.zip" DEST="$DATA_DIR" "$PY" - <<'EOF'
import os, zipfile
# Only the perturbed experiment trees, the processed sets used by the histogram
# figures, the synthetic sets and the source textures are needed.
keep = ("-experiment/", "-processed/", "data/raw/", "data/textures/")
dest = os.environ["DEST"]
z = zipfile.ZipFile(os.environ["ZIP"])
sel = [i for i in z.infolist()
       if not i.filename.startswith("__MACOSX")
       and not i.filename.endswith("/")
       and any(k in i.filename for k in keep)]
print(f"  extracting {len(sel)} entries "
      f"({sum(i.file_size for i in sel)/1e6:.0f} MB) -> {dest}")
for i in sel:
    # Strip the archive's leading "data/" so the tree lands directly in DEST.
    rel = i.filename.split("/", 1)[1] if i.filename.startswith("data/") else i.filename
    target = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with z.open(i) as src, open(target, "wb") as out:
        out.write(src.read())
EOF
release "$ARCHIVE/inputs/all-datasets.zip"

echo "==> checkpoints (seed 1 for the medical sets, alpha=0.10 for synthetic)"
for ds in busi glas heart spleen; do
  materialise "$ARCHIVE/checkpoints/$ds.zip"
  ZIP="$ARCHIVE/checkpoints/$ds.zip" DEST="$CKPT_DIR/$ds" "$PY" - <<'EOF'
import os, zipfile
want = {f"{a}_256_1" for a in
        ("UNet", "AttentionUNet", "UNet++", "NoSkipUNet", "VNet", "NoSkipVNet")}
dest = os.environ["DEST"]
z = zipfile.ZipFile(os.environ["ZIP"])
n = 0
for name in z.namelist():
    if name.startswith("__MACOSX") or name.endswith("/"):
        continue
    parts = name.split("/")
    if len(parts) >= 3 and parts[1] in want and (
            name.endswith(".pth") or name.endswith("config.json")):
        # Drop only the archive's top-level directory (which still carries a
        # legacy version name) and keep the run directory.
        target = os.path.join(dest, *parts[1:])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with z.open(name) as src, open(target, "wb") as out:
            out.write(src.read())
        n += 1
print(f"  {os.path.basename(dest)}: {n} files")
EOF
  release "$ARCHIVE/checkpoints/$ds.zip"
done

for scen in background foreground; do
  materialise "$ARCHIVE/checkpoints/synthetic-$scen.zip"
  ZIP="$ARCHIVE/checkpoints/synthetic-$scen.zip" \
  DEST="$CKPT_DIR/synthetic-$scen" "$PY" - <<'EOF'
import os, zipfile
dest = os.environ["DEST"]
z = zipfile.ZipFile(os.environ["ZIP"])
n = 0
for name in z.namelist():
    if name.startswith("__MACOSX") or name.endswith("/"):
        continue
    if "alphablend_0p10_normal_seed_1/" in name and (
            name.endswith(".pth") or name.endswith("config.json")):
        target = os.path.join(dest, *name.split("/")[1:])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with z.open(name) as src, open(target, "wb") as out:
            out.write(src.read())
        n += 1
print(f"  {os.path.basename(dest)}: {n} files")
EOF
  release "$ARCHIVE/checkpoints/synthetic-$scen.zip"
done

echo
echo "done."
du -sh "$DATA_DIR" "$CKPT_DIR" 2>/dev/null || true
