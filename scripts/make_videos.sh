#!/usr/bin/env bash
# Rebuild every demonstration video loop.
#
# Requires the inputs and checkpoints in place (scripts/fetch_artifacts.sh) plus
# ffmpeg on PATH. Predictions are cached under results/video-cache, so re-running
# to change only the timing or encoding does not repeat inference.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

# Archive quality into results/videos, then smaller copies for the repo.
ARCHIVE_CRF="${ARCHIVE_CRF:-20}"
REPO_CRF="${REPO_CRF:-30}"

"$PY" -m toskipornot.visualization.make_robustness_video \
      --dataset all --crf "$ARCHIVE_CRF"
"$PY" -m toskipornot.visualization.make_synthetic_video \
      --scenario both --crf "$ARCHIVE_CRF"

# Both passes leave their frames behind, so the repo-sized copies are just a
# second encode of the same frames -- no re-rendering, no repeated inference.
# Seconds-per-frame must match the defaults each script used.
mkdir -p docs/videos
reencode() { # frames_dir  out  seconds_per_frame
  ffmpeg -y -loglevel error \
    -framerate "$("$PY" -c "print(1/$3)")" \
    -i "results/videos/$1/frame_%04d.png" \
    -vf "fps=25,pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p -crf "$REPO_CRF" \
    -tune stillimage -movflags +faststart "$2"
}
for d in busi glas heart spleen; do
  reencode "frames_$d" "docs/videos/robustness_$d.mp4" 0.9
done
for s in background foreground; do
  reencode "frames_synthetic_$s" "docs/videos/robustness_synthetic_$s.mp4" 0.42
done

# Frames are large and fully regenerable from the cached predictions.
rm -rf results/videos/frames_*

echo
echo "repo copies:"
ls -la docs/videos/*.mp4
echo "archive copies:"
ls -la results/videos/*.mp4
