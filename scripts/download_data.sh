#!/usr/bin/env bash
# Download the training data that does not ship in git, from the v2.0 release:
#   data/reorient_demos/      SO(3) expert demonstrations for the reverse curriculum (163 MB)
#   ckpts/*/*_models.pt       the world models of the released world-model actors, to fine-tune them (33 MB)
#   bash scripts/download_data.sh [reorient_demos] [wm_models]     (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${IHR_RELEASE_URL:-https://github.com/Msornerrrr/in-hand-rotation-mjlab/releases/download/v2.0}
PARTS=("$@")
[ ${#PARTS[@]} -gt 0 ] || PARTS=(reorient_demos wm_models)
for p in "${PARTS[@]}"; do
  f=$p.tar.gz
  if [ ! -f "$f" ]; then
    echo "[download] $URL/$f"
    curl -L --fail --progress-bar -o "$f.part" "$URL/$f"
    mv "$f.part" "$f"
  fi
  tar -xzf "$f"
  echo "[download] $p extracted"
done
