#!/usr/bin/env bash
set -euo pipefail
# Skip if DSing (already vocals-only)
if [ -d "data/raw/dsing" ]; then
  echo "DSing detected: skipping demucs"
  cp -r data/raw/dsing data/interim/
else
  python -m autolyrics.data.isolate \
    --in data/raw/jamendo \
    --out data/interim/jamendo_vocals \
    --device cuda
fi
