# Author: ramduvvuri
# File: 01_download_data.sh

#!/usr/bin/env bash
set -euo pipefail
python -m autolyrics.data.download --source dsing --out data/raw/dsing || \
python -m autolyrics.data.download --source jamendo --out data/raw/jamendo