#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean CSFC.spec

echo
echo "Build complete."
echo "Executable: $SCRIPT_DIR/dist/CSFC"
