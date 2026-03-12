#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS_NAME=$(uname)

if [ "$OS_NAME" = "Darwin" ]; then
  PYTHON_BIN="python3"
elif [ "$OS_NAME" = "Linux" ]; then
  PYTHON_BIN="python3"
else
    PYTHON_BIN=".venv/bin/python"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean CSFC.spec

echo
echo "Build complete."
echo "Executable: $SCRIPT_DIR/dist/CSFC"
