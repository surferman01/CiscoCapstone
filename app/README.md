
Desktop (no web server) app with:
- **Tabbed dashboard** (Overview, Data Table, Feature Importance, Metrics, Confusion Matrix, ROC, Bins)
- **Fixed-height charts** so the layout never squishes
- **CatBoost** classifier pipeline: train/test split, metrics, confusion matrix, ROC, feature importance

Run (dev):
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # macOS/Linux: source .venv/bin/activate
  pip install -r requirements.txt
  python main.py

Build single executable:
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # macOS/Linux: source .venv/bin/activate
  pip install -r requirements.txt
  ./build_executable.sh

Build output:
  dist/CSFC
  # Windows: dist\CSFC.exe

Notes:
  - The build uses PyInstaller with CSFC.spec so the logo, QSS theme files,
    and native ML libraries are bundled into a single file.
  - Saved runs are stored in the user's app-data directory, not beside the
    executable, so the packaged app can run from a normal installed location.
  - On Linux, PySide6 still depends on system Qt/XCB libraries. If the
    packaged app fails to start, install the missing `libxcb-*` and
    `libxkbcommon-x11` packages on the build machine and rebuild.
