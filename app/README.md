
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
  pip install pyinstaller
  pyinstaller --onefile --windowed --name CSFC main.py
