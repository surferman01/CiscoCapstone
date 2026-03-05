# main.py
import sys, os
import re
import json
import html
import io
import base64
from datetime import datetime
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QThreadPool, QRunnable, QObject

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from analysis import run_analysis, run_analysis_with_artifact, save_model_artifact
from widgets import DropZone, ClickTile


# -----------------------------
# Weak (safe) metadata detection (UI-side suggestions)
# -----------------------------
_ID_NAME_RE = re.compile(
    r"(serial|serno|s/n|sn\b|lot|wafer|unit|device|die|barcode|uuid|guid|hash|mac|imei|imsi|ip\b|hostname|name\b|id\b|identifier|index)",
    re.IGNORECASE,
)
_DATE_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)


def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "run"


def _df_to_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    use = df if max_rows is None else df.head(max_rows)
    out = use.where(pd.notna(use), None).to_dict(orient="records")
    return out


def _results_to_saved_payload(results: dict) -> dict:
    bins_df = results.get("bins", pd.DataFrame())
    shap_imp = results.get("shap_importance", {}) or {}
    shap_json = {}
    if isinstance(shap_imp, dict):
        for k, v in shap_imp.items():
            if isinstance(v, pd.DataFrame):
                shap_json[str(k)] = _df_to_records(v, max_rows=500)

    roc_json = []
    for r in results.get("roc", []) or []:
        roc_json.append(
            {
                "class": str(r.get("class", "")),
                "fpr": [float(x) for x in list(r.get("fpr", []))],
                "tpr": [float(x) for x in list(r.get("tpr", []))],
                "auc": float(r.get("auc", 0.0)),
            }
        )

    metrics = {
        str(k): float(v) if isinstance(v, (int, float)) else v
        for k, v in (results.get("metrics", {}) or {}).items()
    }

    return {
        "model_name": str(results.get("model_name", "")),
        "metrics": metrics,
        "classification_report": results.get("classification_report", {}) or {},
        "classification_report_text": results.get("classification_report_text", ""),
        "meta": results.get("meta", {}) or {},
        "bins": _df_to_records(bins_df, max_rows=None),
        "roc": roc_json,
        "shap_importance": shap_json,
        "visual_plots": results.get("visual_plots", {}) or {},
        # Dashboard table displays up to 1000 rows anyway.
        "dataframe": _df_to_records(results.get("dataframe", pd.DataFrame()), max_rows=1000),
    }


def _saved_payload_to_results(payload: dict) -> dict:
    shap_out = {}
    for k, rows in (payload.get("shap_importance", {}) or {}).items():
        shap_out[str(k)] = pd.DataFrame(rows or [])

    return {
        "model_name": payload.get("model_name", "Saved Run"),
        "metrics": payload.get("metrics", {}) or {},
        "classification_report": payload.get("classification_report", {}) or {},
        "classification_report_text": payload.get("classification_report_text", ""),
        "meta": payload.get("meta", {}) or {},
        "bins": pd.DataFrame(payload.get("bins", []) or []),
        "roc": payload.get("roc", []) or [],
        "shap_importance": shap_out,
        "visual_plots": payload.get("visual_plots", {}) or {},
        "dataframe": pd.DataFrame(payload.get("dataframe", []) or []),
    }


def suggest_drop_columns_weak(df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n == 0:
        return []

    drops = set()
    for c in df.columns:
        name = str(c)

        if _ID_NAME_RE.search(name) or _DATE_NAME_RE.search(name):
            drops.add(c)
            continue

        s = df[c]

        if pd.api.types.is_datetime64_any_dtype(s):
            drops.add(c)
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            if len(non_null) < max(10, int(0.1 * n)):
                continue

            uniq_ratio = non_null.nunique(dropna=True) / max(1, len(non_null))
            num_parse = pd.to_numeric(non_null, errors="coerce")
            numericish_ratio = float(num_parse.notna().mean())

            if uniq_ratio > 0.95 and numericish_ratio < 0.20:
                drops.add(c)

    return sorted(drops)


# -----------------------------
# Column drop selection dialog
# -----------------------------
class ColumnDropDialog(QtWidgets.QDialog):
    def __init__(self, parent, all_columns: list[str], prechecked: set[str]):
        super().__init__(parent)
        self.setWindowTitle("Review columns to exclude from training")
        self.resize(640, 520)

        root = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Checked columns will be EXCLUDED from model training features.\n"
            "Uncheck any column you want to KEEP as a measurement feature."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search columns…")
        root.addWidget(self.search)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        root.addWidget(self.list, stretch=1)

        for c in all_columns:
            item = QtWidgets.QListWidgetItem(str(c))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c in prechecked else Qt.Unchecked)
            self.list.addItem(item)

        btn_row = QtWidgets.QHBoxLayout()
        self.checkAllBtn = QtWidgets.QPushButton("Check all")
        self.uncheckAllBtn = QtWidgets.QPushButton("Uncheck all")
        btn_row.addWidget(self.checkAllBtn)
        btn_row.addWidget(self.uncheckAllBtn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        root.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.checkAllBtn.clicked.connect(self._check_all)
        self.uncheckAllBtn.clicked.connect(self._uncheck_all)
        self.search.textChanged.connect(self._filter)

    def _check_all(self):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked)

    def _uncheck_all(self):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Unchecked)

    def _filter(self, text: str):
        text = (text or "").lower().strip()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def selected_drops(self) -> list[str]:
        drops = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.Checked:
                drops.append(item.text())
        return drops


# -----------------------------
# Pandas model for QTableView
# -----------------------------
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None, max_rows: int = 1000):
        super().__init__(parent)
        self._df = df
        self._max = max_rows

    def rowCount(self, parent=None):
        return min(len(self._df), self._max)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        if role == Qt.DisplayRole:
            v = self._df.iat[idx.row(), idx.column()]
            return "" if pd.isna(v) else str(v)

    def headerData(self, s, orient, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[s]) if orient == Qt.Horizontal else str(s)


# -----------------------------
# Worker / ThreadPool
# -----------------------------
class WorkerSignals(QObject):
    finished = Signal(object)
    status = Signal(str)


class TrainWorker(QRunnable):
    def __init__(self, data_path: str, config: dict):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.status.emit("Starting training...")
            res = run_analysis(self.data_path, self.config)
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.finished.emit(e)


class AnalyzeWorker(QRunnable):
    def __init__(self, data_path: str, artifact_path: str):
        super().__init__()
        self.data_path = data_path
        self.artifact_path = artifact_path
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.status.emit("Running analysis with saved model...")
            res = run_analysis_with_artifact(self.data_path, self.artifact_path, {})
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.finished.emit(e)


# -----------------------------
# Splash Page
# -----------------------------
class SplashPage(QtWidgets.QWidget):
    requestTrain = Signal(str, dict)
    requestLoadTrained = Signal(str)
    requestViewSavedRuns = Signal()
    requestExportSavedRun = Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Cisco Silicon Failure Characterization")
        title.setObjectName("titleBar")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        logo = QtWidgets.QLabel(alignment=Qt.AlignCenter)
        if os.path.exists(logo_path):
            pm = QtGui.QPixmap(logo_path).scaledToWidth(320, Qt.SmoothTransformation)
            logo.setPixmap(pm)
        else:
            logo.setObjectName("heroLogo")
            logo.setText("cisco")
        layout.addWidget(logo)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(18)
        layout.addLayout(columns, stretch=1)

        def make_column(title_text: str) -> QtWidgets.QGroupBox:
            box = QtWidgets.QGroupBox(title_text)
            box.setObjectName("columnBox")
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(14, 14, 14, 14)
            v.setSpacing(12)
            return box

        # Left: Add Data
        left = make_column("Add Data")
        self.drop = DropZone("Add Data\n(drag & drop)")
        self.drop.setMinimumHeight(180)
        self.drop.setMaximumHeight(220)

        self.browseBtn = QtWidgets.QPushButton("Browse Files…")
        self.browseBtn.setMinimumHeight(40)

        self.selectedLabel = QtWidgets.QLabel("")
        self.selectedLabel.setWordWrap(True)
        self.selectedLabel.setProperty("muted", True)

        self.dropSummary = QtWidgets.QLabel("Drop columns: —")
        self.dropSummary.setWordWrap(True)
        self.dropSummary.setProperty("muted", True)

        self.reviewDropsBtn = QtWidgets.QPushButton("Review Drops…")
        self.reviewDropsBtn.setMinimumHeight(36)
        self.reviewDropsBtn.setEnabled(False)

        orLbl = QtWidgets.QLabel("— or —")
        orLbl.setAlignment(Qt.AlignCenter)
        orLbl.setProperty("muted", True)

        left.layout().addWidget(self.drop)
        left.layout().addWidget(orLbl)
        left.layout().addWidget(self.browseBtn)
        left.layout().addWidget(self.selectedLabel)
        left.layout().addWidget(self.dropSummary)
        left.layout().addWidget(self.reviewDropsBtn)
        left.layout().addStretch(1)
        columns.addWidget(left, 1)

        # Middle: Training
        middle = make_column("Training")

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["-- select --", "CatBoost", "XGBoost"])
        self.combo.setMinimumHeight(36)

        self.gpuCheck = QtWidgets.QCheckBox("Use GPU (if available)")
        self.gpuCheck.setChecked(True)

        self.targetBox = QtWidgets.QGroupBox("Target Column")
        tv = QtWidgets.QVBoxLayout(self.targetBox)

        self.targetCombo = QtWidgets.QComboBox()
        self.targetCombo.setEditable(True)
        self.targetCombo.setMinimumHeight(34)
        if self.targetCombo.lineEdit() is not None:
            self.targetCombo.lineEdit().setPlaceholderText(
                "Select a recommended target or type your own..."
            )

        self.targetHint = QtWidgets.QLabel(
            "Upload a dataset to see recommended target columns."
        )
        self.targetHint.setWordWrap(True)
        self.targetHint.setProperty("muted", True)

        tv.addWidget(self.targetCombo)
        tv.addWidget(self.targetHint)

        self.trainBtn = QtWidgets.QPushButton("Train")
        self.trainBtn.setMinimumHeight(44)
        self.trainBtn.setEnabled(False)

        middle.layout().addWidget(QtWidgets.QLabel("Model"))
        middle.layout().addWidget(self.combo)
        middle.layout().addWidget(self.gpuCheck)
        middle.layout().addWidget(self.targetBox)
        middle.layout().addSpacing(8)
        middle.layout().addWidget(self.trainBtn)
        middle.layout().addStretch(1)
        columns.addWidget(middle, 1)

        # Right: Already trained
        right = make_column("Already Trained?")
        hint = QtWidgets.QLabel("(insert file)")
        hint.setAlignment(Qt.AlignCenter)
        hint.setProperty("muted", True)

        self.loadTileBtn = QtWidgets.QPushButton("Browse Trained Artifact…")
        self.loadTileBtn.setMinimumHeight(40)
        self.viewSavedRunsBtn = QtWidgets.QPushButton("View Saved Runs…")
        self.viewSavedRunsBtn.setMinimumHeight(36)
        self.exportSavedRunBtn = QtWidgets.QPushButton("Export Saved Run…")
        self.exportSavedRunBtn.setMinimumHeight(36)

        self.modelLabel = QtWidgets.QLabel("")
        self.modelLabel.setWordWrap(True)
        self.modelLabel.setProperty("muted", True)

        right.layout().addWidget(hint)
        right.layout().addSpacing(8)
        right.layout().addWidget(self.loadTileBtn)
        right.layout().addWidget(self.viewSavedRunsBtn)
        right.layout().addWidget(self.exportSavedRunBtn)
        right.layout().addWidget(self.modelLabel)
        right.layout().addStretch(1)
        columns.addWidget(right, 1)

        columns.setStretch(0, 1)
        columns.setStretch(1, 1)
        columns.setStretch(2, 1)

        # State
        self.data_path = None
        self.recommended_targets: list[str] = []
        self.all_columns: list[str] = []
        self.suggested_drop_cols: list[str] = []
        self.user_drop_cols: list[str] = []

        # Connections
        self.drop.fileDropped.connect(self._on_file_dropped)
        self.browseBtn.clicked.connect(self._on_browse_clicked)
        self.reviewDropsBtn.clicked.connect(self._on_review_drops)

        self.combo.currentIndexChanged.connect(self._update_train_enabled)
        self.targetCombo.currentTextChanged.connect(self._update_train_enabled)
        self.trainBtn.clicked.connect(self._on_train_click)

        self.loadTileBtn.clicked.connect(self._on_load_trained_clicked)
        self.viewSavedRunsBtn.clicked.connect(lambda: self.requestViewSavedRuns.emit())
        self.exportSavedRunBtn.clicked.connect(
            lambda: self.requestExportSavedRun.emit()
        )

        self._update_train_enabled()

    def _recommend_targets(self, df: pd.DataFrame, max_cols: int = 10) -> list[str]:
        candidates = []
        # non-fully-numeric columns
        for c in df.columns:
            s = df[c]
            if s.dropna().empty:
                continue
            all_numeric = pd.to_numeric(s, errors="coerce").notna().all()
            if not all_numeric:
                candidates.append(c)

        # low-cardinality numeric columns
        for c in df.columns:
            if c in candidates:
                continue
            s = df[c].dropna()
            if s.empty:
                continue
            all_numeric = pd.to_numeric(s, errors="coerce").notna().all()
            if all_numeric and s.nunique() <= 20:
                candidates.append(c)

        return candidates[:max_cols]

    def _set_data_path(self, path: str):
        self.data_path = path
        self.selectedLabel.setText(f"Selected: {os.path.basename(path)}")

        try:
            if path.lower().endswith(".csv"):
                df_preview = pd.read_csv(path, nrows=5000)
            else:
                df_preview = pd.read_parquet(path).head(5000)

            self.all_columns = list(df_preview.columns)

            recs = self._recommend_targets(df_preview)
            self.recommended_targets = recs[:]

            self.targetCombo.blockSignals(True)
            self.targetCombo.clear()
            if recs:
                for c in recs:
                    self.targetCombo.addItem(c)
                self.targetCombo.setCurrentIndex(0)
                self.targetHint.setText(
                    "Recommended targets loaded. Select one or type your own."
                )
            else:
                self.targetHint.setText(
                    "No obvious targets detected. Type a column name manually."
                )
            self.targetCombo.blockSignals(False)

            self.suggested_drop_cols = suggest_drop_columns_weak(df_preview)
            self.user_drop_cols = list(self.suggested_drop_cols)

            self.dropSummary.setText(
                f"Drop columns: {len(self.user_drop_cols)} (click Review Drops…)"
            )
            self.reviewDropsBtn.setEnabled(True)

        except Exception as e:
            self.targetHint.setText(f"Could not parse columns: {e}")
            self.all_columns = []
            self.recommended_targets = []
            self.suggested_drop_cols = []
            self.user_drop_cols = []
            self.dropSummary.setText("Drop columns: —")
            self.reviewDropsBtn.setEnabled(False)

        self._update_train_enabled()

    def _on_review_drops(self):
        if not self.all_columns:
            return
        prechecked = set(self.user_drop_cols or self.suggested_drop_cols or [])
        dlg = ColumnDropDialog(self, self.all_columns, prechecked)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.user_drop_cols = dlg.selected_drops()
            self.dropSummary.setText(
                f"Drop columns: {len(self.user_drop_cols)} (customized)"
            )

    def _on_file_dropped(self, path: str):
        if path:
            self._set_data_path(path)

    def _on_browse_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose data file",
            "",
            "Data Files (*.csv *.parquet *.pq);;CSV (*.csv);;Parquet (*.parquet *.pq);;All Files (*.*)",
        )
        if path:
            self._set_data_path(path)

    def _update_train_enabled(self):
        valid_model = self.combo.currentText() in {"CatBoost", "XGBoost"}
        valid_target = bool(self.targetCombo.currentText().strip())
        self.trainBtn.setEnabled(bool(self.data_path) and valid_model and valid_target)

    def _on_train_click(self):
        if not self.data_path:
            QtWidgets.QMessageBox.information(
                self, "No data", "Choose a dataset first."
            )
            return

        model_type = self.combo.currentText()
        if model_type not in {"CatBoost", "XGBoost"}:
            QtWidgets.QMessageBox.information(
                self, "Choose model", "Please select CatBoost or XGBoost."
            )
            return

        target_col = self.targetCombo.currentText().strip()
        if not target_col:
            QtWidgets.QMessageBox.information(
                self, "Target required", "Please select or type a target column."
            )
            return

        # quick validation (preview)
        try:
            if self.data_path.lower().endswith(".csv"):
                df_check = pd.read_csv(self.data_path, nrows=5000)
            else:
                df_check = pd.read_parquet(self.data_path).head(5000)

            if target_col not in df_check.columns:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid target",
                    f"Column '{target_col}' not found in dataset.",
                )
                return

            if df_check[target_col].dropna().nunique() < 2:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid target",
                    f"Target '{target_col}' has < 2 unique values.",
                )
                return

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Validation failed", str(e))
            return

        recs = list(self.recommended_targets or [])
        user_drops = list(self.user_drop_cols or [])

        # Always exclude other recommended targets (except chosen target)
        rec_excludes = [c for c in recs if c != target_col]

        # Merge + never drop chosen target
        exclude_cols = sorted(set(user_drops) | set(rec_excludes))
        exclude_cols = [c for c in exclude_cols if c != target_col]

        config = {
            "model_type": model_type,
            "use_gpu": self.gpuCheck.isChecked(),
            "target_column": target_col,
            "recommended_targets": recs,
            "exclude_columns": exclude_cols,
        }

        self.requestTrain.emit(self.data_path, config)

    def _on_load_trained_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open trained artifact", "", "Model Artifact (*.pkl);;All Files (*.*)"
        )
        if path:
            self.modelLabel.setText(f"Selected: {os.path.basename(path)}")
            self.requestLoadTrained.emit(path)


# -----------------------------
# Training Page
# -----------------------------
class TrainingPage(QtWidgets.QWidget):
    cancelRequested = Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(
            "Cisco Silicon Failure Characterization", alignment=Qt.AlignCenter
        )
        title.setObjectName("titleBar")
        layout.addWidget(title)

        center = QtWidgets.QVBoxLayout()
        msg = QtWidgets.QLabel("Training / Analyzing...", alignment=Qt.AlignCenter)
        f = msg.font()
        f.setPointSize(26)
        msg.setFont(f)
        center.addWidget(msg, alignment=Qt.AlignCenter)

        h = QtWidgets.QHBoxLayout()
        self.spinner = QtWidgets.QProgressBar()
        self.spinner.setRange(0, 0)
        self.spinner.setMaximumWidth(280)
        h.addStretch()
        h.addWidget(self.spinner)
        h.addStretch()
        center.addLayout(h)

        layout.addLayout(center, stretch=1)

        self.modifyBtn = QtWidgets.QPushButton("modify data")
        self.modifyBtn.clicked.connect(lambda: self.cancelRequested.emit())
        layout.addWidget(self.modifyBtn, alignment=Qt.AlignRight)


# -----------------------------
# Dashboard UI widgets
# -----------------------------
class KPIBox(QtWidgets.QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.setObjectName("kpiBox")
        self.setMinimumHeight(110)
        self.setMaximumHeight(130)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        self.value = QtWidgets.QLabel("—", alignment=Qt.AlignCenter)
        f = self.value.font()
        f.setPointSize(16)
        f.setBold(True)
        self.value.setFont(f)

        v.addStretch(1)
        v.addWidget(self.value)
        v.addStretch(1)


class ChartCard(QtWidgets.QGroupBox):
    def __init__(self, title: str, height: int = 380):
        super().__init__(title)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(12, 12, 12, 12)
        self.layout().setSpacing(8)

        self.canvas = FigureCanvas(Figure(figsize=(6, 3), dpi=100))
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.ax = self.canvas.figure.add_subplot(111)
        self.layout().addWidget(self.canvas)

        self.setMinimumHeight(height)
        self.setMaximumHeight(height)


# -----------------------------
# Dashboard Tabs (3 tabs only)
# -----------------------------
class DashboardTabs(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()

        self.dashboard = QtWidgets.QWidget()
        self.tablePage = QtWidgets.QWidget()
        self.fiPage = QtWidgets.QWidget()

        self.addTab(self.dashboard, "Dashboard")
        self.addTab(self.tablePage, "Data Table")
        self.addTab(self.fiPage, "Feature Importance")

        # Dashboard tab
        d = QtWidgets.QVBoxLayout(self.dashboard)
        d.setContentsMargins(12, 12, 12, 12)
        d.setSpacing(16)

        kpi_row = QtWidgets.QHBoxLayout()
        kpi_row.setSpacing(16)

        self.kpiAccuracy = KPIBox("Accuracy")
        self.kpiPrecision = KPIBox("Precision (w)")
        self.kpiRecall = KPIBox("Recall (w)")
        self.kpiF1 = KPIBox("F1 (w)")
        self.kpiModel = KPIBox("Model")
        self.kpiTarget = KPIBox("Target")
        self.kpiClasses = KPIBox("Classes")

        kpi_row.addWidget(self.kpiAccuracy, 1)
        kpi_row.addWidget(self.kpiPrecision, 1)
        kpi_row.addWidget(self.kpiRecall, 1)
        kpi_row.addWidget(self.kpiF1, 1)
        kpi_row.addWidget(self.kpiModel, 1)
        kpi_row.addWidget(self.kpiTarget, 1)
        kpi_row.addWidget(self.kpiClasses, 1)

        kpi_wrap = QtWidgets.QWidget()
        kpi_wrap.setLayout(kpi_row)
        kpi_wrap.setMinimumHeight(130)
        kpi_wrap.setMaximumHeight(140)
        d.addWidget(kpi_wrap)

        self.ovBar = ChartCard("Class Distribution (Test)", height=250)
        self.ovRoc = ChartCard("ROC Curves (Preview)", height=250)
        self.ovTopHist = ChartCard(
            "Top Feature Histograms (PASS vs FAIL, Top 5 SHAP)", height=250
        )
        self.ovProbBox = ChartCard(
            "Model Probability by Group (PASS/FAIL)", height=250
        )

        charts_grid = QtWidgets.QGridLayout()
        charts_grid.setHorizontalSpacing(16)
        charts_grid.setVerticalSpacing(16)
        charts_grid.addWidget(self.ovBar, 0, 0)
        charts_grid.addWidget(self.ovRoc, 0, 1)
        charts_grid.addWidget(self.ovTopHist, 1, 0)
        charts_grid.addWidget(self.ovProbBox, 1, 1)
        charts_grid.setColumnStretch(0, 1)
        charts_grid.setColumnStretch(1, 1)

        charts_wrap = QtWidgets.QWidget()
        charts_wrap.setLayout(charts_grid)
        charts_wrap.setMinimumHeight(530)
        charts_wrap.setMaximumHeight(530)
        d.addWidget(charts_wrap)

        self.perClassCard = QtWidgets.QGroupBox("Per-Class Metrics")
        self.perClassCard.setMinimumHeight(220)
        self.perClassCard.setMaximumHeight(240)
        per_v = QtWidgets.QVBoxLayout(self.perClassCard)
        self.perClassTable = QtWidgets.QTableWidget()
        self.perClassTable.setColumnCount(5)
        self.perClassTable.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1", "Support"]
        )
        self.perClassTable.verticalHeader().setVisible(False)
        self.perClassTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.perClassTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.perClassTable.setAlternatingRowColors(True)
        self.perClassTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        per_v.addWidget(self.perClassTable)
        d.addWidget(self.perClassCard)

        d.addStretch(1)

        # Data Table tab
        tl = QtWidgets.QVBoxLayout(self.tablePage)
        tl.setContentsMargins(12, 12, 12, 12)
        self.table = QtWidgets.QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        tl.addWidget(self.table)

        # Feature Importance tab (scroll)
        fi_outer = QtWidgets.QVBoxLayout(self.fiPage)
        fi_outer.setContentsMargins(0, 0, 0, 0)

        self.fiScroll = QtWidgets.QScrollArea()
        self.fiScroll.setWidgetResizable(True)
        self.fiScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.fiScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        fi_outer.addWidget(self.fiScroll)

        fi_inner = QtWidgets.QWidget()
        self.fiScroll.setWidget(fi_inner)

        fl = QtWidgets.QVBoxLayout(fi_inner)
        fl.setContentsMargins(12, 12, 12, 12)
        fl.setSpacing(12)

        self.fiHeader = QtWidgets.QLabel(
            "Top 20 features by SHAP (one table per class/bin)."
        )
        self.fiHeader.setWordWrap(True)
        fl.addWidget(self.fiHeader)

        self.fiTablesLayout = QtWidgets.QGridLayout()
        self.fiTablesLayout.setHorizontalSpacing(12)
        self.fiTablesLayout.setVerticalSpacing(12)
        self.fiTablesLayout.setAlignment(Qt.AlignTop)
        fl.addLayout(self.fiTablesLayout)

        fl.addStretch(1)

    def _clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            child = item.layout()
            if w:
                w.deleteLater()
            elif child:
                self._clear_layout(child)

    def populate(self, results: dict):
        # Table
        df = results.get("dataframe", pd.DataFrame())
        self.table.setModel(PandasModel(df))

        meta = results.get("meta", {}) or {}
        metrics = results.get("metrics", {}) or {}
        bins = results.get("bins", pd.DataFrame())
        roc_list = results.get("roc", [])
        visual_plots = results.get("visual_plots", {}) or {}
        class_report = results.get("classification_report", {}) or {}

        # KPIs
        self.kpiAccuracy.value.setText(
            f"{float(metrics.get('Accuracy', 0.0)):.4f}"
            if "Accuracy" in metrics
            else "—"
        )
        self.kpiPrecision.value.setText(
            f"{float(metrics.get('Precision_weighted', 0.0)):.4f}"
            if "Precision_weighted" in metrics
            else "—"
        )
        self.kpiRecall.value.setText(
            f"{float(metrics.get('Recall_weighted', 0.0)):.4f}"
            if "Recall_weighted" in metrics
            else "—"
        )
        self.kpiF1.value.setText(
            f"{float(metrics.get('F1_weighted', 0.0)):.4f}"
            if "F1_weighted" in metrics
            else "—"
        )
        self.kpiModel.value.setText(str(meta.get("model", meta.get("model_name", "—"))))
        self.kpiTarget.value.setText(str(meta.get("target_column", "—")))
        self.kpiClasses.value.setText(str(meta.get("num_classes", "—")))

        # Class distribution
        ax = self.ovBar.ax
        ax.clear()
        if (
            isinstance(bins, pd.DataFrame)
            and not bins.empty
            and {"bin_name", "count"}.issubset(bins.columns)
        ):
            ax.bar(bins["bin_name"].astype(str), bins["count"])
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
        self.ovBar.canvas.figure.tight_layout()
        self.ovBar.canvas.draw()

        # ROC
        ax = self.ovRoc.ax
        ax.clear()
        for r in roc_list[:3]:
            ax.plot(r["fpr"], r["tpr"], label=f"{r['class']} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        if roc_list:
            ax.legend()
        self.ovRoc.canvas.figure.tight_layout()
        self.ovRoc.canvas.draw()

        # Top SHAP feature histograms: PASS vs FAIL
        fig_hist = self.ovTopHist.canvas.figure
        fig_hist.clear()
        hist_items = visual_plots.get("top_shap_hist", []) or []
        if not hist_items:
            axh = fig_hist.add_subplot(111)
            axh.text(
                0.5,
                0.5,
                "No top-feature histogram data available.",
                ha="center",
                va="center",
                transform=axh.transAxes,
            )
            axh.set_axis_off()
        else:
            n = min(5, len(hist_items))
            cols = 3 if n > 3 else n
            rows = 2 if n > 3 else 1
            for i, item in enumerate(hist_items[:5]):
                axh = fig_hist.add_subplot(rows, cols, i + 1)
                pass_vals = item.get("pass_values", []) or []
                fail_vals = item.get("fail_values", []) or []

                if pass_vals:
                    axh.hist(
                        pass_vals,
                        bins=24,
                        alpha=0.55,
                        density=True,
                        color="#2fc1ff",
                        label="PASS",
                    )
                if fail_vals:
                    axh.hist(
                        fail_vals,
                        bins=24,
                        alpha=0.55,
                        density=True,
                        color="#ff7b72",
                        label="FAIL",
                    )

                title = str(item.get("feature", "feature"))
                if len(title) > 26:
                    title = title[:23] + "..."
                axh.set_title(title, fontsize=9)
                if i == 0:
                    axh.legend(fontsize=8)
        fig_hist.tight_layout()
        self.ovTopHist.canvas.draw()

        # Probability boxplot by dataset groups with PASS/FAIL grouping
        fig_box = self.ovProbBox.canvas.figure
        fig_box.clear()
        axb = fig_box.add_subplot(111)
        box_records = visual_plots.get("probability_box", []) or []
        if not box_records:
            axb.text(
                0.5,
                0.5,
                "No probability group data available.",
                ha="center",
                va="center",
                transform=axb.transAxes,
            )
            axb.set_axis_off()
        else:
            bdf = pd.DataFrame(box_records)
            if "group" not in bdf.columns and "cal_group" in bdf.columns:
                bdf["group"] = bdf["cal_group"]
            group_order = bdf["group"].dropna().astype(str).drop_duplicates().tolist()
            if not group_order:
                group_order = sorted(bdf["group"].dropna().astype(str).unique().tolist())

            pos = []
            vals = []
            colors = []
            labels = []
            offset = {"PASS": -0.16, "FAIL": 0.16}
            color_map = {"PASS": "#2fc1ff", "FAIL": "#ff7b72"}

            for i, grp in enumerate(group_order):
                base = i + 1
                for pf in ["PASS", "FAIL"]:
                    arr = (
                        bdf[
                            (bdf["group"] == grp) & (bdf["pass_fail"] == pf)
                        ]["probability"]
                        .dropna()
                        .to_numpy()
                    )
                    if len(arr) == 0:
                        continue
                    vals.append(arr)
                    pos.append(base + offset[pf])
                    colors.append(color_map[pf])
                    labels.append(pf)

            if vals:
                bp = axb.boxplot(
                    vals,
                    positions=pos,
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
                for median in bp["medians"]:
                    median.set_linewidth(1.5)

                axb.set_xticks([i + 1 for i in range(len(group_order))])
                axb.set_xticklabels(group_order)
                axb.set_ylim(0.0, 1.0)
                axb.set_xlabel("Distribution Group")
                axb.set_ylabel("Probability")
                legend_items = [
                    Patch(facecolor="#2fc1ff", alpha=0.6, label="PASS"),
                    Patch(facecolor="#ff7b72", alpha=0.6, label="FAIL"),
                ]
                axb.legend(handles=legend_items, fontsize=8)
            else:
                axb.text(
                    0.5,
                    0.5,
                    "No probability values available for boxplot.",
                    ha="center",
                    va="center",
                    transform=axb.transAxes,
                )
                axb.set_axis_off()
        fig_box.tight_layout()
        self.ovProbBox.canvas.draw()

        # Per-class precision / recall / f1
        class_order = []
        if (
            isinstance(bins, pd.DataFrame)
            and not bins.empty
            and "bin_name" in bins.columns
        ):
            class_order = [str(x) for x in bins["bin_name"].tolist()]
        if not class_order and isinstance(class_report, dict):
            for k, v in class_report.items():
                if isinstance(v, dict) and {"precision", "recall", "f1-score"}.issubset(
                    set(v.keys())
                ):
                    class_order.append(str(k))

        rows = []
        for cls in class_order:
            row = class_report.get(cls, {})
            if not isinstance(row, dict):
                continue
            if not {"precision", "recall", "f1-score"}.issubset(set(row.keys())):
                continue
            rows.append(
                (
                    cls,
                    float(row.get("precision", 0.0)),
                    float(row.get("recall", 0.0)),
                    float(row.get("f1-score", 0.0)),
                    int(row.get("support", 0)),
                )
            )

        self.perClassTable.setRowCount(len(rows))
        for i, (cls, p, r, f1, sup) in enumerate(rows):
            self.perClassTable.setItem(i, 0, QtWidgets.QTableWidgetItem(str(cls)))
            self.perClassTable.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{p:.4f}"))
            self.perClassTable.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{r:.4f}"))
            self.perClassTable.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{f1:.4f}"))
            self.perClassTable.setItem(i, 4, QtWidgets.QTableWidgetItem(str(int(sup))))

        # Feature importance tables
        self._clear_layout(self.fiTablesLayout)
        shap_importance = results.get("shap_importance", {}) or {}
        if not isinstance(shap_importance, dict) or not shap_importance:
            msg = QtWidgets.QLabel("No per-class feature importance available.")
            msg.setProperty("muted", True)
            self.fiTablesLayout.addWidget(msg, 0, 0)
            return

        # order by bins if possible
        if (
            isinstance(bins, pd.DataFrame)
            and not bins.empty
            and "bin_name" in bins.columns
        ):
            class_list = [str(x) for x in bins["bin_name"].tolist()]
        else:
            class_list = [str(k) for k in shap_importance.keys()]

        cols_per_row = 3
        added = 0

        for idx, cls_name in enumerate(class_list):
            df_cls = shap_importance.get(cls_name)
            if df_cls is None:
                df_cls = shap_importance.get(str(cls_name).strip().lower())
            if not (isinstance(df_cls, pd.DataFrame) and not df_cls.empty):
                continue

            group = QtWidgets.QGroupBox(f"{cls_name} – Top 20 Features")
            vbox = QtWidgets.QVBoxLayout(group)

            table = QtWidgets.QTableWidget()
            table.setMinimumHeight(630)
            table.setColumnCount(8)
            table.setHorizontalHeaderLabels(
                [
                    "Rank",
                    "Feature",
                    "SHAP |Δ|",
                    "Share (%)",
                    "Direction",
                    "Failure Avg",
                    "PASS/Other Avg",
                    "PASS/Other Std",
                ]
            )
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            table.setAlternatingRowColors(True)
            header = table.horizontalHeader()
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            header.setStretchLastSection(True)

            top = df_cls.head(20).reset_index(drop=True)
            table.setRowCount(len(top))

            for r, row in top.iterrows():
                table.setItem(
                    r, 0, QtWidgets.QTableWidgetItem(str(int(row.get("rank", r + 1))))
                )
                table.setItem(
                    r, 1, QtWidgets.QTableWidgetItem(str(row.get("feature", "")))
                )
                table.setItem(
                    r,
                    2,
                    QtWidgets.QTableWidgetItem(
                        f"{float(row.get('importance', 0.0)):.4f}"
                    ),
                )
                table.setItem(
                    r,
                    3,
                    QtWidgets.QTableWidgetItem(
                        f"{float(row.get('share_pct', 0.0)):.1f}"
                    ),
                )
                table.setItem(
                    r,
                    4,
                    QtWidgets.QTableWidgetItem(
                        f"{float(row.get('direction', 0.0)):.4f}"
                    ),
                )

                fa = row.get("failure_avg")
                pa = row.get("pass_avg")
                ps = row.get("pass_std")

                table.setItem(
                    r,
                    5,
                    QtWidgets.QTableWidgetItem(
                        "" if pd.isna(fa) else f"{float(fa):.4f}"
                    ),
                )
                table.setItem(
                    r,
                    6,
                    QtWidgets.QTableWidgetItem(
                        "" if pd.isna(pa) else f"{float(pa):.4f}"
                    ),
                )
                table.setItem(
                    r,
                    7,
                    QtWidgets.QTableWidgetItem(
                        "" if pd.isna(ps) else f"{float(ps):.4f}"
                    ),
                )

            vbox.addWidget(table)

            row_i = idx // cols_per_row
            col_i = idx % cols_per_row
            self.fiTablesLayout.addWidget(group, row_i, col_i)
            self.fiTablesLayout.setColumnStretch(col_i, 1)
            added += 1

        if added == 0:
            msg = QtWidgets.QLabel(
                "No per-class feature importance tables were generated."
            )
            msg.setProperty("muted", True)
            self.fiTablesLayout.addWidget(msg, 0, 0)


# -----------------------------
# Dashboard Page
# -----------------------------
class DashboardPage(QtWidgets.QWidget):
    requestModify = Signal()
    requestSave = Signal()
    requestSaveRun = Signal()

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(
            "Cisco Silicon Failure Characterization", alignment=Qt.AlignCenter
        )
        title.setObjectName("titleBar")
        root.addWidget(title)

        self.tabs = DashboardTabs()
        root.addWidget(self.tabs, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self.modifyBtn = QtWidgets.QPushButton("modify data")
        self.modifyBtn.clicked.connect(lambda: self.requestModify.emit())
        self.saveBtn = QtWidgets.QPushButton("save model")
        self.saveBtn.clicked.connect(lambda: self.requestSave.emit())
        self.saveRunBtn = QtWidgets.QPushButton("save run")
        self.saveRunBtn.clicked.connect(lambda: self.requestSaveRun.emit())
        self.readOnlyLabel = QtWidgets.QLabel("Viewing saved run (read-only)")
        self.readOnlyLabel.setProperty("muted", True)
        self.readOnlyLabel.hide()
        actions.addStretch()
        actions.addWidget(self.readOnlyLabel)
        actions.addStretch()
        actions.addWidget(self.modifyBtn)
        actions.addWidget(self.saveRunBtn)
        actions.addWidget(self.saveBtn)
        root.addLayout(actions)

    def populate(self, results: dict):
        self.tabs.populate(results)

    def set_mode(
        self,
        *,
        read_only: bool,
        can_save_model: bool,
        can_save_run: bool,
    ):
        self.readOnlyLabel.setVisible(read_only)
        self.saveBtn.setEnabled(can_save_model and not read_only)
        self.saveRunBtn.setEnabled(can_save_run and not read_only)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cisco Silicon Failure Characterization")
        self.resize(1100, 720)

        base_dir = os.path.dirname(__file__)
        self.theme_files = {
            "dark": os.path.join(base_dir, "styles.qss"),
            "light": os.path.join(base_dir, "styles_light.qss"),
        }
        self.current_theme = "dark"
        self._apply_theme(self.current_theme)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.splash = SplashPage()
        self.training = TrainingPage()
        self.dashboard = DashboardPage()

        self.stack.addWidget(self.splash)
        self.stack.addWidget(self.training)
        self.stack.addWidget(self.dashboard)

        self.threadpool = QThreadPool()
        self.last_results = None
        self.viewing_saved_run = False
        self.saved_runs_dir = os.path.join(base_dir, "saved_runs")
        os.makedirs(self.saved_runs_dir, exist_ok=True)

        self.splash.requestTrain.connect(self.start_training)
        self.splash.requestLoadTrained.connect(self.load_trained)
        self.splash.requestViewSavedRuns.connect(self.open_saved_run)
        self.splash.requestExportSavedRun.connect(self.export_saved_run_html)

        self.training.cancelRequested.connect(self.to_splash)
        self.dashboard.requestModify.connect(self.to_splash)
        self.dashboard.requestSave.connect(self.save_current_model)
        self.dashboard.requestSaveRun.connect(self.save_current_run_snapshot)
        self.dashboard.set_mode(read_only=False, can_save_model=False, can_save_run=False)

        self.themeToggle = QtWidgets.QPushButton()
        self.themeToggle.clicked.connect(self._toggle_theme)
        self.themeToggle.setToolTip("Switch between dark and light mode.")
        self.themeToggle.setFixedWidth(42)
        self.statusBar().addPermanentWidget(self.themeToggle)
        self._refresh_theme_toggle_label()

    def _read_stylesheet(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _apply_theme(self, theme: str):
        theme = theme if theme in self.theme_files else "dark"
        qss = self._read_stylesheet(self.theme_files[theme])
        if qss:
            self.setStyleSheet(qss)
            self.current_theme = theme
        self._refresh_theme_toggle_label()

    def _toggle_theme(self):
        next_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme(next_theme)

    def _refresh_theme_toggle_label(self):
        if hasattr(self, "themeToggle"):
            if self.current_theme == "dark":
                self.themeToggle.setText("☀")
                self.themeToggle.setToolTip("Switch to light mode")
            else:
                self.themeToggle.setText("☾")
                self.themeToggle.setToolTip("Switch to dark mode")

    @Slot()
    def to_splash(self):
        self.viewing_saved_run = False
        self.stack.setCurrentIndex(0)

    def start_training(self, data_path: str, config: dict):
        self.viewing_saved_run = False
        self.stack.setCurrentIndex(1)
        worker = TrainWorker(data_path, config)
        worker.signals.finished.connect(self._on_result_ready)
        worker.signals.status.connect(lambda s: self.statusBar().showMessage(s, 3000))
        self.threadpool.start(worker)

    def _on_result_ready(self, payload):
        if isinstance(payload, Exception):
            QtWidgets.QMessageBox.critical(self, "Error", f"{payload}")
            self.stack.setCurrentIndex(0)
            return

        self.last_results = payload
        self.dashboard.populate(payload)
        self.dashboard.set_mode(
            read_only=self.viewing_saved_run,
            can_save_model=bool(payload.get("artifact")),
            can_save_run=True,
        )
        self.stack.setCurrentIndex(2)

    def save_current_model(self):
        if not self.last_results or "artifact" not in self.last_results:
            QtWidgets.QMessageBox.information(self, "No model", "Train a model first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save model artifact",
            "",
            "Model Artifact (*.pkl);;All Files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".pkl"):
            path += ".pkl"

        try:
            save_model_artifact(self.last_results["artifact"], path)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Saved model artifact:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def load_trained(self, artifact_path: str):
        # choose dataset to analyze without retraining
        data_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose dataset to analyze (no retrain)",
            "",
            "Data Files (*.csv *.parquet *.pq);;CSV (*.csv);;Parquet (*.parquet *.pq);;All Files (*.*)",
        )
        if not data_path:
            return

        self.viewing_saved_run = False
        self.stack.setCurrentIndex(1)
        worker = AnalyzeWorker(data_path, artifact_path)
        worker.signals.finished.connect(self._on_result_ready)
        worker.signals.status.connect(lambda s: self.statusBar().showMessage(s, 3000))
        self.threadpool.start(worker)

    def save_current_run_snapshot(self):
        if not self.last_results:
            QtWidgets.QMessageBox.information(self, "No run", "Train a model first.")
            return
        if self.viewing_saved_run:
            QtWidgets.QMessageBox.information(
                self, "Read-only", "Saved runs are read-only and cannot be re-saved."
            )
            return

        default_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Run", "Enter a name for this training run:", text=default_name
        )
        if not ok:
            return
        run_name = (run_name or "").strip()
        if not run_name:
            QtWidgets.QMessageBox.information(self, "Name required", "Run name is required.")
            return

        payload = _results_to_saved_payload(self.last_results)
        snapshot = {
            "schema_version": 1,
            "run_name": run_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "payload": payload,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{_safe_filename(run_name)}"
        out_path = os.path.join(self.saved_runs_dir, f"{base}.json")
        n = 1
        while os.path.exists(out_path):
            out_path = os.path.join(self.saved_runs_dir, f"{base}_{n}.json")
            n += 1

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            QtWidgets.QMessageBox.information(
                self, "Run saved", f"Saved run snapshot:\n{os.path.basename(out_path)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _list_saved_run_files(self) -> list[str]:
        files = []
        try:
            for name in os.listdir(self.saved_runs_dir):
                if name.lower().endswith(".json"):
                    files.append(os.path.join(self.saved_runs_dir, name))
        except Exception:
            return []
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files

    def _read_saved_run_snapshot(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _choose_saved_run_file(self, title: str, prompt: str) -> tuple[str | None, dict | None]:
        files = self._list_saved_run_files()
        if not files:
            QtWidgets.QMessageBox.information(
                self, "No saved runs", "No saved run snapshots found yet."
            )
            return None, None

        labels = []
        label_to_file: dict[str, str] = {}
        for p in files:
            label = os.path.basename(p)
            try:
                data = self._read_saved_run_snapshot(p)
                run_name = str(data.get("run_name", "")).strip()
                saved_at = str(data.get("saved_at", "")).strip()
                model = str(((data.get("payload") or {}).get("model_name", ""))).strip()
                label = f"{run_name or os.path.basename(p)} | {saved_at} | {model or 'run'}"
            except Exception:
                label = os.path.basename(p)
            labels.append(label)
            label_to_file[label] = p

        chosen, ok = QtWidgets.QInputDialog.getItem(
            self, title, prompt, labels, 0, False
        )
        if not ok or not chosen:
            return None, None

        path = label_to_file.get(chosen)
        if not path:
            return None, None

        try:
            snapshot = self._read_saved_run_snapshot(path)
            return path, snapshot
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(e))
            return None, None

    def _fig_to_base64_png(self, fig: Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _build_saved_run_charts(self, results: dict) -> dict[str, str]:
        charts: dict[str, str] = {}

        # Class distribution
        bins = results.get("bins", pd.DataFrame())
        if isinstance(bins, pd.DataFrame) and not bins.empty and {"bin_name", "count"}.issubset(bins.columns):
            fig = Figure(figsize=(7.2, 3.4), dpi=110)
            ax = fig.add_subplot(111)
            ax.bar(bins["bin_name"].astype(str), bins["count"].astype(float), color="#13a8f1")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            fig.tight_layout()
            charts["distribution"] = self._fig_to_base64_png(fig)

        # ROC
        roc_list = results.get("roc", []) or []
        if roc_list:
            fig = Figure(figsize=(7.2, 3.4), dpi=110)
            ax = fig.add_subplot(111)
            for r in roc_list[:6]:
                try:
                    ax.plot(r.get("fpr", []), r.get("tpr", []), label=f"{r.get('class', '')} (AUC={float(r.get('auc', 0.0)):.3f})")
                except Exception:
                    continue
            ax.plot([0, 1], [0, 1], "--", linewidth=1, color="#777")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            if ax.lines:
                ax.legend(fontsize=8)
            fig.tight_layout()
            charts["roc"] = self._fig_to_base64_png(fig)

        visual = results.get("visual_plots", {}) or {}

        # Top SHAP histograms
        hist_items = visual.get("top_shap_hist", []) or []
        if hist_items:
            n = min(5, len(hist_items))
            cols = 3 if n > 3 else n
            rows = 2 if n > 3 else 1
            fig = Figure(figsize=(8.6, 4.6), dpi=110)
            for i, item in enumerate(hist_items[:5]):
                ax = fig.add_subplot(rows, cols, i + 1)
                pass_vals = item.get("pass_values", []) or []
                fail_vals = item.get("fail_values", []) or []
                if pass_vals:
                    ax.hist(pass_vals, bins=24, density=True, alpha=0.55, color="#26bdfd", label="PASS")
                if fail_vals:
                    ax.hist(fail_vals, bins=24, density=True, alpha=0.55, color="#ff7d75", label="FAIL")
                title = str(item.get("feature", "feature"))
                if len(title) > 28:
                    title = title[:25] + "..."
                ax.set_title(title, fontsize=9)
                if i == 0:
                    ax.legend(fontsize=7)
            fig.tight_layout()
            charts["top_hist"] = self._fig_to_base64_png(fig)

        # Probability box by group
        records = visual.get("probability_box", []) or []
        if records:
            bdf = pd.DataFrame(records)
            if "group" not in bdf.columns and "cal_group" in bdf.columns:
                bdf["group"] = bdf["cal_group"]
            if {"group", "pass_fail", "probability"}.issubset(bdf.columns):
                fig = Figure(figsize=(7.2, 3.4), dpi=110)
                ax = fig.add_subplot(111)
                group_order = bdf["group"].dropna().astype(str).drop_duplicates().tolist()
                pos, vals, colors = [], [], []
                offset = {"PASS": -0.16, "FAIL": 0.16}
                color_map = {"PASS": "#2fc1ff", "FAIL": "#ff7b72"}
                for i, grp in enumerate(group_order):
                    base = i + 1
                    for pf in ["PASS", "FAIL"]:
                        arr = (
                            bdf[(bdf["group"].astype(str) == grp) & (bdf["pass_fail"].astype(str) == pf)]["probability"]
                            .dropna()
                            .to_numpy()
                        )
                        if len(arr) == 0:
                            continue
                        vals.append(arr)
                        pos.append(base + offset[pf])
                        colors.append(color_map[pf])
                if vals:
                    bp = ax.boxplot(vals, positions=pos, widths=0.28, patch_artist=True, showfliers=False)
                    for patch, c in zip(bp["boxes"], colors):
                        patch.set_facecolor(c)
                        patch.set_alpha(0.6)
                    ax.set_xticks([i + 1 for i in range(len(group_order))])
                    ax.set_xticklabels(group_order)
                    ax.set_ylim(0.0, 1.0)
                    ax.set_xlabel("Distribution Group")
                    ax.set_ylabel("Probability")
                    ax.legend(
                        handles=[
                            Patch(facecolor="#2fc1ff", alpha=0.6, label="PASS"),
                            Patch(facecolor="#ff7b72", alpha=0.6, label="FAIL"),
                        ],
                        fontsize=8,
                    )
                    fig.tight_layout()
                    charts["prob_box"] = self._fig_to_base64_png(fig)

        return charts

    def _build_saved_run_html(self, snapshot: dict, results: dict) -> str:
        run_name = str(snapshot.get("run_name", "Saved Run"))
        saved_at = str(snapshot.get("saved_at", ""))
        meta = results.get("meta", {}) or {}
        metrics = results.get("metrics", {}) or {}
        charts = self._build_saved_run_charts(results)
        class_report = results.get("classification_report", {}) or {}
        df_preview = results.get("dataframe", pd.DataFrame())
        if not isinstance(df_preview, pd.DataFrame):
            df_preview = pd.DataFrame()
        df_preview = df_preview.head(80)
        preview_html = (
            df_preview.to_html(index=False, border=0, classes="data-table")
            if not df_preview.empty
            else "<p>No table preview available.</p>"
        )

        metric_cards = []
        metric_keys = [
            ("Accuracy", "Accuracy"),
            ("Precision_weighted", "Precision (w)"),
            ("Recall_weighted", "Recall (w)"),
            ("F1_weighted", "F1 (w)"),
        ]
        for k, label in metric_keys:
            v = metrics.get(k, None)
            text = f"{float(v):.4f}" if isinstance(v, (int, float)) else "—"
            metric_cards.append(
                f"<div class='card metric'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(text)}</div></div>"
            )
        metric_cards.append(
            f"<div class='card metric'><div class='label'>Model</div><div class='value compact'>{html.escape(str(results.get('model_name', '—')))}</div></div>"
        )
        metric_cards.append(
            f"<div class='card metric'><div class='label'>Target</div><div class='value compact'>{html.escape(str(meta.get('target_column', '—')))}</div></div>"
        )
        metric_cards.append(
            f"<div class='card metric'><div class='label'>Classes</div><div class='value compact'>{html.escape(str(meta.get('num_classes', '—')))}</div></div>"
        )

        chart_blocks = []
        chart_labels = [
            ("distribution", "Class Distribution"),
            ("roc", "ROC Curves"),
            ("top_hist", "Top SHAP Histograms (PASS/FAIL)"),
            ("prob_box", "Probability by Group (PASS/FAIL)"),
        ]
        for key, label in chart_labels:
            if key in charts:
                chart_blocks.append(
                    f"<div class='card chart'><h3>{html.escape(label)}</h3><img src='data:image/png;base64,{charts[key]}' alt='{html.escape(label)}' /></div>"
                )

        fi_sections = []
        shap_imp = results.get("shap_importance", {}) or {}
        if isinstance(shap_imp, dict):
            for cls_name, df_cls in shap_imp.items():
                if not isinstance(df_cls, pd.DataFrame) or df_cls.empty:
                    continue
                fi_sections.append(
                    f"<details class='card fi'><summary>{html.escape(str(cls_name))} - Top Features</summary><div class='table-wrap'>{df_cls.head(20).to_html(index=False, border=0, classes='data-table')}</div></details>"
                )

        per_class_rows = []
        class_order = []
        bins = results.get("bins", pd.DataFrame())
        if isinstance(bins, pd.DataFrame) and not bins.empty and "bin_name" in bins.columns:
            class_order = [str(x) for x in bins["bin_name"].tolist()]
        if not class_order and isinstance(class_report, dict):
            for k, v in class_report.items():
                if isinstance(v, dict) and {"precision", "recall", "f1-score"}.issubset(set(v.keys())):
                    class_order.append(str(k))
        for cls in class_order:
            row = class_report.get(cls, {})
            if not isinstance(row, dict):
                continue
            if not {"precision", "recall", "f1-score"}.issubset(set(row.keys())):
                continue
            per_class_rows.append(
                {
                    "Class": cls,
                    "Precision": f"{float(row.get('precision', 0.0)):.4f}",
                    "Recall": f"{float(row.get('recall', 0.0)):.4f}",
                    "F1": f"{float(row.get('f1-score', 0.0)):.4f}",
                    "Support": int(row.get("support", 0)),
                }
            )
        per_class_html = (
            pd.DataFrame(per_class_rows).to_html(index=False, border=0, classes="data-table")
            if per_class_rows
            else "<p>No per-class metrics available.</p>"
        )

        meta_items = [
            ("Saved At", saved_at),
            ("Run Name", run_name),
            ("Mode", meta.get("mode", "")),
            ("Model Type", meta.get("model_type", "")),
            ("Data Path", meta.get("data_path", "")),
            ("Rows", meta.get("n_rows", "")),
            ("Columns (Original)", meta.get("n_cols_original", "")),
            ("Columns (After Filter)", meta.get("n_cols_after_filter", "")),
        ]
        meta_html = "".join(
            f"<div><span>{html.escape(str(k))}</span><strong>{html.escape(str(v))}</strong></div>"
            for k, v in meta_items if str(v).strip() != ""
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(run_name)} - Cisco Silicon Failure Characterization</title>
  <style>
    :root {{ --bg:#0f141c; --panel:#171f2b; --ink:#e7edf8; --muted:#9fb0c5; --cyan:#27bcff; --line:#2f3a4d; }}
    body {{ margin:0; font-family:"Avenir Next","Segoe UI",Arial,sans-serif; color:var(--ink); background:radial-gradient(circle at top right,#1d2f46,#0f141c 55%); }}
    .wrap {{ max-width:1320px; margin:28px auto; padding:0 18px 28px; }}
    h1 {{ font-size:30px; margin:6px 0 4px; }}
    .sub {{ color:var(--muted); margin-bottom:14px; }}
    .grid {{ display:grid; gap:14px; }}
    .metrics {{ grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
    .meta {{ grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }}
    .charts {{ grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }}
    .card {{ background:linear-gradient(180deg,#1b2432,#151d2a); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:0 10px 30px rgba(0,0,0,.25); }}
    .metric .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .metric .value {{ font-size:26px; font-weight:700; margin-top:6px; color:#edf5ff; line-height:1.15; overflow-wrap:anywhere; word-break:break-word; white-space:normal; }}
    .metric .value.compact {{ font-size:18px; line-height:1.25; }}
    .meta div {{ display:flex; justify-content:space-between; gap:10px; padding:6px 0; border-bottom:1px solid #253143; align-items:flex-start; }}
    .meta div:last-child {{ border-bottom:none; }}
    .meta span {{ color:var(--muted); }}
    .meta strong {{ max-width:70%; text-align:right; overflow-wrap:anywhere; word-break:break-word; }}
    .chart h3 {{ margin:2px 0 8px; color:#dce7f8; font-size:16px; }}
    .chart img {{ width:100%; border-radius:10px; background:#fff; }}
    .section {{ margin-top:14px; }}
    .table-wrap {{ width:100%; overflow-x:auto; overflow-y:hidden; border-radius:10px; border:1px solid #273549; }}
    .data-table {{ width:max-content; min-width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }}
    .data-table th,.data-table td {{ border:1px solid #304058; padding:6px 8px; text-align:left; }}
    .data-table th {{ background:#1f2b3c; position:sticky; top:0; }}
    details.fi summary {{ cursor:pointer; font-weight:600; color:#dbe8ff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(run_name)}</h1>
    <div class="sub">Saved run report · {html.escape(saved_at)}</div>

    <section class="grid metrics">
      {''.join(metric_cards)}
    </section>

    <section class="grid meta section card">
      {meta_html}
    </section>

    <section class="grid charts section">
      {''.join(chart_blocks) if chart_blocks else "<div class='card'>No chart images available in this snapshot.</div>"}
    </section>

    <section class="section card">
      <h3>Per-Class Metrics</h3>
      <div class="table-wrap">{per_class_html}</div>
    </section>

    <section class="section card">
      <h3>Data Preview</h3>
      <div class="table-wrap">{preview_html}</div>
    </section>

    <section class="section grid">
      {''.join(fi_sections) if fi_sections else "<div class='card'>No feature-importance tables available.</div>"}
    </section>
  </div>
</body>
</html>"""

    def export_saved_run_html(self):
        path, snapshot = self._choose_saved_run_file(
            "Export Saved Run", "Select a saved run to export:"
        )
        if not path or not snapshot:
            return

        try:
            results = _saved_payload_to_results(snapshot.get("payload", {}) or {})
            report_html = self._build_saved_run_html(snapshot, results)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return

        base_name = _safe_filename(
            str(snapshot.get("run_name", "")).strip() or os.path.splitext(os.path.basename(path))[0]
        )
        out_default = os.path.join(self.saved_runs_dir, f"{base_name}.html")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Saved Run to HTML",
            out_default,
            "HTML (*.html);;All Files (*.*)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".html"):
            out_path += ".html"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_html)
            QtWidgets.QMessageBox.information(
                self, "Exported", f"Saved run exported to:\n{out_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def open_saved_run(self):
        path, snapshot = self._choose_saved_run_file(
            "Open Saved Run", "Select a saved run:"
        )
        if not path or not snapshot:
            return

        try:
            payload = snapshot.get("payload", {}) or {}
            results = _saved_payload_to_results(payload)
            results["meta"] = results.get("meta", {}) or {}
            results["meta"]["mode"] = "saved-run"
            results["meta"]["saved_run_name"] = snapshot.get("run_name", "")
            results["meta"]["saved_run_file"] = os.path.basename(path)
            self.viewing_saved_run = True
            self.last_results = results
            self.dashboard.populate(results)
            self.dashboard.set_mode(
                read_only=True,
                can_save_model=False,
                can_save_run=False,
            )
            self.stack.setCurrentIndex(2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Cisco Silicon Failure Characterization")

    # Optional font fallback
    font = QtGui.QFont("Inter")
    if not QtGui.QFontInfo(font).exactMatch():
        font = QtGui.QFont("Helvetica Neue")
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
