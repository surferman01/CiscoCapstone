# main.py
import sys, os
import re
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QThreadPool, QRunnable, QObject

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from analysis import run_analysis
from widgets import DropZone, ClickTile


# -----------------------------
# Weak (safe) metadata detection
# -----------------------------
_ID_NAME_RE = re.compile(
    r"(serial|serno|s/n|sn\b|lot|wafer|unit|device|die|barcode|uuid|guid|hash|mac|imei|imsi|ip\b|hostname|name\b|id\b|identifier|index)",
    re.IGNORECASE,
)
_DATE_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)


def suggest_drop_columns_weak(df: pd.DataFrame) -> list[str]:
    """
    Conservative suggestions:
      - obvious ID/date columns by name
      - datetime dtype
      - mostly-unique string columns that are mostly non-numeric (typical IDs)
    """
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

            # mostly unique string-like AND mostly non-numeric => likely identifier
            if uniq_ratio > 0.95 and numericish_ratio < 0.20:
                drops.add(c)

    return sorted(drops)


# -----------------------------
# Column drop selection dialog
# -----------------------------
class ColumnDropDialog(QtWidgets.QDialog):
    """
    Checklist dialog: checked columns will be excluded from model training features.
    """

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

        # Search
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search columns…")
        root.addWidget(self.search)

        # List
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        root.addWidget(self.list, stretch=1)

        for c in all_columns:
            item = QtWidgets.QListWidgetItem(str(c))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c in prechecked else Qt.Unchecked)
            self.list.addItem(item)

        # Quick actions
        btn_row = QtWidgets.QHBoxLayout()
        self.checkAllBtn = QtWidgets.QPushButton("Check all")
        self.uncheckAllBtn = QtWidgets.QPushButton("Uncheck all")
        btn_row.addWidget(self.checkAllBtn)
        btn_row.addWidget(self.uncheckAllBtn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # OK/Cancel
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

    def set(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()


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


# -----------------------------
# Splash Page
# -----------------------------
class SplashPage(QtWidgets.QWidget):
    requestTrain = Signal(str, dict)
    requestLoadTrained = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        # Title + logo
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

        # --- Three equal columns
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
        self.selectedLabel.setStyleSheet("color:#666;")

        self.dropSummary = QtWidgets.QLabel("Drop columns: —")
        self.dropSummary.setWordWrap(True)
        self.dropSummary.setStyleSheet("color:#666;")

        self.reviewDropsBtn = QtWidgets.QPushButton("Review Drops…")
        self.reviewDropsBtn.setMinimumHeight(36)
        self.reviewDropsBtn.setEnabled(False)

        orLbl = QtWidgets.QLabel("— or —")
        orLbl.setAlignment(Qt.AlignCenter)
        orLbl.setStyleSheet("color:#888;")

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
        self.targetHint.setStyleSheet("color:#666;")

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
        hint.setStyleSheet("color:#777;")

        self.loadTileBtn = QtWidgets.QPushButton("Browse Trained Artifact…")
        self.loadTileBtn.setMinimumHeight(40)

        self.modelLabel = QtWidgets.QLabel("")
        self.modelLabel.setWordWrap(True)
        self.modelLabel.setStyleSheet("color:#666;")

        right.layout().addWidget(hint)
        right.layout().addSpacing(8)
        right.layout().addWidget(self.loadTileBtn)
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

        self._update_train_enabled()

    def _recommend_targets(self, df: pd.DataFrame, max_cols: int = 10) -> list[str]:
        """
        Similar to your snippet:
          - columns that are not fully numeric are good target candidates
          - also include low-cardinality numeric columns as secondary
        """
        candidates = []
        for c in df.columns:
            s = df[c]
            if s.dropna().empty:
                continue
            all_numeric = pd.to_numeric(s, errors="coerce").notna().all()
            if not all_numeric:
                candidates.append(c)

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

        # Parse preview to populate targets + drop suggestions
        try:
            if path.lower().endswith(".csv"):
                df_preview = pd.read_csv(path, nrows=5000)
            else:
                df_preview = pd.read_parquet(path).head(5000)

            self.all_columns = list(df_preview.columns)

            # Recommended targets for target selection
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

            # Suggested metadata drops (weak filter)
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
            self.dropSummary.setText(f"Drop columns: {len(self.user_drop_cols)} (customized)")

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
            QtWidgets.QMessageBox.information(self, "No data", "Choose a dataset first.")
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

        # quick validation (fast preview)
        try:
            if self.data_path.lower().endswith(".csv"):
                df_check = pd.read_csv(self.data_path, nrows=5000)
            else:
                df_check = pd.read_parquet(self.data_path).head(5000)

            if target_col not in df_check.columns:
                QtWidgets.QMessageBox.critical(
                    self, "Invalid target", f"Column '{target_col}' not found in dataset."
                )
                return

            if df_check[target_col].dropna().nunique() < 2:
                QtWidgets.QMessageBox.critical(
                    self, "Invalid target", f"Target '{target_col}' has < 2 unique values."
                )
                return

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Validation failed", str(e))
            return

        recs = list(self.recommended_targets or [])
        user_drops = list(self.user_drop_cols or [])

        # Always exclude other recommended targets (except the chosen target)
        rec_excludes = [c for c in recs if c != target_col]

        # Merge & never drop chosen target
        exclude_cols = sorted(set(user_drops) | set(rec_excludes))
        exclude_cols = [c for c in exclude_cols if c != target_col]

        config = {
            "model_type": model_type,
            "use_gpu": self.gpuCheck.isChecked(),
            "target_column": target_col,

            # analysis.py can use this for additional safe exclusion/reporting
            "recommended_targets": recs,

            # final per-run exclusions (UI-controlled)
            "exclude_columns": exclude_cols,
        }

        # Optional small confirmation (comment out if you don't want popups)
        # preview = "\n".join(exclude_cols[:25])
        # more = "" if len(exclude_cols) <= 25 else f"\n… (+{len(exclude_cols)-25} more)"
        # QtWidgets.QMessageBox.information(
        #     self,
        #     "Training exclusions",
        #     f"Excluding {len(exclude_cols)} columns from training features.\n\n{preview}{more}",
        # )

        self.requestTrain.emit(self.data_path, config)

    def _on_load_trained_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open trained artifact", "", "Pickle/Model Files (*.*)"
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
        title = QtWidgets.QLabel("Cisco Silicon Failure Characterization", alignment=Qt.AlignCenter)
        title.setObjectName("titleBar")
        layout.addWidget(title)

        center = QtWidgets.QVBoxLayout()
        msg = QtWidgets.QLabel("Training your data...", alignment=Qt.AlignCenter)
        f = msg.font()
        f.setPointSize(28)
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
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ax = self.canvas.figure.add_subplot(111)

        self.layout().addWidget(self.canvas)

        # fixed card height so charts don't stretch to fill the entire page
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

        # Dashboard tab (no scroll; fixed-size KPIs and charts)
        d = QtWidgets.QVBoxLayout(self.dashboard)
        d.setContentsMargins(12, 12, 12, 12)
        d.setSpacing(16)

        kpi_row = QtWidgets.QHBoxLayout()
        kpi_row.setSpacing(16)

        self.kpiAccuracy = KPIBox("Accuracy")
        self.kpiPrecision = KPIBox("Precision (w)")
        self.kpiRecall = KPIBox("Recall (w)")
        self.kpiF1 = KPIBox("F1 (weighted)")
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

        self.ovBar = ChartCard("Class Distribution (Test)", height=380)
        self.ovRoc = ChartCard("ROC Curves (Preview)", height=380)

        charts_row = QtWidgets.QHBoxLayout()
        charts_row.setSpacing(16)
        charts_row.addWidget(self.ovBar, 1)
        charts_row.addWidget(self.ovRoc, 1)

        charts_wrap = QtWidgets.QWidget()
        charts_wrap.setLayout(charts_row)
        charts_wrap.setMinimumHeight(380)
        charts_wrap.setMaximumHeight(380)
        d.addWidget(charts_wrap)

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

        self.fiHeader = QtWidgets.QLabel("Top 20 features by SHAP (one table per class/bin).")
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
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.deleteLater()
            elif child_layout:
                self._clear_layout(child_layout)

    def populate(self, results: dict):
        # Data table
        df = results.get("dataframe", pd.DataFrame())
        self.table.setModel(PandasModel(df))

        meta = results.get("meta", {}) or {}
        metrics = results.get("metrics", {}) or {}
        bins = results.get("bins", pd.DataFrame())
        roc_list = results.get("roc", [])

        # KPIs
        self.kpiAccuracy.value.setText(
            f"{float(metrics.get('Accuracy', 0.0)):.4f}" if "Accuracy" in metrics else "—"
        )
        # Precision/Recall (weighted) – best source is classification_report dict if present
        prec_w = None
        rec_w = None

        report = results.get("classification_report")
        if isinstance(report, dict):
            wavg = report.get("weighted avg") or {}
            if isinstance(wavg, dict):
                prec_w = wavg.get("precision")
                rec_w = wavg.get("recall")

        # fallback: allow metrics dict to supply these later if you add them in analysis.py
        if prec_w is None:
            prec_w = metrics.get("Precision_weighted")
        if rec_w is None:
            rec_w = metrics.get("Recall_weighted")

        self.kpiPrecision.value.setText(f"{float(prec_w):.4f}" if prec_w is not None else "—")
        self.kpiRecall.value.setText(f"{float(rec_w):.4f}" if rec_w is not None else "—")

        self.kpiF1.value.setText(
            f"{float(metrics.get('F1_weighted', 0.0)):.4f}" if "F1_weighted" in metrics else "—"
        )
        self.kpiModel.value.setText(str(meta.get("model", meta.get("model_name", "—"))))
        self.kpiTarget.value.setText(str(meta.get("target_column", meta.get("target", "—"))))
        self.kpiClasses.value.setText(str(meta.get("num_classes", meta.get("classes", "—"))))

        # Class distribution
        ax = self.ovBar.ax
        ax.clear()
        if isinstance(bins, pd.DataFrame) and not bins.empty and {"bin_name", "count"}.issubset(bins.columns):
            ax.bar(bins["bin_name"].astype(str), bins["count"])
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
        self.ovBar.canvas.figure.tight_layout()
        self.ovBar.canvas.draw()

        # ROC preview
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

        # Feature importance tables
        self._clear_layout(self.fiTablesLayout)
        shap_importance = results.get("shap_importance", {}) or {}

        if not isinstance(shap_importance, dict) or not shap_importance:
            msg = QtWidgets.QLabel("No per-class feature importance available.")
            msg.setStyleSheet("color:#666;")
            self.fiTablesLayout.addWidget(msg, 0, 0)
            return

        # Prefer ordering from bins
        if isinstance(bins, pd.DataFrame) and not bins.empty and "bin_name" in bins.columns:
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
                ["Rank", "Feature", "SHAP |Δ|", "Share (%)", "Direction", "Failure Avg", "PASS Avg", "PASS Std"]
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
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(int(row.get("rank", r + 1)))))
                table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(row.get("feature", ""))))
                table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{float(row.get('importance', 0.0)):.4f}"))
                table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{float(row.get('share_pct', 0.0)):.1f}"))
                table.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{float(row.get('direction', 0.0)):.4f}"))

                fa = row.get("failure_avg")
                pa = row.get("pass_avg")
                ps = row.get("pass_std")

                table.setItem(r, 5, QtWidgets.QTableWidgetItem("" if pd.isna(fa) else f"{float(fa):.4f}"))
                table.setItem(r, 6, QtWidgets.QTableWidgetItem("" if pd.isna(pa) else f"{float(pa):.4f}"))
                table.setItem(r, 7, QtWidgets.QTableWidgetItem("" if pd.isna(ps) else f"{float(ps):.4f}"))

            vbox.addWidget(table)

            row_i = idx // cols_per_row
            col_i = idx % cols_per_row
            self.fiTablesLayout.addWidget(group, row_i, col_i)
            self.fiTablesLayout.setColumnStretch(col_i, 1)
            added += 1

        if added == 0:
            msg = QtWidgets.QLabel("No per-class feature importance tables were generated.")
            msg.setStyleSheet("color:#666;")
            self.fiTablesLayout.addWidget(msg, 0, 0)


# -----------------------------
# Dashboard Page
# -----------------------------
class DashboardPage(QtWidgets.QWidget):
    requestModify = Signal()
    requestSave = Signal()

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Cisco Silicon Failure Characterization", alignment=Qt.AlignCenter)
        title.setObjectName("titleBar")
        root.addWidget(title)

        self.tabs = DashboardTabs()
        root.addWidget(self.tabs, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self.modifyBtn = QtWidgets.QPushButton("modify data")
        self.modifyBtn.clicked.connect(lambda: self.requestModify.emit())
        self.saveBtn = QtWidgets.QPushButton("save")
        self.saveBtn.clicked.connect(lambda: self.requestSave.emit())
        actions.addStretch()
        actions.addWidget(self.modifyBtn)
        actions.addWidget(self.saveBtn)
        root.addLayout(actions)

    def populate(self, results: dict):
        self.tabs.populate(results)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cisco Silicon Failure Characterization")
        self.resize(1100, 720)

        style = os.path.join(os.path.dirname(__file__), "styles.qss")
        if os.path.exists(style):
            with open(style, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.splash = SplashPage()
        self.training = TrainingPage()
        self.dashboard = DashboardPage()

        self.stack.addWidget(self.splash)
        self.stack.addWidget(self.training)
        self.stack.addWidget(self.dashboard)

        self.threadpool = QThreadPool()

        self.splash.requestTrain.connect(self.start_training)
        self.splash.requestLoadTrained.connect(self.load_trained)

        self.training.cancelRequested.connect(self.to_splash)
        self.dashboard.requestModify.connect(self.to_splash)

    @Slot()
    def to_splash(self):
        self.stack.setCurrentIndex(0)

    def start_training(self, data_path: str, config: dict):
        self.stack.setCurrentIndex(1)
        worker = TrainWorker(data_path, config)
        worker.signals.finished.connect(self._on_trained)
        worker.signals.status.connect(lambda s: self.statusBar().showMessage(s, 3000))
        self.threadpool.start(worker)

    def _on_trained(self, payload):
        if isinstance(payload, Exception):
            QtWidgets.QMessageBox.critical(self, "Error", f"{payload}")
            self.stack.setCurrentIndex(0)
            return
        self.dashboard.populate(payload)
        self.stack.setCurrentIndex(2)

    def load_trained(self, path: str):
        QtWidgets.QMessageBox.information(
            self,
            "Loaded",
            f"Selected trained artifact: {os.path.basename(path)}\n(Wire this to restore a model.)",
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Cisco Silicon Failure Characterization")

    # Optional font fallback to avoid missing Inter warnings (safe even if you keep QSS)
    font = QtGui.QFont("Inter")
    if not QtGui.QFontInfo(font).exactMatch():
        font = QtGui.QFont("Helvetica Neue")
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
