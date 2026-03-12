from __future__ import annotations

import importlib.util
import os
import json

import pandas as pd

from matplotlib.patches import Patch
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from core.preprocessing import suggest_drop_columns_weak
from styles import APP_NAME, resolve_logo_path
from ui.components import (
    ChartCard,
    ColumnDropDialog,
    HyperparameterDialog,
    KPIBox,
    PandasModel,
    coerce_hyperparameters,
    default_hyperparameters,
    summarize_hyperparameters,
)
from widgets import DropZone


class SplashPage(QtWidgets.QWidget):
    requestTrain = Signal(str, dict)
    requestLoadTrained = Signal(str)
    requestViewSavedRuns = Signal()
    requestExportSavedRun = Signal()

    def __init__(self):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(content)

        title = QtWidgets.QLabel(APP_NAME)
        title.setObjectName("titleBar")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        logo_path = resolve_logo_path()
        logo = QtWidgets.QLabel(alignment=Qt.AlignCenter)
        if logo_path.exists():
            pm = QtGui.QPixmap(str(logo_path)).scaledToWidth(320, Qt.SmoothTransformation)
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

        left = make_column("Add Data")
        self.drop = DropZone("Add Data\n(drag & drop)")
        self.drop.setMinimumHeight(180)
        self.drop.setMaximumHeight(220)

        self.browseBtn = QtWidgets.QPushButton("Browse Files...")
        self.browseBtn.setMinimumHeight(40)

        self.selectedLabel = QtWidgets.QLabel("")
        self.selectedLabel.setWordWrap(True)
        self.selectedLabel.setProperty("muted", True)

        self.dropSummary = QtWidgets.QLabel("Drop columns: -")
        self.dropSummary.setWordWrap(True)
        self.dropSummary.setProperty("muted", True)

        self.reviewDropsBtn = QtWidgets.QPushButton("Review Drops...")
        self.reviewDropsBtn.setMinimumHeight(36)
        self.reviewDropsBtn.setEnabled(False)

        orLbl = QtWidgets.QLabel("- or -")
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

        middle = make_column("Training")

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(
            [
                "-- select --",
                "CatBoost",
                "XGBoost",
                "Mega Multiclass XGBoost",
                "Mega OVR XGBoost",
                "Mega Hierarchical XGBoost",
            ]
        )
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

        self.hyperparamSummary = QtWidgets.QLabel("Hyperparameters: Defaults")
        self.hyperparamSummary.setWordWrap(True)
        self.hyperparamSummary.setProperty("muted", True)

        self.hyperparamBtn = QtWidgets.QPushButton("Hyperparameters...")
        self.hyperparamBtn.setMinimumHeight(36)
        self.loadPresetBtn = QtWidgets.QPushButton("Load Hyperparameters...")
        self.loadPresetBtn.setMinimumHeight(36)
        self.savePresetBtn = QtWidgets.QPushButton("Save Current Preset...")
        self.savePresetBtn.setMinimumHeight(36)

        middle.layout().addWidget(QtWidgets.QLabel("Model"))
        middle.layout().addWidget(self.combo)
        middle.layout().addWidget(self.gpuCheck)
        middle.layout().addWidget(self.targetBox)
        middle.layout().addWidget(self.hyperparamSummary)
        middle.layout().addWidget(self.hyperparamBtn)
        middle.layout().addWidget(self.loadPresetBtn)
        middle.layout().addWidget(self.savePresetBtn)
        middle.layout().addSpacing(8)
        middle.layout().addWidget(self.trainBtn)
        middle.layout().addStretch(1)
        columns.addWidget(middle, 1)

        right = make_column("Already Trained?")
        hint = QtWidgets.QLabel("(insert file)")
        hint.setAlignment(Qt.AlignCenter)
        hint.setProperty("muted", True)

        self.loadTileBtn = QtWidgets.QPushButton("Browse Trained Artifact...")
        self.loadTileBtn.setMinimumHeight(40)
        self.viewSavedRunsBtn = QtWidgets.QPushButton("View Saved Runs...")
        self.viewSavedRunsBtn.setMinimumHeight(36)
        self.exportSavedRunBtn = QtWidgets.QPushButton("Export Saved Run...")
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

        self.data_path = None
        self.recommended_targets: list[str] = []
        self.all_columns: list[str] = []
        self.suggested_drop_cols: list[str] = []
        self.user_drop_cols: list[str] = []
        self.hyperparams = default_hyperparameters()

        self.drop.fileDropped.connect(self._on_file_dropped)
        self.browseBtn.clicked.connect(self._on_browse_clicked)
        self.reviewDropsBtn.clicked.connect(self._on_review_drops)
        self.combo.currentIndexChanged.connect(self._update_train_enabled)
        self.combo.currentTextChanged.connect(self._update_hyperparam_summary)
        self.targetCombo.currentTextChanged.connect(self._update_train_enabled)
        self.trainBtn.clicked.connect(self._on_train_click)
        self.hyperparamBtn.clicked.connect(self._on_edit_hyperparams)
        self.loadPresetBtn.clicked.connect(self._on_load_preset)
        self.savePresetBtn.clicked.connect(self._on_save_preset)
        self.loadTileBtn.clicked.connect(self._on_load_trained_clicked)
        self.viewSavedRunsBtn.clicked.connect(lambda: self.requestViewSavedRuns.emit())
        self.exportSavedRunBtn.clicked.connect(
            lambda: self.requestExportSavedRun.emit()
        )

        self._update_hyperparam_summary()
        self._update_train_enabled()

    def _recommend_targets(self, df: pd.DataFrame, max_cols: int = 10) -> list[str]:
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
                f"Drop columns: {len(self.user_drop_cols)} (click Review Drops...)"
            )
            self.reviewDropsBtn.setEnabled(True)

        except Exception as e:
            self.targetHint.setText(f"Could not parse columns: {e}")
            self.all_columns = []
            self.recommended_targets = []
            self.suggested_drop_cols = []
            self.user_drop_cols = []
            self.dropSummary.setText("Drop columns: -")
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
        valid_model = self.combo.currentText() in {
            "CatBoost",
            "XGBoost",
            "Mega Multiclass XGBoost",
            "Mega OVR XGBoost",
            "Mega Hierarchical XGBoost",
        }
        valid_target = bool(self.targetCombo.currentText().strip())
        self.trainBtn.setEnabled(bool(self.data_path) and valid_model and valid_target)

    def _current_model_type(self) -> str:
        return self.combo.currentText().strip()

    def _missing_dependencies_for_training(self, model_type: str) -> list[str]:
        needed = []
        if model_type == "CatBoost":
            needed.append("catboost")
        elif model_type == "XGBoost":
            needed.append("xgboost")
        elif model_type in {
            "Mega Multiclass XGBoost",
            "Mega OVR XGBoost",
            "Mega Hierarchical XGBoost",
        }:
            needed.append("xgboost")
            tuned_keys = {
                "best_params",
                "best_stage1_params",
                "best_stage2_params",
            }
            has_tuned_preset = any(
                key in (self.hyperparams or {}) for key in tuned_keys
            )
            if not has_tuned_preset:
                needed.append("optuna")

        return [
            module
            for module in needed
            if importlib.util.find_spec(module) is None
        ]

    def _update_hyperparam_summary(self):
        model_type = self._current_model_type()
        if model_type.startswith("--"):
            self.hyperparamSummary.setText("Hyperparameters: Select a model first.")
            return
        summary = summarize_hyperparameters(model_type, self.hyperparams)
        self.hyperparamSummary.setText(f"Hyperparameters: {summary}")

    def _on_edit_hyperparams(self):
        model_type = self._current_model_type()
        if model_type.startswith("--"):
            QtWidgets.QMessageBox.information(
                self, "Choose model", "Select a model before editing hyperparameters."
            )
            return

        dlg = HyperparameterDialog(self, model_type, self.hyperparams)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.hyperparams = coerce_hyperparameters(model_type, dlg.values())
            self._update_hyperparam_summary()

    def _current_preset_payload(self) -> dict:
        model_type = self._current_model_type()
        return {
            "schema_version": 1,
            "model_type": model_type,
            "use_gpu": self.gpuCheck.isChecked(),
            "hyperparameters": coerce_hyperparameters(model_type, self.hyperparams),
        }

    def _apply_preset_payload(self, payload: dict):
        model_type = str(payload.get("model_type", "")).strip()
        if model_type in {
            "CatBoost",
            "XGBoost",
            "Mega Multiclass XGBoost",
            "Mega OVR XGBoost",
            "Mega Hierarchical XGBoost",
        }:
            idx = self.combo.findText(model_type)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        if "use_gpu" in payload:
            self.gpuCheck.setChecked(bool(payload.get("use_gpu")))
        self.hyperparams = coerce_hyperparameters(
            self._current_model_type(), payload.get("hyperparameters", {})
        )
        self._update_hyperparam_summary()

    def _on_save_preset(self):
        model_type = self._current_model_type()
        if model_type.startswith("--"):
            QtWidgets.QMessageBox.information(
                self, "Choose model", "Select a model before saving a preset."
            )
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Hyperparameter Preset",
            f"{model_type.lower().replace(' ', '_')}_preset.json",
            "JSON (*.json);;All Files (*.*)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".json"):
            out_path += ".json"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._current_preset_payload(), f, indent=2)
            QtWidgets.QMessageBox.information(
                self, "Preset saved", f"Saved hyperparameter preset:\n{out_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Hyperparameters",
            "",
            "JSON (*.json);;All Files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("Preset file must contain a JSON object.")
            self._apply_preset_payload(payload)
            QtWidgets.QMessageBox.information(
                self, "Preset loaded", f"Loaded hyperparameter preset:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _on_train_click(self):
        if not self.data_path:
            QtWidgets.QMessageBox.information(
                self, "No data", "Choose a dataset first."
            )
            return

        model_type = self.combo.currentText()
        if model_type not in {
            "CatBoost",
            "XGBoost",
            "Mega Multiclass XGBoost",
            "Mega OVR XGBoost",
            "Mega Hierarchical XGBoost",
        }:
            QtWidgets.QMessageBox.information(
                self,
                "Choose model",
                "Please select a model option.",
            )
            return

        missing_modules = self._missing_dependencies_for_training(model_type)
        if missing_modules:
            missing_text = ", ".join(missing_modules)
            QtWidgets.QMessageBox.critical(
                self,
                "Missing dependency",
                f"The selected training mode requires missing Python module(s): {missing_text}.",
            )
            return

        target_col = self.targetCombo.currentText().strip()
        if not target_col:
            QtWidgets.QMessageBox.information(
                self, "Target required", "Please select or type a target column."
            )
            return

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
        rec_excludes = [c for c in recs if c != target_col]
        exclude_cols = sorted(set(user_drops) | set(rec_excludes))
        exclude_cols = [c for c in exclude_cols if c != target_col]

        config = {
            "model_type": model_type,
            "use_gpu": self.gpuCheck.isChecked(),
            "target_column": target_col,
            "recommended_targets": recs,
            "exclude_columns": exclude_cols,
        }
        config.update(coerce_hyperparameters(model_type, self.hyperparams))

        self.requestTrain.emit(self.data_path, config)

    def _on_load_trained_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open trained artifact", "", "Model Artifact (*.pkl);;All Files (*.*)"
        )
        if path:
            self.modelLabel.setText(f"Selected: {os.path.basename(path)}")
            self.requestLoadTrained.emit(path)


class TrainingPage(QtWidgets.QWidget):
    cancelRequested = Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(APP_NAME, alignment=Qt.AlignCenter)
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


class DashboardTabs(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()

        self.dashboard = QtWidgets.QScrollArea()
        self.dashboard.setWidgetResizable(True)
        self.dashboard.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.dashboard.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        dashboardInner = QtWidgets.QWidget()
        self.dashboard.setWidget(dashboardInner)
        self.tablePage = QtWidgets.QWidget()
        self.fiPage = QtWidgets.QWidget()

        self.addTab(self.dashboard, "Dashboard")
        self.addTab(self.tablePage, "Data Table")
        self.addTab(self.fiPage, "Feature Importance")

        d = QtWidgets.QVBoxLayout(dashboardInner)
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

        tl = QtWidgets.QVBoxLayout(self.tablePage)
        tl.setContentsMargins(12, 12, 12, 12)
        self.table = QtWidgets.QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        tl.addWidget(self.table)

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
        df = results.get("dataframe", pd.DataFrame())
        self.table.setModel(PandasModel(df))

        meta = results.get("meta", {}) or {}
        metrics = results.get("metrics", {}) or {}
        bins = results.get("bins", pd.DataFrame())
        roc_list = results.get("roc", [])
        visual_plots = results.get("visual_plots", {}) or {}
        class_report = results.get("classification_report", {}) or {}

        self.kpiAccuracy.value.setText(
            f"{float(metrics.get('Accuracy', 0.0)):.4f}"
            if "Accuracy" in metrics
            else "-"
        )
        self.kpiPrecision.value.setText(
            f"{float(metrics.get('Precision_weighted', 0.0)):.4f}"
            if "Precision_weighted" in metrics
            else "-"
        )
        self.kpiRecall.value.setText(
            f"{float(metrics.get('Recall_weighted', 0.0)):.4f}"
            if "Recall_weighted" in metrics
            else "-"
        )
        self.kpiF1.value.setText(
            f"{float(metrics.get('F1_weighted', 0.0)):.4f}"
            if "F1_weighted" in metrics
            else "-"
        )
        self.kpiModel.value.setText(str(meta.get("model", meta.get("model_name", "-"))))
        self.kpiTarget.value.setText(str(meta.get("target_column", "-")))
        self.kpiClasses.value.setText(str(meta.get("num_classes", "-")))

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
            offset = {"PASS": -0.16, "FAIL": 0.16}
            color_map = {"PASS": "#2fc1ff", "FAIL": "#ff7b72"}

            for i, grp in enumerate(group_order):
                base = i + 1
                for pf in ["PASS", "FAIL"]:
                    arr = (
                        bdf[(bdf["group"] == grp) & (bdf["pass_fail"] == pf)]["probability"]
                        .dropna()
                        .to_numpy()
                    )
                    if len(arr) == 0:
                        continue
                    vals.append(arr)
                    pos.append(base + offset[pf])
                    colors.append(color_map[pf])

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

        self._clear_layout(self.fiTablesLayout)
        shap_importance = results.get("shap_importance", {}) or {}
        if not isinstance(shap_importance, dict) or not shap_importance:
            msg = QtWidgets.QLabel("No per-class feature importance available.")
            msg.setProperty("muted", True)
            self.fiTablesLayout.addWidget(msg, 0, 0)
            return

        if (
            isinstance(bins, pd.DataFrame)
            and not bins.empty
            and "bin_name" in bins.columns
        ):
            class_list = [str(x) for x in bins["bin_name"].tolist()]
        else:
            class_list = [str(k) for k in shap_importance.keys()]

        cols_per_row = 1
        added = 0

        for idx, cls_name in enumerate(class_list):
            df_cls = shap_importance.get(cls_name)
            if df_cls is None:
                df_cls = shap_importance.get(str(cls_name).strip().lower())
            if not (isinstance(df_cls, pd.DataFrame) and not df_cls.empty):
                continue

            reference_label = str(df_cls.get("reference_label", pd.Series(["Other classes"])).iloc[0]).strip()
            if not reference_label:
                reference_label = "Other classes"

            group = QtWidgets.QGroupBox(f"{cls_name} - Top 20 Features")
            vbox = QtWidgets.QVBoxLayout(group)

            table = QtWidgets.QTableWidget()
            table.setMinimumHeight(630)
            table.setColumnCount(8)
            table.setHorizontalHeaderLabels(
                [
                    "Rank",
                    "Feature",
                    "SHAP |D|",
                    "Share (%)",
                    "Direction",
                    "Failure Avg",
                    f"{reference_label} Avg",
                    f"{reference_label} Std",
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

            row_i = added
            col_i = 0
            self.fiTablesLayout.addWidget(group, row_i, col_i)
            self.fiTablesLayout.setColumnStretch(0, 1)
            added += 1

        if added == 0:
            msg = QtWidgets.QLabel(
                "No per-class feature importance tables were generated."
            )
            msg.setProperty("muted", True)
            self.fiTablesLayout.addWidget(msg, 0, 0)


class DashboardPage(QtWidgets.QWidget):
    requestModify = Signal()
    requestSave = Signal()
    requestSaveRun = Signal()
    requestSaveHyperparameters = Signal()

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(APP_NAME, alignment=Qt.AlignCenter)
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
        self.saveHyperparamsBtn = QtWidgets.QPushButton("save hyperparameters")
        self.saveHyperparamsBtn.clicked.connect(
            lambda: self.requestSaveHyperparameters.emit()
        )
        self.readOnlyLabel = QtWidgets.QLabel("Viewing saved run (read-only)")
        self.readOnlyLabel.setProperty("muted", True)
        self.readOnlyLabel.hide()
        actions.addStretch()
        actions.addWidget(self.readOnlyLabel)
        actions.addStretch()
        actions.addWidget(self.modifyBtn)
        actions.addWidget(self.saveHyperparamsBtn)
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
        can_save_hyperparameters: bool = True,
    ):
        self.readOnlyLabel.setVisible(read_only)
        self.saveBtn.setEnabled(can_save_model and not read_only)
        self.saveRunBtn.setEnabled(can_save_run and not read_only)
        self.saveHyperparamsBtn.setEnabled(can_save_hyperparameters and not read_only)
