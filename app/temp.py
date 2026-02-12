import sys, os, json
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QThreadPool, QRunnable, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from analysis import run_analysis
from widgets import DropZone, ClickTile

FIXED_CHART_HEIGHT = 280


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

        # --- Three equal columns -------------------------------------------------
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(18)
        layout.addLayout(columns, stretch=1)

        # Helper to make equal-sized group boxes
        def make_column(title_text: str) -> QtWidgets.QGroupBox:
            box = QtWidgets.QGroupBox(title_text)
            box.setObjectName("columnBox")
            box.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
            )
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(14, 14, 14, 14)
            v.setSpacing(12)
            return box

        # Left: Add Data (drag & drop OR browse)
        left = make_column("Add Data")
        self.drop = DropZone("Add Data\n(drag & drop)")
        self.drop.setMinimumHeight(180)
        self.drop.setMaximumHeight(220)

        self.browseBtn = QtWidgets.QPushButton("Browse Files…")
        self.browseBtn.setMinimumHeight(40)

        self.selectedLabel = QtWidgets.QLabel("")  # shows chosen data file
        self.selectedLabel.setWordWrap(True)
        self.selectedLabel.setStyleSheet("color:#666;")

        # A tiny "or" hint
        orLbl = QtWidgets.QLabel("— or —")
        orLbl.setAlignment(Qt.AlignCenter)
        orLbl.setStyleSheet("color:#888;")

        left.layout().addWidget(self.drop)
        left.layout().addWidget(orLbl)
        left.layout().addWidget(self.browseBtn)
        left.layout().addWidget(self.selectedLabel)
        left.layout().addStretch(1)
        columns.addWidget(left, 1)

        # Middle: Training Type + Train button
        middle = make_column("Training")
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["-- select --", "CatBoost", "XGBoost"])
        self.combo.setMinimumHeight(36)

        self.gpuCheck = QtWidgets.QCheckBox("Use GPU (if available)")
        self.gpuCheck.setChecked(True)  # default on

        self.trainBtn = QtWidgets.QPushButton("Train")
        self.trainBtn.setMinimumHeight(44)
        self.trainBtn.setEnabled(False)  # enabled once a data file is chosen

        middle.layout().addWidget(QtWidgets.QLabel("What Type"))
        middle.layout().addWidget(self.combo)
        middle.layout().addWidget(self.gpuCheck)
        middle.layout().addSpacing(8)
        middle.layout().addWidget(self.trainBtn)
        middle.layout().addStretch(1)
        columns.addWidget(middle, 1)

        # Right: Already trained? Browse artifact
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

        # Keep the three columns equal width
        columns.setStretch(0, 1)
        columns.setStretch(1, 1)
        columns.setStretch(2, 1)

        # --- connections
        self.drop.fileDropped.connect(self._on_file_dropped)
        self.browseBtn.clicked.connect(self._on_browse_clicked)
        self.trainBtn.clicked.connect(self._on_train_click)
        self.loadTileBtn.clicked.connect(self._on_load_trained_clicked)

        self.data_path = None

    # ------------- helpers & slots ---------------------------------------------
    def _set_data_path(self, path: str):
        self.data_path = path
        base = os.path.basename(path)
        self.selectedLabel.setText(f"Selected: {base}")
        self.trainBtn.setEnabled(True)
        self._update_train_enabled()

    def _on_file_dropped(self, path: str):
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
        self.trainBtn.setEnabled(bool(self.data_path) and valid_model)

    def _on_train_click(self):
        if not self.data_path:
            QtWidgets.QMessageBox.information(
                self,
                "No data",
                "Choose a CSV/Parquet file (drag & drop or Browse Files…).",
            )
            return

        model_type = self.combo.currentText()
        if model_type not in {"CatBoost", "XGBoost"}:
            QtWidgets.QMessageBox.information(
                self, "Choose model", "Please select CatBoost or XGBoost."
            )
            return

        config = {
            "model_type": model_type,
            "use_gpu": self.gpuCheck.isChecked(),
            # You can expose more UI toggles later and add them here.
        }

        self.requestTrain.emit(self.data_path, config)

    def _on_load_trained_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open trained artifact",
            "",
            "Pickle/Model Files (*.*)",
        )
        if path:
            self.modelLabel.setText(f"Selected: {os.path.basename(path)}")
            self.requestLoadTrained.emit(path)


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


# ------------- Dashboard with TABS and fixed-height charts -------------
class ChartCard(QtWidgets.QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = FigureCanvas(Figure(figsize=(6, 3), dpi=100))
        self.canvas.setFixedHeight(FIXED_CHART_HEIGHT)
        self.ax = self.canvas.figure.add_subplot(111)
        self.layout().addWidget(self.canvas)


class DashboardTabs(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()
        # Pages
        self.overview = QtWidgets.QWidget()
        self.tablePage = QtWidgets.QWidget()
        self.fiPage = QtWidgets.QWidget()

        self.addTab(self.overview, "Overview")
        self.addTab(self.tablePage, "Data Table")
        self.addTab(self.fiPage, "Feature Importance")

        # --- Overview layout
        ov = QtWidgets.QVBoxLayout(self.overview)
        self.metricsGroup = QtWidgets.QGroupBox("Classification Report")
        metricsLayout = QtWidgets.QVBoxLayout(self.metricsGroup)

        self.reportTable = QtWidgets.QTableWidget(0, 5)
        self.reportTable.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1-Score", "Support"]
        )
        self.reportTable.horizontalHeader().setStretchLastSection(True)
        self.reportTable.verticalHeader().setVisible(False)
        self.reportTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.reportTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.reportTable.setFocusPolicy(Qt.NoFocus)
        metricsLayout.addWidget(self.reportTable)

        self.summaryGrid = QtWidgets.QGridLayout()
        self.summaryGrid.setVerticalSpacing(4)

        self.accuracyLabel = QtWidgets.QLabel("Accuracy:")
        self.accuracyValue = QtWidgets.QLabel("–")
        self.summaryGrid.addWidget(self.accuracyLabel, 0, 0, alignment=Qt.AlignLeft)
        self.summaryGrid.addWidget(self.accuracyValue, 0, 1, alignment=Qt.AlignRight)

        self.macroLabel = QtWidgets.QLabel("Macro Avg:")
        self.macroValues = QtWidgets.QLabel("–")
        self.summaryGrid.addWidget(self.macroLabel, 1, 0, alignment=Qt.AlignLeft)
        self.summaryGrid.addWidget(self.macroValues, 1, 1, alignment=Qt.AlignRight)

        self.weightedLabel = QtWidgets.QLabel("Weighted Avg:")
        self.weightedValues = QtWidgets.QLabel("–")
        self.summaryGrid.addWidget(self.weightedLabel, 2, 0, alignment=Qt.AlignLeft)
        self.summaryGrid.addWidget(self.weightedValues, 2, 1, alignment=Qt.AlignRight)

        self.summaryGrid.setColumnStretch(0, 1)
        self.summaryGrid.setColumnStretch(1, 1)
        metricsLayout.addLayout(self.summaryGrid)
        ov.addWidget(self.metricsGroup)
        self.ovBar = ChartCard("Class Distribution")
        self.ovRoc = ChartCard("ROC Curves")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ovBar, 1)
        row.addWidget(self.ovRoc, 1)
        ov.addLayout(row)

        # --- Table page
        tl = QtWidgets.QVBoxLayout(self.tablePage)
        self.table = QtWidgets.QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        tl.addWidget(self.table)

        # --- Feature Importance
        fl = QtWidgets.QVBoxLayout(self.fiPage)
        self.fiHeader = QtWidgets.QLabel(
            "Top 20 features by SHAP."
        )
        self.fiHeader.setWordWrap(True)
        fl.addWidget(self.fiHeader)

        self.fiTablesLayout = QtWidgets.QGridLayout()
        self.fiTablesLayout.setHorizontalSpacing(12)
        self.fiTablesLayout.setVerticalSpacing(12)
        self.fiTablesLayout.setAlignment(Qt.AlignTop)
        fl.addLayout(self.fiTablesLayout)
        fl.addStretch(1)


class DashboardPage(QtWidgets.QWidget):
    requestModify = Signal()
    requestSave = Signal()

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
        self.saveBtn = QtWidgets.QPushButton("save")
        self.saveBtn.clicked.connect(lambda: self.requestSave.emit())
        actions.addStretch()
        actions.addWidget(self.modifyBtn)
        actions.addWidget(self.saveBtn)
        root.addLayout(actions)

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
        # Table
        df = results.get("dataframe", pd.DataFrame())
        self.tabs.table.setModel(PandasModel(df))

        # Overview bar
        bins = results.get("bins", pd.DataFrame())
        ax = self.tabs.ovBar.ax
        ax.clear()
        if isinstance(bins, pd.DataFrame) and not bins.empty:
            ax.bar(bins["bin_name"].astype(str), bins["count"])
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
        self.tabs.ovBar.canvas.figure.tight_layout()
        self.tabs.ovBar.canvas.draw()

        # Overview ROC preview
        ax = self.tabs.ovRoc.ax
        ax.clear()
        for r in results.get("roc", []):
            ax.plot(r["fpr"], r["tpr"], label=f"{r['class']} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        self.tabs.ovRoc.canvas.figure.tight_layout()
        self.tabs.ovRoc.canvas.draw()

        # Feature importance tables (per class)
        self._clear_layout(self.tabs.fiTablesLayout)
        shap_importance = results.get("shap_importance", {})
        failure_classes = ["cal_1", "cal_2", "other"]
        normalized = {}
        if isinstance(shap_importance, dict):
            for key, df_val in shap_importance.items():
                normalized[str(key).strip().lower()] = df_val

        if normalized:
            added = 0
            for col_idx, cls in enumerate(failure_classes):
                df = normalized.get(cls)
                if not (isinstance(df, pd.DataFrame) and not df.empty):
                    continue
                group = QtWidgets.QGroupBox(f"{cls} – Top 20 Features")
                group.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
                )
                vbox = QtWidgets.QVBoxLayout(group)
                table = QtWidgets.QTableWidget()
                table.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
                )
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
                        "PASS Avg",
                        "PASS Std",
                    ]
                )
                table.verticalHeader().setVisible(False)
                table.setTextElideMode(Qt.ElideNone)
                table.setWordWrap(True)
                header = table.horizontalHeader()
                header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
                header.setStretchLastSection(True)
                table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
                table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
                table.setAlternatingRowColors(True)
                top = df.head(20)
                table.setRowCount(len(top))
                for idx, row in top.reset_index(drop=True).iterrows():
                    table.setItem(
                        idx, 0, QtWidgets.QTableWidgetItem(str(int(row["rank"])))
                    )
                    table.setItem(idx, 1, QtWidgets.QTableWidgetItem(str(row["feature"])))
                    table.setItem(
                        idx,
                        2,
                        QtWidgets.QTableWidgetItem(f"{float(row['importance']):.4f}"),
                    )
                    table.setItem(
                        idx,
                        3,
                        QtWidgets.QTableWidgetItem(f"{float(row.get('share_pct', 0)):.1f}"),
                    )
                    table.setItem(
                        idx,
                        4,
                        QtWidgets.QTableWidgetItem(f"{float(row.get('direction', 0)):.4f}"),
                    )
                    table.setItem(
                        idx,
                        5,
                        QtWidgets.QTableWidgetItem(
                            "" if pd.isna(row.get("failure_avg")) else f"{float(row.get('failure_avg')):.4f}"
                        ),
                    )
                    pass_avg = row.get("pass_avg")
                    pass_std = row.get("pass_std")
                    pass_avg_text = "" if pd.isna(pass_avg) else f"{float(pass_avg):.4f}"
                    pass_std_text = "" if pd.isna(pass_std) else f"{float(pass_std):.4f}"
                    table.setItem(idx, 6, QtWidgets.QTableWidgetItem(pass_avg_text))
                    table.setItem(idx, 7, QtWidgets.QTableWidgetItem(pass_std_text))
                table.resizeColumnsToContents()
                vbox.addWidget(table)
                self.tabs.fiTablesLayout.addWidget(group, 0, col_idx)
                self.tabs.fiTablesLayout.setColumnStretch(col_idx, 1)
                added += 1

            if added == 0:
                msg = QtWidgets.QLabel("No SHAP-based feature importance available.")
                msg.setStyleSheet("color:#666;")
                self.tabs.fiTablesLayout.addWidget(msg, 0, 0)
        else:
            msg = QtWidgets.QLabel("No SHAP-based feature importance available.")
            msg.setStyleSheet("color:#666;")
            self.tabs.fiTablesLayout.addWidget(msg, 0, 0)

        # Overview classification report
        report = results.get("classification_report", {})
        report_table = self.tabs.reportTable
        report_table.setRowCount(0)
        self.tabs.accuracyValue.setText("–")
        self.tabs.macroValues.setText("–")
        self.tabs.weightedValues.setText("–")
        if isinstance(report, dict) and report:
            class_rows = [
                (label, stats)
                for label, stats in report.items()
                if label not in {"accuracy", "macro avg", "weighted avg"}
            ]
            for row_idx, (label, stats) in enumerate(class_rows):
                report_table.insertRow(row_idx)
                cls_item = QtWidgets.QTableWidgetItem(str(label))
                cls_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                report_table.setItem(row_idx, 0, cls_item)

                if isinstance(stats, dict):
                    values = [
                        stats.get("precision", 0.0),
                        stats.get("recall", 0.0),
                        stats.get("f1-score", 0.0),
                        stats.get("support", 0),
                    ]

                for col_idx, val in enumerate(values, start=1):
                    if col_idx < 4:
                        text = f"{float(val):.4f}" if val != "" else ""
                    else:
                        text = f"{int(val)}" if isinstance(val, (int, float)) else ""
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    report_table.setItem(row_idx, col_idx, item)

            accuracy = report.get("accuracy")
            if accuracy is not None:
                self.tabs.accuracyValue.setText(f"{float(accuracy):.4f}")

            macro = report.get("macro avg") or {}
            if isinstance(macro, dict):
                macro_parts = [
                    macro.get("precision"),
                    macro.get("recall"),
                    macro.get("f1-score"),
                    macro.get("support"),
                ]
                macro_text = ", ".join(
                    [
                        part
                        for part in [
                            f"P {macro_parts[0]:.4f}" if macro_parts[0] is not None else "",
                            f"R {macro_parts[1]:.4f}" if macro_parts[1] is not None else "",
                            f"F1 {macro_parts[2]:.4f}" if macro_parts[2] is not None else "",
                            f"N {int(macro_parts[3])}" if macro_parts[3] is not None else "",
                        ]
                        if part
                    ]
                )
                self.tabs.macroValues.setText(macro_text or "–")

            weighted = report.get("weighted avg") or {}
            if isinstance(weighted, dict):
                weighted_parts = [
                    weighted.get("precision"),
                    weighted.get("recall"),
                    weighted.get("f1-score"),
                    weighted.get("support"),
                ]
                weighted_text = ", ".join(
                    [
                        part
                        for part in [
                            f"P {weighted_parts[0]:.4f}" if weighted_parts[0] is not None else "",
                            f"R {weighted_parts[1]:.4f}" if weighted_parts[1] is not None else "",
                            f"F1 {weighted_parts[2]:.4f}" if weighted_parts[2] is not None else "",
                            f"N {int(weighted_parts[3])}" if weighted_parts[3] is not None else "",
                        ]
                        if part
                    ]
                )
                self.tabs.weightedValues.setText(weighted_text or "–")
        else:
            report_table.setRowCount(1)
            placeholder = QtWidgets.QTableWidgetItem("Classification report unavailable.")
            placeholder.setFlags(Qt.ItemIsEnabled)
            report_table.setItem(0, 0, placeholder)
            report_table.setSpan(0, 0, 1, 5)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cisco Silicon Failure Characterization")
        self.resize(1280, 860)
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
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
