from __future__ import annotations

import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt


HYPERPARAMETER_DEFAULTS = {
    "random_state": 67,
    "test_size": 0.2,
    "val_size": 0.2,
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "verbosity": 1,
    "verbose": False,
    "cat_verbose": 100,
    "mega_n_trials": 45,
}

HYPERPARAMETER_FIELDS = {
    "random_state": {
        "label": "Random State",
        "type": "int",
        "min": 0,
        "max": 999999,
    },
    "test_size": {
        "label": "Test Size",
        "type": "float",
        "min": 0.05,
        "max": 0.5,
        "step": 0.05,
        "decimals": 2,
    },
    "val_size": {
        "label": "Validation Size",
        "type": "float",
        "min": 0.05,
        "max": 0.4,
        "step": 0.05,
        "decimals": 2,
    },
    "iterations": {
        "label": "Iterations",
        "type": "int",
        "min": 50,
        "max": 5000,
    },
    "learning_rate": {
        "label": "Learning Rate",
        "type": "float",
        "min": 0.001,
        "max": 1.0,
        "step": 0.01,
        "decimals": 3,
    },
    "depth": {
        "label": "Depth",
        "type": "int",
        "min": 1,
        "max": 16,
    },
    "subsample": {
        "label": "Subsample",
        "type": "float",
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "decimals": 2,
    },
    "colsample_bytree": {
        "label": "Column Sample",
        "type": "float",
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "decimals": 2,
    },
    "verbosity": {
        "label": "XGBoost Verbosity",
        "type": "int",
        "min": 0,
        "max": 3,
    },
    "verbose": {
        "label": "Show Fit Logs",
        "type": "bool",
    },
    "cat_verbose": {
        "label": "CatBoost Verbosity",
        "type": "int",
        "min": 0,
        "max": 500,
    },
    "mega_n_trials": {
        "label": "Optuna Trials",
        "type": "int",
        "min": 5,
        "max": 500,
    },
}

MODEL_HYPERPARAMETERS = {
    "CatBoost": [
        "random_state",
        "test_size",
        "iterations",
        "learning_rate",
        "depth",
        "cat_verbose",
    ],
    "XGBoost": [
        "random_state",
        "test_size",
        "iterations",
        "learning_rate",
        "depth",
        "subsample",
        "colsample_bytree",
        "verbosity",
        "verbose",
    ],
    "Mega Multiclass XGBoost": [
        "random_state",
        "test_size",
        "val_size",
        "mega_n_trials",
    ],
    "Mega OVR XGBoost": [
        "random_state",
        "test_size",
        "val_size",
        "mega_n_trials",
    ],
    "Mega Hierarchical XGBoost": [
        "random_state",
        "test_size",
        "val_size",
        "mega_n_trials",
    ],
}


def default_hyperparameters() -> dict:
    return dict(HYPERPARAMETER_DEFAULTS)


def coerce_hyperparameters(model_type: str, values: dict | None) -> dict:
    coerced = default_hyperparameters()
    values = values or {}
    for key, raw in values.items():
        if key not in HYPERPARAMETER_FIELDS:
            coerced[key] = raw
    for key, spec in HYPERPARAMETER_FIELDS.items():
        if key not in values:
            continue
        raw = values.get(key)
        try:
            if spec["type"] == "int":
                coerced[key] = int(raw)
            elif spec["type"] == "float":
                coerced[key] = float(raw)
            elif spec["type"] == "bool":
                coerced[key] = bool(raw)
        except Exception:
            continue

    return coerced


def summarize_hyperparameters(model_type: str, values: dict | None) -> str:
    values = coerce_hyperparameters(model_type, values)
    extra_keys = [
        k
        for k in values.keys()
        if k not in HYPERPARAMETER_DEFAULTS and values.get(k) not in (None, "", {})
    ]
    if model_type == "CatBoost":
        summary = (
            f"iters={values['iterations']}, lr={values['learning_rate']:.3f}, "
            f"depth={values['depth']}, seed={values['random_state']}"
        )
    elif model_type == "XGBoost":
        summary = (
            f"iters={values['iterations']}, lr={values['learning_rate']:.3f}, "
            f"depth={values['depth']}, subsample={values['subsample']:.2f}"
        )
    elif model_type in MODEL_HYPERPARAMETERS:
        summary = (
            f"seed={values['random_state']}, test={values['test_size']:.2f}, "
            f"val={values['val_size']:.2f}, trials={values['mega_n_trials']}"
        )
    else:
        summary = "Defaults"
    if extra_keys:
        summary += f", tuned={len(extra_keys)}"
    return summary


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
        self.search.setPlaceholderText("Search columns...")
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
        return None

    def headerData(self, s, orient, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[s]) if orient == Qt.Horizontal else str(s)


class KPIBox(QtWidgets.QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.setObjectName("kpiBox")
        self.setMinimumHeight(110)
        self.setMaximumHeight(130)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        self.value = QtWidgets.QLabel("-", alignment=Qt.AlignCenter)
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
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)


class HyperparameterDialog(QtWidgets.QDialog):
    def __init__(self, parent, model_type: str, values: dict | None):
        super().__init__(parent)
        self.setWindowTitle("Training Hyperparameters")
        self.resize(480, 420)
        self._model_type = model_type
        self._values = coerce_hyperparameters(model_type, values)
        self._widgets: dict[str, QtWidgets.QWidget] = {}

        root = QtWidgets.QVBoxLayout(self)

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setProperty("muted", True)
        root.addWidget(self.info)

        self.form = QtWidgets.QFormLayout()
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        root.addLayout(self.form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.RestoreDefaults
        )
        root.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )

        self._rebuild_form()

    def _restore_defaults(self):
        self._values = default_hyperparameters()
        self._rebuild_form()

    def _clear_form(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets = {}

    def _rebuild_form(self):
        self._clear_form()
        field_keys = MODEL_HYPERPARAMETERS.get(self._model_type, [])
        self.info.setText(
            f"Editing hyperparameters for {self._model_type or 'the selected model'}."
        )
        for key in field_keys:
            spec = HYPERPARAMETER_FIELDS[key]
            widget: QtWidgets.QWidget
            value = self._values.get(key, HYPERPARAMETER_DEFAULTS[key])
            if spec["type"] == "int":
                spin = QtWidgets.QSpinBox()
                spin.setRange(int(spec["min"]), int(spec["max"]))
                spin.setValue(int(value))
                widget = spin
            elif spec["type"] == "float":
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(float(spec["min"]), float(spec["max"]))
                spin.setDecimals(int(spec.get("decimals", 2)))
                spin.setSingleStep(float(spec.get("step", 0.1)))
                spin.setValue(float(value))
                widget = spin
            else:
                check = QtWidgets.QCheckBox()
                check.setChecked(bool(value))
                widget = check
            self._widgets[key] = widget
            self.form.addRow(spec["label"], widget)

    def values(self) -> dict:
        out = dict(self._values)
        for key, widget in self._widgets.items():
            spec = HYPERPARAMETER_FIELDS[key]
            if spec["type"] == "int":
                out[key] = int(widget.value())
            elif spec["type"] == "float":
                out[key] = float(widget.value())
            else:
                out[key] = bool(widget.isChecked())
        return out
