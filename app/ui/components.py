from __future__ import annotations

import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt


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
