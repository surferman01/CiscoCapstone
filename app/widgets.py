
from PySide6 import QtCore, QtGui, QtWidgets

class DropZone(QtWidgets.QFrame):
    fileDropped = QtCore.Signal(str)
    def __init__(self, text="Add Data\n(drag & drop)"):
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(text)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        f = self.label.font(); f.setPointSize(18); self.label.setFont(f)
        layout.addWidget(self.label)
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QtGui.QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".csv",".parquet",".pq")):
                self.fileDropped.emit(path); return

class ClickTile(QtWidgets.QFrame):
    clicked = QtCore.Signal()
    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.setObjectName("tile")
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        layout = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel(title)
        tf = self.title.font(); tf.setPointSize(16); tf.setBold(True); self.title.setFont(tf)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.title)
        self.subtitle = QtWidgets.QLabel(subtitle)
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.subtitle)
        layout.addStretch()
    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)
