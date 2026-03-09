from __future__ import annotations

from PySide6 import QtCore
from PySide6.QtCore import QObject, QRunnable, Signal

from analysis import run_analysis, run_analysis_with_artifact


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
