from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QThreadPool, Slot

from analysis import save_model_artifact
from styles import APP_NAME
from ui.pages import DashboardPage, SplashPage, TrainingPage
from ui.saved_runs import (
    build_saved_run_html,
    results_to_saved_payload,
    safe_filename,
    saved_payload_to_results,
)
from ui.workers import AnalyzeWorker, TrainWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(1100, max(720, int(avail.width() * 0.92)))
            h = min(720, max(560, int(avail.height() * 0.88)))
            self.resize(w, h)
        else:
            self.resize(1000, 680)
        self.setMinimumSize(680, 500)

        base_dir = os.path.dirname(os.path.dirname(__file__))
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
                self.themeToggle.setText("LIGHT")
                self.themeToggle.setToolTip("Switch to light mode")
            else:
                self.themeToggle.setText("DARK")
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

        payload = results_to_saved_payload(self.last_results)
        snapshot = {
            "schema_version": 1,
            "run_name": run_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "payload": payload,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{safe_filename(run_name)}"
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

    def _choose_saved_run_file(
        self, title: str, prompt: str
    ) -> tuple[str | None, dict | None]:
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

    def export_saved_run_html(self):
        path, snapshot = self._choose_saved_run_file(
            "Export Saved Run", "Select a saved run to export:"
        )
        if not path or not snapshot:
            return

        try:
            results = saved_payload_to_results(snapshot.get("payload", {}) or {})
            report_html = build_saved_run_html(snapshot, results)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return

        base_name = safe_filename(
            str(snapshot.get("run_name", "")).strip()
            or os.path.splitext(os.path.basename(path))[0]
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
            results = saved_payload_to_results(payload)
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
    app.setApplicationName(APP_NAME)

    font = QtGui.QFont("Inter")
    if not QtGui.QFontInfo(font).exactMatch():
        font = QtGui.QFont("Helvetica Neue")
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
