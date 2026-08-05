#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tool_GUI.py MultiAgent GUI (Agent Tabs) with pinned New Agent tab,
#               per-agent scrollable panel, improved numbering, and top-right logo.
#               FIXES:
#               - New Agent tab: stay on the newly created tab (no jump back to Agent 1)
#               - Progress steps: show "Open" for Excel/other files, open non-HTML via OS
#
#               Further updates (2025-09):
#               - Report viewer: fixed-aspect (16:10) proportional scaling so HTML never
#                 gets squeezed on smaller (laptop) displays. Toggle between "Fixed 16:10"
#                 and "Responsive" from the Report panel.
#               - Run Output layout: default to 50/50 with Parameters on laptops.
#               - "Progress" / "Raw log" tabs: shorter height to free space.
#               - Trimmed utility button heights so the step list gets more room.
#
#               Stability fixes (2025-09-25):
#               - Per-agent run exclusivity and robust QThread cleanup to eliminate
#                 "QThread: Destroyed while thread is still running" crashes when
#                 re-running quickly or with many agents.
#               - Use system monospace font fallback instead of hard-coding
#                 "JetBrains Mono" to avoid noisy font warnings.
#               - Guard closing tabs/window while a worker thread is still stopping.
#
from __future__ import annotations

import os
import sys
import shlex
import subprocess
import threading
import json as _json
import tempfile
import re  # for path auto-detection from progress details
from typing import Dict, Tuple, List, Optional, Callable

from PyQt5.QtCore import (
    Qt, QUrl, QSize, pyqtSignal, QObject, QThread, QTimer, QEasingCurve, QRect
)
from PyQt5.QtGui import (
    QFont, QIcon, QKeySequence, QDesktopServices, QPixmap, QColor, QPainter, QBrush, QFontDatabase
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QToolButton,
    QHBoxLayout, QVBoxLayout, QFrame, QFileDialog, QSplitter, QStyle,
    QLineEdit, QFormLayout, QMessageBox, QToolBar, QSizePolicy, QAction,
    QPlainTextEdit, QTabWidget, QTextEdit, QScrollArea, QTabBar,
    QProgressBar, QSpacerItem, QGridLayout, QStyleFactory, QLayout, QButtonGroup,
    QComboBox
)

from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings  # Qt5 location

QT_API = "PyQt5"
IS_MAC = sys.platform == "darwin"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

SKILL_NAMES = {
    "P1": "Skill 1",
    "P2": "Skill 2",
    "P3": "Skill 3",
}

# ---------------- Small helpers ----------------
def clear_layout(layout: QLayout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)

def shlex_join(cmd: List[str]) -> str:
    try:
        return shlex.join(cmd)  # type: ignore[attr-defined]
    except Exception:
        def q(s: str) -> str:
            if not s:
                return "''"
            if any(ch.isspace() for ch in s) or any(ch in s for ch in "\"'\\$"):
                return "'" + s.replace("'", "'\"'\"'") + "'"
            return s
        return "".join([q(x) + " " for x in cmd]).strip()

def _monospace_font(point_size: int = 10) -> QFont:
    """
    Return a robust monospace font available on the host OS.
    Avoids hard-coded families that may be missing (e.g., JetBrains Mono).
    """
    try:
        f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        f.setPointSize(point_size)
        return f
    except Exception:
        f = QFont("Menlo" if IS_MAC else "Courier New")
        f.setPointSize(point_size)
        return f

# ---- Sample HTML shown initially in the report viewer --------------------
SAMPLE_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Report Preview</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0; padding: 24px; color: #0b3e4a; line-height: 1.5; background:#f6fbfe;}
header { background: linear-gradient(#eaf6fb, #ffffff); border: 1px solid #cbe3ed;
         padding: 18px 20px; border-radius: 14px; }
h1 { margin: 0; font-size: 20px; }
.muted { color: #53727b; }
</style>
</head>
<body>
<header>
  <h1>Report Preview</h1>
  <div class="muted">Use <b>Import</b> to load a local HTML file, or <b>Open in browser</b> to view externally.</div>
</header>
<p class="muted">Drop an .html file into this panel to open it.</p>
<div style="height: 600px"></div>
</body>
</html>
"""

# ---------------- Popup HTML viewer (kept) ----------------
class HtmlPopup(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AgentLnc Viewer")
        self.resize(1100, 700)

        self.view = QWebEngineView()
        self.setCentralWidget(self.view)

        if QWebEngineSettings:
            s = self.view.settings()
            s.setAttribute(QWebEngineSettings.ShowScrollBars, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        tb = QToolBar("Viewer")
        tb.setMovable(False)
        tb.setIconSize(QSize(28, 28))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        toggle_fullscreen = QAction(self.style().standardIcon(QStyle.SP_TitleBarMaxButton), "Full Screen (F11)", self)
        toggle_fullscreen.setShortcut(QKeySequence("F11"))
        toggle_fullscreen.triggered.connect(self._toggle_fullscreen)
        tb.addAction(toggle_fullscreen)

        close_act = QAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "Close (Esc)", self)
        close_act.setShortcut(QKeySequence("Esc"))
        close_act.triggered.connect(self.close)
        tb.addAction(close_act)

        self._is_fullscreen = False

    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self._is_fullscreen = not self._is_fullscreen

    def load_html(self, html: str, base_url: QUrl | None = None):
        self.view.setHtml(html, base_url or QUrl.fromLocalFile(APP_DIR + os.sep))

    def load_url(self, url: QUrl):
        self.view.load(url)

# ---------------- Live raw log popup (optional) ----------------
class LogPopup(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Log - Live Output")
        self.resize(900, 600)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(_monospace_font(10))
        self.view.setMaximumBlockCount(5000)
        self.setCentralWidget(self.view)

        tb = QToolBar("Log")
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        copy_act = QAction(self.style().standardIcon(QStyle.SP_DialogYesButton), "Copy all", self)
        copy_act.triggered.connect(self._copy_all)
        tb.addAction(copy_act)

        save_act = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save...", self)
        save_act.triggered.connect(self._save_as)
        tb.addAction(save_act)

        clear_act = QAction(self.style().standardIcon(QStyle.SP_DialogResetButton), "Clear", self)
        clear_act.triggered.connect(self.view.clear)
        tb.addAction(clear_act)

    def append(self, text: str):
        self.view.appendPlainText(text.rstrip("\n"))

    def _copy_all(self):
        self.view.selectAll()
        self.view.copy()
        self.view.moveCursor(self.view.textCursor().End)

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save log", APP_DIR, "Text Files (*.txt);;All Files (*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.view.toPlainText())

# ---------------- Report panel with fixed-aspect scaler ----------------

class DroppableWebView(QWebEngineView):
    fileDropped = pyqtSignal(QUrl)
    resized = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            self.resized.emit()
        except Exception:
            pass

    def dragEnterEvent(self, e):
        md = e.mimeData()
        accept = False
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile():
                    if os.path.splitext(u.toLocalFile())[1].lower() in (".html", ".htm", ".xhtml"):
                        accept = True
                        break
                elif u.scheme() in ("http", "https"):
                    accept = True
                    break
        e.acceptProposedAction() if accept else e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile() and os.path.splitext(u.toLocalFile())[1].lower() in (".html", ".htm", ".xhtml"):
                    self.fileDropped.emit(QUrl.fromLocalFile(u.toLocalFile()))
                    e.acceptProposedAction()
                    return
                elif u.scheme() in ("http", "https"):
                    self.fileDropped.emit(u)
                    e.acceptProposedAction()
                    return
        e.ignore()



class ReportPanel(QFrame):
    """HTML report viewer without transform-scaling.
    - Renders pages natively in QWebEngineView (no iframe wrappers).
    - Table scrollbars and nested scroll areas behave like a normal browser.
    - Adds Zoom controls and an optional 'Fit width (auto)' mode implemented
      with QWebEngineView.setZoomFactor rather than CSS transforms.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # track last content to allow 'Open in browser' to open original
        self.last_url: QUrl | None = None
        self.last_html: str | None = None
        self.base_url: QUrl = QUrl.fromLocalFile(APP_DIR + os.sep)

        # zoom & fit state
        self.zoom_factor: float = 0.8
        self._applying_fit: bool = False
        self._fit_width_enabled: bool = False

        # view height ratio (can be changed to 0.75 / 0.9 etc.)
        self._view_height_ratio: float = 1

        # ===== Layout =====
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(10)
        self._outer_layout = outer  # save reference for height calculation

        header = QLabel("Report")
        header.setObjectName("h1")
        outer.addWidget(header)
        self.header = header

        self.view = DroppableWebView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if QWebEngineSettings:
            s = self.view.settings()
            s.setAttribute(QWebEngineSettings.ShowScrollBars, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        # --- Fix A: force classic, always-visible scrollbars in the embedded viewer
        # Import here to keep this change self-contained.
        try:
            from PyQt5.QtWebEngineCore import QWebEngineScript  # Qt5
        except Exception:
            QWebEngineScript = None

        # CSS for classic scrollbars + reserved gutter so layout doesn't jump
        css_scrollbars = r"""
        :root, body, * { scrollbar-gutter: stable both-edges; }
        ::-webkit-scrollbar { width: 12px !important; height: 12px !important; }
        ::-webkit-scrollbar-thumb {
          background-color: rgba(90,90,90,.55) !important;
          border-radius: 8px !important;
          border: 2px solid rgba(255,255,255,.6) !important;
        }
        ::-webkit-scrollbar-thumb:hover { background-color: rgba(60,60,60,.65) !important; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,.06) !important; }
        """

        js_inject = f"""
        (function(){{
          try {{
            if (document.getElementById('force-classic-scrollbars')) return;
            var s = document.createElement('style');
            s.id = 'force-classic-scrollbars';
            s.textContent = `{css_scrollbars}`;
            document.documentElement.appendChild(s);
          }} catch(e) {{}}
        }})();
        """

        if QWebEngineScript is not None:
            try:
                script = QWebEngineScript()
                script.setName("ForceClassicScrollbars")
                # Run as soon as DOM is ready so scrollbars are present before user interacts
                script.setInjectionPoint(QWebEngineScript.DocumentReady)
                script.setRunsOnSubFrames(True)  # apply to iframes/subframes as well
                script.setWorldId(QWebEngineScript.MainWorld)  # affect page's main world
                script.setSourceCode(js_inject)
                self.view.page().profile().scripts().insert(script)
            except Exception:
                # Fallback: inject after each load if script API fails for any reason
                self.view.loadFinished.connect(lambda *_: self.view.page().runJavaScript(js_inject))
        else:
            # Fallback when QWebEngineScript isn't available (older builds):
            self.view.loadFinished.connect(lambda *_: self.view.page().runJavaScript(js_inject))

        # make view vertically expandable
        outer.addWidget(self.view, 1)

        self.view.fileDropped.connect(self.load_url)
        self.view.loadFinished.connect(self._on_load_finished)
        try:
            # available on Qt >= 5.7
            self.view.page().contentsSizeChanged.connect(lambda *_: self._maybe_apply_fit_width())
        except Exception:
            pass
        try:
            self.view.resized.connect(self._maybe_apply_fit_width)
        except Exception:
            pass

        # ---- Controls (bottom) ----
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addStretch(1)

        self.import_btn = QToolButton(); self.import_btn.setText("Import")
        self.import_btn.setToolTip("Import a local .html file")
        self.import_btn.setCursor(Qt.PointingHandCursor)
        self.import_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.import_btn.setIconSize(QSize(24,24)); self.import_btn.setProperty("class","pill"); self.import_btn.setMinimumHeight(28)
        controls_layout.addWidget(self.import_btn)

        self.full_btn = QToolButton(); self.full_btn.setText("Open in browser")
        self.full_btn.setToolTip("Open the report in your default browser")
        self.full_btn.setCursor(Qt.PointingHandCursor)
        browser_icon_path = os.path.join(APP_DIR, "icons", "browser.svg")
        self.full_btn.setIcon(QIcon.fromTheme("internet-web-browser", QIcon(browser_icon_path)))
        self.full_btn.setIconSize(QSize(24,24)); self.full_btn.setProperty("class","pill"); self.full_btn.setMinimumHeight(28)
        controls_layout.addWidget(self.full_btn)

        # Fit width toggle
        self.fit_btn = QToolButton(); self.fit_btn.setText("Fit width (auto)")
        self.fit_btn.setCheckable(True); self.fit_btn.setChecked(False)
        self.fit_btn.setCursor(Qt.PointingHandCursor); self.fit_btn.setProperty("class","pill"); self.fit_btn.setMinimumHeight(28)
        controls_layout.addWidget(self.fit_btn)

        # Zoom controls
        self.zoom_out_btn = QToolButton(); self.zoom_out_btn.setText("−"); self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.setCursor(Qt.PointingHandCursor); self.zoom_out_btn.setProperty("class","pill"); self.zoom_out_btn.setMinimumHeight(28); self.zoom_out_btn.setFixedWidth(36)
        controls_layout.addWidget(self.zoom_out_btn)

        self.zoom_reset_btn = QToolButton(); self.zoom_reset_btn.setText("100%")
        self.zoom_reset_btn.setCursor(Qt.PointingHandCursor); self.zoom_reset_btn.setProperty("class","pill"); self.zoom_reset_btn.setMinimumHeight(28)
        controls_layout.addWidget(self.zoom_reset_btn)

        self.zoom_in_btn = QToolButton(); self.zoom_in_btn.setText("+"); self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.setCursor(Qt.PointingHandCursor); self.zoom_in_btn.setProperty("class","pill"); self.zoom_in_btn.setMinimumHeight(28); self.zoom_in_btn.setFixedWidth(36)
        controls_layout.addWidget(self.zoom_in_btn)

        # live zoom label (read-only)
        self.zoom_lbl = QLabel("100%"); self.zoom_lbl.setStyleSheet("color:#53727b; font-weight:600; padding-left:6px;")
        controls_layout.addWidget(self.zoom_lbl)

        # Wrap controls in a QWidget for easier height measurement
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls_layout)
        outer.addWidget(self.controls_widget)

        # wire up actions
        self.import_btn.clicked.connect(self.import_html)
        self.full_btn.clicked.connect(self.open_in_browser)
        self.fit_btn.toggled.connect(self._on_fit_toggled)
        self.zoom_in_btn.clicked.connect(lambda: self._user_zoom(self.zoom_factor + 0.1))
        self.zoom_out_btn.clicked.connect(lambda: self._user_zoom(self.zoom_factor - 0.1))
        self.zoom_reset_btn.clicked.connect(lambda: self._user_zoom(1.0))

        # initial content
        self.load_html(SAMPLE_HTML)

        # On first show/resize, set view height by ratio
        QTimer.singleShot(0, self._apply_view_height)

    # ---------- helpers ----------
    def _user_zoom(self, z: float):
        # Any explicit user zoom disables fit width
        if self._fit_width_enabled:
            self._fit_width_enabled = False
            try:
                self.fit_btn.blockSignals(True)
                self.fit_btn.setChecked(False)
            finally:
                self.fit_btn.blockSignals(False)
        self._apply_zoom(z)

    def _apply_zoom(self, z: float):
        z = max(0.25, min(3.0, float(z)))
        self.zoom_factor = z
        try:
            self.view.setZoomFactor(z)
        except Exception:
            pass
        self.zoom_lbl.setText(f"{int(round(z*100))}%")

    def _on_fit_toggled(self, enabled: bool):
        self._fit_width_enabled = bool(enabled)
        if enabled:
            self._maybe_apply_fit_width()
        else:
            # revert to last explicit zoom (keep current if user had none)
            self._apply_zoom(self.zoom_factor or 1.0)

    def _maybe_apply_fit_width(self):
        if not self._fit_width_enabled:
            return
        # To measure in CSS px consistently, ensure base zoom during measurement.
        if self._applying_fit:
            return
        self._applying_fit = True
        try:
            self.view.setZoomFactor(1.0)
        except Exception:
            pass

        js = """(function(){
            var b=document.body, e=document.documentElement;
            var sw = Math.max(b?b.scrollWidth:0, e?e.scrollWidth:0);
            return sw || 0;
        })();"""
        try:
            self.view.page().runJavaScript(js, self._fit_width_from_scrollwidth)
        except Exception:
            self._applying_fit = False

    def _fit_width_from_scrollwidth(self, scroll_width_css):
        try:
            sw = float(scroll_width_css or 0.0)
        except Exception:
            sw = 0.0
        if sw <= 0.0:
            self._applying_fit = False
            return
        # Compute zoom so that viewport device width maps to the document's CSS width.
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            try:
                dpr = float(self.view.devicePixelRatioF())
            except Exception:
                dpr = 1.0
        vp_w = max(1, int(self.view.size().width()))
        z = vp_w / (dpr * sw)
        z = max(0.25, min(3.0, z))
        try:
            self.view.setZoomFactor(z)
        except Exception:
            pass
        self.zoom_factor = z
        self.zoom_lbl.setText(f"{int(round(z*100))}%")
        self._applying_fit = False

    def _on_load_finished(self, ok: bool):
        # After each content load, set height again (avoids initial sizeHint errors)
        self._apply_view_height()
        if self._fit_width_enabled:
            QTimer.singleShot(0, self._maybe_apply_fit_width)
        else:
            self._apply_zoom(self.zoom_factor or 1.0)

    # ---------- public load methods ----------
    def load_html(self, html: str, base_url: QUrl | None = None):
        self.last_html = html
        self.last_url = None
        self.view.setHtml(html, base_url or self.base_url)

    def load_url(self, url: QUrl):
        self.last_url = url
        self.last_html = None
        self.view.load(url)

    def import_html(self):
        start_dir = os.path.expanduser("~")
        dlg = QFileDialog(self, "Open HTML file", start_dir)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setNameFilters([
            "HTML Files (*.html *.htm *.xhtml *.HTML *.HTM *.XHTML)",
            "All Files (*)",
        ])
        dlg.selectNameFilter("HTML Files (*.html *.htm *.xhtml *.HTML *.HTM *.XHTML)")
        dlg.resize(1000, 680)
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            if path:
                self.load_url(QUrl.fromLocalFile(os.path.abspath(path)))

    def open_in_browser(self):
        def _open_temp_html(html_text: str):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="agentlnc_", mode="w", encoding="utf-8") as tmp:
                tmp.write(html_text or "")
                temp_path = tmp.name
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(temp_path)))

        # Open the original content — not a wrapper
        if self.last_url is not None:
            QDesktopServices.openUrl(self.last_url); return
        if self.last_html is not None:
            _open_temp_html(self.last_html); return
        try:
            self.view.page().toHtml(_open_temp_html)
        except Exception:
            _open_temp_html("<h2>Unable to capture current HTML.</h2>")

    # ===== New: auto-adjust height to 80% =====
    def _apply_view_height(self):
        """Based on current ReportPanel height, compute target view height = 80% of available height."""
        try:
            # Layout margins + widget heights + spacing
            m = self._outer_layout.contentsMargins()
            spacing = self._outer_layout.spacing()

            header_h = self.header.sizeHint().height() if self.header is not None else 0
            controls_h = self.controls_widget.sizeHint().height() if hasattr(self, "controls_widget") else 0

            # outer: [header][view][controls] -> two gaps
            gaps = spacing * 2

            total_margins = m.top() + m.bottom()
            available = self.height() - (total_margins + header_h + controls_h + gaps)
            available = max(0, available)

            target = int(max(200, available * float(self._view_height_ratio)))

            # Use min/max together to give layout a clear height signal
            self.view.setMinimumHeight(target)
            self.view.setMaximumHeight(target)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_view_height()

    def showEvent(self, event):
        super().showEvent(event)
        # On first show, recalc after actual rendering
        QTimer.singleShot(0, self._apply_view_height)

class ProcessRunner(QObject):
    started = pyqtSignal(list)
    stdoutLine = pyqtSignal(str)
    stderrLine = pyqtSignal(str)
    finished = pyqtSignal(int, str, str)  # returncode, full_stdout, full_stderr

    def __init__(self, cmd: List[str], cwd: Optional[str] = None):
        super().__init__()
        self.cmd = cmd; self.cwd = cwd

    def run(self):
        self.started.emit(self.cmd)
        out_buf: List[str] = []; err_buf: List[str] = []
        try:
            env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, universal_newlines=True, env=env,
            )
            def _reader(stream, emit, sink: List[str]):
                for line in iter(stream.readline, ""):
                    sink.append(line); emit(line)
                try: stream.close()
                except Exception: pass
            t_out = threading.Thread(target=_reader, args=(proc.stdout, self.stdoutLine.emit, out_buf), daemon=True)
            t_err = threading.Thread(target=_reader, args=(proc.stderr, self.stderrLine.emit, err_buf), daemon=True)
            t_out.start(); t_err.start()
            rc = proc.wait()
            t_out.join(timeout=0.2); t_err.join(timeout=0.2)
            self.finished.emit(rc, "".join(out_buf), "".join(err_buf))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}\n"
            err_buf.append(msg); self.stderrLine.emit(msg)
            self.finished.emit(-1, "".join(out_buf), "".join(err_buf))

# ---------------- Chips & Progress board ----------------
class Chip(QLabel):
    def __init__(self, text: str, kind: str = "stage", parent=None):
        super().__init__(text, parent)
        self.setMargin(6)
        self.setStyleSheet({
            "stage":  "QLabel { border:1px solid #b9d9ff; background:#e8f3ff; color:#084e8a; border-radius:12px; }",
            "kind":   "QLabel { border:1px solid #b7e4b7; background:#eaf7ea; color:#0f5132; border-radius:12px; }",
            "run":    "QLabel { border:1px solid #c7e5ef; background:#eef8fb; color:#0d6578; border-radius:12px; }",
            "error":  "QLabel { border:1px solid #f4c7c7; background:#fdecec; color:#a12626; border-radius:12px; }",
        }.get(kind, "QLabel { border:1px solid #cbd5e1; background:#eef2f7; color:#334155; border-radius:12px; }"))

class StepWidget(QFrame):
    openPathRequested = pyqtSignal(str)
    def __init__(self, step:int, total:int, title:str, run:str, stage:str, kind:str, parent=None):
        super().__init__(parent)
        self.setObjectName("StepCard"); self.setFrameStyle(QFrame.NoFrame)
        self._step = step; self._total = total; self._last_value = 0

        outer = QVBoxLayout(self); outer.setContentsMargins(12,10,12,10); outer.setSpacing(8)
        # Row 1
        row1 = QHBoxLayout(); row1.setSpacing(8)

        # Dot indicator
        self.dot = QFrame()
        self.dot.setFixedSize(10, 10)
        self.dot.setStyleSheet("background:#295f7b; border-radius:5px;")

        self.title = QLabel(f"Step {step}/{total}: {title}")
        self.title.setStyleSheet("font-weight:600; color:#0b3e4a;")
        row1.addWidget(self.dot); row1.addWidget(self.title, 1)
        meta = QHBoxLayout(); meta.setSpacing(6)
        self.chip_run = Chip(run, "run"); self.chip_stage = Chip(stage or "stage", "stage"); self.chip_kind = Chip(kind or "update", "kind")
        for c in (self.chip_run, self.chip_stage, self.chip_kind): meta.addWidget(c)
        row1.addLayout(meta); outer.addLayout(row1)
        # Detail
        self.detail = QLabel(""); self.detail.setWordWrap(True); self.detail.setStyleSheet("color:#3b5560;")
        outer.addWidget(self.detail)
        # Progress
        self.bar = QProgressBar(); self.bar.setRange(0,0); self.bar.setTextVisible(True); self.bar.setFormat("%p%")
        self.bar.setStyleSheet("""
            QProgressBar { background:#e8f3f8; border:1px solid #cfe5ee; border-radius:6px; text-align:center; height:18px; }
            QProgressBar::chunk { background:#1b7a8e; }
        """)
        outer.addWidget(self.bar)
        # Actions
        act = QHBoxLayout(); act.addStretch(1)
        self.open_btn = QToolButton(); self.open_btn.setText("Open report"); self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.setProperty("class","pill"); self.open_btn.setVisible(False); self.open_btn.setMinimumHeight(28)
        self.open_btn.clicked.connect(self._emit_open)
        act.addWidget(self.open_btn, 0); outer.addLayout(act)

        self.setStyleSheet("""
            QFrame#StepCard { border:1px solid #d8ecf3; border-radius:12px; background:#ffffff; }
            QToolButton[class="pill"] {
                border:1px solid #0b3e4a; border-radius:12px; padding:6px 14px; color:#0b3e4a; background:#eff7fb; min-height:28px;
            }
            QToolButton[class="pill"]:hover { background:#e4f2f8; }
        """)

    @property
    def step(self): return self._step

    def _emit_open(self):
        path = self.open_btn.property("path")
        if path: self.openPathRequested.emit(path)

    def set_detail(self, text: str): self.detail.setText(text or "")
    def set_stage(self, stage:str): self.chip_stage.setText(stage or "stage")
    def set_kind(self, kind:str):
        self.chip_kind.setText(kind or "update")
        if (kind or "").lower() == "error":
            self.chip_kind.setStyleSheet("QLabel { border:1px solid #f4c7c7; background:#fdecec; color:#a12626; border-radius:12px; }")
    def set_title(self, title:str): self.title.setText(f"Step {self._step}/{self._total}: {title}")

    def show_open(self, path:str):
        p = (path or "").strip()
        if not p:
            self.open_btn.setVisible(False)
            self.open_btn.setProperty("path", "")
            return
        ext = os.path.splitext(p)[1].lower()
        self.open_btn.setText("Open report" if ext in (".html", ".htm", ".xhtml") else "Open file")
        self.open_btn.setProperty("path", p)
        self.open_btn.setVisible(True)

    def _animate_to(self, new_val:int):
        new_val = max(0, min(100, int(new_val)))
        if self.bar.maximum() == 0: self.bar.setRange(0,100); self.bar.setValue(self._last_value)
        anim = QPropertyAnimation(self.bar, b"value", self)
        anim.setDuration(220); anim.setStartValue(self._last_value); anim.setEndValue(new_val)
        anim.setEasingCurve(QEasingCurve.InOutCubic); anim.start(QPropertyAnimation.DeleteWhenStopped)
        self._last_value = new_val

    def _set_dot(self, color: str):
        self.dot.setStyleSheet(f"background:{color}; border-radius:5px;")

    # robust path inference for non-HTML artifacts
    def _extract_path_from_payload(self, payload: dict) -> str:
        extra = payload.get("extra") or {}
        for key in ("path", "file", "artifact", "url"):
            v = (extra.get(key) or "").strip()
            if v:
                return v
        blobs = []
        for k in ("detail", "title"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                blobs.append(v)
        blob = " ".join(blobs)
        if not blob:
            return ""
        m = re.search(
            r'((?:[A-Za-z]:[\\/]|/|\.{1,2}[\\/])?[^\s\'"<>|]+?\.(?:xhtml|html|htm|xlsx|xls|csv|tsv|pdf|json|txt|png|jpe?g|svg))',
            blob
        )
        if not m:
            return ""
        p = m.group(1).strip().rstrip('.,;:)]}')
        if p.startswith(("http://", "https://", "file://")):
            return p
        return os.path.abspath(p) if not os.path.isabs(p) else p

    def apply_payload(self, payload:dict):
        self.set_stage((payload.get("stage") or "").lower())
        self.set_kind((payload.get("kind") or "").lower())
        self.set_title(payload.get("title",""))
        self.set_detail(payload.get("detail",""))

        extra = payload.get("extra") or {}
        path = extra.get("path") or self._extract_path_from_payload(payload)
        self.show_open(path)

        kind = (payload.get("kind") or "").lower()
        step = int(payload.get("step") or self._step)
        total= int(payload.get("total") or self._total)
        perc = int(payload.get("perc") or 0)

        base = (max(0, step-1) / max(1, total)) * 100.0
        span = 100.0 / max(1, total)
        local = int(round((perc - base) / span * 100.0))

        if step <= 1 and perc < span*0.15:
            pass
        else:
            self._animate_to(local)

        if kind == "done":
            self._animate_to(100); self._set_dot("#16a34a")
        elif kind == "error":
            self.bar.setRange(0,100); self._animate_to(max(self._last_value, 5))
            self._set_dot("#dc2626")
        else:
            if self.bar.maximum() == 100 and self._last_value < 100:
                self._set_dot("#295f7b")
            elif self.bar.maximum() == 0:
                self._set_dot("#1b7a8e")

class ProgressBoard(QWidget):
    openPathRequested = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(8)

        # Overall header
        head = QFrame(); head.setObjectName("ProgHead")
        hl = QHBoxLayout(head); hl.setContentsMargins(12,12,12,8); hl.setSpacing(12)
        self.run_lbl = QLabel("Run: -"); self.run_lbl.setStyleSheet("font-weight:600; color:#0b3e4a;")
        self.stage_chip = Chip("stage","stage"); self.kind_chip = Chip("update","kind")
        self.elapsed_lbl = QLabel("Elapsed: 0.0s"); self.elapsed_lbl.setStyleSheet("color:#3b5560;")
        hl.addWidget(self.run_lbl); hl.addWidget(self.stage_chip); hl.addWidget(self.kind_chip)
        hl.addStretch(1); hl.addWidget(self.elapsed_lbl)

        self.overall = QProgressBar(); self.overall.setRange(0,100); self.overall.setValue(0)
        self.overall.setTextVisible(True); self.overall.setFormat("Total %p%")
        self.overall.setStyleSheet("""
            QProgressBar { background:#e8f3f8; border:1px solid #cfe5ee; border-radius:8px; height:20px; }
            QProgressBar::chunk { background:#4a8bc6; }
        """)
        over_box = QVBoxLayout(); over_box.setContentsMargins(12,0,12,12); over_box.addWidget(self.overall)

        head_box = QVBoxLayout(); head_box.setContentsMargins(0,0,0,0); head_box.addWidget(head); head_box.addLayout(over_box)
        head_wrap = QFrame(); head_wrap.setLayout(head_box)
        head_wrap.setStyleSheet("QFrame#ProgHead { border:1px solid #d8ecf3; border-radius:12px; background:#ffffff; }")
        root.addWidget(head_wrap)

        # Steps scroller
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        cont = QWidget(); self.steps_layout = QVBoxLayout(cont); self.steps_layout.setContentsMargins(0,0,0,0); self.steps_layout.setSpacing(8)
        self.scroll.setWidget(cont); root.addWidget(self.scroll, 1)
        self.steps_layout.addItem(QSpacerItem(1,1,QSizePolicy.Minimum,QSizePolicy.Expanding))

        self._cards: Dict[int, StepWidget] = {}; self._last_overall = 0

    def clear(self):
        for k in list(self._cards.keys()):
            w = self._cards.pop(k); w.setParent(None); w.deleteLater()
        while self.steps_layout.count() > 1:
            item = self.steps_layout.takeAt(0)
            if item and item.widget(): item.widget().deleteLater()
        self._last_overall = 0; self.overall.setValue(0)
        self.run_lbl.setText("Run: -"); self.stage_chip.setText("stage"); self.kind_chip.setText("update")
        self.elapsed_lbl.setText("Elapsed: 0.0s")

    def _animate_overall(self, new_val:int):
        new_val = max(0, min(100, int(new_val)))
        anim = QPropertyAnimation(self.overall, b"value", self)
        anim.setDuration(220); anim.setStartValue(self._last_overall); anim.setEndValue(new_val)
        anim.setEasingCurve(QEasingCurve.InOutCubic); anim.start(QPropertyAnimation.DeleteWhenStopped)
        self._last_overall = new_val

    def apply(self, payload: dict):
        run = payload.get("run","-"); stage = (payload.get("stage") or "").lower() or "stage"
        kind = (payload.get("kind") or "").lower() or "update"; t = float(payload.get("t") or 0.0)
        perc = int(payload.get("perc") or 0)
        self.run_lbl.setText(f"Run: {run}"); self.stage_chip.setText(stage); self.kind_chip.setText(kind)
        self.elapsed_lbl.setText(f"Elapsed: {t:.1f}s")
        self._animate_overall(perc)

        step = int(payload.get("step") or 0); total = int(payload.get("total") or 0); title = payload.get("title","")
        if step not in self._cards:
            card = StepWidget(step, total, title, run, stage, kind, self)
            card.openPathRequested.connect(self.openPathRequested.emit)
            self.steps_layout.insertWidget(self.steps_layout.count()-1, card)
            self._cards[step] = card
        card = self._cards[step]
        card.apply_payload(payload)

# ---------------- Parameters + Progress + Raw Log ----------------
class ExternalFileRow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self); h.setContentsMargins(0,0,0,0); h.setSpacing(6)
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText("Select a file:")
        self.path_edit.setMinimumHeight(32); self.path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.browse_btn = QToolButton(); self.browse_btn.setText("Browse"); self.browse_btn.setProperty("class","pill"); self.browse_btn.setMinimumHeight(32)
        self.desc_edit = QLineEdit(); self.desc_edit.setPlaceholderText("Description for this file"); self.desc_edit.setMinimumHeight(32)
        self.desc_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.remove_btn = QToolButton(); self.remove_btn.setText("Remove"); self.remove_btn.setProperty("class","pill"); self.remove_btn.setMinimumHeight(32)

        h.addWidget(self.path_edit, 3); h.addWidget(self.browse_btn, 0); h.addWidget(self.desc_edit, 2); h.addWidget(self.remove_btn, 0)
        self.browse_btn.clicked.connect(self._on_browse)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", APP_DIR, "All Files (*)")
        if path: self.path_edit.setText(path)

    def value(self) -> str:
        p = self.path_edit.text().strip(); d = self.desc_edit.text().strip() or "No description"
        return f"{p} (description: {d})" if p else ""

    def is_empty(self) -> bool:
        return not self.path_edit.text().strip() and not self.desc_edit.text().strip()

class ParameterPanel(QFrame):
    runRequested = pyqtSignal(list, str)     # (cmd_list, expected_html_path)
    previewRequested = pyqtSignal(str)       # shell string (not used externally)
    openOutputRequested = pyqtSignal(str)    # expected_html_path (or "")
    openLogRequested = pyqtSignal()

    LABEL_W = 150  # fixed label column width for tidy rows
    INPUT_MIN_W = 420
    DEMO_VALUES: Dict[str, Dict[str, object]] = {
        "P1": {
            "gene_of_interest": "PRKAG2-AS1",
            "user_query": (
                "I want to know the binding protein of PRKAG2-AS1 to inference "
                "its potential function in liver metabolism"
            ),
            "reasoning_function": "Liver metabolism",
            "binding_database": ("NPInter", "starBase", "RNAInter"),
            "literature_searching": ("Similarity", "PubMed"),
            "output_file_name": "PRKAG2-AS1_R1",
            "quotauser": "quota0001",
            "fresh": ("N",),
        },
        "P2": {
            "query": "liver metabolism",
            "gene": "RBFOX2",
            "output": "RBFOX2_liver_metabolism",
            "fresh": ("N",),
        },
        "P3": {
            "rbp": "RBFOX2",
            "lncrna": "PRKAG2-AS1",
            "tissue": "Liver",
            "regulation_type": ("Both",),
            "append_info": (
                "We confirmed PRKAG2-AS1 could bind to RBFOX in human and mouse "
                "liver tissue."
            ),
            "output": "PRKAG2-AS1_RBFOX2_liver_metabolism",
            "fresh": ("N",),
        },
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_mode: str = "P1"
        self.fields: Dict[str, QWidget] = {}
        self.expected_html_path: str = ""
        self._ext_rows: List[ExternalFileRow] = []
        self._ext_box: Optional[QVBoxLayout] = None

        self._demo_restore_timer = QTimer(self)
        self._demo_restore_timer.setSingleShot(True)
        self._demo_restore_timer.timeout.connect(self._refresh_demo_button)

        outer = QVBoxLayout(self); outer.setContentsMargins(2, 8, 8, 8); outer.setSpacing(10)

        self.header = QLabel("Parameters — Skill 1: lncRNA–RBP inference"); self.header.setObjectName("h1")
        outer.addWidget(self.header)

        # ------- Upper panel (form + actions) and lower panel (run output) ------
        self.vsplit = QSplitter(Qt.Vertical); self.vsplit.setChildrenCollapsible(False)

        # Upper container with a form (no inner scroll area)
        self.upper_container = QWidget()
        upper_layout = QVBoxLayout(self.upper_container)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(8)

        self.form_container = QWidget()
        self.form = QFormLayout(self.form_container)
        self.form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.form.setHorizontalSpacing(8)
        self.form.setVerticalSpacing(8)

        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.param_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_scroll.setFrameShape(QFrame.NoFrame)
        self.param_scroll.setWidget(self.form_container)

        upper_layout.addWidget(self.param_scroll, 1)

        action_bar = QHBoxLayout(); action_bar.setContentsMargins(0,2,0,2); action_bar.setSpacing(8)
        self.demo_btn = QToolButton()
        self.demo_btn.setCursor(Qt.PointingHandCursor)
        self.demo_btn.setProperty("class", "demo")
        self.demo_btn.setMinimumHeight(38)
        self.demo_btn.setMinimumWidth(210)
        self.demo_btn.setToolTip("Fill the tested demo parameters only; this does not run the skill.")
        action_bar.addWidget(self.demo_btn)
        action_bar.addStretch(1)
        self.preview_btn = QToolButton(); self.preview_btn.setText("Copy command"); self.preview_btn.setToolTip("Copy the assembled command to clipboard")
        self.preview_btn.setCursor(Qt.PointingHandCursor)
        copy_icon_path = os.path.join(APP_DIR, "icons", "copy.svg")
        self.preview_btn.setIcon(QIcon.fromTheme("edit-copy", QIcon(copy_icon_path)))
        self.preview_btn.setIconSize(QSize(22,22)); self.preview_btn.setProperty("class","pill"); self.preview_btn.setMinimumHeight(32)
        action_bar.addWidget(self.preview_btn)

        self.run_btn = QToolButton(); self.run_btn.setText("Run"); self.run_btn.setToolTip("Execute the pipeline with current parameters")
        self.run_btn.setCursor(Qt.PointingHandCursor); self.run_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_btn.setIconSize(QSize(22,22)); self.run_btn.setProperty("class","pill"); self.run_btn.setMinimumHeight(32)
        action_bar.addWidget(self.run_btn)
        upper_layout.addLayout(action_bar)

        # Lower container (Run output)
        self.log_card = QFrame(); self.log_card.setObjectName("LogCard")
        log_layout = QVBoxLayout(self.log_card); log_layout.setContentsMargins(12,12,12,12); log_layout.setSpacing(10)

        log_header = QHBoxLayout(); lbl = QLabel("Run output"); lbl.setObjectName("h1")
        log_header.addWidget(lbl); log_header.addStretch(1)
        self.log_clear_btn = QToolButton(); self.log_clear_btn.setText("Clear"); self.log_clear_btn.setCursor(Qt.PointingHandCursor)
        self.log_clear_btn.setProperty("class","pill"); self.log_clear_btn.setMinimumHeight(30)
        log_header.addWidget(self.log_clear_btn)
        log_layout.addLayout(log_header)

        self.log_tabs = QTabWidget()
        self.log_tabs.setObjectName("RunLogTabs")
        self.log_tabs.setDocumentMode(True)  # more compact tabs
        # Scoped stylesheet so only this tab bar shrinks
        self.log_tabs.setStyleSheet("""
        #RunLogTabs::pane { border: none; background: transparent; }
        #RunLogTabs QTabBar { qproperty-drawBase: 0; }
        #RunLogTabs QTabBar::tab {
          padding: 6px 12px; margin: 0 6px; font-weight: 600; min-height: 26px;
          border: 1px solid #cfe5ee; border-top-left-radius: 10px; border-top-right-radius: 10px;
          background: #e5eef4; color: #0b3e4a;
        }
        #RunLogTabs QTabBar::tab:selected {
          background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1b7a8e, stop:1 #0f5f6e);
          color: white; border-color: #0f5f6e;
        }
        #RunLogTabs QTabBar::tab:hover:!selected { background: #edf3f8; }
        """)
        self.log_tabs.setTabPosition(QTabWidget.North)

        self.progress_board = ProgressBoard()
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(_monospace_font(10))

        self.log_tabs.addTab(self.progress_board, "Progress")
        self.log_tabs.addTab(self.log_view, "Raw log")
        self.log_tabs.setCurrentIndex(0)

        log_layout.addWidget(self.log_tabs)

        self.vsplit.addWidget(self.upper_container)
        self.vsplit.addWidget(self.log_card)

        # Equal stretch for half-half behavior
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 2)

        # Initial 50/50 split (pixels not ratios)
        def _apply_initial_split():
            h = max(1, self.vsplit.size().height() or self.height())
            top = int(h * 0.40)   # ~50%
            bottom = max(1, h - top)
            self.vsplit.setSizes([top, bottom])
        QTimer.singleShot(0, _apply_initial_split)
        outer.addWidget(self.vsplit, 1)

        # ---- Signals ----
        self.log_clear_btn.clicked.connect(self.clear_log)
        self.demo_btn.clicked.connect(self._load_demo)
        self.preview_btn.clicked.connect(self._on_preview)
        self.run_btn.clicked.connect(self._on_run)

        # ---- Build initial mode ----
        self.show_mode("P1")

    # ---------- helpers ----------
    def _pretty_label(self, s: str) -> str:
        """
        Put anything in parentheses on the next line, and also split around slashes.
        Examples:
          "Append file (optional):" -> "Append file\n(optional):"
          "Gene (symbol):"          -> "Gene\n(symbol):"
          "Reasoning / experiment"  -> "Reasoning /\nexperiment"
        """
        # move parenthetical to new line
        s = re.sub(r"\s*\(([^)]+)\)", r"\n(\1)", s)
        # break after ' / ' so long labels don't overflow
        s = s.replace(" / ", " /\n")
        return s

    def _label(self, text: str) -> QLabel:
        lab = QLabel(self._pretty_label(text))
        lab.setMinimumWidth(self.LABEL_W)
        lab.setMaximumWidth(self.LABEL_W)
        lab.setWordWrap(True)  # allow multiline labels
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _add_line(self, key: str, label: str, placeholder: str = "", tooltip: str = "") -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setMinimumHeight(32)
        # Let the field grow with available width (no fixed minimum width)
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if tooltip:
            edit.setToolTip(tooltip)
        self.fields[key] = edit
        self.form.addRow(self._label(label), edit)
        return edit

    def _add_text(self, key: str, label: str, placeholder: str = "", tooltip: str = "") -> QTextEdit:
        te = QTextEdit()
        te.setPlaceholderText(placeholder)
        # Wrap at widget width (avoid horizontal scrollbar)
        te.setLineWrapMode(QTextEdit.WidgetWidth)
        te.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # we'll grow height instead
        te.setMinimumHeight(56)
        te.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        if tooltip:
            te.setToolTip(tooltip)
        self.fields[key] = te
        self.form.addRow(self._label(label), te)

        # Auto-resize height to content (bounded so the outer page scroll handles overflow)
        def _adjust_height():
            doc = te.document()
            doc.setTextWidth(te.viewport().width())  # match wrap width to viewport
            h = doc.size().height() + te.frameWidth() * 2 + 8
            te.setFixedHeight(int(max(56, min(180, h))))  # grow up to ~260px, no inner scrollbar
        te.textChanged.connect(_adjust_height)
        # also adjust when the widget is first shown and when resized
        try:
            old_resize = te.resizeEvent
            def _on_resize(ev):
                old_resize(ev)
                _adjust_height()
            te.resizeEvent = _on_resize  # type: ignore[assignment]
        except Exception:
            pass
        QTimer.singleShot(0, _adjust_height)

        return te

    def _add_combo(self, key: str, label: str, options: List[str],
                   placeholder: str = "", tooltip: str = "", editable: bool = True) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(editable)
        combo.setMinimumHeight(32)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setMaxVisibleItems(18)
        combo.addItems(options)
        if editable:
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.setCurrentIndex(-1)
            if combo.lineEdit() is not None:
                combo.lineEdit().setPlaceholderText(placeholder)
            if combo.completer() is not None:
                combo.completer().setCaseSensitivity(Qt.CaseInsensitive)
                combo.completer().setFilterMode(Qt.MatchContains)
        if tooltip:
            combo.setToolTip(tooltip)
        self.fields[key] = combo
        self.form.addRow(self._label(label), combo)
        return combo

    def _gtex_tissue_options(self) -> List[str]:
        """Read only the GTEx header so the tissue list follows the bundled database."""
        path = os.path.join(APP_DIR, "database", "GTEx_Tissue.txt")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return [x.strip() for x in fh.readline().rstrip("\r\n").split("\t")[2:] if x.strip()]
        except Exception:
            return ["Liver", "Liver_Hepatocyte", "Brain_Cortex", "Lung", "Pancreas"]

    def _add_file_browse(self, key: str, label: str, placeholder: str = "", multiple: bool = False,
                         save: bool = False, filter_spec: str = "All Files (*)", post_select=None) -> QLineEdit:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setMinimumHeight(32)
        # Grow to available width
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn = QToolButton()
        btn.setText("Browse")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setProperty("class", "pill")
        btn.setMinimumHeight(32)
        h.addWidget(edit, 1)
        h.addWidget(btn, 0)

        def on_click():
            if multiple:
                files, _ = QFileDialog.getOpenFileNames(self, "Select files", APP_DIR, filter_spec)
                if files:
                    edit.setText("|".join(files))
                    post_select and post_select(files)
            elif save:
                path, _ = QFileDialog.getSaveFileName(self, "Select output file", APP_DIR, filter_spec)
                if path:
                    edit.setText(path)
                    post_select and post_select(path)
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", APP_DIR, filter_spec)
                if path:
                    edit.setText(path)
                    post_select and post_select(path)

        btn.clicked.connect(on_click)
        self.fields[key] = edit
        self.form.addRow(self._label(label), w)
        return edit

    def _add_check_group(self, key: str, label: str, options: List[str], checked_by_default: Optional[List[str]] = None):
        cont = QWidget(); h = QHBoxLayout(cont); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
        btns: List[Tuple[str, QToolButton]] = []
        defaults = set(checked_by_default) if checked_by_default is not None else set(options)
        for name in options:
            b = QToolButton(); b.setText(name); b.setCheckable(True); b.setChecked(name in defaults)
            b.setCursor(Qt.PointingHandCursor); b.setProperty("class", "pill-check"); b.setMinimumHeight(32)
            h.addWidget(b); btns.append((name, b))
        h.addStretch(1)
        cont._isCheckGroup = True
        cont._check_buttons = btns
        self.fields[key] = cont
        self.form.addRow(self._label(label), cont)
        return cont

    def _add_external_row(self, path: str = "", description: str = ""):
        if self._ext_box is None:
            return
        row = ExternalFileRow(self)
        row.path_edit.setText(path)
        row.desc_edit.setText(description)
        self._ext_rows.append(row)
        self._ext_box.addWidget(row)
        row.remove_btn.clicked.connect(lambda _checked=False, item=row: self._remove_external_row(item))

    def _remove_external_row(self, row: ExternalFileRow):
        if row not in self._ext_rows:
            return
        if self._ext_box is not None:
            self._ext_box.removeWidget(row)
        self._ext_rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _reset_external_rows(self):
        """Match the tested Skill 1 snapshot, which contains no external files."""
        for row in list(self._ext_rows):
            self._remove_external_row(row)
        self._add_external_row()

    def _set_field_value(self, key: str, value: object):
        widget = self.fields.get(key)
        if widget is None:
            return
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value))
        elif isinstance(widget, QComboBox):
            text = str(value)
            index = widget.findText(text, Qt.MatchFixedString)
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                widget.setEditText(text)
        elif hasattr(widget, "_isCheckGroup"):
            selected = {str(value)} if isinstance(value, str) else {str(item) for item in value}
            buttons = list(getattr(widget, "_check_buttons"))
            # Select targets first so exclusive button groups always retain one choice.
            for name, button in buttons:
                if name in selected:
                    button.setChecked(True)
            for name, button in buttons:
                if name not in selected:
                    button.setChecked(False)

    def _refresh_demo_button(self):
        skill_name = SKILL_NAMES.get(self.current_mode, self.current_mode)
        self.demo_btn.setText(f"★ DEMO · Load {skill_name} Example")
        self.demo_btn.setToolTip(
            f"Fill the tested {skill_name} snapshot example only; this does not start a run."
        )

    def _load_demo(self):
        """Populate the current form from its tested snapshot without starting a run."""
        values = self.DEMO_VALUES[self.current_mode]
        if self.current_mode == "P1":
            self._reset_external_rows()
        for key, value in values.items():
            self._set_field_value(key, value)
        self.param_scroll.verticalScrollBar().setValue(0)
        skill_name = SKILL_NAMES.get(self.current_mode, self.current_mode)
        self.demo_btn.setText(f"✓ DEMO LOADED · {skill_name}")
        self._demo_restore_timer.start(1800)

    # ---- Mode switching ----
    def show_mode(self, mode: str):
        self.current_mode = mode
        self._demo_restore_timer.stop()
        self._refresh_demo_button()
        clear_layout(self.form)
        self.fields.clear()
        self.expected_html_path = ""
        self._ext_rows = []
        self._ext_box = None

        if mode == "P1":
            self.header.setText("Parameters — Skill 1: lncRNA–RBP inference")
            self._add_line("gene_of_interest", "lncRNA symbol:", "e.g., PRKAG2-AS1")
            self._add_line(
                "user_query",
                "Research question:",
                "e.g., Which RBPs bind PRKAG2-AS1 and what functions may result?",
            )
            self._add_line(
                "reasoning_function",
                "Biological context / experiment:",
                "e.g., Liver metabolism; RNA pulldown and knockdown",
            )

            self._add_check_group(
                "binding_database",
                "Binding Database:",
                options=["NPInter", "starBase", "RNAInter"],
                checked_by_default=["NPInter", "starBase", "RNAInter"]
            )

            self._add_check_group(
                "literature_searching",
                "Literature searching methods:",
                options=["Similarity", "PubMed"],
                checked_by_default=["Similarity", "PubMed"]
            )

            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            out_edit = QLineEdit(); out_edit.setPlaceholderText("report_name (no .html)"); out_edit.setMinimumHeight(32)
            out_edit.setMinimumWidth(self.INPUT_MIN_W); out_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            suffix = QLabel(".html"); suffix.setStyleSheet("color:#53727b; font-weight:600; margin-left:6px;")
            h.addWidget(out_edit, 1); h.addWidget(suffix, 0)
            self.fields["output_file_name"] = out_edit
            self.form.addRow(self._label("Output name:"), row)
            def _sync_p1_expected():
                base = out_edit.text().strip()
                base = os.path.splitext(os.path.basename(base))[0]
                self.expected_html_path = os.path.join(APP_DIR, "temp", f"{base}.html") if base else ""
            out_edit.textChanged.connect(_sync_p1_expected)

            ext_container = QWidget(); ext_box = QVBoxLayout(ext_container); ext_box.setContentsMargins(0,0,0,0); ext_box.setSpacing(6)
            self._ext_box = ext_box
            add_btn = QToolButton(); add_btn.setText("Add file"); add_btn.setProperty("class","pill"); add_btn.setMinimumHeight(32)
            add_btn.clicked.connect(lambda: self._add_external_row())
            self._add_external_row()
            self.form.addRow(self._label("External files:"), ext_container)
            self.form.addRow(self._label(""), add_btn)

            q = self._add_line("quotauser", "Quota user (optional):", "Google Search quota id")
            q.setText("quota0001")

            # --- Fresh Mode (Y/N) for Skill 1 ---
            fresh_cont = self._add_check_group(
                "fresh",
                "Fresh Mode:",
                options=["Y", "N"],
                checked_by_default=["N"]
            )
            fresh_group = QButtonGroup(fresh_cont)
            fresh_group.setExclusive(True)
            for _, b in getattr(fresh_cont, "_check_buttons"):
                fresh_group.addButton(b)

        elif mode == "P2":
            self.header.setText("Parameters — Skill 2: RBP evidence check")
            self._add_line(
                "query",
                "Evidence question:",
                "e.g., What experiments support HDLBP function in liver metabolism?",
            )
            self._add_line("gene", "RBP gene symbol:", "e.g., HDLBP")
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            out_edit = QLineEdit(); out_edit.setPlaceholderText("report_name (no .html)"); out_edit.setMinimumHeight(32)
            out_edit.setMinimumWidth(self.INPUT_MIN_W); out_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            suffix = QLabel(".html"); suffix.setStyleSheet("color:#53727b; font-weight:600; margin-left:6px;")
            h.addWidget(out_edit, 1); h.addWidget(suffix, 0)
            self.fields["output"] = out_edit
            self.form.addRow(self._label("Output name:"), row)
            def _sync_expected():
                base = out_edit.text().strip()
                base = os.path.splitext(os.path.basename(base))[0]
                self.expected_html_path = os.path.join(APP_DIR, "temp", f"{base}.html") if base else ""
            out_edit.textChanged.connect(_sync_expected)
            # --- Fresh Mode (Y/N) ---
            fresh_cont = self._add_check_group(
                "fresh",
                "Fresh Mode:",
                options=["Y", "N"],
                checked_by_default=["N"]
            )
            fresh_group = QButtonGroup(fresh_cont)
            fresh_group.setExclusive(True)
            for _, b in getattr(fresh_cont, "_check_buttons"):
                fresh_group.addButton(b)

        else:  # Skill 3
            self.header.setText("Parameters — Skill 3: phenotype inference")
            self._add_line("rbp", "RBP gene symbol:", "e.g., HDLBP")
            self._add_line("lncrna", "lncRNA symbol (optional):", "e.g., PRKAG2-AS1")
            self._add_combo(
                "tissue",
                "GTEx tissue:",
                self._gtex_tissue_options(),
                placeholder="Type or select, e.g., Liver",
                tooltip="Select an exact GTEx tissue or type a keyword. Liver-like inputs also enable bundled DRS evidence.",
            )
            reg_cont = self._add_check_group(
                "regulation_type",
                "Regulation direction:",
                options=["Both", "Up", "Down"],
                checked_by_default=["Both"],
            )
            reg_group = QButtonGroup(reg_cont)
            reg_group.setExclusive(True)
            for _, b in getattr(reg_cont, "_check_buttons"):
                reg_group.addButton(b)
            self._add_text(
                "append_info",
                "Additional context (optional):",
                "Free-text notes about the phenotype, model, condition, or proposed mechanism",
            )

            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            out_edit = QLineEdit(); out_edit.setPlaceholderText("report_name (no .html)"); out_edit.setMinimumHeight(32)
            out_edit.setMinimumWidth(self.INPUT_MIN_W); out_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            suffix = QLabel(".html"); suffix.setStyleSheet("color:#53727b; font-weight:600; margin-left:6px;")
            h.addWidget(out_edit, 1); h.addWidget(suffix, 0)
            self.fields["output"] = out_edit
            self.form.addRow(self._label("Output name:"), row)

            def _sync_expected_p3():
                base = out_edit.text().strip()
                base = os.path.splitext(os.path.basename(base))[0]
                self.expected_html_path = os.path.join(APP_DIR, "temp", f"{base}.html") if base else ""
            out_edit.textChanged.connect(_sync_expected_p3)

            # --- Fresh Mode (Y/N) for Skill 3 ---
            fresh_cont = self._add_check_group(
                "fresh",
                "Fresh Mode:",
                options=["Y", "N"],
                checked_by_default=["N"]
            )
            fresh_group = QButtonGroup(fresh_cont)
            fresh_group.setExclusive(True)
            for _, b in getattr(fresh_cont, "_check_buttons"):
                fresh_group.addButton(b)

    # ---- command assembly & validation ----
    def _values(self) -> Dict[str, str]:
        vals: Dict[str, str] = {}
        for k, w in self.fields.items():
            if isinstance(w, QLineEdit): vals[k] = w.text().strip()
            elif isinstance(w, QTextEdit): vals[k] = w.toPlainText().strip()
            elif isinstance(w, QComboBox): vals[k] = w.currentText().strip()
            elif hasattr(w, "_isCheckGroup"):
                selected = [name for (name, btn) in getattr(w, "_check_buttons") if btn.isChecked()]
                vals[k] = ",".join(selected)
        return vals

    def _validate(self, vals: Dict[str, str]) -> Tuple[bool, str]:
        if self.current_mode == "P1":
            req = ["gene_of_interest", "user_query", "reasoning_function", "binding_database", "literature_searching"]
        elif self.current_mode == "P2":
            req = ["query", "gene", "output"]
        else:
            req = ["rbp", "tissue", "output"]
        missing = [r for r in req if not vals.get(r)]
        if missing: return False, f"Missing required field(s): {', '.join(missing)}"
        return True, ""

    def build_command(self) -> Tuple[List[str], str, Optional[str]]:
        vals = self._values(); ok, msg = self._validate(vals)
        if not ok: return [], "", msg
        script_map = {
            "P1": "skill1_lncRNA_RBP_inference.py",
            "P2": "skill2_RBP_evidence_check.py",
            "P3": "skill3_phenotype_inference.py",
        }
        script_path = os.path.join(APP_DIR, script_map[self.current_mode])
        cmd: List[str] = [sys.executable, script_path]
        if self.current_mode == "P1":
            cmd += ["--gene_of_interest", vals["gene_of_interest"]]
            cmd += ["--user_query", vals["user_query"]]
            cmd += ["--reasoning_function", vals["reasoning_function"]]
            cmd += ["--binding_database", vals["binding_database"]]
            cmd += ["--literature_searching", vals["literature_searching"]]
            if hasattr(self, "_ext_rows"):
                items = [r.value() for r in self._ext_rows if not r.is_empty()]
                if items: cmd += ["--external_information", "|".join(items)]
            output_base = vals.get("output_file_name", ""); expected = self.expected_html_path
            if output_base:
                cmd += ["--output_file_name", os.path.splitext(os.path.basename(output_base))[0]]
                if not expected:
                    expected = os.path.join(APP_DIR, "temp", f"{os.path.splitext(os.path.basename(output_base))[0]}.html")
            if vals.get("quotauser"): cmd += ["--quotauser", vals["quotauser"]]

            fresh_val = (vals.get("fresh") or "N")
            if "," in fresh_val:
                fresh_val = fresh_val.split(",", 1)[0]
            cmd += ["--fresh", fresh_val]

            if not expected: expected = ""
        elif self.current_mode == "P2":
            cmd += ["--query", vals["query"]]
            cmd += ["--gene", vals["gene"]]
            cmd += ["--output", vals["output"]]
            fresh_val = (vals.get("fresh") or "N")
            if "," in fresh_val:
                fresh_val = fresh_val.split(",", 1)[0]
            cmd += ["--fresh", fresh_val]
            output_name = os.path.basename(vals.get("output", ""))
            _, output_ext = os.path.splitext(output_name)
            output_html = output_name if output_ext.lower() in (".html", ".htm", ".xhtml") else output_name + ".html"
            expected = os.path.join(APP_DIR, "temp", output_html) if output_name else self.expected_html_path
        else:
            cmd += ["--rbp", vals["rbp"]]
            if vals.get("lncrna"): cmd += ["--lncrna", vals["lncrna"]]
            cmd += ["--tissue", vals["tissue"]]
            regulation_map = {
                "Both": "all",
                "Up": "up_regulation",
                "Down": "down_regulation",
            }
            selected_regulation = vals.get("regulation_type") or "Both"
            cmd += ["--regulation_type", regulation_map.get(selected_regulation, "all")]
            if vals.get("append_info"): cmd += ["--append_info", vals["append_info"]]
            fresh_val = (vals.get("fresh") or "N")
            if "," in fresh_val:
                fresh_val = fresh_val.split(",", 1)[0]
            cmd += ["--fresh", fresh_val]

            output_base = vals.get("output", "").strip()
            cmd += ["--output", output_base]
            output_stem = os.path.splitext(os.path.basename(output_base))[0]
            expected = os.path.join(APP_DIR, "temp", f"{output_stem}.html")
        if expected and not os.path.splitext(expected)[1]: expected += ".html"
        return cmd, expected, None

    def _on_preview(self):
        cmd, expected, err = self.build_command()
        if err: QMessageBox.warning(self, "Missing parameters", err); return
        shell = shlex_join(cmd); QApplication.clipboard().setText(shell)
        tip = "Copied to clipboard:\n\n" + shell
        if expected: tip += f"\n\n(Expected report: {expected})"
        QMessageBox.information(self, "Command copied", tip)

    def _on_run(self):
        cmd, expected, err = self.build_command()
        if err: QMessageBox.warning(self, "Missing parameters", err); return
        self.clear_log(); self.log_tabs.setCurrentIndex(0)
        self.runRequested.emit(cmd, expected)

    def append_log(self, text: str):
        if not text: return
        self.log_view.appendPlainText(text.rstrip("\n"))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def append_agent_event(self, payload: dict):
        self.progress_board.apply(payload)

    def clear_agent(self):
        self.progress_board.clear()

    def clear_log(self):
        self.log_view.clear(); self.clear_agent()

# ---------------- Agent Workspace ----------------
class AgentWorkspace(QWidget):
    """One agent: pipeline selector + parameters + report; owns its own runner."""
    titleChanged = pyqtSignal(str)        # to update tab title
    runningChanged = pyqtSignal(bool)     # to inform main for run counters

    def __init__(self, agent_index:int, request_slot: Callable[[], bool], release_slot: Callable[[], None], parent=None):
        super().__init__(parent)
        self.agent_index = agent_index
        self._request_slot = request_slot
        self._release_slot = release_slot
        self._running = False
        self._current_mode = "P1"
        self._expected_report_path: str = ""
        self._report_watch = QTimer(self); self._report_watch.setInterval(800); self._report_watch.timeout.connect(self._try_open_expected_report)

        # ---- Outer scroller (Agent panel) ----
        root = QVBoxLayout(self); root.setContentsMargins(6,6,6,6); root.setSpacing(6)
        self.agent_scroll = QScrollArea(); self.agent_scroll.setWidgetResizable(True)
        root.addWidget(self.agent_scroll, 1)
        self.agent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.agent_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.agent_panel = QFrame()
        self.agent_panel.setObjectName("AgentPanel")
        self.agent_scroll.setWidget(self.agent_panel)

        panel = QVBoxLayout(self.agent_panel); panel.setContentsMargins(12,12,12,12); panel.setSpacing(10)

        # Pipeline selector row (per agent)
        selector = QHBoxLayout(); selector.setContentsMargins(0,0,0,0); selector.setSpacing(8)
        self.btn_p1 = QPushButton("Skill 1: lncRNA–RBP Inference")
        self.btn_p2 = QPushButton("Skill 2: RBP Evidence Check")
        self.btn_p3 = QPushButton("Skill 3: Phenotype Inference")
        for b in (self.btn_p1, self.btn_p2, self.btn_p3):
            b.setProperty("class", "teal"); b.setCursor(Qt.PointingHandCursor); b.setMinimumHeight(36); b.setMinimumWidth(230)
            selector.addWidget(b)
        selector.addStretch(1)
        panel.addLayout(selector)

        # Split: parameters | report
        self.splitter = QSplitter(Qt.Horizontal); self.splitter.setChildrenCollapsible(False)
        panel.addWidget(self.splitter, 1)

        self.param_panel = ParameterPanel()
        self.report_panel = ReportPanel()

        self.splitter.addWidget(self.param_panel)
        self.splitter.addWidget(self.report_panel)
        self.splitter.setSizes([520, 760])

        # Activation flash label
        self._flash = QLabel("Active agent switched"); self._flash.setAlignment(Qt.AlignCenter)
        self._flash.setStyleSheet("background: rgba(27,122,142,0.12); color:#0b3e4a; border:1px solid #7dcde0; border-radius:10px; padding:6px;")
        self._flash.setVisible(False)
        panel.addWidget(self._flash)

        # Connect signals
        self.btn_p1.clicked.connect(lambda: self.switch_mode("P1"))
        self.btn_p2.clicked.connect(lambda: self.switch_mode("P2"))
        self.btn_p3.clicked.connect(lambda: self.switch_mode("P3"))

        self.param_panel.runRequested.connect(self.run_command)
        self.param_panel.progress_board.openPathRequested.connect(self.open_output)
        self.param_panel.openOutputRequested.connect(self.open_output)

        # Initial mode banner
        self.switch_mode("P1")

        # Runner slots
        self._runner_thread: Optional[QThread] = None
        self._runner_obj: Optional[ProcessRunner] = None

    # ----- Presentation helpers -----
    def _banner_html(self, msg: str) -> str:
        banner = f"<div style='padding:10px;background:#eef8fb;border:1px solid #c7e5ef;border-radius:10px;color:#0d6578;'>{msg}</div>"
        return SAMPLE_HTML.replace("<body>", f"<body>\n{banner}", 1)

    def switch_mode(self, mode: str):
        self._current_mode = mode
        self.param_panel.show_mode(mode)
        info_map = {
            "P1": "Skill 1 infers candidate lncRNA-binding RBPs and potential functions from local interaction databases, similarity search, and literature.",
            "P2": "Skill 2 checks experimental evidence for an RBP using citation filtering, PMC full text, and structured evidence extraction.",
            "P3": "Skill 3 infers tissue-aware phenotype mechanisms from RBP/lncRNA context, GTEx, GWAS, PPI, assays, optional liver DRS, and PMC evidence.",
        }
        self.report_panel.load_html(self._banner_html(info_map.get(mode, "Mode changed.")))
        self.titleChanged.emit(self._tab_title())

    def _tab_title(self) -> str:
        return f"Agent {self.agent_index} {SKILL_NAMES.get(self._current_mode, self._current_mode)}"

    # ----- Actions -----
    def open_output(self, path: str):
        """Open outputs: HTML inside the viewer; others via OS default app; URLs are supported."""
        if path:
            p = path.strip()
            u = QUrl.fromUserInput(p)
            if u.isValid() and u.scheme() in ("http", "https", "file"):
                if u.isLocalFile():
                    local = u.toLocalFile()
                    ext = os.path.splitext(local)[1].lower()
                    if ext in (".html", ".htm", ".xhtml"):
                        self.report_panel.load_url(QUrl.fromLocalFile(os.path.abspath(local)))
                    else:
                        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(local)))
                else:
                    QDesktopServices.openUrl(u)
                return
            if os.path.exists(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in (".html", ".htm", ".xhtml"):
                    self.report_panel.load_url(QUrl.fromLocalFile(os.path.abspath(p)))
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(p)))
                return
        chosen, _ = QFileDialog.getOpenFileName(self, "Open output", APP_DIR,
                                                "All Files (*);;HTML Files (*.html *.htm *.xhtml)")
        if chosen:
            ext = os.path.splitext(chosen)[1].lower()
            if ext in (".html", ".htm", ".xhtml"):
                self.report_panel.load_url(QUrl.fromLocalFile(os.path.abspath(chosen)))
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(chosen)))

    def run_command(self, cmd: List[str], expected_html_path: str):
        script = cmd[1] if len(cmd) > 1 else ""
        if not os.path.exists(script):
            QMessageBox.critical(self, "Script not found",
                                 f"Cannot find script:\n{script}\n\nPlace it next to this tool or adjust the path.")
            return

        # Per-agent exclusivity: don't allow a new run while the previous thread is still finishing.
        if self._runner_thread is not None and not self._runner_thread.isFinished():
            QMessageBox.information(self, "Agent is finishing",
                                    "This agent is still finishing the previous run. Please wait a moment.")
            return

        # Concurrency gate (max 4 running at once; managed by MainWindow)
        if not self._request_slot():
            QMessageBox.information(self, "Run limit reached",
                                    "Maximum of 4 agents can run at the same time.\nStart this run after one completes.")
            return

        shell = shlex_join(cmd)
        self._expected_report_path = expected_html_path or ""
        if self._expected_report_path:
            self._report_watch.start()

        # Start runner
        self._runner_thread = QThread()
        self._runner_obj = ProcessRunner(cmd, cwd=APP_DIR)
        self._runner_obj.moveToThread(self._runner_thread)

        # Update running flag and tab title; also disable run/preview
        self._set_running(True)

        def _push(line: str, is_err: bool = False):
            s = line.rstrip("\n")
            if s.startswith("@@PROGRESS "):
                try:
                    payload = _json.loads(s[len("@@PROGRESS "):])
                    self.param_panel.append_agent_event(payload)
                    one_liner = (
                        f"Step {payload.get('step','?')}/{payload.get('total','?')}: "
                        f"{payload.get('title','')} | {float(payload.get('t',0.0)):.2f}s   "
                        f"{payload.get('run','')} {(payload.get('stage') or '').lower()} "
                        f"{(payload.get('kind') or '').lower()}"
                    )
                    self.param_panel.append_log(one_liner)
                    return
                except Exception:
                    pass
            prefix = "[stderr] " if is_err else ""
            self.param_panel.append_log(prefix + s)

        # Wire up signals
        self._runner_thread.started.connect(self._runner_obj.run)
        self._runner_obj.stdoutLine.connect(lambda s: _push(s, False))
        self._runner_obj.stderrLine.connect(lambda s: _push(s, True))

        # Cleanup and lifecycle management
        self._runner_obj.finished.connect(self._runner_thread.quit)
        self._runner_thread.finished.connect(self._on_thread_stopped)
        self._runner_thread.finished.connect(self._runner_thread.deleteLater)
        self._runner_obj.finished.connect(self._runner_obj.deleteLater)

        # Worker finished -> UI/report handling
        self._runner_obj.finished.connect(lambda rc, out, err, sh=shell: self._on_worker_finished(rc, out, err, sh))

        self._runner_thread.start()

    def _on_worker_finished(self, rc: int, out: str, err: str, shell: str):
        # Pop a dialog with details
        text = f"$ {shell}\n\n[exit code] {rc}\n\n[stdout]\n{(out or '-').strip()}\n\n[stderr]\n{(err or '-').strip()}"
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Run completed" if rc == 0 else "Run completed (with errors)")
        dlg.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        dlg.setIcon(QMessageBox.Information if rc == 0 else QMessageBox.Warning)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setDetailedText(text)
        dlg.setText(f"Process finished with exit code {rc}.\nClick 'Show Details' to see logs.")
        dlg.exec()

        # Attempt to open expected report if present now, otherwise keep watcher alive briefly
        opened_now = False
        for cand in self._resolve_report_candidate_paths(self._expected_report_path):
            if os.path.exists(cand):
                self.report_panel.load_url(QUrl.fromLocalFile(os.path.abspath(cand)))
                opened_now = True
                break
        if not opened_now and self._expected_report_path:
            self._report_watch.start()  # continue watching briefly

        # Release slot and re-enable controls (thread will finish imminently via quit())
        self._set_running(False)
        self._release_slot()

    def _on_thread_stopped(self):
        """Final thread cleanup after QThread has fully stopped."""
        try:
            if self._runner_thread is not None and self._runner_thread.isFinished():
                pass  # nothing else to do; deleteLater already scheduled
        except Exception:
            pass
        # Drop references
        self._runner_thread = None
        self._runner_obj = None

    def _set_running(self, state: bool):
        self._running = state
        # Disable/enable buttons to avoid accidental double-runs
        try:
            self.param_panel.run_btn.setEnabled(not state)
            self.param_panel.preview_btn.setEnabled(not state)
            self.param_panel.demo_btn.setEnabled(not state)
        except Exception:
            pass
        self.titleChanged.emit(self._tab_title())
        self.runningChanged.emit(state)

    def _resolve_report_candidate_paths(self, expected: str) -> list:
        if not expected: return []
        paths = []
        p = expected
        if not os.path.isabs(p): p = os.path.join(APP_DIR, p)
        base, ext = os.path.splitext(p)
        paths.append(p)
        if ext.lower() != ".html": paths.append(base + ".html")
        else: paths.append(p + ".html")
        temp_dir = os.path.join(APP_DIR, "temp")
        paths.append(os.path.join(temp_dir, os.path.basename(base) + ".html"))
        paths.append(os.path.join(temp_dir, os.path.basename(p)))
        uniq = []; seen = set()
        for x in paths:
            x = os.path.abspath(x)
            if x not in seen:
                uniq.append(x); seen.add(x)
        return uniq

    def _try_open_expected_report(self):
        for cand in self._resolve_report_candidate_paths(self._expected_report_path):
            if os.path.exists(cand):
                self.report_panel.load_url(QUrl.fromLocalFile(cand))
                self._report_watch.stop()
                return

# ---------------- Pinned TabBar (first tab is New Agent, not closable/movable) ----------------
class PinnedTabBar(QTabBar):
    newAgentRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self._pin_first_tab = True
        self.setElideMode(Qt.ElideRight)

    def mousePressEvent(self, event):
        idx = self.tabAt(event.pos())
        if idx == 0:
            event.accept()
            QTimer.singleShot(0, self.newAgentRequested.emit)
            return
        super().mousePressEvent(event)

    def tabInserted(self, index: int):
        super().tabInserted(index)
        self._apply_pinned_style()

    def _apply_pinned_style(self):
        if self.count() > 0:
            try:
                self.setTabButton(0, QTabBar.RightSide, None)
                self.setTabButton(0, QTabBar.LeftSide, None)
            except Exception:
                pass

    def tabSizeHint(self, index):
        s = super().tabSizeHint(index)
        if index == 0:
            return QSize(max(s.width(), 130), s.height())
        return s

# ---------------- Main window with Agent Tabs ----------------
class MainWindow(QMainWindow):
    MAX_PARALLEL_RUNS = 4  # <= change here if you want a different cap

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgentLnc MultiAgent")
        self.resize(1360, 860)

        # Concurrency limiter
        self._active_runs = 0

        # Green running icon (circle)
        self._icon_running = self._make_circle_icon(QColor(22, 163, 74))  # green
        self._icon_none = QIcon()

        container = QWidget(); self.setCentralWidget(container)
        root = QVBoxLayout(container); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)

        # ---------- Agent Tabs ----------
        self.agent_tabs = QTabWidget()
        self.agent_tabs.setDocumentMode(True)
        self.agent_tabs.setMovable(False)
        self.agent_tabs.setTabsClosable(True)
        self.agent_tabs.setElideMode(Qt.ElideRight)

        # Use pinned tab bar
        bar = PinnedTabBar(self.agent_tabs)
        # --- Put logo into the tab row (top-right corner)
        logo_path = os.path.join(APP_DIR, "icons", "logo.png")

        corner = QWidget(self.agent_tabs)
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(8, 4, 12, 4)
        corner_layout.setSpacing(0)

        logo_lbl = QLabel(corner)
        logo_lbl.setObjectName("HeaderLogo")
        if os.path.exists(logo_path):
            pm = QPixmap(logo_path)
            pm = pm.scaledToHeight(40, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pm)
            logo_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            logo_lbl.adjustSize()

        corner_layout.addWidget(logo_lbl)
        self.agent_tabs.setCornerWidget(corner, Qt.TopRightCorner)

        bar.setExpanding(False)
        bar.setUsesScrollButtons(True)

        self.agent_tabs.setTabBar(bar)
        self.agent_tabs.tabCloseRequested.connect(self._on_tab_close)
        bar.newAgentRequested.connect(self._add_agent)

        # Add tabs directly to root layout
        root.addWidget(self.agent_tabs, 1)

        # ---------- First pinned tab (New Agent) then first agent ----------
        self._insert_pinned_new_tab()
        self._add_agent(first=True)

        # Status bar
        self._status = self.statusBar()
        self._update_status()

        # Styling
        self.setStyleSheet("""
            QMainWindow { background: #f6fbfe; }

            QTabWidget::pane { border: none; background: #ffffff; }

            QTabBar::tab {
                padding: 10px 16px;
                margin: 2px;
                font-weight: 600;
                border: 1px solid #cfe5ee; border-radius: 8px;
                background: #eaf3f8; color: #0b3e4a;
                min-height: 40px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1b7a8e, stop:1 #0f5f6e);
                color: white; border-color: #0f5f6e;
            }
            QTabBar::tab:first {
                background: #0f5f6e; color: #ffffff; font-weight: 700;
            }
            QTabBar::tab:first:hover { background: #147a8c; }

            QPushButton[class="teal"], QToolButton[class="pill"] {
                color: white; border: none; border-radius: 16px; padding: 8px 14px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1b7a8e, stop:1 #0f5f6e);
            }
            QPushButton[class="teal"]:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2195ab, stop:1 #147a8c);
            }
            QToolButton[class="pill"] {
                border: 1px solid #0b3e4a; border-radius: 12px; padding: 6px 12px; color: #0b3e4a; background: #eff7fb;
            }
            QToolButton[class="pill"]:hover { background: #e4f2f8; }

            QToolButton[class="demo"] {
                border: 2px solid #c56a00; border-radius: 12px; padding: 7px 16px;
                background: #fff1c7; color: #713b00; font-weight: 700;
            }
            QToolButton[class="demo"]:hover { background: #ffe29a; }
            QToolButton[class="demo"]:pressed { background: #ffd170; }
            QToolButton[class="demo"]:disabled {
                background: #f2f2f2; color: #8a8a8a; border-color: #c7c7c7;
            }

            QToolButton[class="pill-check"] {
                border: 1px solid #b9dbe7; border-radius: 12px; padding: 6px 12px; color: #0b3e4a; background: #f4fafd;
            }
            QToolButton[class="pill-check"]:hover { background: #e9f5fb; }
            QToolButton[class="pill-check"]:checked {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1b7a8e, stop:1 #0f5f6e);
                color: white; border: none;
            }
            QToolButton[class="pill-check"]:checked:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2195ab, stop:1 #147a8c);
            }

            QFrame#Card {
                background: #ffffff;
                border: 1px solid #cfe5ee;
                border-radius: 18px;
            }
            QFrame#LogCard {
                background: #ffffff;
                border: 1px solid #cfe5ee;
                border-radius: 12px;
            }
            QLabel#h1 { color: #0b3e4a; font-size: 18px; font-weight: 600; }
            QLineEdit, QTextEdit {
                border: 1px solid #b9dbe7; border-radius: 6px; padding: 6px 8px; background: #fcfeff;
            }
            QToolTip { background: #0b3e4a; color: #ffffff; border: none; }

            QFrame#AgentPanel {
                background: #ffffff; border: none; border-radius: 0;
            }
        """)

        # React to current tab changes to update status
        self.agent_tabs.currentChanged.connect(self._on_current_tab_changed)

    def _make_circle_icon(self, color: QColor, diameter: int = 10) -> QIcon:
        pm = QPixmap(diameter, diameter)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, diameter, diameter)
        painter.end()
        return QIcon(pm)

    # ----- Agent management -----
    def _insert_pinned_new_tab(self):
        dummy = QWidget()
        dummy.setDisabled(True)
        self.agent_tabs.insertTab(0, dummy, "New Agent")
        try:
            self.agent_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
            self.agent_tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        except Exception:
            pass

    def _current_agent_numbers(self) -> List[int]:
        nums: List[int] = []
        for i in range(1, self.agent_tabs.count()):  # skip pinned 0
            w = self.agent_tabs.widget(i)
            if isinstance(w, AgentWorkspace):
                nums.append(w.agent_index)
        return nums

    def _next_agent_number(self) -> int:
        used = set(self._current_agent_numbers())
        n = 1
        while n in used:
            n += 1
        return n

    def _ensure_close_icon(self, index: int):
        if index <= 0:
            return
        tb = self.agent_tabs.tabBar()
        btn = tb.tabButton(index, QTabBar.RightSide)
        if not isinstance(btn, QToolButton):
            close_btn = QToolButton()
            close_btn.setAutoRaise(True)
            close_btn.setIcon(self.style().standardIcon(QStyle.SP_DockWidgetCloseButton))
            tb.setTabButton(index, QTabBar.RightSide, close_btn)
        else:
            close_btn = btn
            close_btn.setIcon(self.style().standardIcon(QStyle.SP_DockWidgetCloseButton))
            close_btn.setText("")
        w = self.agent_tabs.widget(index)
        try:
            close_btn.clicked.disconnect()
        except Exception:
            pass
        close_btn.clicked.connect(lambda: self._on_tab_close(self.agent_tabs.indexOf(w)))

    def _add_agent(self, first: bool=False):
        idx_num = self._next_agent_number()
        ws = AgentWorkspace(
            agent_index=idx_num,
            request_slot=self._try_acquire_slot,
            release_slot=self._release_slot,
            parent=self
        )
        ws.titleChanged.connect(lambda title, w=ws: self._set_tab_title_for(ws, title))
        ws.runningChanged.connect(lambda running, w=ws: self._on_agent_running_changed(w, running))

        insert_at = self.agent_tabs.count()
        i = self.agent_tabs.insertTab(insert_at, ws, f"Agent {idx_num} Skill 1")
        QTimer.singleShot(0, lambda i=i: self.agent_tabs.setCurrentIndex(i))

        self._ensure_close_icon(i)
        self.agent_tabs.setTabIcon(i, self._icon_none)

        if not first:
            self._status.showMessage(f"Created Agent {idx_num}")

    def _on_agent_running_changed(self, ws: AgentWorkspace, running: bool):
        i = self.agent_tabs.indexOf(ws)
        if i >= 0:
            self.agent_tabs.setTabIcon(i, self._icon_running if running else self._icon_none)
        self._update_status()

    def _set_tab_title_for(self, ws: AgentWorkspace, title: str):
        i = self.agent_tabs.indexOf(ws)
        if i >= 0:
            self.agent_tabs.setTabText(i, title)
            self.agent_tabs.setTabIcon(i, self._icon_running if ws._running else self._icon_none)

    def _on_tab_close(self, index: int):
        if index == 0:
            return
        w = self.agent_tabs.widget(index)
        if isinstance(w, AgentWorkspace):
            # Block closing if running or thread still finishing
            if w._running or (w._runner_thread is not None and not w._runner_thread.isFinished()):
                QMessageBox.information(self, "Agent is busy", "This agent is currently running or stopping and cannot be closed.")
                return
        self.agent_tabs.removeTab(index)
        self._update_status()

    def _on_current_tab_changed(self, idx: int):
        if idx <= 0:
            self._status.showMessage("Click New Agent to create a new agent.")
            return
        w = self.agent_tabs.widget(idx)
        if isinstance(w, AgentWorkspace):
            self._status.showMessage(f"Viewing Agent {w.agent_index}")

    # ----- Concurrency management -----
    def _try_acquire_slot(self) -> bool:
        if self._active_runs < self.MAX_PARALLEL_RUNS:
            self._active_runs += 1
            self._update_status()
            return True
        return False

    def _release_slot(self):
        self._active_runs = max(0, self._active_runs - 1)
        self._update_status()

    def _update_status(self):
        cur = self.agent_tabs.currentIndex()
        cur_txt = ""
        if cur > 0:
            w = self.agent_tabs.widget(cur)
            if isinstance(w, AgentWorkspace):
                cur_txt = f" | Viewing Agent {w.agent_index}"
        self.statusBar().showMessage(f"{self._active_runs}/{self.MAX_PARALLEL_RUNS} Agents active{cur_txt}")

    # ----- Close handling -----
    def closeEvent(self, event):
        # Prevent closing while any agent is still running or has a thread finishing
        for i in range(1, self.agent_tabs.count()):
            w = self.agent_tabs.widget(i)
            if isinstance(w, AgentWorkspace):
                if w._running or (w._runner_thread is not None and not w._runner_thread.isFinished()):
                    QMessageBox.information(self, "Runs in progress",
                                            "Please wait for all running agents to finish before exiting.")
                    event.ignore()
                    return
        super().closeEvent(event)

# ---- main ----
def main():
    try:
        QApplication.setStyle(QStyleFactory.create("Fusion"))
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("AgentLnc - Multi-Agent")
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
