from __future__ import annotations

from pathlib import Path

APP_NAME = "Cisco Silicon Failure Characterization"
WINDOW_WIDTH = 1380
WINDOW_HEIGHT = 860

COLORS = {
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
    "primary": "#0057B8",
    "text": "#1F2937",
    "muted": "#6B7280",
}


def choose_base_font(families: list[str]) -> str:
    names = {name.lower() for name in families}
    if "inter" in names:
        return "Inter"
    if "helvetica neue" in names:
        return "Helvetica Neue"
    return "Arial"


def resolve_logo_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "logo.png"


def resolve_qss_path() -> Path:
    return Path(__file__).resolve().parent / "styles.qss"
