from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


ROOT = Path.cwd()

datas = [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "styles.qss"), "."),
    (str(ROOT / "styles_light.qss"), "."),
]

binaries = []
for package_name in ("xgboost", "catboost"):
    try:
        binaries += collect_dynamic_libs(package_name)
    except Exception:
        pass

hiddenimports = [
    "optuna",
    "imblearn.combine",
    "imblearn.over_sampling",
]


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CSFC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
