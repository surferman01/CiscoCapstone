from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


ROOT = Path.cwd()

datas = [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "styles.qss"), "."),
    (str(ROOT / "styles_light.qss"), "."),
]

for package_name in ("xgboost", "catboost"):
    try:
        package_dir = Path(find_spec(package_name).origin).parent
    except Exception:
        continue

    version_file = package_dir / "VERSION"
    if version_file.exists():
        datas.append((str(version_file), package_name))

binaries = []
for package_name in ("xgboost", "catboost"):
    try:
        binaries += collect_dynamic_libs(package_name)
    except Exception:
        pass
    try:
        datas += collect_data_files(package_name)
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
