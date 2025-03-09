# First, create a setup.py file for packaging

import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {
    "packages": ["colorama","tqdm","opencv-python", "numpy", "psutil", "queue", "threading", "argparse", "subprocess", "platform"],
    "excludes": [],
    "include_files": []
}

# Base for Windows or Unix
base = None
if sys.platform == "win32":
    base = "Console"

setup(
    name="FrameExtractor",
    version="1.0",
    description="GPU-Accelerated Video Frame Extractor",
    options={"build_exe": build_exe_options},
    executables=[Executable("frame_extractor.py", base=base, target_name="frame_extractor")]
)

# Alternative pyinstaller spec file (frame_extractor.spec)
"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['frame_extractor.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['queue', 'psutil', 'cv2', 'platform'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='frame_extractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
