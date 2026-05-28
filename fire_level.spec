# PyInstaller spec — builds FlameHeightAnalysis.exe on Windows
# Run:  pyinstaller fire_level.spec

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PIL._tkinter_finder',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        'matplotlib.figure',
        'pandas',
        'numpy',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'notebook', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FlameHeightAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # replace with 'icon.ico' if you add one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlameHeightAnalysis',
)
