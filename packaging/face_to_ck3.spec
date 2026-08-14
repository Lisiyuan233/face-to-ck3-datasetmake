# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH).resolve().parent
embedded_root = project_root / "build" / "packaging" / "embedded"
checkpoint = embedded_root / "best.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(
        f"missing compact checkpoint: {checkpoint}; run packaging/build_exe.ps1"
    )

datas = [
    (
        str(checkpoint),
        "runs/convnext_tiny_multiview_identifiability_v5_small_clean_finetune",
    ),
    (
        str(
            project_root
            / "runs"
            / "convnext_tiny_multiview_identifiability_v5_small_clean_finetune"
            / "test-field-improvement.csv"
        ),
        "runs/convnext_tiny_multiview_identifiability_v5_small_clean_finetune",
    ),
    (
        str(
            project_root
            / "face_to_ck3_dataset_male_v2"
            / "recommended_training_schema.json"
        ),
        "face_to_ck3_dataset_male_v2",
    ),
    (
        str(project_root / "face_to_ck3_dataset_male_v2" / "dna_schema_full.json"),
        "face_to_ck3_dataset_male_v2",
    ),
    (
        str(project_root / "face_to_ck3_dataset_male_v2" / "dna" / "face_0001.txt"),
        "face_to_ck3_dataset_male_v2/dna",
    ),
    (
        str(
            project_root
            / "face_to_ck3_dataset_male_v2"
            / "processed_multiview"
            / "manifest.json"
        ),
        "face_to_ck3_dataset_male_v2/processed_multiview",
    ),
]

a = Analysis(
    [str(project_root / "face_to_ck3_gui.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PIL._tkinter_finder",
        "torchvision.models.convnext",
        "torchvision.transforms.functional",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceToCK3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
