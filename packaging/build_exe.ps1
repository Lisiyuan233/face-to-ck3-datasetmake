param(
    [string]$Python = "",
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultPython = Join-Path $ProjectRoot ".python-build\python.exe"

if (-not $Python) {
    if (Test-Path -LiteralPath $DefaultPython) {
        $Python = $DefaultPython
    } else {
        $Command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $Command) {
            throw "Windows Python was not found. Pass Python 3.10+ with -Python, or install it at $DefaultPython"
        }
        $Python = $Command.Source
    }
}

& $Python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed" }

if (-not $SkipDependencyInstall) {
    & $Python -m pip install --upgrade -r (Join-Path $PSScriptRoot "requirements-exe.txt")
    if ($LASTEXITCODE -ne 0) { throw "Failed to install EXE build dependencies" }
    & $Python -m pip install --upgrade --index-url "https://download.pytorch.org/whl/cpu" torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the CPU PyTorch packages" }
}

$SourceCheckpoint = Join-Path $ProjectRoot "runs\convnext_tiny_multiview_identifiability_v5_small_clean_finetune\best.pt"
$EmbeddedCheckpoint = Join-Path $ProjectRoot "build\packaging\embedded\best.pt"
& $Python (Join-Path $ProjectRoot "tools\export_inference_checkpoint.py") $SourceCheckpoint $EmbeddedCheckpoint
if ($LASTEXITCODE -ne 0) { throw "Failed to export the inference-only checkpoint" }

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "face_to_ck3.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
} finally {
    Pop-Location
}

$Exe = Join-Path $ProjectRoot "dist\FaceToCK3.exe"
if (-not (Test-Path -LiteralPath $Exe)) { throw "Expected output was not created: $Exe" }
$SizeMiB = [math]::Round((Get-Item -LiteralPath $Exe).Length / 1MB, 1)
Write-Host "Created $Exe ($SizeMiB MiB)"
