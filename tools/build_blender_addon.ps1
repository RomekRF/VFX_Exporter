Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "blender_addon\rf_vfx_tools"
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null

$zip = Join-Path $dist "rf_vfx_tools.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -Force -LiteralPath $zip }

Compress-Archive -Path $src -DestinationPath $zip -Force

$za = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
  $entries = $za.Entries | ForEach-Object { $_.FullName }
  if (-not ($entries -contains "rf_vfx_tools/__init__.py")) { throw "ZIP invalid: missing rf_vfx_tools/__init__.py" }
  if (-not ($entries -contains "rf_vfx_tools/vendor/vfx2obj.py")) { throw "ZIP invalid: missing vendor/vfx2obj.py" }
} finally { $za.Dispose() }

Write-Host "Wrote:" $zip -ForegroundColor Green
