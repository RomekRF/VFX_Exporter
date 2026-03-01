Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\Users\Romek\OneDrive\Desktop\vfx2obj_build'
$src  = Join-Path $root 'blender_addon\rf_vfx_tools'
$dist = Join-Path $root 'dist'
New-Item -ItemType Directory -Force -Path $dist | Out-Null

$zip = Join-Path $dist 'rf_vfx_tools.zip'
if (Test-Path -LiteralPath $zip) { Remove-Item -Force -LiteralPath $zip }

Compress-Archive -LiteralPath $src\* -DestinationPath $zip -Force

Write-Host "Wrote: $zip" -ForegroundColor Green
