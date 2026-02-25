Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
  [string]$Scale = "10000",
  [string[]]$Inputs = @(
    ".\dist\vfx\jeep.vfx",
    ".\dist\vfx\Cutscene09_fx.vfx",
    ".\dist\vfx\APC.vfx"
  )
)

function Assert([bool]$cond, [string]$msg) {
  if (-not $cond) { throw $msg }
}

function NearZero3([object]$arr, [double]$eps = 0.01) {
  if ($null -eq $arr) { return $true }
  $a = @($arr)
  if ($a.Count -ne 3) { return $false }
  return ([Math]::Abs([double]$a[0]) -le $eps -and [Math]::Abs([double]$a[1]) -le $eps -and [Math]::Abs([double]$a[2]) -le $eps)
}

Write-Host "=== Smoke Test: VFX -> glTF (center + bake-origin) ===" -ForegroundColor Cyan
Write-Host ("Scale={0}" -f $Scale) -ForegroundColor DarkCyan

$py = Join-Path (Get-Location).Path "vfx2obj.py"
Assert (Test-Path -LiteralPath $py) "Missing vfx2obj.py at repo root."

# Compile sanity
& python -m py_compile $py | Out-Null
Assert ($LASTEXITCODE -eq 0) "py_compile failed"

$fail = 0
foreach ($inRel in $Inputs) {
  $inPath = Resolve-Path -LiteralPath $inRel -ErrorAction Stop
  $base = [IO.Path]::Combine([IO.Path]::GetDirectoryName($inPath.Path), [IO.Path]::GetFileNameWithoutExtension($inPath.Path))
  $gltfPath = $base + ".gltf"

  Write-Host "`n--- EXPORT ---" -ForegroundColor Cyan
  Write-Host $inPath.Path -ForegroundColor DarkCyan

  $out = & python $py --gltf --scale $Scale --center --bake-origin $inPath.Path 2>&1 | Out-String
  $out | Out-Host
  Assert ($out -notmatch "Ignoring unknown option") "Unknown option detected in output (CLI parse regression)."

  Assert (Test-Path -LiteralPath $gltfPath) ("Missing output glTF: {0}" -f $gltfPath)

  $g = (Get-Content -LiteralPath $gltfPath -Raw -Encoding UTF8) | ConvertFrom-Json -Depth 100
  $nodes = @($g.nodes)

  # Root sanity
  Assert (@($g.scenes[0].nodes).Count -eq 1) "scene0.nodes should have exactly 1 root"
  $ri = [int]$g.scenes[0].nodes[0]
  Assert ($ri -ge 0 -and $ri -lt $nodes.Count) "Invalid root index"
  $rootNode = $nodes[$ri]
  Assert ($rootNode.name -eq "__VFX_ROOT__") "Root node name is not __VFX_ROOT__"
  Assert (NearZero3 $rootNode.translation 0.01) "Root translation is not near 0,0,0"
  Assert (NearZero3 $rootNode.rotation 0.01) "Root rotation not identity-ish (expected baked to geometry)"

  # Mesh node TRS should be zero
  $meshNodes = @()
  for ($i=0; $i -lt $nodes.Count; $i++) {
    if ($nodes[$i].PSObject.Properties.Name -contains "mesh") { $meshNodes += $nodes[$i] }
  }
  Assert ($meshNodes.Count -ge 1) "No mesh nodes found"
  foreach ($mn in $meshNodes) {
    Assert (NearZero3 $mn.translation 0.01) ("Mesh node '{0}' translation not near 0,0,0" -f $mn.name)
  }

  # Geometry center check using accessor min/max (all POSITION accessors referenced by primitives)
  $accessors = @($g.accessors)
  $meshes = @($g.meshes)

  $checked = 0
  foreach ($m in $meshes) {
    foreach ($p in @($m.primitives)) {
      if ($p.attributes.PSObject.Properties.Name -contains "POSITION") {
        $ai = [int]$p.attributes.POSITION
        $acc = $accessors[$ai]
        if (($acc.PSObject.Properties.Name -contains "min") -and ($acc.PSObject.Properties.Name -contains "max")) {
          $mn = @($acc.min); $mx = @($acc.max)
          if ($mn.Count -eq 3 -and $mx.Count -eq 3) {
            $cx = ([double]$mn[0] + [double]$mx[0]) / 2.0
            $cy = ([double]$mn[1] + [double]$mx[1]) / 2.0
            $cz = ([double]$mn[2] + [double]$mx[2]) / 2.0
            Assert ([Math]::Abs($cx) -le 0.05) ("Geom center X not near 0 (acc {0}): {1}" -f $ai, $cx)
            Assert ([Math]::Abs($cy) -le 0.05) ("Geom center Y not near 0 (acc {0}): {1}" -f $ai, $cy)
            Assert ([Math]::Abs($cz) -le 0.05) ("Geom center Z not near 0 (acc {0}): {1}" -f $ai, $cz)
            $checked++
          }
        }
      }
    }
  }
  Assert ($checked -ge 1) "No POSITION accessors with min/max were checked (unexpected)."

  Write-Host "PASS ✅" -ForegroundColor Green
}

Write-Host "`nALL PASSED ✅" -ForegroundColor Green
