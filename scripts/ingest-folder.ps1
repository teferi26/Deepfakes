param(
  [Parameter(Mandatory=$true)][string]$BaseUrl,
  [Parameter(Mandatory=$true)][string]$ApiKey,
  [Parameter(Mandatory=$true)][ValidateSet('ai_generated','not_ai_generated')][string]$Label,
  [Parameter(Mandatory=$true)][string]$Folder,
  [int]$MaxFiles = 0
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not (Test-Path -LiteralPath $Folder)) {
  throw "Folder not found: $Folder"
}

$files = Get-ChildItem -LiteralPath $Folder -File -Recurse | Where-Object {
  $_.Extension -match '^(?i)\.(jpg|jpeg|png|webp)$'
}

if ($MaxFiles -gt 0) {
  $files = $files | Select-Object -First $MaxFiles
}

if (-not $files -or $files.Count -eq 0) {
  throw "No image files found under: $Folder"
}

Write-Host "Uploading $($files.Count) files as label=$Label ..." -ForegroundColor Cyan

$ok = 0
$fail = 0
$idx = 0

foreach ($f in $files) {
  $idx++
  try {
    $ct = 'image/jpeg'
    if ($f.Extension -match '^(?i)\.png$') { $ct = 'image/png' }
    elseif ($f.Extension -match '^(?i)\.webp$') { $ct = 'image/webp' }

    $resp = & curl.exe -sS -X POST "$BaseUrl/v1/dataset/upload?label=$Label" -H "X-API-Key: $ApiKey" -F "file=@$($f.FullName);type=$ct"

    if ($LASTEXITCODE -ne 0) {
      throw "curl failed"
    }

    $null = $resp | ConvertFrom-Json
    $ok++
  } catch {
    $fail++
  }

  if (($idx % 25) -eq 0 -or $idx -eq $files.Count) {
    Write-Host "[$idx/$($files.Count)] ok=$ok fail=$fail" -ForegroundColor Gray
  }
}

Write-Host "Done. ok=$ok fail=$fail" -ForegroundColor Cyan
