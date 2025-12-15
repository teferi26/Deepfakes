param(
  [Parameter(Mandatory=$true)][string]$BaseUrl,
  [Parameter(Mandatory=$true)][string]$ApiKey,
  [Parameter(Mandatory=$true)][ValidateSet('ai_generated','not_ai_generated')][string]$Label,
  [Parameter(Mandatory=$true)][string]$ZipPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not (Test-Path -LiteralPath $ZipPath)) {
  throw "ZIP not found: $ZipPath"
}

Write-Host "Uploading ZIP: $ZipPath (label=$Label)" -ForegroundColor Cyan

$resp = & curl.exe -sS -X POST "$BaseUrl/v1/dataset/upload-zip?label=$Label" -H "X-API-Key: $ApiKey" -F "file=@$ZipPath;type=application/zip"

if ($LASTEXITCODE -ne 0) {
  throw "curl failed"
}

$resp | ConvertFrom-Json | ConvertTo-Json -Depth 10
