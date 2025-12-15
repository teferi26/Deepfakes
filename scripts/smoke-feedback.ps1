$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

Write-Host "== Smoke test: auth -> upload -> job -> feedback ==" -ForegroundColor Cyan

$email = "test+$(Get-Date -Format 'yyyyMMddHHmmss')@example.com"
$pass = 'TestPassw0rd!'

Write-Host "Registering: $email" -ForegroundColor Gray
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/auth/register' -ContentType 'application/json' -Body (@{ email = $email; password = $pass } | ConvertTo-Json) | Out-Null

Write-Host "Logging in..." -ForegroundColor Gray
$token = (Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/auth/login' -ContentType 'application/json' -Body (@{ email = $email; password = $pass } | ConvertTo-Json)).access_token
if (-not $token) { throw 'No access_token returned' }

$tmp = Join-Path $env:TEMP 'fraud-detector-sample.jpg'
Write-Host "Downloading sample image -> $tmp" -ForegroundColor Gray
$downloadUrls = @(
  'https://picsum.photos/seed/realtest/800/600.jpg',
  'https://placehold.co/800x600.jpg'
)
$downloaded = $false
foreach ($u in $downloadUrls) {
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $tmp -MaximumRedirection 5 -Headers @{ 'User-Agent' = 'Mozilla/5.0' } | Out-Null
    $downloaded = $true
    break
  } catch {
    Write-Host "Download failed: $u" -ForegroundColor DarkYellow
  }
}
if (-not $downloaded) { throw 'Failed to download a sample image (all URLs failed)' }

Write-Host "Uploading..." -ForegroundColor Gray
$uploadJson = & curl.exe -sS -X POST "http://localhost:8000/v1/upload" -H "Authorization: Bearer $token" -F "file=@$tmp;type=image/jpeg"
if (-not $uploadJson) { throw 'Upload failed: empty response' }
$upload = $uploadJson | ConvertFrom-Json
$jobId = $upload.job_id
if (-not $jobId) { throw 'No job_id returned from upload' }
Write-Host "Job: $jobId" -ForegroundColor Yellow

$deadline = (Get-Date).AddMinutes(5)
$result = $null
while ((Get-Date) -lt $deadline) {
  $job = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/v1/jobs/$jobId" -Headers @{ Authorization = "Bearer $token" }
  if ($job.status -in @('success','failure')) { $result = $job; break }
  Start-Sleep -Seconds 2
}
if (-not $result) { throw "Job timeout: $jobId" }

$prob = $result.result.probability
$decision = $result.result.ai_decision
$analysisId = $result.result.analysis_id

Write-Host "Status: $($result.status)" -ForegroundColor Green
Write-Host "probability: $prob" -ForegroundColor Gray
Write-Host "ai_decision: $decision" -ForegroundColor Gray
Write-Host "analysis_id: $analysisId" -ForegroundColor Gray

if (-not $analysisId) {
  throw 'Missing analysis_id in job result (frontend feedback needs this)'
}

Write-Host "Submitting feedback..." -ForegroundColor Gray
$fb = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/history/$analysisId/feedback" -Headers @{ Authorization = "Bearer $token" } -ContentType 'application/json' -Body (@{ label = 'not_ai_generated'; comment = 'Manual smoke test feedback' } | ConvertTo-Json)

Write-Host "Feedback saved." -ForegroundColor Green
Write-Host "Calibration:" -ForegroundColor Cyan
$fb.calibration | ConvertTo-Json -Depth 6
