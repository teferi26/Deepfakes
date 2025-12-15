$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Seeds multiple labeled samples so the calibrator can train.
# Uses two sources: a random photo (likely real) and a synthetic face (AI).

$email = "seed+$(Get-Date -Format 'yyyyMMddHHmmss')@example.com"
$pass = 'TestPassw0rd!'

Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/auth/register' -ContentType 'application/json' -Body (@{ email = $email; password = $pass } | ConvertTo-Json) | Out-Null
$token = (Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/auth/login' -ContentType 'application/json' -Body (@{ email = $email; password = $pass } | ConvertTo-Json)).access_token

function Invoke-AnalyzeAndFeedback {
  param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][ValidateSet('ai_generated','not_ai_generated')][string]$Label
  )

  $tmp = Join-Path $env:TEMP ("seed-" + [Guid]::NewGuid().ToString() + '.jpg')
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $tmp -MaximumRedirection 5 -Headers @{ 'User-Agent' = 'Mozilla/5.0' } | Out-Null

  $uploadJson = & curl.exe -sS -X POST "http://localhost:8000/v1/upload" -H "Authorization: Bearer $token" -F "file=@$tmp;type=image/jpeg"
  $upload = $uploadJson | ConvertFrom-Json
  $jobId = $upload.job_id

  $deadline = (Get-Date).AddMinutes(5)
  $result = $null
  while ((Get-Date) -lt $deadline) {
    $job = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/v1/jobs/$jobId" -Headers @{ Authorization = "Bearer $token" }
    if ($job.status -in @('success','failure')) { $result = $job; break }
    Start-Sleep -Seconds 2
  }
  if (-not $result) { throw "Timeout job: $jobId" }
  if (-not $result.result.analysis_id) { throw 'Missing analysis_id' }

  Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/history/$($result.result.analysis_id)/feedback" -Headers @{ Authorization = "Bearer $token" } -ContentType 'application/json' -Body (@{ label = $Label } | ConvertTo-Json) | Out-Null

  return [pscustomobject]@{
    url = $Url
    label = $Label
    job_id = $jobId
    analysis_id = $result.result.analysis_id
    probability = $result.result.probability
    ai_decision = $result.result.ai_decision
  }
}

# Try to collect enough labeled samples to trigger training (>=10 per class).
$realUrl = 'https://picsum.photos/seed/realcalib/800/600.jpg'
$aiUrl = 'https://thispersondoesnotexist.com/'

$rows = @()
for ($i=0; $i -lt 10; $i++) {
  $rows += Invoke-AnalyzeAndFeedback -Url $realUrl -Label 'not_ai_generated'
}
for ($i=0; $i -lt 10; $i++) {
  $rows += Invoke-AnalyzeAndFeedback -Url $aiUrl -Label 'ai_generated'
}

$rows | Select-Object label, probability, ai_decision, analysis_id | Format-Table -AutoSize

Write-Host "Done. Wait ~60s and new analyses should start using the calibrator." -ForegroundColor Cyan
