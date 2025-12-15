param(
  [Parameter(Mandatory=$true)][string]$BaseUrl,
  [Parameter(Mandatory=$true)][string]$ApiKey,
  [int]$MaxRuns = 3,
  [int]$PollSeconds = 5
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-Job {
  param([Parameter(Mandatory=$true)][string]$JobId)
  return (Invoke-RestMethod -Method Get -Uri "$BaseUrl/v1/jobs/$JobId" -Headers @{ 'X-API-Key' = $ApiKey })
}

for ($run=1; $run -le $MaxRuns; $run++) {
  Write-Host "Run $run/${MaxRuns}: enqueue train" -ForegroundColor Cyan
  $enq = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/dataset/train" -Headers @{ 'X-API-Key' = $ApiKey }
  $jobId = $enq.job_id
  Write-Host "job_id=$jobId" -ForegroundColor Gray

  $deadline = (Get-Date).AddMinutes(30)
  while ((Get-Date) -lt $deadline) {
    $job = Get-Job -JobId $jobId
    if ($job.status -eq 'success') {
      $job.result | ConvertTo-Json -Depth 20
      if ($job.result.status -eq 'trained') {
        Write-Host "Trained and persisted." -ForegroundColor Green
        return
      }
      if ($job.result.reason -eq 'below_target_accuracy') {
        Write-Host "Below target. Add more/cleaner data and retry." -ForegroundColor Yellow
        break
      }
      Write-Host "Train skipped: $($job.result.reason)" -ForegroundColor Yellow
      break
    }
    if ($job.status -eq 'failure') {
      $job | ConvertTo-Json -Depth 20
      throw "Training job failed"
    }
    Start-Sleep -Seconds $PollSeconds
  }
}

Write-Host "Stopped after $MaxRuns runs." -ForegroundColor Yellow
