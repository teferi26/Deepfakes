# Automated Training Pipeline
# Target: ≤3% inconclusives, ≥97% accuracy
# Run this from the project root directory

$ErrorActionPreference = "Stop"
$ProjectPath = "c:\Users\tefer\OneDrive\Documentos\SPRINGMARKET\CLIENTES\PAGINAS WEB\PERITCAIONES.IO"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AUTONOMOUS TRAINING PIPELINE" -ForegroundColor Cyan
Write-Host "  Target: ≤3% inconclusives, ≥97% accuracy" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Navigate to project directory
Set-Location $ProjectPath
Write-Host "[1/5] Changed to project directory: $ProjectPath" -ForegroundColor Green

# Step 2: Rebuild and start containers
Write-Host ""
Write-Host "[2/5] Rebuilding API and Worker containers with new MLP architecture..." -ForegroundColor Yellow
docker compose up -d --build api worker
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build containers!" -ForegroundColor Red
    exit 1
}
Write-Host "Containers rebuilt successfully!" -ForegroundColor Green

# Step 3: Wait for services to be ready
Write-Host ""
Write-Host "[3/5] Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check if API is responding
Write-Host "Checking API health..."
$maxRetries = 10
$retry = 0
$apiReady = $false
while ($retry -lt $maxRetries -and -not $apiReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $apiReady = $true
            Write-Host "API is ready!" -ForegroundColor Green
        }
    } catch {
        $retry++
        Write-Host "  Waiting for API... ($retry/$maxRetries)" -ForegroundColor DarkGray
        Start-Sleep -Seconds 5
    }
}

if (-not $apiReady) {
    Write-Host "WARNING: API health check failed, but continuing..." -ForegroundColor Yellow
}

# Step 4: Check worker logs
Write-Host ""
Write-Host "[4/5] Checking worker status..." -ForegroundColor Yellow
docker compose logs worker --tail 20

# Step 5: Run the aggressive training loop
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "[5/5] Starting AGGRESSIVE Training Loop" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor White
Write-Host "  1. Seed massive training data (up to 10k per class)" -ForegroundColor White
Write-Host "  2. Train MLP head with hyperparameter search" -ForegroundColor White
Write-Host "  3. Evaluate and find optimal thresholds" -ForegroundColor White
Write-Host "  4. Repeat until targets are met" -ForegroundColor White
Write-Host ""
Write-Host "Targets: ≤3% inconclusive rate, ≥97% accuracy on decided samples" -ForegroundColor Yellow
Write-Host ""

# Run the aggressive training loop
docker compose exec -T api python scripts/aggressive-train.py `
    --email admin@example.com `
    --password "admin123" `
    --target-samples 10000 `
    --eval-samples 500 `
    --max-rounds 8

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "✅ TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "   Targets have been met!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Training ended with exit code: $exitCode" -ForegroundColor Yellow
    Write-Host "   Check logs for details." -ForegroundColor Yellow
}
Write-Host "============================================" -ForegroundColor Cyan

# Show final worker logs
Write-Host ""
Write-Host "Final worker logs:" -ForegroundColor Yellow
docker compose logs worker --tail 50
