$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Build e Teste Local (CPU)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar Rust
Write-Host "[1/5] Verificando Rust..." -ForegroundColor Yellow
$rustVersion = rustc --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Rust: $rustVersion" -ForegroundColor Green
} else {
    Write-Host "ERRO - Rust nao encontrado!" -ForegroundColor Red
    exit 1
}

# 2. Build CPU
Write-Host "[2/5] Build CPU..." -ForegroundColor Yellow
cargo build --release --features cpu
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Build concluido!" -ForegroundColor Green
} else {
    Write-Host "ERRO - Build falhou!" -ForegroundColor Red
    exit 1
}

# 3. Testar 85m
Write-Host "[3/5] Testando 85m..." -ForegroundColor Yellow
& ./target/release/ptbr-slm.exe benchmark --model-size 85m --seq-len 64 --num-iterations 3
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - 85m passou!" -ForegroundColor Green
} else {
    Write-Host "ERRO - 85m falhou!" -ForegroundColor Red
    exit 1
}

# 4. Testar 400m
Write-Host "[4/5] Testando 400m (lento em CPU)..." -ForegroundColor Yellow
& ./target/release/ptbr-slm.exe benchmark --model-size 400m --seq-len 32 --num-iterations 1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - 400m passou!" -ForegroundColor Green
} else {
    Write-Host "AVISO - 400m falhou (normal em CPU)" -ForegroundColor Yellow
}

# 5. Info binario
Write-Host "[5/5] Info binario..." -ForegroundColor Yellow
if (Test-Path "target/release/ptbr-slm.exe") {
    $size = [math]::Round((Get-Item target/release/ptbr-slm.exe).Length / 1MB, 2)
    Write-Host "Binario: $size MB" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== TESTE LOCAL CONCLUIDO ===" -ForegroundColor Green
