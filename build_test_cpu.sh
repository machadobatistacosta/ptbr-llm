#!/bin/bash
# Script para build e teste local (CPU) antes de ir para Kaggle

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  🧪 Build e Teste Local (CPU)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para print colorido
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 1. Verificar Rust
echo "[1/5] Verificando Rust..."
if ! command -v rustc &> /dev/null; then
    print_error "Rust não encontrado! Instale: https://rustup.rs/"
    exit 1
fi
RUST_VERSION=$(rustc --version)
print_success "Rust encontrado: $RUST_VERSION"
echo ""

# 2. Build CPU
echo "[2/5] Build CPU..."
print_info "Compilando com features CPU..."
cargo build --release --features cpu 2>&1 | tee build.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Build concluído!"
else
    print_error "Build falhou! Ver build.log"
    exit 1
fi
echo ""

# 3. Testar modelo pequeno (85m)
echo "[3/5] Testando modelo 85m..."
print_info "Criando modelo 85m e testando forward..."
./target/release/ptbr-slm benchmark --model-size 85m --seq-len 64 --num-iterations 3 2>&1 | tee test_85m.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "85m OK!"
else
    print_error "85m falhou! Ver test_85m.log"
    exit 1
fi
echo ""

# 4. Testar modelo médio (400m) - pode ser lento em CPU
echo "[4/5] Testando modelo 400m..."
print_warning "400m pode demorar em CPU..."
if ./target/release/ptbr-slm benchmark --model-size 400m --seq-len 32 --num-iterations 1 2>&1 | tee test_400m.log; then
    print_success "400m OK!"
else
    print_error "400m falhou! Ver test_400m.log"
    print_warning "Isso pode ser normal em CPU - verifique memória disponível"
fi
echo ""

# 5. Verificar tamanho do binário
echo "[5/5] Verificando binário..."
BINARY_SIZE=$(du -h target/release/ptbr-slm | cut -f1)
print_success "Binário: $BINARY_SIZE"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Teste Local Concluído!"
echo "═══════════════════════════════════════════════════════════"
echo ""
print_info "Próximos passos para Kaggle:"
echo "  1. Execute: ./build_kaggle.sh para build CUDA"
echo "  2. Siga checklist em KAGGLE_CHECKLIST.md"
echo ""
