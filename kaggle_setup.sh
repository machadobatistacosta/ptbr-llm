#!/bin/bash
# Script para setup no Kaggle Notebook

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 Setup PTBR-SLM no Kaggle"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 1. Verificar ambiente
echo "[1/4] Verificando ambiente..."
echo "  Python: $(python --version)"
echo "  CUDA: $(nvcc --version 2>/dev/null || echo 'não encontrado')"
echo "  GPUs:"
nvidia-smi --list-gpus 2>/dev/null || echo "    ⚠️  nvidia-smi não disponível"
echo ""

# 2. Instalar Rust (se necessário)
if ! command -v rustc &> /dev/null; then
    echo "[2/4] Instalando Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.cargo/env
    echo "  ✅ Rust instalado: $(rustc --version)"
else
    echo "[2/4] Rust já instalado: $(rustc --version)"
fi
echo ""

# 3. Clone/Build projeto
echo "[3/4] Clonando e buildando projeto..."
if [ ! -d "ptbr-slm" ]; then
    # Se já tem o código no notebook, ajuste o caminho
    echo "  ℹ️  Assumindo código já presente no diretório atual"
    cd ptbr-slm 2>/dev/null || echo "  ℹ️  Executando no diretório raiz do projeto"
fi

echo "  🔨 Build com CUDA (isso pode demorar 10-20 min)..."
CARGO_BUILD_JOBS=4 cargo build --release --features cuda
echo "  ✅ Build concluído!"
echo ""

# 4. Teste rápido
echo "[4/4] Teste rápido..."
if [ -f "/kaggle/input" ]; then
    echo "  ℹ️  Dataset encontrado em /kaggle/input"
else
    echo "  ⚠️  Dataset não encontrado - certifique-se de anexar o dataset!"
fi

echo ""
echo "✅ Setup concluído!"
echo ""
echo "Próximos passos:"
echo "  1. Verifique se o dataset está anexado (/kaggle/input)"
echo "  2. Execute: ./target/release/ptbr-slm info --model-size 400m"
echo "  3. Comece o treinamento seguindo KAGGLE_CHECKLIST.md"
echo ""
