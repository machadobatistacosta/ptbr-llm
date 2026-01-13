#!/bin/bash
# Verifica requisitos de VRAM para diferentes modelos e configurações

echo "═══════════════════════════════════════════════════════════"
echo "  📊 Verificação de Requisitos de VRAM"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Carregar função do Rust (simulação)
# Em produção, isso viria do binário compilado

print_vram_info() {
    local model=$1
    local batch=$2
    local seq=$3
    
    echo "  Modelo: $model | Batch: $batch | Seq Len: $seq"
    ./target/release/ptbr-slm info --model-size $model 2>/dev/null | grep "VRAM" || echo "    VRAM: (executar: ./target/release/ptbr-slm info --model-size $model)"
    echo ""
}

echo "📋 T4 GPU: 16GB VRAM disponível"
echo ""

echo "🔍 Configurações Recomendadas:"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Modelo 400m"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_vram_info "400m" "2" "256"
print_vram_info "400m" "1" "512"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Modelo 800m"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_vram_info "800m" "1" "256"
print_vram_info "800m" "1" "128"
echo ""

echo "✅ Recomendações:"
echo "  • 400m: batch=2, seq_len=256 ✅ Seguro"
echo "  • 800m: batch=1, seq_len=256 ⚠️  Pode ser apertado"
echo "  • 800m: batch=1, seq_len=128 ✅ Mais seguro"
echo ""
