"""
verify_dataset.py - Verifica integridade do tokenizer.json e train.bin
"""
import json
import struct
import os

print("=" * 60)
print("  VERIFICAÇÃO DE INTEGRIDADE DO DATASET")
print("=" * 60)

# 1. Verificar tokenizer.json
print("\n📋 Verificando tokenizer.json...")
try:
    with open("data/tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    vocab_size = len(data.get("id_to_token", []))
    merges = len(data.get("merges", []))
    special = data.get("special_tokens", {})
    
    print(f"   ✅ JSON válido!")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Merges: {merges}")
    print(f"   Special tokens: {special}")
    
    # Verificar tokens importantes
    bos = special.get("[BOS]", "NÃO ENCONTRADO")
    eos = special.get("[EOS]", "NÃO ENCONTRADO")
    print(f"   BOS ID: {bos}")
    print(f"   EOS ID: {eos}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# 2. Verificar train.bin
print("\n📋 Verificando train.bin...")
try:
    path = "data/train.bin"
    size = os.path.getsize(path)
    tokens = size // 2  # uint16
    
    with open(path, "rb") as f:
        # Primeiros 20 tokens
        first_tokens = []
        for _ in range(20):
            b = f.read(2)
            if len(b) == 2:
                first_tokens.append(struct.unpack("<H", b)[0])
        
        # Últimos 20 tokens
        f.seek(max(0, size - 40))
        last_tokens = []
        for _ in range(20):
            b = f.read(2)
            if len(b) == 2:
                last_tokens.append(struct.unpack("<H", b)[0])
    
    print(f"   ✅ Arquivo válido!")
    print(f"   Tamanho: {size / 1e9:.2f} GB")
    print(f"   Tokens: {tokens:,}")
    print(f"   Primeiros 20: {first_tokens}")
    print(f"   Últimos 20: {last_tokens}")
    
    # Verificar se tem BOS/EOS
    if 258 in first_tokens:
        print(f"   ✅ BOS (258) encontrado no início")
    else:
        print(f"   ⚠️ BOS (258) NÃO encontrado no início")
    
    if 259 in last_tokens:
        print(f"   ✅ EOS (259) encontrado no final")
    else:
        print(f"   ⚠️ EOS (259) NÃO encontrado no final")
    
    # Verificar distribuição
    print("\n📊 Amostragem aleatória...")
    import random
    random.seed(42)
    samples = []
    with open(path, "rb") as f:
        for _ in range(100):
            pos = random.randint(0, tokens - 1) * 2
            f.seek(pos)
            b = f.read(2)
            if len(b) == 2:
                samples.append(struct.unpack("<H", b)[0])
    
    max_tok = max(samples)
    min_tok = min(samples)
    print(f"   Min token: {min_tok}")
    print(f"   Max token: {max_tok}")
    
    if max_tok < 32000:
        print(f"   ✅ Todos tokens dentro do vocab (< 32000)")
    else:
        print(f"   ❌ ERRO: Token {max_tok} fora do vocab!")

except Exception as e:
    print(f"   ❌ ERRO: {e}")

print("\n" + "=" * 60)
print("  VERIFICAÇÃO CONCLUÍDA")
print("=" * 60)
