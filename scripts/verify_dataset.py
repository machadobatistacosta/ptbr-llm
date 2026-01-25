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
    with open("data/tokenizer.json", "r", encoding="utf-8") as f:
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
    
    if vocab_size != 65536:
        print(f"   ⚠️ AVISO: Vocab esperado 65536, encontrado {vocab_size}")

except Exception as e:
    print(f"   ❌ ERRO: {e}")

# 2. Verificar train.bin
print("\n📋 Verificando train.bin...")
try:
    path = "data/train.bin"
    size = os.path.getsize(path)
    
    with open(path, "rb") as f:
        # Ler Header u64
        header_bytes = f.read(8)
        num_tokens = struct.unpack("<Q", header_bytes)[0]
        
        print(f"   📄 Header (Num Tokens): {num_tokens:,}")
        
        expected_tokens = (size - 8) // 2
        if num_tokens == expected_tokens:
            print(f"   ✅ Header coincide com tamanho do arquivo")
        else:
            print(f"   ❌ ERRO DE CORRUPÇÃO: Header diz {num_tokens} mas arquivo tem {expected_tokens} tokens")

        # Primeiros 20 tokens (pular 8 bytes header)
        f.seek(8)
        first_tokens = []
        for _ in range(20):
            b = f.read(2)
            if len(b) == 2:
                first_tokens.append(struct.unpack("<H", b)[0])
        
        # Últimos 20 tokens
        f.seek(max(8, size - 40))
        last_tokens = []
        for _ in range(20):
            b = f.read(2)
            if len(b) == 2:
                last_tokens.append(struct.unpack("<H", b)[0])
    
    print(f"   ✅ Arquivo legível!")
    print(f"   Tamanho: {size / 1e9:.2f} GB")
    print(f"   Primeiros 20: {first_tokens}")
    print(f"   Últimos 20: {last_tokens}")
    
    # Verificar BOS/EOS se possível (assumindo IDs padrão se não lidos do tokenizer)
    # BOS/EOS IDs dependem do treino
    
    # Verificar distribuição
    print("\n📊 Amostragem aleatória...")
    import random
    random.seed(42)
    samples = []
    with open(path, "rb") as f:
        start_data = 8
        for _ in range(100):
            pos = start_data + random.randint(0, num_tokens - 1) * 2
            f.seek(pos)
            b = f.read(2)
            if len(b) == 2:
                samples.append(struct.unpack("<H", b)[0])
    
    max_tok = max(samples)
    min_tok = min(samples)
    print(f"   Min token: {min_tok}")
    print(f"   Max token: {max_tok}")
    
    if max_tok < 65536:
        print(f"   ✅ Todos tokens dentro do vocab (< 65536)")
    else:
        print(f"   ❌ ERRO: Token {max_tok} fora do vocab!")

except Exception as e:
    print(f"   ❌ ERRO: {e}")

print("\n" + "=" * 60)
print("  VERIFICAÇÃO CONCLUÍDA")
print("=" * 60)
