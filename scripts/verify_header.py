"""
verify_header.py - Verifica o header de 8 bytes do train.bin
"""
import struct
import os

path = "data/train.bin"
file_size = os.path.getsize(path)

with open(path, 'rb') as f:
    # Lê header de 8 bytes
    header_bytes = f.read(8)
    
    print("=" * 60)
    print("  VERIFICAÇÃO DO HEADER train.bin")
    print("=" * 60)
    
    print(f"\nBytes brutos (hex): {header_bytes.hex()}")
    print(f"Bytes brutos (dec): {list(header_bytes)}")
    
    # Interpreta como u64 little-endian
    num_tokens_header = struct.unpack('<Q', header_bytes)[0]
    print(f"\nTokens declarados no header: {num_tokens_header:,}")
    
    # Calcula tokens reais
    data_size = file_size - 8
    real_tokens = data_size // 2
    print(f"Tokens reais no arquivo: {real_tokens:,}")
    
    # Compara
    if num_tokens_header == real_tokens:
        print(f"\n✅ HEADER VÁLIDO! Match perfeito.")
    else:
        diff = real_tokens - num_tokens_header
        print(f"\n❌ MISMATCH! Diferença: {diff:,} tokens")
    
    # Lê primeiros tokens após header
    print(f"\n📊 Primeiros 10 tokens:")
    first_tokens = []
    for _ in range(10):
        b = f.read(2)
        if len(b) == 2:
            first_tokens.append(struct.unpack('<H', b)[0])
    print(f"   {first_tokens}")
    
    # Verifica BOS (258)
    if first_tokens[0] == 258:
        print(f"   ✅ Primeiro token é BOS (258)")
    else:
        print(f"   ⚠️ Primeiro token NÃO é BOS: {first_tokens[0]}")
    
    # Lê últimos tokens
    f.seek(file_size - 20)
    last_tokens = []
    for _ in range(10):
        b = f.read(2)
        if len(b) == 2:
            last_tokens.append(struct.unpack('<H', b)[0])
    
    print(f"\n📊 Últimos 10 tokens:")
    print(f"   {last_tokens}")
    
    if 259 in last_tokens:
        print(f"   ✅ EOS (259) encontrado no final")
    else:
        print(f"   ⚠️ EOS (259) NÃO encontrado no final")

print(f"\n📁 Tamanho do arquivo: {file_size:,} bytes ({file_size/1e9:.2f} GB)")
print("=" * 60)
