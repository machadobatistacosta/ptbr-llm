#!/usr/bin/env python3
"""Validação RÁPIDA - apenas checa se arquivos abrem em UTF-8"""
import os
from pathlib import Path

sources = {
    "planalto_clean": Path("data/planalto_clean"),
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("🚀 VALIDAÇÃO RÁPIDA (encoding UTF-8)")
print("=" * 60)

total_files = 0
total_bytes = 0
errors = []

for name, path in sources.items():
    if not path.exists():
        print(f"❌ {name}: NÃO ENCONTRADO")
        continue
    
    files = list(path.glob("*.txt"))
    source_bytes = 0
    
    for f in files:
        total_files += 1
        size = f.stat().st_size
        source_bytes += size
        
        # Tenta ler como UTF-8
        try:
            with open(f, 'r', encoding='utf-8') as file:
                _ = file.read(1000)  # Lê só início
        except UnicodeDecodeError as e:
            errors.append((name, f.name, str(e)))
    
    total_bytes += source_bytes
    print(f"✅ {name:20s} {len(files):3d} files  {source_bytes/1024/1024:6.1f} MB")

print("=" * 60)
print(f"Total: {total_files} arquivos, {total_bytes/1024/1024/1024:.2f} GB")

if errors:
    print(f"\n❌ {len(errors)} ERROS DE ENCODING:")
    for source, file, err in errors[:5]:
        print(f"  {source}/{file}: {err}")
else:
    print(f"\n✅ TODOS ARQUIVOS OK - UTF-8 válido")
