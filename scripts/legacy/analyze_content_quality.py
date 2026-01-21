#!/usr/bin/env python3
"""Avalia QUALIDADE real do conteúdo dos txts"""
import random
from pathlib import Path

samples = {
    "planalto_clean": 2,
    "wiki_clean": 3,
}

print("=" * 70)
print("📖 ANÁLISE DE QUALIDADE DE CONTEÚDO")
print("=" * 70)

for source, count in samples.items():
    source_path = Path(f"data/{source}")
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    sample_files = random.sample(files, min(count, len(files)))
    
    for f in sample_files:
        print(f"\n{'=' * 70}")
        print(f"📄 {source}/{f.name}")
        print("=" * 70)
        
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Mostra CONTEÚDO REAL
        print("\n🔍 INÍCIO (primeiros 1000 chars):")
        print(content[:1000])
        
        print(f"\n🔍 MEIO (sample do meio):")
        mid = len(content) // 2
        print(content[mid:mid+1000])
        
        # Estatísticas
        lines = content.split('\n')
        words = content.split()
        
        print(f"\n📊 STATS:")
        print(f"  Total chars: {len(content):,}")
        print(f"  Total linhas: {len(lines):,}")
        print(f"  Total palavras: {len(words):,}")
        print(f"  Avg chars/linha: {len(content)/len(lines):.1f}")
        
        # Diversidade vocabular (primeiras 5k palavras)
        sample_words = words[:5000]
        unique = len(set(w.lower() for w in sample_words))
        print(f"  Vocabulário único (em 5k palavras): {unique}")
        print(f"  Diversidade: {unique/len(sample_words)*100:.1f}%")

print(f"\n{'=' * 70}")
print("👆 AVALIE VOCÊ: O conteúdo é BOM para treinar LLM?")
print("=" * 70)
