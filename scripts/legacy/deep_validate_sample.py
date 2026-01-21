#!/usr/bin/env python3
"""
Validação profunda de AMOSTRA de arquivos txt
"""
import random
from pathlib import Path
import re

# Amostra estratégica
samples = {
    "planalto_clean": 2,      # 2 dos 15
    "wiki_clean": 5,          # 5 dos 132
    "wikibooks_clean": 1,     # 1 dos 2
    "wikinews_clean": 1,      # 1 dos 4
    "wikisource_clean": 2,    # 2 dos 11
}

print("=" * 70)
print("🔬 VALIDAÇÃO PROFUNDA - AMOSTRA DE TXT FILES")
print("=" * 70)

issues = []

for source, count in samples.items():
    source_path = Path(f"data/{source}")
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    sample_files = random.sample(files, min(count, len(files)))
    
    print(f"\n📁 {source}:")
    for f in sample_files:
        print(f"  Verificando {f.name}...")
        
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Amostra: início + meio + fim
            total_chars = len(content)
            sample_text = content[:2000] + content[total_chars//2:total_chars//2+2000] + content[-2000:]
            
            # Checagens
            mojibake_count = len(re.findall(r'[Ãâ€][£©³ºµ§¢\s]+', sample_text))
            wiki_markup = len(re.findall(r'\{\{|\[\[|\}\}|\]\]', sample_text))
            html_tags = len(re.findall(r'<[^>]+>', sample_text))
            
            file_issues = []
            if mojibake_count > 5:
                file_issues.append(f"Mojibake: {mojibake_count} ocorrências")
            if wiki_markup > 10:
                file_issues.append(f"Wiki markup: {wiki_markup} tags")
            if html_tags > 5:
                file_issues.append(f"HTML: {html_tags} tags")
            
            if file_issues:
                print(f"    ⚠️ {' | '.join(file_issues)}")
                issues.append((source, f.name, file_issues))
            else:
                print(f"    ✅ OK")
                
        except Exception as e:
            print(f"    ❌ Erro: {e}")
            issues.append((source, f.name, [f"Erro: {e}"]))

print(f"\n{'=' * 70}")
print(f"📊 RESULTADO:")
if not issues:
    print("✅ TODOS ARQUIVOS DA AMOSTRA OK - Prosseguir")
else:
    print(f"⚠️ {len(issues)} arquivos com problemas:")
    for source, fname, probs in issues[:5]:
        print(f"  - {source}/{fname}: {probs[0]}")
    
    critical = sum(1 for _, _, probs in issues if any('Mojibake' in p for p in probs))
    if critical > 2:
        print(f"\n❌ CRÍTICO: {critical} arquivos com mojibake - CORRIGIR antes de prosseguir")
    else:
        print(f"\n⚠️ Problemas menores - Avaliar se aceita")

print("=" * 70)
