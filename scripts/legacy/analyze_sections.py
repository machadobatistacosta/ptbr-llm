#!/usr/bin/env python3
"""
Analisa arquivos e identifica SEÇÕES ruins dentro de cada um
Em vez de deletar arquivo inteiro, mostra O QUE é lixo dentro dele
"""
import re
from pathlib import Path
from typing import List, Tuple

# Padrões de lixo
JUNK_PATTERNS = [
    (r"é uma comuna francesa.*?Estende-se por uma área", "Listas de comunas francesas"),
    (r"é um município.*?da Bulgária", "Municípios bulgários"),
    (r"é um município.*?da Romênia", "Municípios romenos"),
    (r"é uma cidade.*?da Hungria", "Cidades húngaras"),
]

def analyze_file_sections(filepath: Path) -> Tuple[int, int, List[str]]:
    """
    Retorna: (chars_total, chars_lixo, exemplos_lixo)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    total_chars = len(content)
    junk_chars = 0
    junk_examples = []
    
    for pattern, description in JUNK_PATTERNS:
        matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
        if matches:
            for match in matches[:3]:  # Primeiros 3 exemplos
                sample = match.group(0)[:100]
                junk_examples.append(f"{description}: {sample}...")
            
            junk_chars += sum(len(m.group(0)) for m in matches)
    
    return total_chars, junk_chars, junk_examples

sources = {
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
   "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("=" * 80)
print("🔬 ANÁLISE DE SEÇÕES - Identificando lixo DENTRO dos arquivos")
print("=" * 80)

results = []

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    print(f"\n📁 {source_name}: analisando {len(files)} arquivos...")
    
    for f in files:
        total, junk, examples = analyze_file_sections(f)
        
        if junk > 0:
            junk_pct = (junk / total) * 100
            results.append({
                'source': source_name,
                'file': f.name,
                'total_kb': total / 1024,
                'junk_kb': junk / 1024,
                'junk_pct': junk_pct,
                'examples': examples
            })

# Ordena por % de lixo
results.sort(key=lambda x: x['junk_pct'], reverse=True)

print(f"\n{'=' * 80}")
print(f"📊 ARQUIVOS COM LIXO DETECTADO ({len(results)} arquivos):")
print(f"{'=' * 80}\n")

# Classifica
to_delete = []  # >90% lixo
to_clean = []   # 10-90% lixo
mostly_good = []  # <10% lixo

for r in results:
    if r['junk_pct'] > 90:
        to_delete.append(r)
    elif r['junk_pct'] > 10:
        to_clean.append(r)
    else:
        mostly_good.append(r)

print(f"🗑️  DELETAR (>90% lixo): {len(to_delete)} arquivos")
for r in to_delete[:10]:
    print(f"  {r['junk_pct']:5.1f}% lixo | {r['source']}/{r['file']}")
    for ex in r['examples'][:2]:
        print(f"    → {ex}")

print(f"\n🧹 LIMPAR (10-90% lixo): {len(to_clean)} arquivos")
for r in to_clean[:10]:
    print(f"  {r['junk_pct']:5.1f}% lixo | {r['source']}/{r['file']}")
    for ex in r['examples'][:1]:
        print(f"    → {ex}")

print(f"\n✅ OK (<10% lixo): {len(mostly_good)} arquivos")

# Salva lista de ações
with open("data/CLEANUP_PLAN.txt", 'w', encoding='utf-8') as f:
    f.write("# PLANO DE LIMPEZA\n\n")
    
    f.write(f"## DELETAR ({len(to_delete)} arquivos - >90% lixo)\n")
    for r in to_delete:
        f.write(f"{r['source']}/{r['file']}\n")
    
    f.write(f"\n## LIMPAR ({len(to_clean)} arquivos - contém lixo mas também conteúdo bom)\n")
    for r in to_clean:
        f.write(f"{r['source']}/{r['file']} - {r['junk_pct']:.1f}% lixo\n")
        for ex in r['examples'][:2]:
            f.write(f"  # {ex}\n")

print(f"\n💾 Plano salvo em: data/CLEANUP_PLAN.txt")
print(f"\n{'=' * 80}")
print("✅ Análise completa. Revise CLEANUP_PLAN.txt")
print("=" * 80)
