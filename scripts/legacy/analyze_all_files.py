#!/usr/bin/env python3
"""Analisa TODOS os 164 arquivos e classifica por qualidade"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class FileAnalysis:
    source: str
    filename: str
    size_kb: float
    repetitiveness: float  # 0-1, maior = mais repetitivo
    foreign_content: float  # 0-1, maior = mais estrangeiro
    diversity: float  # 0-1, maior = melhor
    quality_score: float  # 0-100
    
sources = {
    "planalto_clean": Path("data/planalto_clean"),
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

# Padrões de lixo
REPETITIVE_PATTERNS = [
    r"é uma comuna francesa",
    r"na região administrativa",
    r"estende-se por uma área",
    r"no departamento",
]

FOREIGN_KEYWORDS = [
    "frança", "francês", "francesa", "paris",
    "espanha", "espanhol", "madrid",
    "itália", "italiano", "roma",
    "estados unidos", "americano",
]

def analyze_file(filepath: Path, source: str) -> FileAnalysis:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    size_kb = len(content) / 1024
    words = content.lower().split()
    
    # Repetitividade (% de ocorrências de padrões repetitivos)
    repetitive_count = sum(len(re.findall(p, content, re.IGNORECASE)) for p in REPETITIVE_PATTERNS)
    repetitiveness = min(1.0, repetitive_count / 100)
    
    # Conteúdo estrangeiro
    foreign_count = sum(words.count(kw) for kw in FOREIGN_KEYWORDS)
    foreign_content = min(1.0, foreign_count / max(1, len(words) / 100))
    
    # Diversidade vocabular (primeiras 5k palavras)
    sample = words[:5000] if len(words) > 5000 else words
    if sample:
        unique = len(set(sample))
        diversity = unique / len(sample)
    else:
        diversity = 0.0
    
    # Score de qualidade (0-100)
    # Penaliza repetitividade e conteúdo estrangeiro, premia diversidade
    quality_score = 100 * (
        (1 - repetitiveness) * 0.4 +
        (1 - foreign_content) * 0.3 +
        diversity * 0.3
    )
    
    return FileAnalysis(
        source=source,
        filename=filepath.name,
        size_kb=size_kb,
        repetitiveness=repetitiveness,
        foreign_content=foreign_content,
        diversity=diversity,
        quality_score=quality_score
    )

print("=" * 80)
print("🔬 ANÁLISE COMPLETA DE QUALIDADE - 164 ARQUIVOS")
print("=" * 80)

all_analyses: List[FileAnalysis] = []

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    print(f"\n📁 {source_name}: {len(files)} arquivos")
    
    for f in files:
        analysis = analyze_file(f, source_name)
        all_analyses.append(analysis)

# Estatísticas
print(f"\n{'=' * 80}")
print(f"📊 RESULTADOS:")
print(f"Total analisado: {len(all_analyses)} arquivos")

# Classifica por qualidade
excellent = [a for a in all_analyses if a.quality_score >= 70]
good = [a for a in all_analyses if 50 <= a.quality_score < 70]
poor = [a for a in all_analyses if 30 <= a.quality_score < 50]
garbage = [a for a in all_analyses if a.quality_score < 30]

print(f"\n📈 DISTRIBUIÇÃO DE QUALIDADE:")
print(f"  EXCELENTE (>=70): {len(excellent)} arquivos")
print(f"  BOM (50-70):      {len(good)} arquivos")
print(f"  FRACO (30-50):    {len(poor)} arquivos")
print(f"  LIXO (<30):       {len(garbage)} arquivos")

# Piores 20 arquivos
print(f"\n🗑️  TOP 20 PIORES ARQUIVOS (CANDIDATOS A REMOÇÃO):")
worst = sorted(all_analyses, key=lambda x: x.quality_score)[:20]
for a in worst:
    print(f"  {a.quality_score:5.1f} | {a.source:20s} | {a.filename:30s} | Rep:{a.repetitiveness:.2f} For:{a.foreign_content:.2f} Div:{a.diversity:.2f}")

# Salva lista de remoção
remove_list_path = Path("data/REMOVE_LIST.txt")
with open(remove_list_path, 'w', encoding='utf-8') as f:
    f.write("# Arquivos de baixa qualidade (score < 30)\n")
    f.write(f"# Total: {len(garbage)} arquivos\n\n")
    for a in sorted(garbage, key=lambda x: x.quality_score):
        f.write(f"{a.source}/{a.filename}\n")

print(f"\n💾 Lista de remoção salva em: {remove_list_path}")
print(f"   {len(garbage)} arquivos marcados para remoção")

# Stats finais
total_size = sum(a.size_kb for a in all_analyses)
remove_size = sum(a.size_kb for a in garbage)
keep_size = total_size - remove_size

print(f"\n📦 DADOS:")
print(f"  Total:   {total_size/1024:.1f} MB ({len(all_analyses)} arquivos)")
print(f"  Remover: {remove_size/1024:.1f} MB ({len(garbage)} arquivos)")
print(f"  Manter:  {keep_size/1024:.1f} MB ({len(all_analyses) - len(garbage)} arquivos)")

print(f"\n{'=' * 80}")
print(f"✅ Análise completa. Revise REMOVE_LIST.txt e confirme remoção.")
print("=" * 80)
