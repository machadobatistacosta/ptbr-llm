#!/usr/bin/env python3
"""
Análise PROFUNDA e REAL dos arquivos txt
Mostra conteúdo REAL para decisão informada
"""
import random
from pathlib import Path
from collections import Counter
import re

def analyze_file_deep(filepath: Path) -> dict:
    """Análise profunda de um arquivo"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    words = content.lower().split()
    
    # Estatísticas básicas
    total_lines = len(lines)
    total_words = len(words)
    total_chars = len(content)
    
    # Top palavras (excluindo muito curtas)
    word_freq = Counter(w for w in words if len(w) > 3)
    top_words = word_freq.most_common(20)
    
    # Padrões repetitivos - ABRANGENTE
    patterns = {
        # Geografia estrangeira (mas manter se tiver contexto)
        "comuna_francesa": len(re.findall(r"é uma comuna francesa", content, re.IGNORECASE)),
        "municipio_bulgaria": len(re.findall(r"é um município.*?Bulgária", content, re.IGNORECASE)),
        "municipio_romenia": len(re.findall(r"é um município.*?Rom[eê]nia", content, re.IGNORECASE)),
        "municipio_hungria": len(re.findall(r"é um município.*?Hungria", content, re.IGNORECASE)),
        "municipio_polonia": len(re.findall(r"é um município.*?Pol[oô]nia", content, re.IGNORECASE)),
        "cidade_turquia": len(re.findall(r"é uma cidade.*?Turquia", content, re.IGNORECASE)),
        "estende_area": len(re.findall(r"Estende-se por uma área", content, re.IGNORECASE)),
        "regiao_administrativa": len(re.findall(r"na região administrativa", content, re.IGNORECASE)),
        
        # Wiki markup/metadata
        "infobox": len(re.findall(r"\{\{", content)),
        "wiki_links": len(re.findall(r"\[\[", content)),
        "categorias": len(re.findall(r"\[\[Categoria:", content, re.IGNORECASE)),
        
        # Stubs/templates
        "esbocos": len(re.findall(r"esboço|stub", content, re.IGNORECASE)),
        "sem_fontes": len(re.findall(r"sem fontes|carece de fontes", content, re.IGNORECASE)),
        
        # Listas sem contexto
        "lista_simples": len(re.findall(r"^\* [A-Z]", content, re.MULTILINE)),
        
        # Conteúdo não-português (manter mas detectar)
        "ingles": len(re.findall(r"\b(the|and|of|in|to|for|is|was|with)\b", content, re.IGNORECASE)),
        "frances": len(re.findall(r"\b(le|la|les|de|et|dans|pour|avec)\b", content, re.IGNORECASE)),
    }
    
    # Amostra de conteúdo
    samples = {
        'inicio': lines[:20],
        'meio': lines[total_lines//2:total_lines//2+20] if total_lines > 40 else [],
        'fim': lines[-20:] if total_lines > 20 else []
    }
    
    # NÃO julgar - apenas REPORTAR o que tem
    # Usuário decide se é problema ou conhecimento válido
    
    report = []
    
    # Reportar padrões repetitivos (pode ser legítimo ou não)
    for pattern_name, count in patterns.items():
        if count > 50:
            report.append({
                'type': 'REPORT',
                'category': pattern_name,
                'count': count,
                'note': 'Alta frequência - pode ser lista válida ou repetição inútil'
            })
    
    # Reportar características do arquivo
    if total_words < 100:
        report.append({
            'type': 'REPORT',
            'category': 'artigo_curto',
            'count': total_words,
            'note': 'Artigo muito curto - pode ser stub ou resumo válido'
        })
    
    # Reportar presença de markup
    if patterns.get('infobox', 0) > 20:
        report.append({
            'type': 'REPORT',
            'category': 'wiki_markup',
            'count': patterns.get('infobox', 0),
            'note': 'Contém markup wiki - pode precisar limpeza ou ser intencional'
        })
    
    return {
        'stats': {
            'lines': total_lines,
            'words': total_words,
            'chars': total_chars,
            'size_kb': total_chars / 1024
        },
        'top_words': top_words,
        'samples': samples,
        'patterns': patterns,
        'report': report  # Neutral reporting, not problem flagging
    }

sources = {
    "planalto_clean": Path("data/planalto_clean"),
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("=" * 80)
print("🔬 ANÁLISE PROFUNDA E REAL - Amostra com conteúdo completo")
print("=" * 80)

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    
    print(f"\n{'━' * 80}")
    print(f"📁 {source_name.upper()} - Analisando TODOS os {len(files)} arquivos")
    print(f"{'━' * 80}")
    
    for f in files:  # TODOS os arquivos
        print(f"\n📄 {f.name}")
        print("─" * 80)
        
        analysis = analyze_file_deep(f)
        
        # Stats
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"  Linhas: {analysis['stats']['lines']:,}")
        print(f"  Palavras: {analysis['stats']['words']:,}")
        print(f"  Tamanho: {analysis['stats']['size_kb']:.1f} KB")
        
        # Top palavras
        print(f"\n🔤 TOP 10 PALAVRAS:")
        for word, count in analysis['top_words'][:10]:
            print(f"  {word:20s} {count:5d}x")
        
        # Conteúdo real
        print(f"\n📖 INÍCIO (primeiras 10 linhas):")
        for i, line in enumerate(analysis['samples']['inicio'][:10], 1):
            preview = line[:70] + "..." if len(line) > 70 else line
            print(f"  {i:2d}: {preview}")
        
        if analysis['samples']['meio']:
            print(f"\n📖 MEIO (amostra):")
            for i, line in enumerate(analysis['samples']['meio'][:5], 1):
                preview = line[:70] + "..." if len(line) > 70 else line
                print(f"  {i:2d}: {preview}")
        
        # Report (neutral)
        if analysis['report']:
            print(f"\n📋 CARACTERÍSTICAS DETECTADAS:")
            for item in analysis['report']:
                print(f"  • {item['category']}: {item['count']} ocorrências")
                print(f"    ℹ️  {item['note']}")
        
        # Mostrar todos os padrões detectados 
        print(f"\n📊 PADRÕES ENCONTRADOS:")
        for pattern, count in sorted(analysis['patterns'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  • {pattern}: {count}x")
        
        print("\n" + "─" * 80)

print(f"\n{'=' * 80}")
print("📝 ANÁLISE COMPLETA")
print("=" * 80)
print("\nRevise o conteúdo acima e decida:")
print("  - Arquivos para DELETAR (muito repetitivos/inúteis)")
print("  - Arquivos para LIMPAR (têm lixo mas também conteúdo bom)")
print("  - Arquivos para MANTER (conteúdo de qualidade)")
print("=" * 80)
