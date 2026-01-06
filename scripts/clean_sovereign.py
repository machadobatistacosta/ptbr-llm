#!/usr/bin/env python3
"""
clean_sovereign_v11.py - ESCRITA INCREMENTAL (sem MemoryError)
"""

import re
from pathlib import Path
from collections import Counter

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ═══════════════════════════════════════════════════════════════
# PADRÕES
# ═══════════════════════════════════════════════════════════════

CODE_PATTERNS = [
    (r'<\?php', 'PHP'),
    (r'\$wg[A-Z]\w+', 'MediaWiki'),
    (r'function\s+\w+\s*\([^)]*\)\s*\{', 'JS/PHP func'),
    (r'include_once\s*\(', 'PHP include'),
    (r'#include\s*<', 'C include'),
    (r'public\s+static\s+void', 'Java'),
    (r'def\s+__\w+__\s*\(', 'Python dunder'),
    (r'using\s+namespace\s+std', 'C++ using'),
    (r'printf\s*\(\s*"', 'C printf'),
    (r'cout\s*<<', 'C++ cout'),
    (r'System\.out\.print', 'Java print'),
    (r'return\s+\$\w+', 'PHP return'),
    (r'\}\s*else\s*\{', 'else block'),
]

IBERIAN_PATTERNS = [
    r'freguesia\s+portuguesa',
    r'município\s+de\s+portugal',
    r'povoação\s+portuguesa',
    r'aldeia\s+portuguesa',
    r'vila\s+portuguesa',
    r'cidade\s+portuguesa',
    r'\bfreguesia\s+de\b',
    r'\bconcelho\s+de\b',
    r'\bdistrito\s+de\s+(lisboa|porto|braga|coimbra|faro|aveiro|leiria|setúbal|évora|beja|bragança|castelo\s+branco|guarda|portalegre|santarém|viana|vila\s+real|viseu)\b',
    r'\bem\s+(lisboa|coimbra|porto|braga|guimarães|évora|faro|aveiro|leiria|setúbal)\b',
    r'\bde\s+(lisboa|coimbra|porto|braga|guimarães|évora|faro|aveiro)\b(?!\s+alegre)',
    r'\buniversidade\s+de\s+(coimbra|lisboa|porto)\b',
    r'\bacademia\s+das\s+ciências\b',
    r'\baçores\b',
    r'\bilha\s+da\s+madeira\b',
    r'escritor\s+português',
    r'poeta\s+português',
    r'autor\s+português',
    r'artista\s+português',
    r'político\s+português',
    r'literatura\s+portuguesa',
    r'história\s+de\s+portugal',
    r'nascido\s+em\s+lisboa',
    r'nasceu\s+em\s+lisboa',
    r'morreu\s+em\s+lisboa',
    r'comunidade\s+autónoma',
    r'comunidad\s+autónoma',
    r'província\s+de\s+(girona|barcelona|madrid|valencia|sevilla|zaragoza)',
    r'comarca\s+de\s+(alt|baix|garrotxa)',
    r'\bcatalunha\b',
    r'\bcataluña\b',
    r'\bcatalunya\b',
    r'\bpaís\s+basco\b',
    r'\bgaliza\b',
    r'\bandaluzia\b',
    r'título\s+em\s+portugal',
    r'código\s+postal\s+\d{4}',
]

# ═══════════════════════════════════════════════════════════════
# FUNÇÕES
# ═══════════════════════════════════════════════════════════════

def fix_encoding(text: str) -> str:
    if HAS_FTFY:
        return ftfy.fix_text(text)
    return text


def clean_wiki(text: str) -> str:
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'[\{\}]', '', text)
    text = re.sub(r'\|\s*\w+\s*=\s*[^|\n]*', ' ', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[(File|Imagem|Ficheiro|Arquivo|Category|Categoria):[^\]]*\]\]', '', text, flags=re.I)
    text = re.sub(r'\[\[[^\]|]*\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub(r'\b(thumb|miniatura|miniaturadaimagem|\d+px|direita|esquerda|right|left|center)\b', '', text, flags=re.I)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def has_code(text: str) -> bool:
    for pattern, _ in CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def is_iberian(text: str) -> bool:
    text_lower = text.lower()
    for pattern in IBERIAN_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_good_paragraph(text: str) -> bool:
    text = text.strip()
    if len(text) < 50:
        return False
    words = text.split()
    if len(words) < 10:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.6:
        return False
    if not re.search(r'[.!?]', text):
        return False
    if text.count('|') > 3 or text.count('=') > 5:
        return False
    return True


def process_file(filepath: Path, output_handle, stats: Counter):
    """Processa arquivo e ESCREVE DIRETAMENTE no output."""
    try:
        raw = filepath.read_text(encoding='utf-8', errors='replace')
    except:
        return
    
    text = fix_encoding(raw)
    paragraphs = re.split(r'\n+', text)
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < 30:
            continue
        
        para = clean_wiki(para)
        if len(para) < 30:
            continue
        
        stats['total'] += 1
        
        if has_code(para):
            stats['code'] += 1
            continue
        
        if is_iberian(para):
            stats['iberian'] += 1
            continue
        
        if not is_good_paragraph(para):
            stats['low_quality'] += 1
            continue
        
        # ESCREVE DIRETAMENTE - não acumula na memória!
        output_handle.write(para + '\n\n')
        stats['kept'] += 1


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python script.py <input_dir> <output_file>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    stats = Counter()
    
    files = sorted(input_dir.glob('*.txt'))
    
    print("=" * 60)
    print("  🇧🇷 LIMPEZA SOBERANA V11")
    print("  Escrita Incremental (sem MemoryError)")
    print("=" * 60)
    print(f"  📂 Input:  {input_dir}")
    print(f"  📄 Output: {output_file}")
    print(f"  📚 Arquivos: {len(files)}")
    print("=" * 60)
    
    # ABRE ARQUIVO UMA VEZ E ESCREVE INCREMENTALMENTE
    with open(output_file, 'w', encoding='utf-8') as out:
        for i, f in enumerate(files):
            process_file(f, out, stats)
            
            if (i + 1) % 10 == 0:
                pct = stats['kept'] / max(stats['total'], 1) * 100
                print(f"  [{i+1:3d}/{len(files)}] ✅ {stats['kept']:,} ({pct:.1f}%)")
    
    total = max(stats['total'], 1)
    
    # Tamanho do arquivo
    file_size = output_file.stat().st_size / 1024 / 1024
    
    print("\n" + "=" * 60)
    print("  📊 RESULTADO FINAL")
    print("=" * 60)
    print(f"  Total parágrafos: {stats['total']:>12,}")
    print(f"  ❌ Código:         {stats['code']:>12,} ({stats['code']/total*100:5.1f}%)")
    print(f"  ❌ Ibérico:        {stats['iberian']:>12,} ({stats['iberian']/total*100:5.1f}%)")
    print(f"  ❌ Qualidade:      {stats['low_quality']:>12,} ({stats['low_quality']/total*100:5.1f}%)")
    print(f"  ✅ MANTIDOS:       {stats['kept']:>12,} ({stats['kept']/total*100:5.1f}%)")
    print("-" * 60)
    print(f"  📦 Tamanho final: {file_size:.1f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()