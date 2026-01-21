# scripts/filter_v14_quick_clean.py

"""
Filtro V14 - Limpeza rápida do V13
Remove só metadata Wikipedia que passou
"""

import re
from pathlib import Path

# Patterns de metadata Wikipedia
WIKI_GARBAGE = [
    r'~{3,}',  # ~~~~~ timestamps
    r'\d+h\d+min\s+de\s+\d+',  # 23h36min de 21
    r'(?i)\bdisc\s+\d+h',  # disc 23h
    r'(?i)\(UTC\)',
    r'(?i)\bvotação\b.*\b(dias?|prorrog)',
    r'(?i)\bpáginas?\s+você\s+vigia',
    r'(?i)\bdireito\s+de\s+imagem\b.*\bclube\b',
    r'(?i)\bseleção\s+(australiana|americana|inglesa|francesa|alemã|italiana|japonesa|chinesa)',
    r'(?i)\b(the\s+cw|nbc|cbs|abc|fox|hbo)\b',
    r'(?i)\bgangue\b.*\b(rua|street)\b',
    r'(?i)\bshaolin\b',
]

GARBAGE_COMPILED = [re.compile(p) for p in WIKI_GARBAGE]

def is_garbage(line: str) -> bool:
    for pattern in GARBAGE_COMPILED:
        if pattern.search(line):
            return True
    return False

def main():
    input_path = "data/sovereign/corpus_v13_generous.txt"
    output_path = "data/sovereign/corpus_v14_clean.txt"
    
    print("🧹 Filtro V14 - Limpeza rápida")
    print(f"📂 Lendo: {input_path}")
    
    kept = 0
    removed = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            if is_garbage(line):
                removed += 1
            else:
                fout.write(line)
                kept += 1
            
            if (i + 1) % 500000 == 0:
                print(f"  Processadas: {i+1:,}...")
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Mantidas: {kept:,}")
    print(f"❌ Removidas: {removed:,}")
    print(f"📦 Tamanho: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()