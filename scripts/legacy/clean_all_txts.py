#!/usr/bin/env python3
"""
Limpeza automática COMPLETA - Remove TODO o lixo dos txts
"""
import re
from pathlib import Path
import shutil

# Padrões de lixo a remover
JUNK_PATTERNS = [
    # Listas de municípios/comunas estrangeiras
    r"[^\n]*é uma comuna francesa[^\n]*\n",
    r"[^\n]*é um município.*?(Bulgária|Romênia|Hungria|Polônia)[^\n]*\n",
    r"[^\n]*é uma cidade.*?(Bulgária|Romênia|Hungria|Polônia)[^\n]*\n",
    r"[^\n]*Estende-se por uma área de[^\n]*\n",
    
    # Stubs muito curtos (< 50 chars por linha repetidos)
    r"^.{1,50}\n(?=.{1,50}\n.{1,50}\n)",
    
    # Info boxes wiki vazios ou quebrados
    r"\{\{[^\}]*\}\}",
    r"\[\[[^\]]*\]\]",
]

def clean_file(filepath: Path) -> tuple[int, int]:
    """
    Limpa arquivo removendo padrões de lixo
    Retorna: (chars_antes, chars_depois)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        original = f.read()
    
    cleaned = original
    for pattern in JUNK_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove linhas vazias excessivas (mais de 2 seguidas)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove espaços em branco no final das linhas
    cleaned = re.sub(r' +\n', '\n', cleaned)
    
    chars_before = len(original)
    chars_after = len(cleaned)
    
    # Só salva se houve mudança significativa (>1% redução)
    if chars_after < chars_before * 0.99:
        # Backup
        backup_path = filepath.with_suffix('.txt.bak')
        shutil.copy2(filepath, backup_path)
        
        # Salva limpo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    
    return chars_before, chars_after

sources = {
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("=" * 80)
print("🧹 LIMPEZA AUTOMÁTICA COMPLETA - Removendo TODO o lixo")
print("=" * 80)

total_before = 0
total_after = 0
cleaned_count = 0

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    print(f"\n📁 {source_name}: limpando {len(files)} arquivos...")
    
    for f in files:
        before, after = clean_file(f)
        total_before += before
        total_after += after
        
        if after < before * 0.99:
            cleaned_count += 1
            reduction = ((before - after) / before) * 100
            print(f"  ✅ {f.name}: -{reduction:.1f}% ({before/1024:.1f}KB → {after/1024:.1f}KB)")

print(f"\n{'=' * 80}")
print(f"📊 RESULTADO:")
print(f"  Arquivos processados: {sum(len(list(p.glob('*.txt'))) for p in sources.values() if p.exists())}")
print(f"  Arquivos limpos: {cleaned_count}")
print(f"  Tamanho antes: {total_before/1024/1024:.1f} MB")
print(f"  Tamanho depois: {total_after/1024/1024:.1f} MB")
print(f"  Redução: {((total_before - total_after)/total_before)*100:.1f}%")
print(f"\n💾 Backups salvos como *.txt.bak")
print("=" * 80)
