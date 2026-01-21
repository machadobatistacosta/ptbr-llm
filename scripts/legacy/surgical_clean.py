#!/usr/bin/env python3
"""
LIMPEZA CIRÚRGICA
Remove padrões de "lixo geográfico" repetitivo, mantendo o resto.
"""
import re
from pathlib import Path
import os

# Padrões exatos do lixo que vimos (listas de cidades/comunas)
# Se uma linha/parágrafo contém isso E pouco mais, é lixo.
GARBAGE_PATTERNS = [
    r"é uma comuna francesa na região administrativa",
    r"é um município da .*?Estende-se por uma área",
    r"é uma cidade da .*?Estende-se por uma área",
    r"é uma povoação da .*?Estende-se por uma área",
]

def clean_file_surgical(filepath: Path) -> tuple[int, int]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_lines = len(lines)
    new_lines = []
    
    for line in lines:
        is_garbage = False
        # Verifica se linha bate com padrão de lixo
        for pattern in GARBAGE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Confirma se é linha curta/boilerplate (evita deletar artigo completo sobre Paris)
                if len(line) < 500: # Boilerplates costumam ser curtos
                    is_garbage = True
                    break
        
        if not is_garbage:
            new_lines.append(line)
    
    # Se arquivo ficou muito pequeno (<1KB) ou vazio, marca para deleção
    content = "".join(new_lines)
    if len(content.strip()) < 500: # Menos de 500 chars sobrando = stub inútil
        return original_lines, -1 # Código para deletar
        
    if len(new_lines) < original_lines:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
    return original_lines, len(new_lines)

sources = {
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
    # Planalto não limpamos (é sagrado)
}

print("=" * 80)
print("🧹 LIMPEZA CIRÚRGICA - Removendo boilerplates geográficos")
print("=" * 80)

total_deleted_files = 0
total_cleaned_lines = 0

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    print(f"\n📁 {source_name}: processando {len(files)} arquivos...")
    
    for f in files:
        orig, final = clean_file_surgical(f)
        
        if final == -1:
            print(f"  🗑️ DELETADO (ficou vazio/stub): {f.name}")
            os.remove(f)
            total_deleted_files += 1
        elif final < orig:
            removed = orig - final
            total_cleaned_lines += removed
            # print(f"  ✨ Limpo: {f.name} (-{removed} linhas)")

print(f"\n{'=' * 80}")
print(f"📊 RESULTADO FINAL:")
print(f"  Arquivos deletados (stubs/vazios): {total_deleted_files}")
print(f"  Linhas de lixo removidas: {total_cleaned_lines}")
print("=" * 80)
