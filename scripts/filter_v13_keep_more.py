# scripts/filter_v13_keep_more.py

"""
Filtro V13 - Manter Mais, Remover Menos
Estratégia: Remove só lixo óbvio, mantém conteúdo ambíguo
"""

import re
from pathlib import Path
from collections import Counter

# ============================================
# REMOVER APENAS ISSO (lista curta e específica)
# ============================================

REMOVER_PATTERNS = [
    # Países/idiomas específicos (não Brasil)
    r'(?i)\b(portugal|português|portuguesa|lisboa|porto|coimbra)\b(?!.*brasil)',
    r'(?i)\b(estados\s+unidos|americano|americana|washington|california)\b',
    r'(?i)\b(inglaterra|inglês|inglesa|londres|reino\s+unido|britânico)\b',
    r'(?i)\b(frança|francês|francesa|paris)\b',
    r'(?i)\b(alemanha|alemão|alemã|berlim)\b',
    r'(?i)\b(espanha|espanhol|espanhola|madrid|barcelona)\b',
    r'(?i)\b(itália|italiano|italiana|roma|milão)\b',
    r'(?i)\b(japão|japonês|japonesa|tóquio)\b',
    r'(?i)\b(china|chinês|chinesa|pequim)\b',
    
    # Ortografia PT-PT
    r'(?i)\bactriz\b',
    r'(?i)\bactor\b', 
    r'(?i)\bacção\b',
    r'(?i)\bdirecção\b',
    r'(?i)\bselecção\b',
    
    # Metadata Wikipedia
    r'(?i)\bdiscussão\b.*\bpágina\b',
    r'(?i)\beliminar\s+artigo\b',
    r'(?i)\bvotação\s+para\b',
    r'(?i)\bconsenso\s+unânime\b',
    r'(?i)\busuário\s*:',
    r'(?i)\bwikipédia\s*:',
    r'(?i)\bpredefinição\s*:',
    r'(?i)\bcategoria\s*:',
    
    # Lixo técnico
    r'formato_áudio\s*=',
    r'p_transmissão\s*=',
    r'\|\s*nome\s*=',
    r'\|\s*imagem\s*=',
    r'\{\s*\|',
    r'\|\s*\}',
]

REMOVER_COMPILED = [re.compile(p) for p in REMOVER_PATTERNS]

def deve_remover(linha: str) -> tuple[bool, str]:
    """Retorna (remover, motivo) - só remove o claramente ruim."""
    linha = linha.strip()
    
    # Muito curta
    if len(linha) < 80:
        return True, "curta"
    
    # Poucas palavras
    if len(linha.split()) < 10:
        return True, "poucas_palavras"
    
    # Verifica patterns de remoção
    for pattern in REMOVER_COMPILED:
        if pattern.search(linha):
            return True, "pattern_ruim"
    
    # Proporção de letras muito baixa
    letras = sum(1 for c in linha if c.isalpha())
    total = len(linha.replace(' ', ''))
    if total > 0 and letras / total < 0.65:
        return True, "pouco_texto"
    
    # Muitos caracteres especiais
    especiais = sum(1 for c in linha if c in '|{}[]<>=')
    if especiais > len(linha) * 0.05:
        return True, "muito_especial"
    
    # MANTÉM tudo o resto
    return False, "ok"

def processar_v13(input_path: str, output_path: str):
    """Processa corpus com filtro V13."""
    
    print("=" * 60)
    print("🇧🇷 FILTRO V13 - MANTER MAIS CONTEÚDO")
    print("=" * 60)
    print(f"📂 Lendo: {input_path}")
    
    stats = Counter()
    linhas_boas = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, linha in enumerate(f):
            remover, motivo = deve_remover(linha)
            
            if remover:
                stats[f"removido_{motivo}"] += 1
            else:
                stats["mantido"] += 1
                linhas_boas.append(linha)
            
            if (i + 1) % 500000 == 0:
                print(f"  Processadas: {i+1:,}...")
    
    total = sum(stats.values())
    
    print(f"\n📊 Estatísticas V13:")
    print(f"  Total entrada: {total:,}")
    print()
    
    for motivo, count in stats.most_common():
        pct = count / total * 100
        status = "✅" if motivo == "mantido" else "❌"
        print(f"  {status} {motivo}: {count:,} ({pct:.1f}%)")
    
    mantidas = stats["mantido"]
    print(f"\n✅ Mantidas: {mantidas:,} ({mantidas/total*100:.1f}%)")
    
    # Salvar
    print(f"\n💾 Salvando: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(linhas_boas)
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"📦 Tamanho: {size_mb:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ CONCLUÍDO!")
    print("=" * 60)

if __name__ == "__main__":
    # Rodar no corpus V3 original (antes do filtro agressivo)
    processar_v13(
        "data/sovereign/corpus_v3.txt",
        "data/sovereign/corpus_v13_generous.txt"
    )