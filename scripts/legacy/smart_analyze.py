#!/usr/bin/env python3
"""
Limpeza INTELIGENTE - Remove padrões repetitivos, mantém conhecimento
"""
from pathlib import Path
from collections import defaultdict
import re
from difflib import SequenceMatcher

def get_template_signature(text: str) -> str:
    """
    Extrai 'assinatura' do template removendo nomes próprios
    Para detectar textos com mesma estrutura
    """
    # Remove nomes próprios (palavras capitalizadas)
    sig = re.sub(r'\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b', 'NAME', text)
    # Remove números
    sig = re.sub(r'\d+', 'NUM', sig)
    # Remove pontuação variável
    sig = re.sub(r'[\.,:;!?]+', '.', sig)
    return sig[:500]  # Primeiros 500 chars da assinatura

def is_template_duplicate(texts: list) -> bool:
    """
    Verifica se lista de textos são duplicatas de template
    """
    if len(texts) < 10:
        return False
    
    signatures = [get_template_signature(t) for t in texts]
    
    # Se 80%+ das assinaturas são muito similares (>90%), é template
    matches = 0
    for i, sig1 in enumerate(signatures[:20]):  # Amostra de 20
        for sig2 in signatures[i+1:21]:
            similarity = SequenceMatcher(None, sig1, sig2).ratio()
            if similarity > 0.9:
                matches += 1
    
    threshold = (20 * 19 / 2) * 0.8  # 80% dos pares
    return matches > threshold

def analyze_file(filepath: Path) -> dict:
    """Analisa arquivo para padrões repetitivos"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Divide em parágrafos
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Agrupa por assinatura de template
    template_groups = defaultdict(list)
    for para in paragraphs:
        if len(para) < 50:  # Muito curto, ignora
            continue
        sig = get_template_signature(para)
        template_groups[sig].append(para)
    
    # Encontra grupos duplicados
    duplicate_groups = {sig: paras for sig, paras in template_groups.items() 
                       if len(paras) >= 5}  # 5+ cópias = suspeito
    
    total_paras = len(paragraphs)
    duplicate_paras = sum(len(paras) for paras in duplicate_groups.values())
    
    return {
        'total_paragraphs': total_paras,
        'duplicate_paragraphs': duplicate_paras,
        'duplicate_ratio': duplicate_paras / total_paras if total_paras > 0 else 0,
        'duplicate_groups': duplicate_groups
    }

# Analisa todos os arquivos
sources = {
    "wiki_clean": Path("data/wiki_clean"),
    "wikibooks_clean": Path("data/wikibooks_clean"),
    "wikinews_clean": Path("data/wikinews_clean"),
    "wikisource_clean": Path("data/wikisource_clean"),
}

print("=" * 80)
print("🔬 ANÁLISE INTELIGENTE - Detectando padrões repetitivos")
print("=" * 80)

problematic_files = []

for source_name, source_path in sources.items():
    if not source_path.exists():
        continue
    
    files = list(source_path.glob("*.txt"))
    print(f"\n📁 {source_name}: analisando TODOS os {len(files)} arquivos...")
    
    for f in files:  # TODOS os arquivos, não amostra
        analysis = analyze_file(f)
        
        if analysis['duplicate_ratio'] > 0.5:  # >50% duplicatas
            problematic_files.append({
                'source': source_name,
                'file': f.name,
                'ratio': analysis['duplicate_ratio'],
                'total': analysis['total_paragraphs'],
                'dupes': analysis['duplicate_paragraphs']
            })
            print(f"  ⚠️ {f.name}: {analysis['duplicate_ratio']*100:.1f}% repetitivo")

print(f"\n{'=' * 80}")
print(f"📊 RESULTADO (TODOS os {sum(len(list(p.glob('*.txt'))) for p in sources.values() if p.exists())} arquivos):")
print(f"  Arquivos problemáticos (>50% repetição): {len(problematic_files)}")

if problematic_files:
    print(f"\n🗑️ TOP 10 MAIS REPETITIVOS:")
    for item in sorted(problematic_files, key=lambda x: x['ratio'], reverse=True)[:10]:
        print(f"  {item['ratio']*100:5.1f}% | {item['source']:20s} | {item['file']}")
    
    print(f"\n💡 RECOMENDAÇÃO:")
    print(f"  - Arquivos com >80% repetição: DELETAR")
    print(f"  - Arquivos com 50-80% repetição: LIMPAR parágrafos duplicados")
    print(f"  - Arquivos com <50% repetição: MANTER como estão")
else:
    print(f"\n✅ Nenhum arquivo altamente repetitivo encontrado em TODOS os arquivos!")

print("=" * 80)
