#!/usr/bin/env python3
"""
Filtro V12 - Corpus Ultra Limpo Brasileiro
Roda sobre o output do V11 para limpeza adicional

Uso: python scripts/filter_v12_ultra_clean.py
"""

import re
import sys
from pathlib import Path
from collections import Counter

# ============================================
# PADRÕES DE LIXO WIKIPEDIA
# ============================================

WIKIPEDIA_GARBAGE_PATTERNS = [
    # Discussões e metadata
    r'(?i)\b(discussão|discussões)\b.*\b(página|usuário|artigo)\b',
    r'(?i)\beliminar\s+(a\s+)?página\b',
    r'(?i)\bvotação\s+(para|de)\b',
    r'(?i)\bconsenso\s+(unânime|para)\b',
    r'(?i)\bPE\s+eliminando\b',
    r'(?i)\bencerro\s+assim\b',
    r'(?i)\b(favor|contra)\b.*\bvoto\b',
    r'(?i)\bpedido\s+de\s+(eliminação|proteção|bloqueio)\b',
    r'(?i)\bredirecionamento\s+(para|de)\b',
    r'(?i)\busuário(\s*:|.*discussão)\b',
    r'(?i)\b(talk|user)\s*:\b',
    r'(?i)\bwikipédia\s*:\b',
    r'(?i)\bpredefinição\s*:\b',
    r'(?i)\bcategoria\s*:\b',
    r'(?i)\b\d+h\d+min\b',  # Timestamps: 00h54min
    r'(?i)\b(UTC|BRT)\b',
    
    # Infoboxes e templates que escaparam
    r'formato_áudio\s*=',
    r'p_transmissão\s*=',
    r'ult_transmissão\s*=',
    r'episódios\s*=',
    r'\|\s*nome\s*=',
    r'\|\s*imagem\s*=',
    r'\|\s*legenda\s*=',
    r'{\s*\|',  # Início de tabela
    r'\|\s*}',  # Fim de tabela
    r'\d+i\s+formato',  # 1080i formato
    r'\bpx\b.*\bpx\b',  # dimensões de imagem
    
    # Instruções de edição
    r'(?i)aqui\s+deve-?se\s+ficar\b',
    r'(?i)caso\s+não\s+saiba\b',
    r'(?i)use\s+o\s+modelo\b',
    r'(?i)ver\s+também\s*:',
    r'(?i)ligações\s+externas\s*:',
    r'(?i)referências\s*:',
    r'(?i)notas\s+e\s+referências',
    r'(?i)bibliografia\s*:',
    r'(?i)\bstub\b',
    r'(?i)\besboço\b',
    
    # Fragmentos numéricos/técnicos
    r'^\s*\d+\s*:\s*\w+',  # "1: Hommaru..."
    r'\(\s*vermelho\s*\)',
    r'\(\s*azul\s*\)',
    r'\(\s*verde\s*\)',
    r'\(\s*amarelo\s*\)',
    r'^\s*[\|\-\+\=]{3,}',  # Linhas de tabela
    
    # Código e markup
    r'<ref\b',
    r'</ref>',
    r'\[\[arquivo:',
    r'\[\[ficheiro:',
    r'\[\[imagem:',
    r'\[\[file:',
    r'\[\[image:',
    r'thumb\s*\|',
    r'\{\{[a-z]+\s*\|',  # Templates
]

# ============================================
# CONTEÚDO ESTRANGEIRO (mais agressivo)
# ============================================

ESTRANGEIRO_EXTRA_PATTERNS = [
    # Empresas estrangeiras
    r'(?i)\bmistral\s+ai\b',
    r'(?i)\bopenai\b',
    r'(?i)\bgoogle\s+(inc|llc|cloud|ai)\b',
    r'(?i)\bmicrosoft\s+(corp|azure)\b',
    r'(?i)\bapple\s+(inc|computer)\b',
    r'(?i)\bamazon\s+(inc|web|aws)\b',
    r'(?i)\bmeta\s+(platforms|ai)\b',
    r'(?i)\bnvidia\b',
    r'(?i)\bintel\s+corp\b',
    r'(?i)\bibm\s+corp\b',
    
    # Instituições estrangeiras
    r'(?i)\bjet\s+propulsion\s+laboratory\b',
    r'(?i)\bnasa\b',
    r'(?i)\bcern\b',
    r'(?i)\bmit\b(?!\s+(do|de|da))',  # MIT mas não "mit do..."
    r'(?i)\bharvard\b',
    r'(?i)\bstanford\b',
    r'(?i)\boxford\b',
    r'(?i)\bcambridge\b(?!\s+analytica)',
    
    # Nomes estrangeiros específicos
    r'(?i)\bwayne\s+ratliff\b',
    r'(?i)\bhommaru\b',
    r'(?i)\bsamurai\b',
    r'(?i)\bshogun\b',
    r'(?i)\bkaiser\b',
    r'(?i)\breichstag\b',
    
    # Castelos e lugares específicos estrangeiros
    r'(?i)\bcastle\b',
    r'(?i)\bchâteau\b',
    r'(?i)\bpalazzo\b',
    r'(?i)\bschloss\b',
    
    # Contextos claramente não-brasileiros
    r'(?i)\bprimeira\s+guerra\s+mundial\b(?!.*brasil)',
    r'(?i)\bsegunda\s+guerra\s+mundial\b(?!.*(brasil|feb|pracinhas))',
    r'(?i)\bguerra\s+fria\b(?!.*brasil)',
    r'(?i)\bholocausto\b',
    r'(?i)\bnazismo\b',
    r'(?i)\bfascismo\b(?!.*brasil)',
    
    # Ortografia de Portugal
    r'(?i)\bactriz\b',
    r'(?i)\bactor\b',
    r'(?i)\bacção\b',
    r'(?i)\bdirecção\b',
    r'(?i)\bselecção\b',
    r'(?i)\bóptimo\b',
    r'(?i)\bfacto\b',
    r'(?i)\bcontacto\b',
    r'(?i)\baptismo\b',
]

# ============================================
# INDICADORES BRASILEIROS FORTES
# ============================================

BRASIL_FORTE_PATTERNS = [
    # Políticos brasileiros
    r'(?i)\b(lula|dilma|bolsonaro|temer)\b',
    r'(?i)\bfernando\s+henrique\b',
    r'(?i)\b(collor|sarney|itamar)\b',
    r'(?i)\bgetúlio\s+vargas\b',
    r'(?i)\bjuscelino\s+kubitschek\b',
    r'(?i)\bdom\s+pedro\s+(i|ii|primeiro|segundo)\b',
    r'(?i)\bprincesa\s+isabel\b',
    r'(?i)\btiradentes\b',
    r'(?i)\bcastello?\s+branco\b',
    r'(?i)\bcosta\s+e\s+silva\b',
    r'(?i)\bmédici\b',
    r'(?i)\bgeisel\b',
    r'(?i)\bfigueiredo\b',
    r'(?i)\btancredo\s+neves\b',
    
    # Instituições brasileiras
    r'(?i)\bpetrobras\b',
    r'(?i)\bvale\s+(do\s+rio\s+doce)?\b',
    r'(?i)\bembraer\b',
    r'(?i)\bbndes\b',
    r'(?i)\bcaixa\s+econômica\b',
    r'(?i)\bbanco\s+(do\s+)?brasil\b',
    r'(?i)\bitaú\b',
    r'(?i)\bbradesco\b',
    r'(?i)\bsantander\s+brasil\b',
    r'(?i)\brede\s+globo\b',
    r'(?i)\btv\s+(globo|record|sbt|bandeirantes|cultura)\b',
    r'(?i)\bfolha\s+de\s+s\.?\s*paulo\b',
    r'(?i)\bestadão\b',
    r'(?i)\bo\s+globo\b',
    
    # Universidades brasileiras
    r'(?i)\b(usp|unicamp|ufrj|ufmg|ufrgs)\b',
    r'(?i)\b(ufpr|ufsc|ufba|ufscar|unifesp)\b',
    r'(?i)\buniversidade\s+(de\s+)?são\s+paulo\b',
    r'(?i)\buniversidade\s+federal\b',
    r'(?i)\bpuc[\s-]+(rio|sp|rs|pr|mg)\b',
    r'(?i)\bfgv\b',
    r'(?i)\binsper\b',
    
    # Futebol brasileiro
    r'(?i)\bcampeonato\s+brasileiro\b',
    r'(?i)\bbrasileirão\b',
    r'(?i)\bcopa\s+do\s+brasil\b',
    r'(?i)\blibertadores\b.*\b(brasil|brasileiro)\b',
    r'(?i)\b(flamengo|corinthians|palmeiras)\b',
    r'(?i)\bsão\s+paulo\s+f\.?c\.?\b',
    r'(?i)\bsantos\s+f\.?c\.?\b',
    r'(?i)\b(grêmio|internacional)\b.*\b(porto\s+alegre|gaúcho)\b',
    r'(?i)\b(cruzeiro|atlético)\s+(mineiro|mg)\b',
    r'(?i)\b(vasco|botafogo|fluminense)\b',
    r'(?i)\bseleção\s+brasileira\b',
    r'(?i)\bcbf\b',
    r'(?i)\b(pelé|ronaldo\s+nazário|ronaldinho|neymar|zico|romário)\b',
    r'(?i)\bmaracanã\b',
    
    # Legislação brasileira
    r'(?i)\blei\s+(federal|estadual|municipal)\b',
    r'(?i)\bconsolidação\s+das\s+leis\s+do\s+trabalho\b',
    r'(?i)\b(clt)\b',
    r'(?i)\bconstituição\s+(federal|de\s+1988|brasileira)\b',
    r'(?i)\bcódigo\s+(civil|penal|processo)\s+brasileiro\b',
    r'(?i)\bsupremo\s+tribunal\s+federal\b',
    r'(?i)\bstf\b',
    r'(?i)\bstj\b',
    r'(?i)\btse\b',
    r'(?i)\btst\b',
    r'(?i)\btcu\b',
    r'(?i)\bministério\s+público\b',
    r'(?i)\bdefensoria\s+pública\b',
    r'(?i)\badvocacia[\s-]geral\s+da\s+união\b',
    r'(?i)\bcâmara\s+dos\s+deputados\b',
    r'(?i)\bsenado\s+federal\b',
    r'(?i)\bcongresso\s+nacional\b',
    
    # Geografia e cultura brasileira
    r'(?i)\bamazônia\b',
    r'(?i)\bpantanal\b',
    r'(?i)\bcerrado\b',
    r'(?i)\bcaatinga\b',
    r'(?i)\bmata\s+atlântica\b',
    r'(?i)\bfernando\s+de\s+noronha\b',
    r'(?i)\bchapada\s+(diamantina|dos\s+veadeiros|guimarães)\b',
    r'(?i)\brio\s+(amazonas|são\s+francisco|paraná)\b',
    r'(?i)\bcarnaval\b',
    r'(?i)\bsamba\b',
    r'(?i)\bbossa\s+nova\b',
    r'(?i)\bforró\b',
    r'(?i)\bfesta\s+junina\b',
    r'(?i)\bsão\s+joão\b.*\bnordeste\b',
    r'(?i)\bfeijoada\b',
    r'(?i)\bcapoeira\b',
    
    # Estados e cidades (menção direta)
    r'(?i)\b(são\s+paulo|rio\s+de\s+janeiro|brasília|salvador|belo\s+horizonte)\b',
    r'(?i)\b(fortaleza|manaus|curitiba|recife|porto\s+alegre)\b',
    r'(?i)\bestado\s+de\s+(são\s+paulo|rio|minas|bahia|paraná)\b',
    r'(?i)\b(gaúcho|paulista|carioca|mineiro|baiano|nordestino)\b',
    
    # Documentos brasileiros
    r'(?i)\b(cpf|cnpj)\b',
    r'(?i)\bcarteira\s+de\s+trabalho\b',
    r'(?i)\b(inss|fgts|pis|pasep)\b',
    r'(?i)\bdiário\s+oficial\b',
]

# ============================================
# COMPILAR PATTERNS
# ============================================

GARBAGE_COMPILED = [re.compile(p) for p in WIKIPEDIA_GARBAGE_PATTERNS + ESTRANGEIRO_EXTRA_PATTERNS]
BRASIL_FORTE_COMPILED = [re.compile(p) for p in BRASIL_FORTE_PATTERNS]

# Indicadores básicos do V11 (simplificados)
BRASIL_BASICO = re.compile(
    r'(?i)\b(brasil|brasileiro|brasileira|brasileiros|brasileiras)\b'
)

ESTADOS_BR = re.compile(
    r'(?i)\b(acre|alagoas|amapá|amazonas|bahia|ceará|espírito\s+santo|goiás|'
    r'maranhão|mato\s+grosso|minas\s+gerais|pará|paraíba|paraná|pernambuco|'
    r'piauí|rio\s+grande|rondônia|roraima|santa\s+catarina|sergipe|tocantins)\b'
)

# ============================================
# FUNÇÕES DE FILTRO
# ============================================

def tem_lixo(texto: str) -> bool:
    """Verifica se tem lixo Wikipedia ou conteúdo estrangeiro."""
    for pattern in GARBAGE_COMPILED:
        if pattern.search(texto):
            return True
    return False

def tem_brasil_forte(texto: str) -> bool:
    """Verifica se tem indicador brasileiro forte."""
    for pattern in BRASIL_FORTE_COMPILED:
        if pattern.search(texto):
            return True
    return False

def tem_brasil_basico(texto: str) -> bool:
    """Verifica indicadores básicos de Brasil."""
    return bool(BRASIL_BASICO.search(texto) or ESTADOS_BR.search(texto))

def calcular_qualidade(texto: str) -> float:
    """Calcula score de qualidade do texto."""
    score = 0.0
    
    # Comprimento adequado
    if len(texto) > 200:
        score += 0.2
    if len(texto) > 500:
        score += 0.1
    
    # Proporção de letras
    letras = sum(1 for c in texto if c.isalpha())
    total = len(texto.replace(' ', ''))
    if total > 0:
        proporcao = letras / total
        if proporcao > 0.8:
            score += 0.3
        elif proporcao > 0.7:
            score += 0.2
    
    # Palavras por frase (coerência)
    frases = texto.split('.')
    palavras_por_frase = [len(f.split()) for f in frases if f.strip()]
    if palavras_por_frase:
        media = sum(palavras_por_frase) / len(palavras_por_frase)
        if 10 < media < 30:  # Frases bem formadas
            score += 0.2
    
    # Poucos caracteres especiais
    especiais = sum(1 for c in texto if c in '|{}[]<>=')
    if especiais < len(texto) * 0.02:
        score += 0.2
    
    return score

def linha_valida_v12(linha: str) -> tuple[bool, str]:
    """Avaliação V12 rigorosa."""
    linha = linha.strip()
    
    # Muito curta (100 chars mínimo)
    if len(linha) < 100:
        return False, "muito_curta"
    
    # Poucas palavras (15 mínimo)
    palavras = linha.split()
    if len(palavras) < 15:
        return False, "poucas_palavras"
    
    # Tem lixo Wikipedia/estrangeiro específico
    if tem_lixo(linha):
        return False, "lixo"
    
    # Brasil forte = sempre manter
    if tem_brasil_forte(linha):
        return True, "brasil_forte"
    
    # Brasil básico + qualidade boa = manter
    if tem_brasil_basico(linha):
        qualidade = calcular_qualidade(linha)
        if qualidade >= 0.5:
            return True, "brasileiro"
        else:
            return False, "baixa_qualidade"
    
    # Sem indicador brasileiro = remover
    return False, "sem_brasil"

def processar_v12(input_path: str, output_path: str):
    """Processa corpus com filtro V12."""
    
    print("=" * 60)
    print("🇧🇷 FILTRO V12 - CORPUS ULTRA LIMPO")
    print("=" * 60)
    print(f"📂 Lendo: {input_path}")
    
    if not Path(input_path).exists():
        print(f"❌ Arquivo não encontrado: {input_path}")
        sys.exit(1)
    
    stats = Counter()
    linhas_boas = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, linha in enumerate(f):
            manter, motivo = linha_valida_v12(linha)
            stats[motivo] += 1
            
            if manter:
                linhas_boas.append(linha)
            
            if (i + 1) % 100000 == 0:
                print(f"  Processadas: {i+1:,}...")
    
    total = sum(stats.values())
    
    print(f"\n📊 Estatísticas V12:")
    print(f"  Total entrada: {total:,}")
    print()
    
    mantidas = 0
    for motivo, count in stats.most_common():
        pct = count / total * 100
        if motivo in ["brasil_forte", "brasileiro"]:
            status = "✅"
            mantidas += count
        else:
            status = "❌"
        print(f"  {status} {motivo}: {count:,} ({pct:.1f}%)")
    
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
    print(f"\nPróximos passos:")
    print(f"  1. Verificar amostra:")
    print(f"     python -c \"import random; lines=open('{output_path}').readlines(); [print(s[:100]) for s in random.sample(lines,10)]\"")
    print(f"  2. Retreinar tokenizer:")
    print(f"     .\\target\\release\\ptbr-llm.exe train-tokenizer --input {output_path} --output data/tokenizer_v12/")
    print(f"  3. Tokenizar e treinar")
    
    return len(linhas_boas)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    input_file = "data/sovereign/corpus_v11_brasil.txt"
    output_file = "data/sovereign/corpus_v12_ultra.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    processar_v12(input_file, output_file)