#!/usr/bin/env python3
"""
Filtro V11 - Corpus 100% Brasileiro
Estratégia: Manter APENAS linhas com indicadores brasileiros
Remover TUDO que seja ambíguo ou estrangeiro
"""

import re
import sys
from pathlib import Path
from collections import Counter

# ============================================
# INDICADORES POSITIVOS (manter se tiver)
# ============================================

ESTADOS_BRASILEIROS = [
    'acre', 'alagoas', 'amapá', 'amazonas', 'bahia', 'ceará',
    'distrito federal', 'espírito santo', 'goiás', 'maranhão',
    'mato grosso', 'mato grosso do sul', 'minas gerais', 'pará',
    'paraíba', 'paraná', 'pernambuco', 'piauí', 'rio de janeiro',
    'rio grande do norte', 'rio grande do sul', 'rondônia',
    'roraima', 'santa catarina', 'são paulo', 'sergipe', 'tocantins',
    # Siglas
    'ac', 'al', 'ap', 'am', 'ba', 'ce', 'df', 'es', 'go', 'ma',
    'mt', 'ms', 'mg', 'pa', 'pb', 'pr', 'pe', 'pi', 'rj', 'rn',
    'rs', 'ro', 'rr', 'sc', 'sp', 'se', 'to'
]

CIDADES_GRANDES = [
    'são paulo', 'rio de janeiro', 'brasília', 'salvador', 'fortaleza',
    'belo horizonte', 'manaus', 'curitiba', 'recife', 'porto alegre',
    'belém', 'goiânia', 'guarulhos', 'campinas', 'são luís', 'maceió',
    'natal', 'teresina', 'campo grande', 'joão pessoa', 'cuiabá',
    'aracaju', 'florianópolis', 'vitória', 'macapá', 'porto velho',
    'boa vista', 'rio branco', 'palmas'
]

TERMOS_BRASILEIROS = [
    # Identificadores diretos
    'brasil', 'brasileiro', 'brasileira', 'brasileiros', 'brasileiras',
    
    # Instituições
    'governo federal', 'governo estadual', 'prefeitura',
    'congresso nacional', 'senado federal', 'câmara dos deputados',
    'supremo tribunal federal', 'stf', 'stj', 'tse', 'tst', 'tcu',
    'ministério público', 'defensoria pública',
    
    # Legislação brasileira
    'constituição federal', 'código civil brasileiro', 'código penal brasileiro',
    'clt', 'consolidação das leis do trabalho', 'lei federal',
    'lei estadual', 'lei municipal', 'medida provisória',
    'decreto-lei', 'emenda constitucional',
    
    # Moeda e economia
    'real', 'reais', 'r$', 'banco central', 'bacen', 'bndes',
    'petrobras', 'vale', 'itaú', 'bradesco', 'banco do brasil',
    
    # Cultura e história
    'carnaval', 'samba', 'bossa nova', 'capoeira', 'feijoada',
    'império brasileiro', 'república brasileira', 'proclamação da república',
    'independência do brasil', 'dom pedro', 'getúlio vargas', 'jk',
    'juscelino kubitschek', 'tancredo neves', 'collor', 'lula', 'dilma',
    'fernando henrique', 'fhc', 'bolsonaro',
    
    # Geografia
    'amazônia', 'pantanal', 'cerrado', 'caatinga', 'mata atlântica',
    'rio amazonas', 'rio são francisco', 'rio paraná',
    'serra do mar', 'chapada diamantina', 'fernando de noronha',
    
    # Esporte
    'seleção brasileira', 'cbf', 'campeonato brasileiro', 'brasileirão',
    'corinthians', 'palmeiras', 'flamengo', 'santos', 'grêmio',
    'internacional', 'cruzeiro', 'atlético mineiro', 'vasco',
    'botafogo', 'fluminense', 'são paulo fc',
    
    # Educação
    'enem', 'vestibular', 'usp', 'unicamp', 'ufrj', 'ufmg',
    'ufrgs', 'ufpr', 'ufsc', 'ufba', 'unb', 'unifesp',
    
    # Documentos
    'cpf', 'cnpj', 'rg', 'carteira de trabalho', 'ctps',
    'inss', 'fgts', 'pis', 'pasep'
]

# Compilar regex de termos brasileiros
BRASIL_PATTERNS = [
    re.compile(r'\b' + re.escape(termo) + r'\b', re.IGNORECASE)
    for termo in ESTADOS_BRASILEIROS + CIDADES_GRANDES + TERMOS_BRASILEIROS
]

# ============================================
# INDICADORES NEGATIVOS (remover se tiver)
# ============================================

PAISES_ESTRANGEIROS = [
    # Europa
    'portugal', 'português', 'portuguesa', 'portugueses', 'portuguesas',
    'espanha', 'espanhol', 'espanhola', 'madrid', 'barcelona',
    'frança', 'francês', 'francesa', 'paris', 'lyon',
    'alemanha', 'alemão', 'alemã', 'berlim', 'munique',
    'itália', 'italiano', 'italiana', 'roma', 'milão',
    'inglaterra', 'inglês', 'inglesa', 'londres', 'manchester',
    'reino unido', 'britânico', 'britânica',
    'holanda', 'países baixos', 'holandês', 'amsterdã',
    'bélgica', 'belga', 'bruxelas',
    'suíça', 'suíço', 'suíça', 'genebra', 'zurique',
    'áustria', 'austríaco', 'viena',
    'grécia', 'grego', 'grega', 'atenas',
    'rússia', 'russo', 'russa', 'moscou',
    'polônia', 'polonês', 'varsóvia',
    'suécia', 'sueco', 'estocolmo',
    'noruega', 'norueguês', 'oslo',
    'dinamarca', 'dinamarquês', 'copenhague',
    'finlândia', 'finlandês', 'helsinque',
    'irlanda', 'irlandês', 'dublin',
    'escócia', 'escocês', 'edimburgo',
    
    # América (exceto Brasil)
    'estados unidos', 'norte-americano', 'norte-americana', 'americano',
    'eua', 'usa', 'washington', 'nova york', 'new york', 'los angeles',
    'califórnia', 'texas', 'flórida', 'chicago', 'boston', 'miami',
    'argentina', 'argentino', 'argentina', 'buenos aires',
    'chile', 'chileno', 'chilena', 'santiago',
    'uruguai', 'uruguaio', 'uruguaia', 'montevidéu',
    'paraguai', 'paraguaio', 'paraguaia', 'assunção',
    'colômbia', 'colombiano', 'colombiana', 'bogotá',
    'venezuela', 'venezuelano', 'venezuelana', 'caracas',
    'peru', 'peruano', 'peruana', 'lima',
    'bolívia', 'boliviano', 'boliviana', 'la paz',
    'equador', 'equatoriano', 'quito',
    'méxico', 'mexicano', 'mexicana', 'cidade do méxico',
    'canadá', 'canadense', 'toronto', 'vancouver', 'montreal',
    'cuba', 'cubano', 'cubana', 'havana',
    
    # Ásia
    'japão', 'japonês', 'japonesa', 'tóquio', 'osaka',
    'china', 'chinês', 'chinesa', 'pequim', 'xangai',
    'coreia', 'coreano', 'coreana', 'seul',
    'índia', 'indiano', 'indiana', 'nova délhi', 'mumbai',
    'tailândia', 'tailandês', 'bangkok',
    'vietnã', 'vietnamita', 'hanói',
    'indonésia', 'indonésio', 'jacarta',
    'filipinas', 'filipino', 'manila',
    'malásia', 'malaio', 'kuala lumpur',
    'cingapura', 'singapura',
    'taiwan', 'taiwanês',
    'hong kong',
    
    # Oceania
    'austrália', 'australiano', 'australiana', 'sydney', 'melbourne',
    'nova zelândia', 'neozelandês', 'auckland',
    
    # África
    'áfrica do sul', 'sul-africano', 'joanesburgo',
    'egito', 'egípcio', 'cairo',
    'marrocos', 'marroquino', 'casablanca',
    'nigéria', 'nigeriano', 'lagos',
    'quênia', 'queniano', 'nairobi',
    
    # Oriente Médio
    'israel', 'israelense', 'tel aviv', 'jerusalém',
    'arábia saudita', 'saudita', 'riad',
    'emirados árabes', 'dubai', 'abu dhabi',
    'irã', 'iraniano', 'teerã',
    'iraque', 'iraquiano', 'bagdá',
    'turquia', 'turco', 'turca', 'istambul', 'ancara'
]

# Termos que indicam conteúdo estrangeiro
TERMOS_ESTRANGEIROS = [
    # Ortografia de Portugal
    'actriz', 'actor', 'acção', 'direcção', 'selecção',
    'óptimo', 'baptismo', 'facto', 'contacto',
    
    # Instituições estrangeiras
    'premier league', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
    'nba', 'nfl', 'mlb', 'nhl',
    'oscar', 'grammy', 'emmy', 'tony award',
    'harvard', 'mit', 'stanford', 'oxford', 'cambridge',
    'nasa', 'cia', 'fbi', 'nsa',
    'união europeia', 'parlamento europeu', 'otan', 'nato',
    
    # Moedas estrangeiras
    'dólar', 'euro', 'libra', 'iene', 'yuan', 'peso',
    
    # Expressões em outros idiomas
    'the ', ' of ', ' and ', ' the', ' for ', ' with ',
    'el ', 'la ', 'los ', 'las ', 'del ', 'der ', 'die ', 'das ',
    'le ', 'les ', 'des ', 'du '
]

# Compilar regex de termos estrangeiros
ESTRANGEIRO_PATTERNS = [
    re.compile(r'\b' + re.escape(termo) + r'\b', re.IGNORECASE)
    for termo in PAISES_ESTRANGEIROS + TERMOS_ESTRANGEIROS
]

# ============================================
# FUNÇÕES DE FILTRO
# ============================================

def tem_indicador_brasileiro(texto: str) -> bool:
    """Verifica se o texto tem algum indicador brasileiro."""
    for pattern in BRASIL_PATTERNS:
        if pattern.search(texto):
            return True
    return False

def tem_indicador_estrangeiro(texto: str) -> bool:
    """Verifica se o texto tem algum indicador estrangeiro."""
    for pattern in ESTRANGEIRO_PATTERNS:
        if pattern.search(texto):
            return True
    return False

def linha_valida(linha: str) -> tuple[bool, str]:
    """
    Avalia se uma linha deve ser mantida.
    Retorna (manter, motivo)
    """
    linha = linha.strip()
    
    # Linha vazia ou muito curta
    if len(linha) < 50:
        return False, "muito_curta"
    
    # Poucas palavras
    palavras = linha.split()
    if len(palavras) < 8:
        return False, "poucas_palavras"
    
    # Tem indicador estrangeiro? REMOVER
    if tem_indicador_estrangeiro(linha):
        return False, "estrangeiro"
    
    # Tem indicador brasileiro? MANTER
    if tem_indicador_brasileiro(linha):
        return True, "brasileiro"
    
    # Ambíguo - verificar proporção de letras
    letras = sum(1 for c in linha if c.isalpha())
    total = len(linha.replace(' ', ''))
    if total > 0 and letras / total < 0.7:
        return False, "pouco_texto"
    
    # Ambíguo mas parece ok - MANTER com cautela
    # (podemos ser mais agressivos e remover aqui também)
    return False, "ambiguo"  # Mudança: agora remove ambíguos também

def processar_corpus(input_path: str, output_path: str):
    """Processa o corpus e salva versão limpa."""
    
    print(f"📂 Lendo: {input_path}")
    
    stats = Counter()
    linhas_boas = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        total = 0
        for linha in f:
            total += 1
            manter, motivo = linha_valida(linha)
            stats[motivo] += 1
            
            if manter:
                linhas_boas.append(linha)
            
            if total % 500000 == 0:
                print(f"  Processadas: {total:,} linhas...")
    
    print(f"\n📊 Estatísticas de Filtragem:")
    print(f"  Total original: {total:,}")
    for motivo, count in stats.most_common():
        pct = count / total * 100
        status = "✅" if motivo == "brasileiro" else "❌"
        print(f"  {status} {motivo}: {count:,} ({pct:.1f}%)")
    
    print(f"\n✅ Mantidas: {len(linhas_boas):,} ({len(linhas_boas)/total*100:.1f}%)")
    
    # Salvar
    print(f"\n💾 Salvando: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(linhas_boas)
    
    # Tamanho do arquivo
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"📦 Tamanho: {size_mb:.1f} MB")
    
    return len(linhas_boas)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    input_file = "data/sovereign/corpus_v3.txt"
    output_file = "data/sovereign/corpus_v11_brasil.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("=" * 60)
    print("🇧🇷 FILTRO V11 - CORPUS 100% BRASILEIRO")
    print("=" * 60)
    
    processar_corpus(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("✅ CONCLUÍDO!")
    print("=" * 60)
    print(f"\nPróximos passos:")
    print(f"  1. Verificar amostra: head -20 {output_file}")
    print(f"  2. Retreinar tokenizer com corpus limpo")
    print(f"  3. Tokenizar e treinar modelo do zero")