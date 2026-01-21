import os
import glob
import re
import shutil
import hashlib
import unicodedata
import html  # <--- V14: Biblioteca nativa para decodificar &amp;, &quot;, etc.

# ==============================================================================
# CONFIGURAÇÕES E REGEX COMPILADOS (ENGINE V14)
# ==============================================================================

# 1. Configurações de Qualidade
MIN_LINE_LENGTH = 25 
MAX_SYMBOL_RATIO = 0.30 

# 2. Regex de Limpeza Profunda ("Vassoura Hidráulica")
RE_HTML_TAGS = re.compile(r'<[^>]+>') 
RE_CURLY_BRACES = re.compile(r'\{\{.*?\}\}') 
RE_WIKI_LINK = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]') 
RE_WIKI_REF = re.compile(r'\[\d+\]') 
RE_PX_DIMENSIONS = re.compile(r'\b\d+x?\d*px\b', re.IGNORECASE) 
RE_CATEGORY_LINK = re.compile(r'\[\[(Categoria|Category|File|Image|Imagem|Ficheiro|Arquivo):.*?\]\]', re.IGNORECASE)
RE_INVISIBLE_CHARS = re.compile(r'[\u200b\u200c\u200d\ufeff\ufffd]')
RE_WIKI_BOLD_ITALIC = re.compile(r"'{2,3}") 
RE_MULTI_SPACE = re.compile(r'\s+')
RE_URLS = re.compile(r'http\S+')

# --- V14: Novos Regex de Polimento Linguístico ---
# Remove hifens soltos de OCR (ex: "cons- tituição" -> "constituição")
RE_OCR_HYPHEN = re.compile(r'(?<=[a-zà-ú])-\s+(?=[a-zà-ú])') 
# Garante espaço em termos jurídicos (ex: "Art.1º" -> "Art. 1º")
RE_LEGAL_SPACE = re.compile(r'(Art\.|§|Inciso)\s*(\d)', re.IGNORECASE)

# 3. Regex para Detecção de Lixo
RE_UPLOAD_LOG = re.compile(r'^\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]{2} \d{4} .*? carregado.*$', re.IGNORECASE)
RE_FILE_PREFIX = re.compile(r'^(File|Ficheiro|Arquivo|Image|Imagem):', re.IGNORECASE)

# 4. Mapa de Correção do Planalto
PLANALTO_MAP = {
    'ę': 'ê', 'ş': 'º', 'ŕ': 'à', 'ă': 'ã', 'č': 'è', 'ľ': 'ç',
    'î': 'í', 'đ': 'ð', '': '' 
}

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def fix_planalto_corruption(text):
    for bad, good in PLANALTO_MAP.items():
        text = text.replace(bad, good)
    return text

def get_line_hash(line):
    return hashlib.md5(line.encode('utf-8')).hexdigest()

def is_code_or_script_line(line):
    # JavaScript/CSS indicators
    js_indicators = [
        'var ', 'function ', 'mw.', 'document.', '.innerHTML', 'getElementById',
        'mw.config', 'mw.loader', '.load(', '.find(', '.attr(',
        'if (', 'if(', 'return;', '});', '{return;', '};',
        'position:absolute', 'float:left', 'text-align:', 'background-color:',
        'width:', 'padding:', 'margin:', '/*', '*/', '//',
        'self.ws_messages', '.messages[', 'ws_messages',
        'progress_V', 'progress_T', 'progress_C', 'progress_MS', 'progress_OCR',
        'progress_X', 'progress_L', 'progress_A', 'progress_MSA',
        'corr_list', 'corr_one', 'corr_many', 'iwtrans', 'compare_texts'
    ]
    
    # Lua/MediaWiki Module indicators
    lua_indicators = [
        'local ', 'function(', 'return ', 'elseif ', 'then$',
        'table.insert', 'table.concat', 'table.remove', 'table.sort',
        'pairs(', 'ipairs(', 'in pairs', 'in ipairs',
        'for _, ', 'for i, ', 'for k, ',
        'string.match', 'string.sub', 'string.find', 'string.gsub',
        'string.format', 'string.lower', 'string.upper', 'string.len',
        ':match(', ':sub(', ':gsub(', ':find(', ':lower(', ':upper(',
        'math.sin', 'math.cos', 'math.tan', 'math.floor', 'math.ceil',
        'math.abs', 'math.max', 'math.min', 'math.random',
        'tostring(', 'tonumber(', 'type(',
        ' .. ', '~= ', '== ', ' or ', ' and ', ' #',
        'require(', "require('", 'require("', 'Módulo:', 'Module:', 'sandbox',
        'frame:', 'frame.', 'getParent()', '.args', 'args[',
        'mList.', 'mTableTools.', 'mMessageBox.',
        "cfg['", 'cfg["', 'cfg.', 'pargs', 'knownflag',
        '#cats', 'cats[', 'boxArgs.', 'tStyles',
        'titleObj.', 'protectionLevels', 'prefixedText',
        'roundPrec', 'fetchWikidata', 'mergePoints', 'yesno(',
        'getConfig(', 'makeWikitextError', 'makeLink(',
        'formatDate', 'parseDate', 'checkType(',
        'trimall', 'removeBlanks', 'compressSparseArray',
        "-- ", "--[[", "]]--",
        ' = function(', ' = {}', ' = {',
    ]
    
    all_indicators = js_indicators + lua_indicators
    for indicator in all_indicators:
        if indicator in line:
            return True
    
    if re.search(r"'[a-zA-Z_]+'\s*:\s*['\"]", line): return True
    if re.search(r'"[a-zA-Z_]+"\s*:\s*[\'"]', line): return True
    if re.search(r'^local\s+\w+\s*=', line): return True
    
    stripped = line.rstrip()
    if stripped.endswith(' end') or stripped == 'end': return True
    if stripped.endswith(']]') and '[[' not in line: return True
        
    code_chars = sum(1 for c in line if c in '{}();=[]')
    if len(line) > 0 and (code_chars / len(line)) > 0.15: return True
    return False

def is_wiki_garbage_line(line):
    stripped = line.strip()
    if not stripped: return True
    
    if stripped.startswith("|") or stripped.startswith("!"): return True
    if "}}" in stripped and "=" in stripped: return True
    if RE_FILE_PREFIX.match(stripped): return True
    if RE_UPLOAD_LOG.match(stripped): return True
    if re.match(r"^'[\w\?]+':", stripped): return True
    
    # V14: Lista expandida com termos do Planalto e Wikisource
    mediawiki_patterns = [
        # Wikisource / Meta
        ':Categoria:', 'Votações de eliminação', 'Documentação incompleta',
        'Esplanada · ', 'Café dos novatos', 'Pedidos a administradores',
        'Especial:Statistics', 'Para mais informações de como pesquisar',
        'Se você não encontrar', 'Se possível, carregue arquivos',
        'Use o formulário abaixo', 'Os formatos ideais são',
        'Para mais estatísticas sobre', 'Originais (:Categoria:',
        '(?) Não revisadas', 'Existência confirmada',
        'Nota: O Wikilivros é', 'GNU Free Documentation',
        'você só pode fazer isso se for proprietário',
        'lista de páginas vigiadas', 'Modificações futuras nesta página',
        'Se você quiser remover a página', 'clique em "desinteressar-se"',
        'Exposição combinada de carregamento', 'Não foi possível reverter',
        'Você está prestes a eliminar permanentemente',
        'Por favor, confirme que você realmente',
        'Ocorreu um erro de sintaxe', 'MySQL retornou',
        'É possível exportar o texto e o histórico',
        'Trancar a base de dados suspenderá',
        'Utilizando o seguinte formulário poderá renomear',
        'Seguiu uma hiperligação para um artigo que ainda não existe',
        'Consulte a página de ajuda', 'projetos irmãos do Wikinotícias',
        'Commons (imagens e media)', 'Wikivoyage (guia de viagem)',
        'Wikispecies (directório', 'Wikinotícias não possui uma página',
        'Especial:Import', 'software MediaWiki',
        
        # V14: Planalto Garbage
        'Este texto não substitui o publicado', 
        '(Redação dada pela Lei', '(Revogado pela Lei',
        'Republicado por ter saído com incorreção',
        'Mensagem de veto', 'Vide Lei nº', 'Vide Decreto nº',
        'Subchefia para Assuntos Jurídicos', 'Presidência da República',
        'Casa Civil',
        
        # V14: Notícias/Livros
        'Página seguinte', 'Página anterior', 'Sumário:',
        'Questão 1', 'Questão 2', 'Resposta:',
        'Fonte: Agência', 'Fonte: Wiki', 'Foto:',
        '---SÃO PAULO', '---BRASÍLIA', '---RIO DE JANEIRO'
    ]
    for pattern in mediawiki_patterns:
        if pattern in stripped: return True
    
    if re.match(r'^[a-z]{2,3}:[A-Z][a-zà-ú]+:', stripped): return True
    if re.search(r'\$[1-9]', stripped): return True
    if 'link=' in stripped: return True
    if re.search(r'\bWP:[A-Z]', stripped): return True
    if re.search(r'\d{1,2}h\d{2}min de \d+ de \w+ de \d{4} \(UTC\)', stripped): return True
    if 'Prazo mínimo do debate:' in stripped: return True
    if 'período de consenso' in stripped: return True
    
    junk_prefixes = [
        "assentamento_tipo", "Aviso aos leitores", "thumb", "miniaturadaimagem",
        "direita", "esquerda", "centro", "Category:", "Categoria:", 
        "Especial:", "Wikipédia:", "Ajuda:"
    ]
    for prefix in junk_prefixes:
        if stripped.lower().startswith(prefix.lower()): return True
    
    if '\ufffd' in stripped: return True
        
    return False

def calculate_symbol_ratio(line):
    if not line: return 0
    weird_symbols = sum(1 for c in line if c in '|/_*{}[]<>=;():#@&^%$!~`\\')
    return weird_symbols / len(line)

def clean_line_content(line):
    """Pipeline principal de higienização da string (V14 Final)."""
    
    # V14: Decodificação de Entidades HTML (&nbsp; -> " ", &quot; -> ", etc.)
    # Isso tem que vir ANTES da normalização unicode
    line = html.unescape(line)
    
    # 1. Normalização Unicode
    line = unicodedata.normalize('NFKC', line)
    
    # 2. Remoção Cirúrgica
    line = RE_INVISIBLE_CHARS.sub('', line)
    line = RE_HTML_TAGS.sub('', line)
    line = RE_CURLY_BRACES.sub('', line)
    line = RE_CATEGORY_LINK.sub('', line)
    
    # 3. Resolução de Links e Formatação
    line = RE_WIKI_LINK.sub(r'\1', line)
    line = RE_WIKI_BOLD_ITALIC.sub('', line)
    
    # V14: Fix de OCR e Jurídico
    # Junta palavras separadas por hífen e espaço (ex: "cons- tituição")
    line = RE_OCR_HYPHEN.sub('', line) 
    # Coloca espaço em Art. e Parágrafos (ex: "Art.5" -> "Art. 5")
    line = RE_LEGAL_SPACE.sub(r'\1 \2', line)
    
    # 4. Limpezas Legadas
    line = RE_WIKI_REF.sub('', line)
    line = RE_PX_DIMENSIONS.sub('', line)
    line = RE_URLS.sub('', line)
    line = re.sub(r'\bthumb\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\bminiaturadaimagem\b', '', line, flags=re.IGNORECASE)
    
    # Limpeza de orfãos
    line = line.replace('{{', '').replace('}}', '')
    line = line.replace('[[', '').replace(']]', '')
    line = line.replace('{|', '').replace('|}', '')
    line = line.replace('|', ' ')
    
    # 5. Normalização Final
    line = RE_MULTI_SPACE.sub(' ', line).strip()
    
    return line

# ==============================================================================
# PROCESSAMENTO DE ARQUIVOS
# ==============================================================================

def process_file(filepath, output_dir, global_seen_hashes):
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        cleaned_lines = []
        is_lei_file = filename.startswith('lei_')
        
        for line in lines:
            # A. Correção Planalto
            if is_lei_file or 'ę' in line or 'ş' in line:
                line = fix_planalto_corruption(line)
            
            stripped = line.strip()
            if not stripped: continue
            
            # B. Filtros de Lixo
            if is_code_or_script_line(stripped): continue
            if is_wiki_garbage_line(stripped): continue
            
            # C. Limpeza Profunda
            cleaned = clean_line_content(stripped)
            if not cleaned: continue
            
            # D. Qualidade
            if len(cleaned) < MIN_LINE_LENGTH: continue
            if calculate_symbol_ratio(cleaned) > MAX_SYMBOL_RATIO: continue
            
            # E. Deduplicação (Hash)
            line_hash = get_line_hash(cleaned)
            if line_hash in global_seen_hashes:
                continue
            
            global_seen_hashes.add(line_hash)
            cleaned_lines.append(cleaned)
            
        if cleaned_lines:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_lines))
            return True
        return False
        
    except Exception as e:
        print(f"Erro no arquivo {filepath}: {e}")
        return False

def main():
    input_dir = os.path.join("data", "tokenizer_full_input")
    output_dir = os.path.join("data", "tokenizer_full_input_cleaned")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"--- INICIANDO LIMPEZA V14 (FINAL RELEASE) ---")
    print(f"Arquivos: {len(files)}")
    print(f"Features V14: HTML Unescape + OCR Fix + Legal Format Fix")
    
    global_seen_hashes = set()
    
    count = 0
    for i, file in enumerate(files):
        if process_file(file, output_dir, global_seen_hashes):
            count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Progresso: {i + 1}/{len(files)} arquivos. (Linhas únicas: {len(global_seen_hashes):,})")
                
    print(f"--- CONCLUÍDO ---")
    print(f"Arquivos válidos: {count}")
    print(f"Total de linhas únicas: {len(global_seen_hashes):,}")

if __name__ == "__main__":
    main()
