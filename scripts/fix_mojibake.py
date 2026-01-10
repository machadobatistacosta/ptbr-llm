import os

# Lista de arquivos para consertar
TARGETS = [
    "data/wikibooks_clean/wiki_000.txt",
    "data/wikinews_clean/wiki_000.txt",
    "data/wiki_clean/wiki_000.txt",
    "data/wiki_clean/wiki_003.txt",
    "data/wiki_clean/wiki_004.txt",
    "data/wiki_clean/wiki_015.txt",
    "data/wiki_clean/wiki_046.txt",
    "data/wiki_clean/wiki_047.txt",
    "data/wiki_clean/wiki_084.txt",
    "data/wiki_clean/wiki_101.txt"
]

def fix_text(text):
    # Tabela de substituição manual (mais segura que ftfy automágico)
    replacements = {
        "Ã£": "ã", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã§": "ç", "Ãª": "ê", "Ã´": "ô", "Ã¡": "á", "Ã¢": "â",
        "Ã ": "à", "Ã¨": "è", "Ã±": "ñ", "Ãµ": "õ", "Ã¶": "ö",
        "Ã¼": "ü", "Ã\x81": "Á", "Ã\x89": "É", "Ã\x8d": "Í", 
        "Ã\x93": "Ó", "Ã\x9a": "Ú", "Ã\x87": "Ç", "Ã\x82": "Â",
        "Ã\x8a": "Ê", "Ã\x94": "Ô", "Ã\x80": "À"
    }
    
    # Faz 2 passadas para garantir casos aninhados
    for _ in range(2):
        for bad, good in replacements.items():
            text = text.replace(bad, good)
            
    return text

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} não encontrado (pulando)")
        return

    print(f"🔧 Consertando: {filepath}...", end=" ")
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        fixed = fix_text(content)
        
        # Só salva se mudou algo
        if fixed != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed)
            print("✅ SALVO")
        else:
            print("⏹️ Sem mudanças")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    print("🛠️ INICIANDO REPARO DE ENCODING (MOJIBAKE)...")
    for t in TARGETS:
        process_file(t)
    print("🏁 Concluído.")