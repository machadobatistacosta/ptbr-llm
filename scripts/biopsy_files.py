import os

# Lista de suspeitos (baseada no seu relatório anterior)
SUSPECTS = [
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

def biopsy(filepath):
    print(f"\n🔍 BIÓPSIA EM: {filepath}")
    print("-" * 60)
    
    if not os.path.exists(filepath):
        print("❌ Arquivo não encontrado (já foi movido?)")
        return

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Procura Mojibake (UTF-8 lido como Latin-1)
        # Ex: "coração" -> "coraÃ§Ã£o"
        mojibake_samples = []
        if "Ã" in content:
            # Pega contexto de 20 chars ao redor do erro
            import re
            matches = list(re.finditer(r".{0,20}(Ã[£©³ºµ§\s¢]+).{0,20}", content))
            for m in matches[:5]: # Mostra os 5 primeiros
                mojibake_samples.append(m.group(0).replace('\n', ' '))
        
        if mojibake_samples:
            print("🔴 ENCODING QUEBRADO DETECTADO:")
            for sample in mojibake_samples:
                print(f"   ...{sample}...")
        else:
            print("✅ Encoding parece OK (falso positivo?)")
            
        # Mostra as primeiras 3 linhas para ver se é inglês ou lixo
        print("\n📄 INÍCIO DO ARQUIVO:")
        print(content[:300].replace('\n', ' | '))

    except Exception as e:
        print(f"❌ Erro ao ler: {e}")

if __name__ == "__main__":
    print("🚑 INICIANDO BIÓPSIA DE ARQUIVOS CORROMPIDOS")
    for s in SUSPECTS:
        biopsy(s)