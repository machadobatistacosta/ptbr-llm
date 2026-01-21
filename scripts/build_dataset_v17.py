"""
build_dataset_v17.py - Gerador de Dataset V17
Gera JSONL e train.bin a partir do dataset limpo V14.

Uso:
  python scripts/build_dataset_v17.py

Saída:
  - data/dataset.jsonl  (formato {"text": "..."})
  - data/train.bin      (tokens uint16)
"""

import os
import glob
import json
import struct
import random
from collections import Counter

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

INPUT_DIR = os.path.join("data", "tokenizer_full_input_cleaned")
OUTPUT_JSONL = os.path.join("data", "dataset.jsonl")
OUTPUT_RAW = os.path.join("data", "train_raw.txt")  # Texto concatenado para tokenização
OUTPUT_BIN = os.path.join("data", "train.bin")

# Qualidade
MIN_CHARS_PER_DOC = 100  # Documentos muito curtos são descartados
MAX_NUMERIC_RATIO = 0.3  # Se mais de 30% são dígitos, descarta

# ==============================================================================
# FUNÇÕES DE VALIDAÇÃO
# ==============================================================================

def is_quality_text(text):
    """Valida se o texto é de qualidade suficiente."""
    if len(text) < MIN_CHARS_PER_DOC:
        return False
    
    # Conta proporção de dígitos
    digits = sum(1 for c in text if c.isdigit())
    if len(text) > 0 and (digits / len(text)) > MAX_NUMERIC_RATIO:
        return False
    
    # Verifica se tem pelo menos algumas palavras
    words = text.split()
    if len(words) < 5:
        return False
    
    return True

def get_stats(texts):
    """Calcula estatísticas do dataset."""
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    return {
        "docs": len(texts),
        "chars": total_chars,
        "words": total_words,
        "avg_words_per_doc": total_words / len(texts) if texts else 0
    }

# ==============================================================================
# PROCESSAMENTO PRINCIPAL
# ==============================================================================

def main():
    print("=" * 60)
    print("  BUILD DATASET V17")
    print("=" * 60)
    
    # 1. Leitura dos arquivos limpos
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"\n📁 Arquivos encontrados: {len(files)}")
    
    all_texts = []
    rejected = {"short": 0, "numeric": 0, "few_words": 0}
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Divide em parágrafos/documentos
            paragraphs = content.split('\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(para) < MIN_CHARS_PER_DOC:
                    rejected["short"] += 1
                    continue
                
                digits = sum(1 for c in para if c.isdigit())
                if len(para) > 0 and (digits / len(para)) > MAX_NUMERIC_RATIO:
                    rejected["numeric"] += 1
                    continue
                
                words = para.split()
                if len(words) < 5:
                    rejected["few_words"] += 1
                    continue
                
                all_texts.append(para)
                
        except Exception as e:
            print(f"⚠️ Erro lendo {filepath}: {e}")
    
    print(f"\n📊 Textos válidos: {len(all_texts):,}")
    print(f"   ❌ Rejeitados (curto): {rejected['short']:,}")
    print(f"   ❌ Rejeitados (numérico): {rejected['numeric']:,}")
    print(f"   ❌ Rejeitados (poucas palavras): {rejected['few_words']:,}")
    
    # 2. Embaralhar para melhor distribuição
    print("\n🔀 Embaralhando dados...")
    random.seed(42)
    random.shuffle(all_texts)
    
    # 3. Estatísticas
    stats = get_stats(all_texts)
    print(f"\n📈 Estatísticas:")
    print(f"   Documentos: {stats['docs']:,}")
    print(f"   Total Chars: {stats['chars']:,}")
    print(f"   Total Words: {stats['words']:,}")
    print(f"   Média palavras/doc: {stats['avg_words_per_doc']:.1f}")
    
    # 4. Gerar JSONL
    print(f"\n📄 Gerando {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for text in all_texts:
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    jsonl_size = os.path.getsize(OUTPUT_JSONL) / (1024 * 1024)
    print(f"   ✅ Tamanho: {jsonl_size:.2f} MB")
    
    # 5. Gerar texto raw (para tokenização externa)
    print(f"\n📄 Gerando {OUTPUT_RAW}...")
    with open(OUTPUT_RAW, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    raw_size = os.path.getsize(OUTPUT_RAW) / (1024 * 1024)
    print(f"   ✅ Tamanho: {raw_size:.2f} MB")
    
    # 6. Resumo final
    print("\n" + "=" * 60)
    print("  ✅ BUILD COMPLETO")
    print("=" * 60)
    print(f"\n  Saídas geradas:")
    print(f"    - {OUTPUT_JSONL} ({jsonl_size:.2f} MB)")
    print(f"    - {OUTPUT_RAW} ({raw_size:.2f} MB)")
    print(f"\n  📊 Dataset: {stats['docs']:,} docs | {stats['words']:,} palavras")
    print(f"\n  ⚡ Próximo passo: Tokenizar train_raw.txt para train.bin")

if __name__ == "__main__":
    main()
