
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np
import struct
import os

# ============================================
# CONFIGURAÇÃO
# ============================================
OUTPUT_DIR = "/kaggle/working/data"
TOKENIZER_PATH = "/kaggle/working/data/tokenizer.json"
MAX_TOKENS = 5_000_000_000  # 5B tokens (meta fase 2)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega tokenizer (se não existir, usa um dummy para teste local/debug)
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
except:
    print(f"⚠️ Tokenizer não encontrado em {TOKENIZER_PATH}. Certifique-se de estar no ambiente correto.")
    # Fallback ou Exit

# ============================================
# OPÇÃO 1: OSCAR (Recomendado - Mais limpo)
# ============================================
def download_oscar():
    """
    OSCAR 23.01 PT (~5-8GB texto)
    ~2-3B tokens após tokenização
    """
    print("📚 Carregando OSCAR-2301 (Streaming)...")
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        "pt",
        split="train",
        streaming=True,  # NÃO baixa tudo de uma vez
        trust_remote_code=True
    )
    return dataset

# ============================================
# TOKENIZAÇÃO + ESCRITA BINÁRIA
# ============================================
def process_and_save(dataset, output_path, max_tokens):
    """
    Tokeniza em streaming e salva no formato binário
    compatível com seu dataset.rs
    """
    token_count = 0
    temp_path = output_path + ".tmp"
    
    print(f"🚀 Iniciando processamento...")
    print(f"🎯 Meta: {max_tokens:,} tokens")
    
    with open(temp_path, "wb") as f:
        # Placeholder para header (será preenchido depois)
        f.write(struct.pack("<Q", 0))
        
        for i, example in enumerate(dataset):
            # Pega o texto
            text = example.get("text", "")
            
            # Filtros de qualidade básicos
            if len(text) < 100:        # Muito curto
                continue
            if len(text) > 100_000:    # Muito longo (possível lixo)
                continue
            
            # Tokeniza
            encoded = tokenizer.encode(text)
            tokens = encoded.ids
            
            # Escreve tokens como u16 LE
            for token_id in tokens:
                if token_id < 65536:
                    f.write(struct.pack("<H", token_id))
                    token_count += 1
            
            # Progress
            if i % 10_000 == 0:
                print(f"  📊 Docs: {i:,} | Tokens: {token_count:,} | "
                      f"Progress: {token_count/max_tokens*100:.1f}%")
            
            if token_count >= max_tokens:
                print(f"✅ Meta atingida!")
                break
    
    # Reescreve header com contagem real
    with open(temp_path, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<Q", token_count))
    
    # Renomeia
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)
    
    print(f"\n{'='*50}")
    print(f"✅ Dataset salvo: {output_path}")
    print(f"📊 Total tokens: {token_count:,}")
    print(f"💾 Tamanho: {os.path.getsize(output_path)/1e9:.2f} GB")
    print(f"{'='*50}")
    
    return token_count

# ============================================
# SPLIT TRAIN/VAL
# ============================================
def create_val_split(input_path, train_path, val_path, val_ratio=0.05):
    """
    Separa 5% para validação
    """
    print(f"🔪 Criando Split Train/Val ({val_ratio*100}%)...")
    with open(input_path, "rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        # Cuidado aqui com RAM: Se o dataset for maior que a RAM, 
        # essa abordagem de ler tudo falha. 
        # Para 2GB (4B tokens n, espera, 2B tokens * 2 bytes = 4GB) deve caber em 13GB RAM.
        all_tokens = np.frombuffer(f.read(), dtype=np.uint16)
    
    assert len(all_tokens) == count, f"Count mismatch: Header {count} vs Read {len(all_tokens)}"
    
    val_size = int(count * val_ratio)
    train_size = count - val_size
    
    # Train
    with open(train_path, "wb") as f:
        f.write(struct.pack("<Q", train_size))
        all_tokens[:train_size].tofile(f)
    
    # Val
    with open(val_path, "wb") as f:
        f.write(struct.pack("<Q", val_size))
        all_tokens[train_size:].tofile(f)
    
    print(f"✅ Train: {train_size:,} tokens saved to {train_path}")
    print(f"✅ Val:   {val_size:,} tokens saved to {val_path}")

# ============================================
# EXECUÇÃO
# ============================================
if __name__ == "__main__":
    # Escolhe o dataset
    dataset = download_oscar()
    
    # Processa
    raw_path = os.path.join(OUTPUT_DIR, "raw_ptbr.bin")
    total = process_and_save(
        dataset, 
        raw_path, 
        max_tokens=2_000_000_000  # Começa com 2B para testar
    )
    
    # Split
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")
    create_val_split(raw_path, train_path, val_path)
    
    # Limpeza opcional (comentar se quiser manter o raw)
    # os.remove(raw_path)
    print("\n🎉 Pipeline concluído! Arquivos prontos para treino.")
