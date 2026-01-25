
import glob
import os
import json
import struct
from tokenizers import ByteLevelBPETokenizer

def build_assets():
    print("🚀 Iniciando Build de Assets RÁPIDO (Python Engine)...")
    
    # === CONFIG ===
    INPUT_DIR = "data/tokenizer_full_input_cleaned"
    OUTPUT_DIR = "data"
    VOCAB_SIZE = 65536
    MIN_FREQ = 5
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    # Ordenar para determinismo
    files.sort()
    
    print(f"  Files: {len(files)}")
    
    # === 1. TREINAR TOKENIZER ===
    print("\n[1/3] Treinando Tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    
    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS
    )
    
    # Validar IDs
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    print(f"  BOS ID: {bos_id}")
    print(f"  EOS ID: {eos_id}")
    
    # === 2. EXPORTAR TOKENIZER (Custom Format) ===
    print("\n[2/3] Salvando data/tokenizer.json (Formato Rust)...")
    
    # Extrair vocab e merges
    vocab_map = tokenizer.get_vocab()
    # Merges não são expostos diretamente na API Python fácil, 
    # mas podemos salvar e reler.
    os.makedirs("data/temp_tok_build", exist_ok=True)
    tokenizer.save_model("data/temp_tok_build")
    
    # Ler merges.txt
    merges_list = []
    with open("data/temp_tok_build/merges.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        start = 1 if lines[0].startswith("#") else 0
        for line in lines[start:]:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                if a in vocab_map and b in vocab_map:
                    merges_list.append((vocab_map[a], vocab_map[b]))
    
    # Construir id_to_token (bytes)
    sorted_vocab = sorted(vocab_map.items(), key=lambda x: x[1])
    id_to_token = []
    for token_str, _ in sorted_vocab:
        # Codificar UTF-8 (compatível com Ġ chars do ByteLevel)
        id_to_token.append(list(token_str.encode("utf-8")))
        
    st_map = {t: vocab_map[t] for t in SPECIAL_TOKENS if t in vocab_map}
    
    final_struct = {
        "id_to_token": id_to_token,
        "merges": merges_list,
        "special_tokens": st_map
    }
    
    # === 3. TOKENIZAR DATASET (Parallel) ===
    print("\n[3/3] Gerando data/train.bin (Parallel Batch)...")
    
    out_bin = os.path.join(OUTPUT_DIR, "train.bin")
    
    # Read all files into memory (1.6GB)
    print("  Lendo arquivos para memória...")
    texts = []
    # Usar tqdm se possível, mas simples loop ok
    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f_in:
            texts.append(f_in.read())
            
    print(f"  Tokenizando {len(texts)} arquivos em paralelo...")
    # encode_batch usa paralelismo Rust
    encodings = tokenizer.encode_batch(texts)
    
    print("  Escrevendo binário...")
    with open(out_bin, "wb") as f_out:
        f_out.write(struct.pack("<Q", 0))
        
        total_tokens = 0
        import array
        
        for enc in encodings:
            ids = enc.ids
            
            # Write BOS
            f_out.write(struct.pack("<H", bos_id))
            
            # Write IDs
            arr = array.array('H', ids)
            if os.sys.byteorder == 'big':
                arr.byteswap()
            arr.tofile(f_out)
            
            # Write EOS
            f_out.write(struct.pack("<H", eos_id))
            
            count = 1 + len(ids) + 1
            total_tokens += count
            
        f_out.seek(0)
        f_out.write(struct.pack("<Q", total_tokens))
        f_out.flush()
        
    print(f"\n  ✅ Finalizado!")
    print(f"  Total Tokens: {total_tokens} ({total_tokens/1e6:.1f}M)")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree("data/temp_tok_build")
    except:
        pass

if __name__ == "__main__":
    build_assets()
