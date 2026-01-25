
import glob
import os
import json
import struct
from tokenizers import ByteLevelBPETokenizer, Tokenizer

def build_assets_robust():
    print("🚀 Iniciando Build de Assets ROBUSTO (Python Engine)...")
    
    # === CONFIG ===
    INPUT_DIR = "data/tokenizer_full_input_cleaned"
    OUTPUT_DIR = "data"
    VOCAB_SIZE = 65536
    MIN_FREQ = 5
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    BATCH_SIZE = 1 # Arquivos por batch (MÁXIMA ROBUSTEZ - 1 por vez com flush)
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    files.sort()
    
    print(f"  Files: {len(files)}")
    
    # === 1. TREINAR OU CARREGAR TOKENIZER ===
    tokenizer_path = "data/temp_tok_build"
    # Verificar validade do cache
    has_cache = os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "vocab.json"))
    
    if has_cache:
        print("\n[1/3] Carregando Tokenizer existente (skip training)...")
        tokenizer = ByteLevelBPETokenizer.from_file(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt")
        )
    else:
        print("\n[1/3] Treinando Tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=files,
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQ,
            show_progress=True,
            special_tokens=SPECIAL_TOKENS
        )
        os.makedirs("data/temp_tok_build", exist_ok=True)
        tokenizer.save_model("data/temp_tok_build")
        
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    print(f"  BOS ID: {bos_id}")
    print(f"  EOS ID: {eos_id}")
    
    # === 2. EXPORTAR TOKENIZER ===
    print("\n[2/3] Salvando data/tokenizer.json...")
    vocab_map = tokenizer.get_vocab()
    merges_list = []
    
    # Se carregou do cache, ler de lá. Se treinou, também ler de lá (salvou antes).
    with open("data/temp_tok_build/merges.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        start = 1 if lines[0].startswith("#") else 0
        for line in lines[start:]:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                if a in vocab_map and b in vocab_map:
                    merges_list.append((vocab_map[a], vocab_map[b]))
                    
    sorted_vocab = sorted(vocab_map.items(), key=lambda x: x[1])
    id_to_token = [list(token.encode("utf-8")) for token, _ in sorted_vocab]
    st_map = {t: vocab_map[t] for t in SPECIAL_TOKENS if t in vocab_map}
    
    final_struct = {
        "id_to_token": id_to_token,
        "merges": merges_list,
        "special_tokens": st_map
    }
    
    with open(os.path.join(OUTPUT_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(final_struct, f, ensure_ascii=False)
        
    # === 3. TOKENIZAR DATASET (Batched & Flushed) ===
    print("\n[3/3] Gerando data/train.bin (Batched)...")
    out_bin = os.path.join(OUTPUT_DIR, "train.bin")
    
    import array
    total_tokens = 0
    
    with open(out_bin, "wb") as f_out:
        f_out.write(struct.pack("<Q", 0))
        
        # Process in chunks of BATCH_SIZE files
        for i in range(0, len(files), BATCH_SIZE):
            batch_files = files[i : i + BATCH_SIZE]
            current_file = batch_files[0]
            print(f"  [{i+1}/{len(files)}] Processando {os.path.basename(current_file)}...")
            
            texts = []
            for fp in batch_files:
                with open(fp, "r", encoding="utf-8", errors="replace") as f_in:
                    texts.append(f_in.read())
            
            # Encode batch
            encodings = tokenizer.encode_batch(texts)
            
            # Write batch
            tokens_in_batch = 0
            for enc in encodings:
                ids = enc.ids
                f_out.write(struct.pack("<H", bos_id))
                arr = array.array('H', ids)
                if os.sys.byteorder == 'big': arr.byteswap()
                arr.tofile(f_out)
                f_out.write(struct.pack("<H", eos_id))
                
                count = 1 + len(ids) + 1
                total_tokens += count
                tokens_in_batch += count
            
            f_out.flush()
            
            # Clear memory
            del texts
            del encodings
            
            # Log periodicamente
            if (i+1) % 10 == 0:
                print(f"    - Tokens acumulados: {total_tokens/1e6:.1f}M")
            
        f_out.seek(0)
        f_out.write(struct.pack("<Q", total_tokens))
        
    print(f"\n  ✅ Finalizado!")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Tamanho estimado: {total_tokens * 2 / 1024 / 1024:.1f} MB")
    
    # Cleanup (Optional: Keep temp for debugging)
    # import shutil
    # try:
    #     shutil.rmtree("data/temp_tok_build")
    # except:
    #     pass

if __name__ == "__main__":
    build_assets_robust()
